import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision
from opencv_transforms import transforms
from dataset import dct_transforms
import datetime
import os
import time
import argparse

from dataset import dataset_cv2
from configs import wpt_config as wpt_config
from model import wpt_net
from dataset import wpt_transforms
from utils.statistics import ProgressMeter, AverageMeter, accuracy
from utils.utils import *

parser = argparse.ArgumentParser(description="WPT")
parser.add_argument('--arch', type=str, default='mobilenet')
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=4e-5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--lr_decay', type=str, default='cos')
parser.add_argument('--lr_decay_gamma', type=float, default=0.98)
parser.add_argument('--need_log', type=str_to_bool, default=False)
parser.add_argument('--log_desc_prefix', type=str, default='test_wpt')
parser.add_argument('--root_dir', type=str, default='~/caltech256')
parser.add_argument('--random_seed', type=int, default=None)
parser.add_argument('--num_classes', type=int, default=257)
parser.add_argument('--input_size', type=int, default=448)
parser.add_argument('--wpt_level', type=int, default=2)
parser.add_argument('--wpt_config_index', type=int, default=0)
parser.add_argument('--use_attention', type=str_to_bool, default=False)
parser.add_argument('--use_gate', type=str_to_bool, default=True)
parser.add_argument('--gate_weight', type=float, default=0.1)
parser.add_argument('--wavelet', type=str, default='db1')
parser.add_argument('--resume_checkpoint', type=str, default=None)

config = parser.parse_args()

deterministic = False
if config.random_seed is not None:
    deterministic = True
    print("Deterministic mode. This might be slow.")

make_deterministic(deterministic=deterministic, seed=config.random_seed)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    log_dir = './logs/'
    if not config.need_log:
        log_dir = './trash/'
    log_dir = log_dir + config.log_desc_prefix + '-' + time_str
    writer = SummaryWriter(log_dir=log_dir)

    writer.add_text("args", str(vars(config)))

    input_channels = 0
    wpt_mask = wpt_config.config_list[config.wpt_config_index]['mask']
    wpt_mask = np.array(wpt_mask, dtype=np.bool)
    if isinstance(wpt_mask, np.ndarray):
        input_channels = wpt_mask.sum()

    mean = wpt_config.mean_std_list[config.wpt_config_index]['mean']
    std = wpt_config.mean_std_list[config.wpt_config_index]['std']

    if config.arch == 'mobilenet':
        model = wpt_net.WPTNet(num_classes=config.num_classes, input_channels=input_channels, use_attention=config.use_attention, use_gate=config.use_gate)
    elif config.arch == 'resnet50':
        model = wpt_net.wpt_resnet_50(input_channels, num_classes=config.num_classes, use_attention=config.use_attention, use_gate=config.use_gate)
    elif config.arch == 'resnet18':
        model = wpt_net.wpt_resnet_18(input_channels, num_classes=config.num_classes)


    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), config.lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)

    resume_epoch = 0
    best_acc1 = 0
    if config.resume_checkpoint is not None:
        # resume training
        checkpoint = torch.load('checkpoints/' + config.resume_checkpoint)
        resume_epoch = checkpoint['epoch']
        model_dict = checkpoint['state_dict']
        optimizer_dict = checkpoint['optimizer']
        best_acc1 = checkpoint['best_acc1']

        model.load_state_dict(model_dict)
        optimizer.load_state_dict(optimizer_dict)

    torch.backends.cudnn.benchmark = True

    train_dir = os.path.join(config.root_dir, 'train')
    val_dir = os.path.join(config.root_dir, 'val')

    should_down_sample = config.arch != 'mobilenet' and config.wpt_level == 2 # if resnet && level 2, then downsample is required

    # use dct transforms to avoid bugs in opencv_transform
    train_transform = transforms.Compose([
        wpt_transforms.BGRToYCbCr(),
        dct_transforms.RandomResizedCrop(config.input_size, scale=(0.08, 1.0)),
        dct_transforms.RandomHorizontalFlip(),
        wpt_transforms.WPT(wavelet=config.wavelet, ch=3, lvl=config.wpt_level),
        wpt_transforms.WPTMask(wpt_mask),
        wpt_transforms.WPTDownSample2() if should_down_sample else wpt_transforms.IdentityTransform(),
        wpt_transforms.WPTToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)) # 32, 32
    ])
        
    val_transform = transforms.Compose([
        wpt_transforms.BGRToYCbCr(),
        dct_transforms.Resize(int(config.input_size / 0.875)), # 896 / 1024, same as DCT
        dct_transforms.CenterCrop(config.input_size),
        wpt_transforms.WPT(wavelet=config.wavelet, ch=3, lvl=config.wpt_level),
        wpt_transforms.WPTMask(wpt_mask),
        wpt_transforms.WPTDownSample2() if should_down_sample else wpt_transforms.IdentityTransform(),
        wpt_transforms.WPTToTensor(),
        transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std))
    ])

    loader = 'opencv'
    train_dataset = dataset_cv2.ImageFolderCustom(train_dir, loader=loader, transform=train_transform, num_cls=config.num_classes, half_data=False)

    val_dataset = dataset_cv2.ImageFolderCustom(val_dir, loader='opencv', transform=val_transform, num_cls=config.num_classes, half_data=False)

    worker_init_fn = seed_worker if deterministic else None

    train_loader = torch.utils.data.DataLoader(train_dataset, worker_init_fn=worker_init_fn, batch_size=config.batch_size, pin_memory=True, shuffle=True, num_workers=config.num_workers)

    val_loader = torch.utils.data.DataLoader(val_dataset, worker_init_fn=worker_init_fn, batch_size=config.batch_size, pin_memory=True, shuffle=False, num_workers=config.num_workers)

    for epoch in range(resume_epoch, config.num_epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, writer)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, epoch, writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        filename = 'checkpoints/last_' + config.log_desc_prefix + '.pt'
        bestname = 'checkpoints/best_' + config.log_desc_prefix + '.pt'
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filename=filename, best_name=bestname)

    print(f"Best acc1: {best_acc1}")

    writer.close()


def train(train_loader, model, criterion, optimizer, epoch, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        adjust_learning_rate(optimizer, epoch, i, len(train_loader))

        # measure data loading time
        data_time.update(time.time() - end)

        # convert to GPU
        images = images.to(device)
        target = target.to(device)

        # compute output
        output = model(images)
        if config.use_attention or config.use_gate:
            output, activations = output

        loss = criterion(output, target)

        if config.use_gate: # gate activation
                acts = torch.tensor([0.]).to(device)
                for ga in activations:
                    acts += torch.mean(ga)

                acts = torch.mean(acts / len(activations))
                loss += acts * config.gate_weight

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.display(i)

    writer_add_scalars(writer, 'train', {
        "top1_acc": top1.avg,
        "top5_acc": top5.avg,
        "loss": losses.avg,
        "batch_time": batch_time.avg
    }, epoch)

    new_lr = optimizer.param_groups[0]['lr']
    writer.add_scalar("learning_rate", new_lr, epoch)


def validate(val_loader, model, criterion, epoch, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.to(device)
            target = target.to(device)

            # compute output
            output = model(images)
            if config.use_attention or config.use_gate:
                output, activations = output

            loss = criterion(output, target)

            if config.use_gate: # gate activation
                acts = torch.tensor([0.]).to(device)
                for ga in activations:
                    acts += torch.mean(ga)

                acts = torch.mean(acts / len(activations))
                loss += acts * config.gate_weight

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 100 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

        writer_add_scalars(writer, 'val', {
            "top1_acc": top1.avg,
            "top5_acc": top5.avg,
            "loss": losses.avg
        }, epoch)

    return top1.avg

def adjust_learning_rate(optimizer, epoch, iteration, num_iter, warmup=False):
    num_epochs = config.num_epochs

    warmup_epoch = 5 if warmup else 0
    warmup_iter = warmup_epoch * num_iter
    current_iter = iteration + epoch * num_iter
    max_iter = num_epochs * num_iter

    if config.lr_decay == 'step':
        lr = config.lr * (config.lr_decay_gamma ** ((current_iter - warmup_iter) // (max_iter - warmup_iter)))
    elif config.lr_decay == 'cos':
        lr = config.lr * (1 + cos(pi * (current_iter - warmup_iter) / (max_iter - warmup_iter))) / 2
    elif config.lr_decay == 'linear':
        lr = config.lr * (1 - (current_iter - warmup_iter) / (max_iter - warmup_iter))
    elif config.lr_decay == 'schedule':
        count = sum([1 for s in config.schedule if s <= epoch])
        lr = config.lr * pow(config.lr_decay_gamma, count)
    else:
        raise ValueError('Unknown lr mode {}'.format(config.lr_decay))

    if epoch < warmup_epoch:
        lr = config.lr * current_iter / warmup_iter

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    main()