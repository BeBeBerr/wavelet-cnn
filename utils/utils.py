import torch
import shutil
from math import cos, pi
import numpy as np
import random

def str_to_bool(value):
    if value.lower() in {'false', 'f', '0', 'no', 'n'}:
        return False
    elif value.lower() in {'true', 't', '1', 'yes', 'y'}:
        return True
    raise ValueError(f'{value} is not a valid boolean value')

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', best_name='model_best.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)

def writer_add_scalars(writer, prefix, kvs, epoch):
    for k, v in kvs.items():
        writer.add_scalar(f"{prefix}/{k}", v, epoch)

def make_deterministic(deterministic=False, seed=4):
    if not deterministic:
        torch.backends.cudnn.benchmark = True
        return

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    # avoid randomness in multi-process data loading
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
     