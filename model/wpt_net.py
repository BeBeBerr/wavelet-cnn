import torch
from torchvision import models
import torch.nn as nn
from .channel_atten import CBAM

from .gate import GateModule
from .gumbel import GumbleSoftmax
from .se_block import SE_Block
from .init_utils import *

class WPTNet(models.MobileNetV2):
    def __init__(self, input_channels=32, num_classes=1000, use_attention=False, use_gate=False):
        super(WPTNet, self).__init__()

        self.use_attention = use_attention
        self.use_gate = use_gate # use gumbel softmax gate

        # remove first conv layer
        self.features = self.features[1:]

        classifier_input_features = self.classifier[1].in_features

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(classifier_input_features, num_classes),
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        if self.use_attention:
            self.channel_attention = CBAM(input_channels=input_channels, reduction_ratio=1.0)

        if input_channels != 32:
            temp_layer = nn.Conv2d(input_channels, input_channels, 3, 1, 1, groups=input_channels, bias=False)
            self.features[0].conv[0][0] = temp_layer
            temp_layer = nn.BatchNorm2d(input_channels)
            self.features[0].conv[0][1] = temp_layer
            temp_layer = nn.Conv2d(input_channels, self.features[0].conv[1].out_channels, 1, 1, 0, bias=False)
            self.features[0].conv[1] = temp_layer

        if self.use_gate:
            self.input_gate = GateModule(input_channels)

            for name, m in self.named_modules():
                if 'inp_gate_l' in str(name):
                    m.weight.data.normal_(0, 0.001)
                    m.bias.data[::2].fill_(0.1)
                    m.bias.data[1::2].fill_(2)
                elif 'inp_gate' in str(name):
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)


        

    def forward(self, x):
        if self.use_attention:
            # use channel attention for the inputs
            x, atten_score = self.channel_attention(x, return_score=True)
        elif self.use_gate:
            # use gumbel softmax gate
            x, atten_score = self.input_gate(x)

        x = self.features(x)
        x = x.mean(3).mean(2) # ~= avg pool, 4dims = b,c,h,w
        x = self.classifier(x)

        if self.use_attention or self.use_gate:
            return x, atten_score
            
        return x

class WPTResNet(models.ResNet):
    def __init__(self, input_channels, block, layers, use_attention=False, use_gate=False, **kwargs):
        super().__init__(block, layers, **kwargs)

        self.use_gate = use_gate # use gumbel softmax gate
        self.use_attention = use_attention

        if self.use_attention:
            self.channel_attention = CBAM(input_channels=input_channels, reduction_ratio=1.0)
    
        if input_channels < 64:
            out_ch = self.layer1[0].conv1.out_channels
            kernel_size = self.layer1[0].conv1.kernel_size
            temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=kernel_size, stride=1, bias=False) # NOTE: In DCT source code, res50 has a bug here, kernel size = 3, but when copying weight, kernel_size became 1. They probably mixed res18 and res50 by mistake
            temp_layer.weight.data = self.layer1[0].conv1.weight.data[:, :input_channels]
            self.layer1[0].conv1 = temp_layer

            # although conventional res18's BasicBlock doesn't have 'downsample',
            # we do need downsample here, otherwise,
            # the residual add operation will throw error
            downsample = self.layer1[0].downsample

            # TODO: didn't copy weights here, check if really needed
            # temp_layer.weight.data = 
            if downsample is None:
                # res18
                out_ch = self.layer1[0].conv2.out_channels
                temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=3, stride=1, bias=False)
                downsample = nn.Sequential(
                    temp_layer,
                    self._norm_layer(64 * block.expansion),
                )
                self.layer1[0].downsample = downsample
            else:
                # res50
                out_ch = self.layer1[0].downsample[0].out_channels
                temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=1, stride=1, bias=False)
                temp_layer.weight.data = self.layer1[0].downsample[0].weight.data[:, :input_channels]
                self.layer1[0].downsample[0] = temp_layer
            
        else:
            out_ch = self.layer1[0].conv1.out_channels
            kernel_size = self.layer1[0].conv1.kernel_size
            temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=kernel_size, stride=1, bias=False)
            kaiming_init(temp_layer)
            self.layer1[0].conv1 = temp_layer

            out_ch = self.layer1[0].downsample[0].out_channels
            temp_layer = nn.Conv2d(input_channels, out_ch, kernel_size=1, stride=1, bias=False)
            kaiming_init(temp_layer)
            self.layer1[0].downsample[0] = temp_layer

        if self.use_gate:
            self.input_gate = GateModule(input_channels)

            for name, m in self.named_modules():
                if 'inp_gate_l' in str(name):
                    m.weight.data.normal_(0, 0.001)
                    m.bias.data[::2].fill_(0.1)
                    m.bias.data[1::2].fill_(2)
                elif 'inp_gate' in str(name):
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)
                    elif isinstance(m, nn.BatchNorm2d):
                        constant_init(m, 1)

    def _forward_impl(self, x):
        # x = self.conv1(x)
        # x = self.bn1(x)
        # x = self.relu(x)

        # x = self.maxpool(x) # without maxpool, then wptdownsample is needed in pre-processing

        if self.use_attention:
            # use channel attention for the inputs
            x, atten_score = self.channel_attention(x, return_score=True)
        elif self.use_gate:
            # use gumbel softmax gate
            x, atten_score = self.input_gate(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        if self.use_attention or self.use_gate:
            return x, atten_score

        return x

def wpt_resnet_18(input_channels, **kwargs):
    model = WPTResNet(input_channels, models.resnet.BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def wpt_resnet_50(input_channels, **kwargs):
    model = WPTResNet(input_channels, models.resnet.Bottleneck, [3, 4, 6, 3], **kwargs)
    return model




if __name__ == '__main__':
    m = models.resnet18()
    print()
