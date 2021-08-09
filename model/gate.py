import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool1d
from .gumbel import GumbleSoftmax

class GateModuleCBAM(nn.Module):
    def __init__(self, in_ch, reduction_ratio=1.0):
        super(GateModuleCBAM, self).__init__()

        self.in_ch = in_ch
        self.reduction_ratio = reduction_ratio

        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.hidden_dim = int(self.in_ch / self.reduction_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(self.in_ch, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.in_ch)
        )

        
        self.inp_gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)

    def forward(self, x, temperature=1.):
        avg_pool = self.avg_pool(x)  # t2
        max_pool = self.max_pool(x)

        avg_pool = avg_pool.view(avg_pool.shape[0], -1)
        max_pool = max_pool.view(max_pool.shape[0], -1)

        avg_pool = self.mlp(avg_pool)
        max_pool = self.mlp(max_pool)

        pool_sum = avg_pool + max_pool
        hatten = pool_sum.unsqueeze(2).unsqueeze(3)

        hatten_d = self.inp_gate(hatten)  # t3
        hatten_d = self.inp_gate_l(hatten_d)  # t4
        hatten_d = hatten_d.reshape(hatten_d.size(0), self.in_ch, 2, 1)
        hatten_d = self.inp_gs(hatten_d, temp=temperature, force_hard=True)

        x = x * hatten_d[:, :, 1].unsqueeze(2)

        return x, hatten_d[:, :, 1]

class GateModule(nn.Module):
    def __init__(self, in_ch, kernel_size=28, doubleGate=False, dwLA=False):
        super(GateModule, self).__init__()

        self.doubleGate, self.dwLA = doubleGate, dwLA
        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch

        if dwLA:
            if doubleGate:
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch,
                              bias=True),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=1, padding=0, groups=in_ch, bias=True),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch,
                                        bias=True)
        else:
            if doubleGate:
                reduction = 4
                self.inp_att = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch // reduction, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(in_ch // reduction, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.Sigmoid()
                )

            self.inp_gate = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True),
            )
            self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)

    def forward(self, x, temperature=1.):
        hatten = self.avg_pool(x)  # t2
        hatten_d = self.inp_gate(hatten)  # t3
        hatten_d = self.inp_gate_l(hatten_d)  # t4
        hatten_d = hatten_d.reshape(hatten_d.size(0), self.in_ch, 2, 1)
        hatten_d = self.inp_gs(hatten_d, temp=temperature, force_hard=True)

        x = x * hatten_d[:, :, 1].unsqueeze(2)

        return x, hatten_d[:, :, 1]
