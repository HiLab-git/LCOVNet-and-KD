# -*- coding: utf-8 -*-
from __future__ import print_function, division
from re import X

import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint
from nnunet.network_architecture.neural_network import SegmentationNetwork
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.utilities.nd_softmax import softmax_helper
import torch.nn.functional as F


class UnetBlock_Encode(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.

class UnetBlock_Down(nn.Module):
    def __init__(self, in_channels):
        super(UnetBlock_Down, self).__init__()

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.dense = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            h_sigmoid()
        )

    def forward(self, x):
        # x = self.conv1(x)
        x = self.pool(x)
        batch, channels, height, width, depth = x.size()
        out = F.avg_pool3d(x, kernel_size=[height, width, depth]).view(batch, -1)
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1, 1)
        return out * x


class UnetBlock_SE(nn.Module):
    def __init__(self, in_channels):
        super(UnetBlock_SE, self).__init__()

        self.dense1 = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            h_sigmoid()
        )

        self.dense2 = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            h_sigmoid()
        )

    def forward(self, x):
        # x = self.conv1(x)
        batch, channels, height, width, depth = x.size()

        out = F.avg_pool3d(x, kernel_size=[height, width, depth]).view(batch, -1)
        out = self.dense1(out)
        out = out.view(batch, channels, 1, 1, 1)
        out =  out * x

        out1 = F.avg_pool3d(out, kernel_size=[height, width, depth]).view(batch, -1)
        out1 = self.dense2(out1)
        out1 = out1.view(batch, channels, 1, 1, 1)
        return out * out1


class UnetBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Up, self).__init__()
        self.conv = self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channel, kernel_size=1,
                      padding=0, groups=1),
            nn.BatchNorm3d(out_channel),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.up = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.up(x)
        return x


class HRNet(SegmentationNetwork):
    def __init__(self, C_in=32, n_classes=17, m=1, is_ds=True):
        super(HRNet, self).__init__()
        self.m = m
        self.num_classes = n_classes
        self.in_chns = C_in
        self.n_class = n_classes
        self.inchn = 16
        self._deep_supervision = is_ds
        self.do_ds = is_ds        # self.ft_chns = [self.inchn, self.inchn*2,
        #                 self.inchn*4, self.inchn*8, self.inchn*8]  # 最初始设置 O
        self.ft_chns = [self.inchn*2,
                        self.inchn*4, self.inchn*8, self.inchn*8]  # A
        # print(self.ft_chns)

        self.conv1 = UnetBlock_Encode(self.m, self.ft_chns[0])
        self.conv2 = UnetBlock_Encode(self.ft_chns[0], self.ft_chns[1])
        self.conv3 = UnetBlock_Encode(self.ft_chns[1], self.ft_chns[2])
        self.conv4 = UnetBlock_Encode(self.ft_chns[2], self.ft_chns[3])
        self.conv5 = UnetBlock_Encode(self.ft_chns[3], self.ft_chns[2])
        self.conv6 = UnetBlock_Encode(self.ft_chns[2], self.ft_chns[1])
        self.conv7 = UnetBlock_Encode(self.ft_chns[1], self.ft_chns[0])

        self.segout = nn.Conv3d(self.ft_chns[0], self.n_class, kernel_size=1, padding=0, groups=1)


        self.final_nonlin = softmax_helper
        # self.do_ds = False
        self.upscale_logits_ops = []
        for m in range(len(self.ft_chns)-1):
            self.upscale_logits_ops.append(lambda x: x)

        self.weightInitializer = InitWeights_He(1e-2)
        self.apply(self.weightInitializer)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5 + x3)
        x7 = self.conv7(x6 + x2)
        x = self.segout(x7 + x1)

        # return tuple([x])
        if self._deep_supervision and self.do_ds:
            return tuple([x])
        else:
            return x
        # return segout