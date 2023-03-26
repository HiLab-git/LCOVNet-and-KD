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
        # print(x.shape)
        # input()
        x = self.conv2(x)
        return x


class UnetBlock_Encode_BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode_BottleNeck, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=3,
                      padding=3, dilation=3),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=3,
                      padding=6, dilation=6),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=3,
                      padding=12, dilation=12),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv4 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=3,
                      padding=18, dilation=18),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x1 = self.conv1(x)
        # print(x.shape, x1.shape)
        x2 = self.conv2(x + x1)
        # print(x2.shape)
        # input()
        x3 = self.conv3(x + x1 + x2)
        x4 = self.conv4(x + x1 + x2 + x3)

        return x1 + x2 + x3 + x4


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


class UnetBlock_Skip(nn.Module):
    def __init__(self, in_channels):
        super(UnetBlock_Skip, self).__init__()
        # print(in_channels)
        # input()
        self.dense = nn.Sequential(
            nn.Linear(in_channels, in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // 2, in_channels),
            h_sigmoid()
        )

    def forward(self, x):
        # x = self.pool(x)
        batch, channels, height, width, depth = x.size()
        out = F.avg_pool3d(x, kernel_size=[height, width, depth]).view(batch, -1)
        # print(out.shape)
        # input()
        out = self.dense(out)
        out = out.view(batch, channels, 1, 1, 1)
        return out * x



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


class SNet(SegmentationNetwork):
    def __init__(self, C_in=16, n_classes=11, m=3, is_ds=True):
        super(SNet, self).__init__()
        self.m = m
        self.num_classes = n_classes
        self.in_chns = C_in
        self.n_class = n_classes
        self.inchn = 16
        self._deep_supervision = is_ds
        self.do_ds = is_ds        # self.ft_chns = [self.inchn, self.inchn*2,
        #                 self.inchn*4, self.inchn*8, self.inchn*8]  # 最初始设置 O
        # self.ft_chns = [self.inchn, self.inchn*2,
        #                 self.inchn*4, self.inchn*8, self.inchn*8]  # A
        self.ft_chns = [self.inchn*2,
                        self.inchn*4, self.inchn*8, self.inchn*8]  # A

        print(self.ft_chns)
        self.resolution_level = len(self.ft_chns)
        self.final_nonlin = softmax_helper
        self.do_ds = False

        self.Encode_block1 = UnetBlock_Encode(self.m, self.ft_chns[0])
        self.down1 = UnetBlock_Down(self.ft_chns[0])
        self.skip1 = UnetBlock_Skip(self.ft_chns[0])

        self.Encode_block2 = UnetBlock_Encode(self.ft_chns[0], self.ft_chns[1])
        self.down2 = UnetBlock_Down(self.ft_chns[1])
        self.skip2 = UnetBlock_Skip(self.ft_chns[1])

        self.Encode_block3 = UnetBlock_Encode_BottleNeck(self.ft_chns[1], self.ft_chns[1])

        # self.down5 = UnetBlock_Down()
        self.up2 = UnetBlock_Up(self.ft_chns[1], self.ft_chns[1])
        self.Decode_block2 = UnetBlock_Encode(self.ft_chns[1] * 2, self.ft_chns[1])
        self.up1 = UnetBlock_Up(self.ft_chns[1], self.ft_chns[0])
        self.Dncode_block1 = UnetBlock_Encode(self.ft_chns[0] * 2, self.ft_chns[0])
        self.segout1 = nn.Conv3d(
            self.ft_chns[0], self.n_class, kernel_size=1, padding=0)

        self.upscale_logits_ops = []
        for m in range(len(self.ft_chns)-1):
            self.upscale_logits_ops.append(lambda x: x)

        self.weightInitializer = InitWeights_He(1e-2)
        self.apply(self.weightInitializer)

    def forward(self, x):
        x1 = self.Encode_block1(x)

        x2 = self.down1(x1)

        x2 = self.Encode_block2(x2)

        x3 = self.down2(x2)
        
        x3 = self.Encode_block3(x3)

        x4 = self.up2(x3)
        # print(x2.shape)
        # print(x4.shape)
        # #self.skip1(x2).shape)

        # input()
        x4 = torch.cat((x4, self.skip2(x2)), dim=1)
        x4 = self.Decode_block2(x4)

        x5 = self.up1(x4)
        x5 = torch.cat((x5, self.skip1(x1)), dim=1)
        x5 = self.Dncode_block1(x5)

        x5 = self.segout1(x5)


        if (self.do_ds == True):
            return tuple([x5])
        else:
            return x5
