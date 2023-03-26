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


class UnetBlock_Encode(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        # self.conv2_2 = nn.Sequential(
        #     nn.AvgPool3d(kernel_size=4, stride=2, padding=1),
        #     nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
        #               padding=0),
        #     nn.BatchNorm3d(self.out_chns),
        #     nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        # )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)

        x = self.conv2_1(x)
        # x2 = self.conv2_2(x)
        # x2 = torch.sigmoid(x2)
        # x = x1 + x2 * x
        return x


class UnetBlock_Encode_b(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode_b, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=self.out_chns,),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        return x


class UnetBlock_Encode_BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode_BottleNeck, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=(1, 1, 3),
                      padding=(0, 0, 1)),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=(3, 3, 1),
                      padding=(1, 1, 0), groups=self.out_chns),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2_2 = nn.Sequential(
            # nn.AvgPool3d(kernel_size=4, stride=2),
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=1,
                      padding=0),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU6(inplace=True),
            nn.Dropout(p=0.2)

            # nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        )

    def forward(self, x):
        x = self.conv1(x)

        x1 = self.conv2_1(x)
        x2 = self.conv2_2(x)
        x2 = torch.sigmoid(x2)
        x = x1 + x2 * x
        return x


class UnetBlock_Down(nn.Module):
    def __init__(self):
        super(UnetBlock_Down, self).__init__()

        self.avg_pool = nn.MaxPool3d(kernel_size=2, stride=2)
        # (kernel_size=2, stride=2)
        # self.max_pool = nn.MaxPool3d(kernel_size=2)

    def forward(self, x):
        # x = self.conv1(x)
        x = self.avg_pool(x)
        return x


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

class LCOV_Net(SegmentationNetwork):
    def __init__(self, C_in=32, n_classes=17, m=1, is_ds=True):
        super(LCOV_Net, self).__init__()
        self.m = m
        self.num_classes = n_classes
        self.in_chns = C_in
        self.n_class = n_classes
        self.inchn = 32
        self._deep_supervision = is_ds
        self.do_ds = is_ds        # self.ft_chns = [self.inchn, self.inchn*2,
        #                 self.inchn*4, self.inchn*8, self.inchn*8]  # 最初始设置 O
        self.ft_chns = [self.inchn, self.inchn*2,
                        self.inchn*4, self.inchn*8, self.inchn*8]  # A
        print(self.ft_chns)
        self.resolution_level = len(self.ft_chns)
        self.final_nonlin = softmax_helper
        self.do_ds = True

        self.Encode_block1 = UnetBlock_Encode(self.m, self.ft_chns[0])
        self.down1 = UnetBlock_Down()

        self.Encode_block2 = UnetBlock_Encode(self.ft_chns[0], self.ft_chns[1])
        self.down2 = UnetBlock_Down()

        self.Encode_block3 = UnetBlock_Encode(self.ft_chns[1], self.ft_chns[2])
        self.down3 = UnetBlock_Down()

        self.Encode_block4 = UnetBlock_Encode(self.ft_chns[2], self.ft_chns[3])
        self.down4 = UnetBlock_Down()

        self.Encode_BottleNeck_block5 = UnetBlock_Encode_b(
            self.ft_chns[3], self.ft_chns[4])
        # self.down5 = UnetBlock_Down()

        self.up1 = UnetBlock_Up(self.ft_chns[4], self.ft_chns[3])
        self.Decode_block1 = UnetBlock_Encode(
            self.ft_chns[3]*2, self.ft_chns[3])
        # self.segout1 = nn.Conv3d(
        #     self.ft_chns[3], self.n_class, kernel_size=1, padding=0)

        self.up2 = UnetBlock_Up(self.ft_chns[3], self.ft_chns[2])
        self.Decode_block2 = UnetBlock_Encode(
            self.ft_chns[2]*2, self.ft_chns[2])
        # self.segout2 = nn.Conv3d(
        #     self.ft_chns[2], self.n_class, kernel_size=1, padding=0)

        self.up3 = UnetBlock_Up(self.ft_chns[2], self.ft_chns[1])
        self.Decode_block3 = UnetBlock_Encode(
            self.ft_chns[1]*2, self.ft_chns[1])
        # self.segout3 = nn.Conv3d(
        #     self.ft_chns[1], self.n_class, kernel_size=1, padding=0)

        self.up4 = UnetBlock_Up(self.ft_chns[1], self.ft_chns[0])
        self.Decode_block4 = UnetBlock_Encode(
            self.ft_chns[0]*2, self.ft_chns[0])
        self.segout4 = nn.Conv3d(
            self.ft_chns[0], self.n_class, kernel_size=1, padding=0)

        self.upscale_logits_ops = []
        for m in range(len(self.ft_chns)-1):
            self.upscale_logits_ops.append(lambda x: x)

        self.weightInitializer = InitWeights_He(1e-2)
        self.apply(self.weightInitializer)

    def forward(self, x):
        # x = x.
        _x1 = self.Encode_block1(x)
        x1 = self.down1(_x1)

        _x2 = self.Encode_block2(x1)
        x2 = self.down2(_x2)

        _x3 = self.Encode_block3(x2)
        x3 = self.down2(_x3)

        _x4 = self.Encode_block4(x3)
        x4 = self.down2(_x4)

        x5 = self.Encode_BottleNeck_block5(x4)

        x6 = self.up1(x5)
        x6 = torch.cat((x6, _x4), dim=1)
        x6 = self.Decode_block1(x6)
        # segout1 = self.upscale_logits_ops[0](self.segout1(x6))

        x7 = self.up2(x6)
        x7 = torch.cat((x7, _x3), dim=1)
        # print(x7.shape, _x3.shape)
        x7 = self.Decode_block2(x7)
        # segout2 = self.upscale_logits_ops[1](self.segout2(x7))

        x8 = self.up3(x7)
        x8 = torch.cat((x8, _x2), dim=1)
        x8 = self.Decode_block3(x8)
        # segout3 = self.upscale_logits_ops[2](self.segout3(x8))

        x9 = self.up4(x8)
        x9 = torch.cat((x9, _x1), dim=1)
        x9 = self.Decode_block4(x9)
        # segout1 = self.upscale_logits_ops[0](self.segout1(x6))
        # segout2 = self.upscale_logits_ops[1](self.segout2(x7))
        # segout3 = self.upscale_logits_ops[2](self.segout3(x8))
        segout4 = self.upscale_logits_ops[3](self.segout4(x9))
        self.do_ds = False
        if (self.do_ds == True):
            return tuple([segout4, segout3, segout2, segout1])
        else:
            return segout4
