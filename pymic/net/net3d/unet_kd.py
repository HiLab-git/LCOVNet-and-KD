# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from torch.utils.checkpoint import checkpoint


class UnetBlock_Encode(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Encode, self).__init__()

        self.in_chns = in_channels
        self.out_chns = out_channel

        self.conv1 = nn.Sequential(
            nn.Conv3d(self.in_chns, self.out_chns, kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

        self.conv2 = nn.Sequential(
            nn.Conv3d(self.out_chns, self.out_chns, kernel_size=3,
                      padding=1),
            nn.BatchNorm3d(self.out_chns),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class UnetBlock_Down(nn.Module):
    def __init__(self):
        super(UnetBlock_Down, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.max_pool(x)
        return x


class UnetBlock_Up(nn.Module):
    def __init__(self, in_channels, out_channel):
        super(UnetBlock_Up, self).__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose3d(in_channels, out_channel,
                               kernel_size=2, stride=2),
            nn.BatchNorm3d(out_channel),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class UNet_KD(nn.Module):
    def __init__(self, C_in=32, n_classes=17, m=1, ds = True):
        super(UNet_KD, self).__init__()
        self.m = m
        self.in_chns = C_in
        self.n_class = n_classes
        self.inchn = 32
        self.num_classes = n_classes
        # self.ft_chns = [self.inchn, self.inchn*2,
        #                 self.inchn*4, self.inchn*8, self.inchn*8]  # 最初始设置 O
        self.ft_chns = [self.inchn, self.inchn*2,
                        self.inchn*4, self.inchn*8, self.inchn*8]  # A
        self.resolution_level = len(self.ft_chns)

        self.do_ds = ds

        self.Encode_block1 = UnetBlock_Encode(self.m, self.ft_chns[0])
        self.down1 = UnetBlock_Down()

        self.Encode_block2 = UnetBlock_Encode(self.ft_chns[0], self.ft_chns[1])
        self.down2 = UnetBlock_Down()

        self.Encode_block3 = UnetBlock_Encode(self.ft_chns[1], self.ft_chns[2])
        self.down3 = UnetBlock_Down()

        self.Encode_block4 = UnetBlock_Encode(self.ft_chns[2], self.ft_chns[3])
        self.down4 = UnetBlock_Down()

        self.Encode_BottleNeck_block5 = UnetBlock_Encode(
            self.ft_chns[3], self.ft_chns[4])
        # self.down5 = UnetBlock_Down()

        self.up1 = UnetBlock_Up(self.ft_chns[4], self.ft_chns[3])
        self.Decode_block1 = UnetBlock_Encode(
            self.ft_chns[3]*2, self.ft_chns[3])
        self.segout1 = nn.Conv3d(
            self.ft_chns[3], self.n_class, kernel_size=1, padding=0)

        self.up2 = UnetBlock_Up(self.ft_chns[3], self.ft_chns[2])
        self.Decode_block2 = UnetBlock_Encode(
            self.ft_chns[2]*2, self.ft_chns[2])
        self.segout2 = nn.Conv3d(
            self.ft_chns[2], self.n_class, kernel_size=1, padding=0)

        self.up3 = UnetBlock_Up(self.ft_chns[2], self.ft_chns[1])
        self.Decode_block3 = UnetBlock_Encode(
            self.ft_chns[1]*2, self.ft_chns[1])
        self.segout3 = nn.Conv3d(
            self.ft_chns[1], self.n_class, kernel_size=1, padding=0)

        self.up4 = UnetBlock_Up(self.ft_chns[1], self.ft_chns[0])
        self.Decode_block4 = UnetBlock_Encode(
            self.ft_chns[0]*2, self.ft_chns[0])
        self.segout4 = nn.Conv3d(
            self.ft_chns[0], self.n_class, kernel_size=1, padding=0)

    def forward(self, x):
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
        segout1 = self.segout1(x6)

        x7 = self.up2(x6)
        x7 = torch.cat((x7, _x3), dim=1)
        # print(x7.shape, _x3.shape)
        x7 = self.Decode_block2(x7)
        segout2 = self.segout2(x7)

        x8 = self.up3(x7)
        x8 = torch.cat((x8, _x2), dim=1)
        x8 = self.Decode_block3(x8)
        segout3 = self.segout3(x8)

        x9 = self.up4(x8)
        x9 = torch.cat((x9, _x1), dim=1)
        x9 = self.Decode_block4(x9)
        segout4 = self.segout4(x9)

        if (self.do_ds == True):
            return [segout4, segout3, segout2, segout1, x9, x8, x7, x6]
        else:
            return segout4