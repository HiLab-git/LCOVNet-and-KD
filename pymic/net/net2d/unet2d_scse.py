# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
import numpy as np 
from pymic.net.net2d.scse2d import *

class ConvScSEBlock(nn.Module):
    """
    Two convolutional blocks followed by `ChannelSpatialSELayer`.
    Each block consists of `Conv2d` + `BatchNorm2d` + `LeakyReLU`.
    A dropout layer is used between the wo blocks.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(ConvScSEBlock, self).__init__()
        self.conv_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(dropout_p),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            ChannelSpatialSELayer(out_channels)
        )
       
    def forward(self, x):
        return self.conv_conv(x)

class DownBlock(nn.Module):
    """Downsampling followed by `ConvScSEBlock`.

    :param in_channels: (int) Input channel number.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    """
    def __init__(self, in_channels, out_channels, dropout_p):
        super(DownBlock, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvScSEBlock(in_channels, out_channels, dropout_p)

        )

    def forward(self, x):
        return self.maxpool_conv(x)

class UpBlock(nn.Module):
    """Up-sampling followed by `ConvScSEBlock` in U-Net structure.
    
    :param in_channels1: (int) Input channel number for low-resolution feature map.
    :param in_channels2: (int) Input channel number for high-resolution feature map.
    :param out_channels: (int) Output channel number.
    :param dropout_p: (int) Dropout probability.
    :param bilinear: (bool) Use bilinear for up-sampling or not.
    """
    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p, 
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv2d(in_channels1, in_channels2, kernel_size = 1)
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels1, in_channels2, kernel_size=2, stride=2)
        self.conv = ConvScSEBlock(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet2D_ScSE(nn.Module):
    """
    Combining 2D U-Net with SCSE module.

    * Reference: Abhijit Guha Roy, Nassir Navab, Christian Wachinger:
      Recalibrating Fully Convolutional Networks With Spatial and Channel 
      "Squeeze and Excitation" Blocks. 
      `IEEE Trans. Med. Imaging 38(2): 540-549 (2019). <https://ieeexplore.ieee.org/document/8447284>`_

    Parameters are given in the `params` dictionary, and should include the
    following fields:

    :param in_chns: (int) Input channel number.
    :param feature_chns: (list) Feature channel for each resolution level. 
      The length should be 5, such as [16, 32, 64, 128, 256].
    :param dropout: (list) The dropout ratio for each resolution level. 
      The length should be the same as that of `feature_chns`.
    :param class_num: (int) The class number for segmentation task. 
    :param bilinear: (bool) Using bilinear for up-sampling or not. 
        If False, deconvolution will be used for up-sampling.
    """
    def __init__(self, params):
        super(UNet2D_ScSE, self).__init__()
        self.params    = params
        self.in_chns   = self.params['in_chns']
        self.ft_chns   = self.params['feature_chns']
        self.dropout   = self.params['dropout']
        self.n_class   = self.params['class_num']
        self.bilinear  = self.params['bilinear']
        assert(len(self.ft_chns) == 5)

        self.in_conv= ConvScSEBlock(self.in_chns, self.ft_chns[0], self.dropout[0])
        self.down1  = DownBlock(self.ft_chns[0], self.ft_chns[1], self.dropout[1])
        self.down2  = DownBlock(self.ft_chns[1], self.ft_chns[2], self.dropout[2])
        self.down3  = DownBlock(self.ft_chns[2], self.ft_chns[3], self.dropout[3])
        self.down4  = DownBlock(self.ft_chns[3], self.ft_chns[4], self.dropout[4])
        self.up1 = UpBlock(self.ft_chns[4], self.ft_chns[3], self.ft_chns[3], dropout_p = self.dropout[3]) 
        self.up2 = UpBlock(self.ft_chns[3], self.ft_chns[2], self.ft_chns[2], dropout_p = self.dropout[2]) 
        self.up3 = UpBlock(self.ft_chns[2], self.ft_chns[1], self.ft_chns[1], dropout_p = self.dropout[1]) 
        self.up4 = UpBlock(self.ft_chns[1], self.ft_chns[0], self.ft_chns[0], dropout_p = self.dropout[0]) 
    
        self.out_conv = nn.Conv2d(self.ft_chns[0], self.n_class,  
            kernel_size = 3, padding = 1)

    def forward(self, x):
        x_shape = list(x.shape)
        if(len(x_shape) == 5):
          [N, C, D, H, W] = x_shape
          new_shape = [N*D, C, H, W]
          x = torch.transpose(x, 1, 2)
          x = torch.reshape(x, new_shape)
        x0 = self.in_conv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        output = self.out_conv(x)

        if(len(x_shape) == 5):
            new_shape = [N, D] + list(output.shape)[1:]
            output = torch.reshape(output, new_shape)
            output = torch.transpose(output, 1, 2)
        return output

if __name__ == "__main__":
    params = {'in_chns':4,
              'feature_chns':[2, 8, 32, 48, 64],
              'dropout':  [0, 0, 0.3, 0.4, 0.5],
              'class_num': 2,
              'bilinear': True}
    Net = UNet2D_ScSE(params)
    Net = Net.double()

    x  = np.random.rand(4, 4, 10, 96, 96)
    xt = torch.from_numpy(x)
    xt = torch.tensor(xt)
    
    y = Net(xt)
    print(len(y.size()))
    y = y.detach().numpy()
    print(y.shape)