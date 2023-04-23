# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class DiceLoss(AbstractSegLoss):
    '''
    Dice loss for segmentation tasks.
    The parameters should be written in the `params` dictionary, and it has the
    following fields:

    :param `loss_softmax`: (bool) Apply softmax to the prediction of network or not. 
    '''
    def __init__(self, params = None):
        super(DiceLoss, self).__init__(params)

    def forward(self, loss_input_dict):
        predict = loss_input_dict['prediction']
        soft_y  = loss_input_dict['ground_truth']
        
        if(isinstance(predict, (list, tuple))):
            predict = predict[0]
        if(self.softmax):
            predict = nn.Softmax(dim = 1)(predict)
        predict = reshape_tensor_to_2D(predict)
        soft_y  = reshape_tensor_to_2D(soft_y) 
        dice_score = get_classwise_dice(predict, soft_y)
        dice_loss  = 1.0 - dice_score.mean()
        return dice_loss