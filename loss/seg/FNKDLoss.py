# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice
from torch import Tensor


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class FNKD_Loss(nn.Module):
    """
    Feature Normalized Knowledge Distillation for Image Classification
    FNKD_Loss
    """

    def __init__(self, T):
        super(FNKD_Loss, self).__init__()
        self.alpha = 0.9
        self.T = T
        ce_kwargs = {}
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, soft_outputs, teacher_outputs, f):
        # kd_loss = loss_fn_kd(
        #     soft_outputs, labels, teacher_outputs, self.alpha, self.T)
        t = 2
        f_L2norm = torch.norm(f)
        q_fn = F.log_softmax((t * teacher_outputs) / f_L2norm, dim=1)
        to_kd = F.softmax((t * soft_outputs) / f_L2norm, dim=1)
        KD_ce_loss = self.ce(
            q_fn, to_kd[:, 0].long())
        return KD_ce_loss