# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch import Tensor
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class RobustCrossEntropyLoss(AbstractSegLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class MSKDCAKDLoss(AbstractSegLoss):
    def __init__(self):
        super(CAKDLoss, self).__init__()
        ce_kwargs = {}
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)

    def forward(self, student_outputs, teacher_outputs):
        loss = 0
        w = [0.4, 0.2, 0.2, 0.2]
        for i in range(0,4):
            loss += w[i] * (0.1 * CAKD(student_outputs[i], teacher_outputs[i]) 
            + 0.2 * FNKD(student_outputs[i], teacher_outputs[i], student_feature[i+4], teacher_feature[i+4]))
        return loss


    def CAKD(self, student_outputs, teacher_outputs);
        [B, C, D, W, H] = student_outputs.shape

        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D*W*H)

        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        teacher_outputs = teacher_outputs.reshape(B, C, D*W*H)

        
        with autocast(enabled=False):
            student_outputs = torch.bmm(student_outputs, student_outputs.permute(
                0, 2, 1))
            teacher_outputs = torch.bmm(teacher_outputs, teacher_outputs.permute(
                0, 2, 1))

        Similarity_loss = (F.cosine_similarity(student_outputs[0, :, :], teacher_outputs[0, :, :], dim=0) +
                           F.cosine_similarity(
            student_outputs[1, :, :], teacher_outputs[1, :, :], dim=0))/2
        loss = -torch.mean(Similarity_loss)  # loss = 0 fully same
        return loss

    def FNKD(self, student_outputs, teacher_outputs, student_feature, teacher_feature):
        student_L2norm = torch.norm(student_feature)
        teacher_L2norm = torch.norm(teacher_feature)
        q_fn = F.log_softmax(teacher_outputs / teacher_L2norm, dim=1)
        to_kd = F.softmax(student_outputs / student_L2norm, dim=1)
        KD_ce_loss = self.ce(
            q_fn, to_kd[:, 0].long())
        return KD_ce_loss