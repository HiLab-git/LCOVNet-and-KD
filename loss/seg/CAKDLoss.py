# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import torch.nn as nn
from pymic.loss.seg.abstract import AbstractSegLoss
from pymic.loss.seg.util import reshape_tensor_to_2D, get_classwise_dice

class CAKDLoss(nn.Module):
    # 知识蒸馏
    def __init__(self, T=30):
        super(CAKDLoss, self).__init__()

    def forward(self, student_outputs, teacher_outputs):
        [B, C, D, W, H] = student_outputs.shape

        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D*W*H)

        teacher_outputs = F.softmax(teacher_outputs, dim=1)
        teacher_outputs = teacher_outputs.reshape(B, C, D*W*H)

        from torch.cuda.amp import autocast
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