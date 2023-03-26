#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch.nn.functional as F
import torch
import torch.nn as nn
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np

class CAKD(nn.Module):
    # 知识蒸馏
    def __init__(self, T=30):
        super(CAKD, self).__init__()

    def forward(self, student_outputs, teacher_outputs):
        [B, C, D, W, H] = student_outputs.shape


        student_outputs = F.softmax(student_outputs, dim=1)
        student_outputs = student_outputs.reshape(B, C, D*W*H)

        teacher_outputs = F.softmax(teacher_outputs/self.T, dim=1)
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
        loss = -torch.mean(Similarity_loss)
        return loss

class FNKD(nn.Module):
    """
    Feature Normalized Knowledge Distillation for Image Classification
    FNKD_Loss
    """

    def __init__(self):
        super(FNKD, self).__init__()

    def forward(self, soft_outputs, teacher_outputs, s_f, t_f):
        tf_L2norm = torch.norm(t_f)
        sf_L2norm = torch.norm(s_f)

        teacher = (2 * teacher_outputs) / tf_L2norm
        student = (2 * soft_outputs) / sf_L2norm

        tensor_dim = len(teacher.size())
        num_class = list(teacher.size())[1]
        
        teacher_perm = teacher.permute(0, 2, 3, 4, 1)
        student_perm = student.permute(0, 2, 3, 4, 1)


        teacher_perm = torch.reshape(teacher_perm, (-1, num_class))
        student_perm = torch.reshape(student_perm, (-1, num_class))
        # print(student_perm.shape, teacher_perm.shape)
        # input()

        entroy = nn.CrossEntropyLoss()
        loss = entroy(student_perm, teacher_perm)
        # print(KD_ce_loss)
        # input()
        return loss

