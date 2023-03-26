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


from copy import deepcopy
from nnunet.utilities.nd_softmax import softmax_helper
from torch import nn
import torch
import numpy as np
from nnunet.network_architecture.initialization import InitWeights_He
from nnunet.network_architecture.neural_network import SegmentationNetwork
import torch.nn.functional


class ConvDropoutNormNonlin(nn.Module):
    """
    fixes a bug in ConvDropoutNormNonlin where lrelu was used regardless of nonlin. Bad.
    """

    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvDropoutNormNonlin, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1,
                           'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        self.conv = self.conv_op(
            input_channels, output_channels, **self.conv_kwargs)
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
                'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))


class ConvDropoutNonlinNorm(ConvDropoutNormNonlin):
    def forward(self, x):
        x = self.conv(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.instnorm(self.lrelu(x))


class StackedConvLayers(nn.Module):
    def __init__(self, input_feature_channels, output_feature_channels, num_convs,
                 conv_op=nn.Conv2d, conv_kwargs=None,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, first_stride=None, basic_block=ConvDropoutNormNonlin, key=None):
        '''
        stacks ConvDropoutNormLReLU layers. initial_stride will only be applied to first layer in the stack. The other parameters affect all layers
        :param input_feature_channels:
        :param output_feature_channels:
        :param num_convs:
        :param dilation:
        :param kernel_size:
        :param padding:
        :param dropout:
        :param initial_stride:
        :param conv_op:
        :param norm_op:
        :param dropout_op:
        :param inplace:
        :param neg_slope:
        :param norm_affine:
        :param conv_bias:
        '''
        self.input_channels = input_feature_channels
        self.output_channels = output_feature_channels
        # print()
        # print('num_convs {} input_feature_channels {} output_feature_channels '
        #       '{}'.format(num_convs, input_feature_channels, output_feature_channels))
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        if conv_kwargs is None:
            conv_kwargs = {'kernel_size': 3, 'stride': 1,
                           'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op

        if first_stride is not None:
            self.conv_kwargs_first_conv = deepcopy(conv_kwargs)
            self.conv_kwargs_first_conv['stride'] = first_stride
        else:
            self.conv_kwargs_first_conv = conv_kwargs

        # if key == 1:
        #     self.conv_kwargs_first_conv = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True}

        super(StackedConvLayers, self).__init__()
        self.blocks = nn.Sequential(
            *([basic_block(input_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs_first_conv,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs)] +
              [basic_block(output_feature_channels, output_feature_channels, self.conv_op,
                           self.conv_kwargs,
                           self.norm_op, self.norm_op_kwargs, self.dropout_op, self.dropout_op_kwargs,
                           self.nonlin, self.nonlin_kwargs) for _ in range(num_convs - 1)]))

    def forward(self, x):
        return self.blocks(x)


def print_module_training_status(module):
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv3d) or isinstance(module, nn.Dropout3d) or \
            isinstance(module, nn.Dropout2d) or isinstance(module, nn.Dropout) or isinstance(module, nn.InstanceNorm3d) \
            or isinstance(module, nn.InstanceNorm2d) or isinstance(module, nn.InstanceNorm1d) \
            or isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm3d) or isinstance(module,
                                                                                                      nn.BatchNorm1d):
        print(str(module), module.training)


class Upsample(nn.Module):
    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=False):
        super(Upsample, self).__init__()
        self.align_corners = align_corners
        self.mode = mode
        self.scale_factor = scale_factor
        self.size = size

    def forward(self, x):
        return nn.functional.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                                         align_corners=self.align_corners)


class HLNet(SegmentationNetwork):
    DEFAULT_BATCH_SIZE_3D = 2
    DEFAULT_PATCH_SIZE_3D = (64, 192, 160)
    SPACING_FACTOR_BETWEEN_STAGES = 2
    BASE_NUM_FEATURES_3D = 30
    MAX_NUMPOOL_3D = 999
    MAX_NUM_FILTERS_3D = 320

    DEFAULT_PATCH_SIZE_2D = (256, 256)
    BASE_NUM_FEATURES_2D = 30
    DEFAULT_BATCH_SIZE_2D = 50
    MAX_NUMPOOL_2D = 999
    MAX_FILTERS_2D = 480

    use_this_for_batch_size_computation_2D = 19739648
    use_this_for_batch_size_computation_3D = 520000000  # 505789440

    def __init__(self, input_channels, base_num_features, num_classes, num_pool, num_conv_per_stage=2,
                 feat_map_mul_on_downscale=2, conv_op=nn.Conv2d,
                 norm_op=nn.BatchNorm2d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout2d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None, deep_supervision=True, dropout_in_localization=False,
                 final_nonlin=softmax_helper, weightInitializer=InitWeights_He(1e-2), pool_op_kernel_sizes=None,
                 conv_kernel_sizes=None,
                 upscale_logits=False, convolutional_pooling=False, convolutional_upsampling=False,
                 max_num_features=None, basic_block=ConvDropoutNormNonlin,
                 seg_output_use_bias=False):
        """
        basically more flexible than v1, architecture is the same

        Does this look complicated? Nah bro. Functionality > usability

        This does everything you need, including world peace.

        Questions? -> f.isensee@dkfz.de
        """
        super(HLNet, self).__init__()
        self.convolutional_upsampling = convolutional_upsampling
        self.convolutional_pooling = convolutional_pooling
        self.upscale_logits = upscale_logits
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}

        self.conv_kwargs = {'stride': 1, 'dilation': 1, 'bias': True}

        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        self.weightInitializer = weightInitializer
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.dropout_op = dropout_op
        self.num_classes = num_classes
        self.final_nonlin = final_nonlin
        self._deep_supervision = deep_supervision
        self.do_ds = deep_supervision

        if conv_op == nn.Conv2d:
            upsample_mode = 'bilinear'
            pool_op = nn.MaxPool2d
            transpconv = nn.ConvTranspose2d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3)] * (num_pool + 1)
        elif conv_op == nn.Conv3d:
            upsample_mode = 'trilinear'
            pool_op = nn.MaxPool3d
            transpconv = nn.ConvTranspose3d
            if pool_op_kernel_sizes is None:
                pool_op_kernel_sizes = [(2, 2, 2)] * num_pool
            if conv_kernel_sizes is None:
                conv_kernel_sizes = [(3, 3, 3)] * (num_pool + 1)
        else:
            raise ValueError(
                "unknown convolution dimensionality, conv op: %s" % str(conv_op))

        self.input_shape_must_be_divisible_by = np.prod(
            pool_op_kernel_sizes, 0, dtype=np.int64)
        self.pool_op_kernel_sizes = pool_op_kernel_sizes
        self.conv_kernel_sizes = conv_kernel_sizes

        self.conv_pad_sizes = []
        for krnl in self.conv_kernel_sizes:
            self.conv_pad_sizes.append([1 if i == 3 else 0 for i in krnl])

        if max_num_features is None:
            if self.conv_op == nn.Conv3d:
                self.max_num_features = self.MAX_NUM_FILTERS_3D
            else:
                self.max_num_features = self.MAX_FILTERS_2D
        else:
            self.max_num_features = max_num_features

        self.conv_blocks_context = []
        self.conv_blocks_localization = []
        self.td = []
        self.tu = []
        self.seg_outputs = []

        output_features = base_num_features
        input_features = input_channels
        # print(num_pool)

        self.feature_chns = [3, 16, 32, 64, 128, 256]
        self.conv_kwargs['kernel_size'] = 3
        self.conv_kwargs['padding'] = 1

        self.block1 = StackedConvLayers(self.feature_chns[0], self.feature_chns[1], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block)

        self.block2 = StackedConvLayers(self.feature_chns[1], self.feature_chns[2], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block, key=1)

        self.block3 = StackedConvLayers(self.feature_chns[2], self.feature_chns[3], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block, key=1)

        self.block4 = StackedConvLayers(self.feature_chns[3]+1, self.feature_chns[4], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block, key=1)

        if not dropout_in_localization:
            old_dropout_p = self.dropout_op_kwargs['p']
            self.dropout_op_kwargs['p'] = 0.0
        # self.block6 = StackedConvLayers(self.feature_chns[4]*2, self.feature_chns[4], num_conv_per_stage,
        #                                                     self.conv_op, self.conv_kwargs, self.norm_op,
        #                                                     self.norm_op_kwargs, self.dropout_op,
        #                                                     self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
        #                                                     first_stride=None, basic_block=basic_block)

        self.block5 = StackedConvLayers(self.feature_chns[3]*2, self.feature_chns[3], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block)

        self.block6 = StackedConvLayers(self.feature_chns[2]*2, self.feature_chns[2], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block)

        self.block7 = StackedConvLayers(self.feature_chns[1]*2, self.feature_chns[1], num_conv_per_stage,
                                        self.conv_op, self.conv_kwargs, self.norm_op,
                                        self.norm_op_kwargs, self.dropout_op,
                                        self.dropout_op_kwargs, self.nonlin, self.nonlin_kwargs,
                                        first_stride=None, basic_block=basic_block)

        self.downsample = nn.MaxPool3d(kernel_size=2, stride=2)
        # self.conv_kwargs_stride_two = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'dilation': 1, 'bias': True}
        # self._downsample = basic_block(self.feature_chns[4], self.feature_chns[4], self.conv_op,
        #                    self.conv_kwargs_stride_two,
        #                    self.norm_op, self.norm_op_kwargs,
        #                    self.dropout_op, self.dropout_op_kwargs,
        #                    self.nonlin, self.nonlin_kwargs)

        # self.up5 = transpconv(self.feature_chns[5], self.feature_chns[4], kernel_size=2,
        #                                   stride=2, bias=False)

        self.up4 = transpconv(self.feature_chns[4], self.feature_chns[3], kernel_size=2,
                              stride=2, bias=False)

        self.up3 = transpconv(self.feature_chns[3], self.feature_chns[2], kernel_size=2,
                              stride=2, bias=False)

        self.up2 = transpconv(self.feature_chns[2], self.feature_chns[1], kernel_size=2,
                              stride=2, bias=False)

        self.conv_kwargs_first_conv = {
            'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.conv1 = basic_block(self.feature_chns[0], self.feature_chns[1], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.conv2 = basic_block(self.feature_chns[1]*2, self.feature_chns[2], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.conv3 = basic_block(self.feature_chns[2]*2, self.feature_chns[3], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.conv4 = basic_block(self.feature_chns[3]*2, self.feature_chns[4], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.conv5 = basic_block(self.feature_chns[4]*2, self.feature_chns[3], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.conv6 = basic_block(self.feature_chns[3]*2, self.feature_chns[2], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.conv7 = basic_block(self.feature_chns[2]*2, self.feature_chns[1], self.conv_op,
                                 self.conv_kwargs_first_conv,
                                 self.norm_op, self.norm_op_kwargs,
                                 self.dropout_op, self.dropout_op_kwargs,
                                 self.nonlin, self.nonlin_kwargs)

        self.ds_conv4 = basic_block(self.feature_chns[4], 2, self.conv_op,
                                    self.conv_kwargs_first_conv,
                                    self.norm_op, self.norm_op_kwargs,
                                    self.dropout_op, self.dropout_op_kwargs,
                                    self.nonlin, self.nonlin_kwargs)

        self.ds_conv3 = basic_block(self.feature_chns[3], 2, self.conv_op,
                                    self.conv_kwargs_first_conv,
                                    self.norm_op, self.norm_op_kwargs,
                                    self.dropout_op, self.dropout_op_kwargs,
                                    self.nonlin, self.nonlin_kwargs)

        self.ds_conv2 = basic_block(self.feature_chns[2], 2, self.conv_op,
                                    self.conv_kwargs_first_conv,
                                    self.norm_op, self.norm_op_kwargs,
                                    self.dropout_op, self.dropout_op_kwargs,
                                    self.nonlin, self.nonlin_kwargs)

        self.ds_conv1 = basic_block(self.feature_chns[1]*2, 2, self.conv_op,
                                    self.conv_kwargs_first_conv,
                                    self.norm_op, self.norm_op_kwargs,
                                    self.dropout_op, self.dropout_op_kwargs,
                                    self.nonlin, self.nonlin_kwargs)

        self.HR2 = transpconv(self.feature_chns[2], self.feature_chns[2], kernel_size=2,
                              stride=2, bias=False)

        self.HR3 = nn.Sequential(
            transpconv(self.feature_chns[3], self.feature_chns[3], kernel_size=2,
                       stride=2, bias=False),
            transpconv(self.feature_chns[3], self.feature_chns[3], kernel_size=2,
                       stride=2, bias=False)
        )

        self.HR4 = nn.Sequential(
            transpconv(self.feature_chns[4], self.feature_chns[4], kernel_size=2,
                       stride=2, bias=False),
            transpconv(self.feature_chns[4], self.feature_chns[4], kernel_size=2,
                       stride=2, bias=False),
            transpconv(self.feature_chns[4], self.feature_chns[4], kernel_size=2,
                       stride=2, bias=False)
        )

        self.HR5 = nn.Sequential(
            transpconv(self.feature_chns[3], self.feature_chns[3], kernel_size=2,
                       stride=2, bias=False),
            transpconv(self.feature_chns[3], self.feature_chns[3], kernel_size=2,
                       stride=2, bias=False)
        )

        self.HR6 = nn.Sequential(
            transpconv(self.feature_chns[2], self.feature_chns[2], kernel_size=2,
                       stride=2, bias=False)
        )
        # self.upscale_logits_ops = []
        # cum_upsample = np.cumprod(np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        # for usl in range(num_pool - 1):
        #     if self.upscale_logits:
        #         self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
        #                                                 mode=upsample_mode))
        #     else:
        #         self.upscale_logits_ops.append(lambda x: x)

        if not dropout_in_localization:
            self.dropout_op_kwargs['p'] = old_dropout_p

        # register all modules properly
        self.seg_outputs = nn.ModuleList(self.seg_outputs)
        # if self.upscale_logits:
        #     self.upscale_logits_ops = nn.ModuleList(
        #         self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here

        if self.weightInitializer is not None:
            self.apply(self.weightInitializer)
            # self.apply(print_module_training_status)

        self.upscale_logits_ops = []
        pool_op_kernel_sizes = [[2, 2, 2], [2, 2, 2], [2, 2, 2]]
        cum_upsample = np.cumprod(
            np.vstack(pool_op_kernel_sizes), axis=0)[::-1]
        for usl in range(3):
            if self.upscale_logits:
                # print('Upsample')
                self.upscale_logits_ops.append(Upsample(scale_factor=tuple([int(i) for i in cum_upsample[usl + 1]]),
                                                        mode=upsample_mode))
            else:
                # print('lambda')
                self.upscale_logits_ops.append(lambda x: x)
        # print(self.upscale_logits_ops)
        if self.upscale_logits:
            self.upscale_logits_ops = nn.ModuleList(
                self.upscale_logits_ops)  # lambda x:x is not a Module so we need to distinguish here
        # print(self.upscale_logits_ops)

    def forward(self, x):
        seg_outputs = []

        courseseg_map = x[:, x.shape[1]-1:, :, :, :]
        courseseg_map = (courseseg_map > 0)
        courseseg_map = courseseg_map.float()
        # print(x.shape)
        x = x[:, :x.shape[1]-1, :, :, :]
        # print('x.shape', x.shape)
        # print(x.shape)
        x1 = self.block1(x)
        x1_down = self.downsample(x1)

        x2 = self.block2(x1_down)
        x2_down = self.downsample(x2)

        x3 = self.block3(x2_down)
        x3_down = self.downsample(x3)

        bottom_x_shape = x3_down.shape
        ds_course = nn.functional.interpolate(courseseg_map, size=bottom_x_shape[2:],
                                              scale_factor=None, mode='trilinear', align_corners=True)
        x3_down = torch.cat((x3_down, ds_course), dim=1)  # x_ds_coursemap_cat

        x4 = self.block4(x3_down)
        # x4_down = self._downsample(x4)
        # print('x4.shape',x4.shape)

        x4_up = self.up4(x4)
        x4_cat = torch.cat((x4_up, x3), dim=1)
        x5 = self.block5(x4_cat)

        x5_up = self.up3(x5)
        x5_cat = torch.cat((x5_up, x2), dim=1)
        x6 = self.block6(x5_cat)

        x6_up = self.up2(x6)
        x6_cat = torch.cat((x6_up, x1), dim=1)
        x7 = self.block7(x6_cat)

        x_above_1 = self.conv1(x)
        _x_above_1 = torch.cat((x_above_1, x1), dim=1)

        x_above_2 = self.conv2(_x_above_1)
        x2_HR = self.HR2(x2)
        _x_above_2 = torch.cat((x_above_2, x2_HR), dim=1)

        x_above_3 = self.conv3(_x_above_2)
        x3_HR = self.HR3(x3)
        _x_above_3 = torch.cat((x_above_3, x3_HR), dim=1)

        x_above_4 = self.conv4(_x_above_3)
        x4_HR = self.HR4(x4)
        _x_above_4 = torch.cat((x_above_4, x4_HR), dim=1)

        x_above_5 = self.conv5(_x_above_4)
        x5_HR = self.HR5(x5)
        _x_above_5 = torch.cat((x_above_5 + x_above_3, x5_HR), dim=1)

        x_above_6 = self.conv6(_x_above_5)
        x6_HR = self.HR6(x6)
        _x_above_6 = torch.cat((x_above_6 + x_above_2, x6_HR), dim=1)

        x_above_7 = self.conv7(_x_above_6)
        x_above_7 = torch.cat((x_above_7 + x_above_1, x7), dim=1)

        # seg_outputs.append(self.final_nonlin(self.ds_conv4(x4)))

        seg_outputs.append(self.final_nonlin(self.ds_conv3(x5)))

        seg_outputs.append(self.final_nonlin(self.ds_conv2(x6)))

        seg_outputs.append(self.final_nonlin(self.ds_conv1(x_above_7)))
        #
        # for i, j in zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1]):
        #     print(i(j).shape)
        if self._deep_supervision and self.do_ds:
            return tuple([seg_outputs[-1]] + [i(j) for i, j in
                                              zip(list(self.upscale_logits_ops)[::-1], seg_outputs[:-1][::-1])])
        else:
            return seg_outputs[-1]

    @staticmethod
    def compute_approx_vram_consumption(patch_size, num_pool_per_axis, base_num_features, max_num_features,
                                        num_modalities, num_classes, pool_op_kernel_sizes, deep_supervision=False,
                                        conv_per_stage=2):
        """
        This only applies for num_conv_per_stage and convolutional_upsampling=True
        not real vram consumption. just a constant term to which the vram consumption will be approx proportional
        (+ offset for parameter storage)
        :param deep_supervision:
        :param patch_size:
        :param num_pool_per_axis:
        :param base_num_features:
        :param max_num_features:
        :param num_modalities:
        :param num_classes:
        :param pool_op_kernel_sizes:
        :return:
        """
        if not isinstance(num_pool_per_axis, np.ndarray):
            num_pool_per_axis = np.array(num_pool_per_axis)

        npool = len(pool_op_kernel_sizes)

        map_size = np.array(patch_size)
        tmp = np.int64((conv_per_stage * 2 + 1) * np.prod(map_size, dtype=np.int64) * base_num_features +
                       num_modalities * np.prod(map_size, dtype=np.int64) +
                       num_classes * np.prod(map_size, dtype=np.int64))

        num_feat = base_num_features

        for p in range(npool):
            for pi in range(len(num_pool_per_axis)):
                map_size[pi] /= pool_op_kernel_sizes[p][pi]
            num_feat = min(num_feat * 2, max_num_features)
            # conv_per_stage + conv_per_stage for the convs of encode/decode and 1 for transposed conv
            num_blocks = (conv_per_stage * 2 +
                          1) if p < (npool - 1) else conv_per_stage
            tmp += num_blocks * np.prod(map_size, dtype=np.int64) * num_feat
            if deep_supervision and p < (npool - 2):
                tmp += np.prod(map_size, dtype=np.int64) * num_classes
            # print(p, map_size, num_feat, tmp)
        return tmp
