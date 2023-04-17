# -*- coding: utf-8 -*-
from __future__ import print_function, division

import torch
import json
import math
import random
import numpy as np
from scipy import ndimage
from pymic.transform.abstract_transform import AbstractTransform
from pymic.util.image_process import *

class CenterCrop(AbstractTransform):
    """
    Crop the given image at the center.
    Input shape should be [C, D, H, W] or [C, H, W].

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `CenterCrop_output_size`: (list or tuple) The output size. 
        [D, H, W] for 3D images and [H, W] for 2D images.
        If D is None, then the z-axis is not cropped.
    :param `CenterCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        super(CenterCrop, self).__init__(params)
        self.output_size = params['CenterCrop_output_size'.lower()]
        self.inverse = params.get('CenterCrop_inverse'.lower(), True)

    def _get_crop_param(self, sample):
        input_shape = sample['image'].shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        temp_output_size = self.output_size
        if(input_dim == 3 and self.output_size[0] is None):
            # note that output size is [D, H, W] and input is [C, D, H, W]
            temp_output_size = [input_shape[1]] + self.output_size[1:]

        crop_margin = [input_shape[i + 1] - temp_output_size[i]\
            for i in range(input_dim)]
        crop_min = [int(item/2) for item in crop_margin]
        crop_max = [crop_min[i] + temp_output_size[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['CenterCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        return sample, crop_min, crop_max

    def __call__(self, sample):
        image = sample['image']
        sample, crop_min, crop_max = self._get_crop_param(sample)

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        sample['image'] = image_t
        
        if('label' in sample and self.task == 'segmentation'):
            label = sample['label']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            sample['label'] = label
        if('pixel_weight' in sample and self.task == 'segmentation'):
            weight = sample['pixel_weight']
            crop_max[0] = weight.shape[0]
            weight = crop_ND_volume_with_bounding_box(weight, crop_min, crop_max)
            sample['pixel_weight'] = weight
        return sample

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['CenterCrop_Param'], list) or \
            isinstance(sample['CenterCrop_Param'], tuple)):
            params = json.loads(sample['CenterCrop_Param'][0]) 
        else:
            params = json.loads(sample['CenterCrop_Param']) 
        return params

    def inverse_transform_for_prediction(self, sample):
        params = self._get_param_for_inverse_transform(sample)
        origin_shape = params[0]
        crop_min     = params[1]
        crop_max     = params[2]
        predict = sample['predict']
        if(isinstance(predict, tuple) or isinstance(predict, list)):
            output_predict = []
            for predict_i in predict:
                origin_shape_i   = list(predict_i.shape[:2]) + origin_shape[1:]
                output_predict_i = np.zeros(origin_shape_i, predict_i.dtype)
                crop_min_i = [0, 0] + crop_min[1:]
                crop_max_i = list(predict_i.shape[:2]) + crop_max[1:]
                output_predict_i = set_ND_volume_roi_with_bounding_box_range(output_predict_i,
                    crop_min_i, crop_max_i, predict_i)
                output_predict.append(output_predict_i)
        else:
            origin_shape   = list(predict.shape[:2]) + origin_shape[1:]
            output_predict = np.zeros(origin_shape, predict.dtype)
            crop_min = [0, 0] + crop_min[1:]
            crop_max = list(predict.shape[:2]) + crop_max[1:]
            output_predict = set_ND_volume_roi_with_bounding_box_range(output_predict,
                crop_min, crop_max, predict)
        
        sample['predict'] = output_predict
        return sample

class CropWithBoundingBox(CenterCrop):
    """
    Crop the image (shape [C, D, H, W] or [C, H, W]) based on a bounding box.
    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `CropWithBoundingBox_start`: (None, or list/tuple) The start index 
        along each spatial axis. If None, calculate the start index automatically 
        so that the cropped region is centered at the non-zero region.
    :param `CropWithBoundingBox_output_size`: (None or tuple/list): 
        Desired spatial output size.
        If None, set it as the size of bounding box of non-zero region.
    :param `CropWithBoundingBox_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        self.start       = params['CropWithBoundingBox_start'.lower()]
        self.output_size = params['CropWithBoundingBox_output_size'.lower()]
        self.inverse     = params.get('CropWithBoundingBox_inverse'.lower(), True)
        self.task = params['task']
        
    def _get_crop_param(self, sample):
        image = sample['image']
        input_shape = sample['image'].shape
        input_dim   = len(input_shape) - 1
        bb_min, bb_max = get_ND_bounding_box(image)
        bb_min, bb_max = bb_min[1:], bb_max[1:]
        if(self.start is None):
            if(self.output_size is None):
                crop_min, crop_max = bb_min, bb_max
            else:
                assert(len(self.output_size) == input_dim)
                crop_min = [int((bb_min[i] + bb_max[i] + 1)/2) - int(self.output_size[i]/2) \
                    for i in range(input_dim)]
                crop_min = [max(0, crop_min[i]) for i in range(input_dim)]
                crop_max = [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        else:
            assert(len(self.start) == input_dim)
            crop_min = self.start
            if(self.output_size is None):
                assert(len(self.output_size) == input_dim)
                crop_max = [crop_min[i] + bb_max[i] - bb_min[i] \
                    for i in range(input_dim)]
            else:
                crop_max =  [crop_min[i] + self.output_size[i] for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['CropWithBoundingBox_Param'] = json.dumps((input_shape, crop_min, crop_max))   
        print("for crop", crop_min, crop_max)
        return sample, crop_min, crop_max

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['CropWithBoundingBox_Param'], list) or \
            isinstance(sample['CropWithBoundingBox_Param'], tuple)):
            params = json.loads(sample['CropWithBoundingBox_Param'][0]) 
        else:
            params = json.loads(sample['CropWithBoundingBox_Param']) 
        return params
        

class RandomCrop(CenterCrop):
    """Randomly crop the input image (shape [C, D, H, W] or [C, H, W]).

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomCrop_output_size`: (list/tuple) Desired output size [D, H, W] or [H, W].
        The output channel is the same as the input channel. 
        If D is None for 3D images, the z-axis is not cropped.
    :param `RandomCrop_foreground_focus`: (optional, bool) 
        If true, allow crop around the foreground. Default is False.
    :param `RandomCrop_foreground_ratio`: (optional, float) 
        Specifying the probability of foreground focus cropping when 
        `RandomCrop_foreground_focus` is True.
    :param `RandomCrop_mask_label`: (optional, None, or list/tuple) 
        Specifying the foreground labels for foreground focus cropping when 
        `RandomCrop_foreground_focus` is True.
    :param `RandomCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `True`.
    """
    def __init__(self, params):
        self.output_size = params['RandomCrop_output_size'.lower()]
        self.fg_focus    = params.get('RandomCrop_foreground_focus'.lower(), False)
        self.fg_ratio    = params.get('RandomCrop_foreground_ratio'.lower(), 0.5)
        self.mask_label  = params.get('RandomCrop_mask_label'.lower(), [1])
        self.inverse     = params.get('RandomCrop_inverse'.lower(), True)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        if(self.mask_label is not None):
            assert isinstance(self.mask_label, (list, tuple))

    def _get_crop_param(self, sample):
        image       = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == len(self.output_size))
        temp_output_size = self.output_size
        if(input_dim == 3 and self.output_size[0] is None):
            # note that output size is [D, H, W] and input is [C, D, H, W]
            temp_output_size = [input_shape[1]] + self.output_size[1:]

        crop_margin = [input_shape[i + 1] - temp_output_size[i]\
            for i in range(input_dim)]
        crop_min = [0 if item == 0 else random.randint(0, item) for item in crop_margin]
        if(self.fg_focus and random.random() < self.fg_ratio):
            label = sample['label']
            mask  = np.zeros_like(label)
            for temp_lab in self.mask_label:
                mask = np.maximum(mask, label == temp_lab)
            if(mask.sum() == 0):
                bb_min = [0] * (input_dim + 1)
                bb_max = mask.shape
            else:
                bb_min, bb_max = get_ND_bounding_box(mask)
            bb_min, bb_max = bb_min[1:], bb_max[1:]
            crop_min = [random.randint(bb_min[i], bb_max[i]) - int(temp_output_size[i]/2) \
                for i in range(input_dim)]
            crop_min = [max(0, item) for item in crop_min]
            crop_min = [min(crop_min[i], input_shape[i+1] - temp_output_size[i]) \
                for i in range(input_dim)]

        crop_max = [crop_min[i] + temp_output_size[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['RandomCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        return sample, crop_min, crop_max

    def _get_param_for_inverse_transform(self, sample):
        if(isinstance(sample['RandomCrop_Param'], list) or \
            isinstance(sample['RandomCrop_Param'], tuple)):
            params = json.loads(sample['RandomCrop_Param'][0]) 
        else:
            params = json.loads(sample['RandomCrop_Param']) 
        return params

class RandomResizedCrop(CenterCrop):
    """
    Randomly crop the input image (shape [C, H, W]). Only 2D images are supported.

    The arguments should be written in the `params` dictionary, and it has the
    following fields:

    :param `RandomResizedCrop_output_size`: (list/tuple) Desired output size [H, W].
        The output channel is the same as the input channel. 
    :param `RandomResizedCrop_scale`: (list/tuple) Range of scale, e.g. (0.08, 1.0).
    :param `RandomResizedCrop_ratio`: (list/tuple) Range of aspect ratio, e.g. (0.75, 1.33).
    :param `RandomResizedCrop_inverse`: (optional, bool) Is inverse transform needed for inference.
        Default is `False`. Currently, the inverse transform is not supported, and 
        this transform is assumed to be used only during training stage. 
    """
    def __init__(self, params):
        self.output_size = params['RandomResizedCrop_output_size'.lower()]
        self.scale       = params['RandomResizedCrop_scale'.lower()]
        self.ratio       = params['RandomResizedCrop_ratio'.lower()]
        self.inverse     = params.get('RandomResizedCrop_inverse'.lower(), False)
        self.task        = params['Task'.lower()]
        assert isinstance(self.output_size, (list, tuple))
        assert isinstance(self.scale, (list, tuple))
        assert isinstance(self.ratio, (list, tuple))
        
    def _get_crop_param(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        assert(input_dim == 2)
        assert(input_dim == len(self.output_size))
        
        scale = self.scale[0] + random.random()*(self.scale[1] - self.scale[0])
        ratio = self.ratio[0] + random.random()*(self.ratio[1] - self.ratio[0])
        crop_w = input_shape[-1] * scale 
        crop_h = crop_w * ratio
        crop_h = min(crop_h, input_shape[-2])
        output_shape = [int(crop_h), int(crop_w)]

        crop_margin = [input_shape[i + 1] - output_shape[i]\
            for i in range(input_dim)]
        crop_min = [random.randint(0, item) for item in crop_margin]
        crop_max = [crop_min[i] + output_shape[i] \
            for i in range(input_dim)]
        crop_min = [0] + crop_min
        crop_max = list(input_shape[0:1]) + crop_max
        sample['RandomResizedCrop_Param'] = json.dumps((input_shape, crop_min, crop_max))
        return sample, crop_min, crop_max

    def __call__(self, sample):
        image = sample['image']
        input_shape = image.shape
        input_dim   = len(input_shape) - 1
        sample, crop_min, crop_max = self._get_crop_param(sample)

        image_t = crop_ND_volume_with_bounding_box(image, crop_min, crop_max)
        crp_shape = image_t.shape
        scale = [(self.output_size[i] + 0.0)/crp_shape[1:][i] for i in range(input_dim)]
        scale = [1.0] + scale
        image_t = ndimage.interpolation.zoom(image_t, scale, order = 1)
        sample['image'] = image_t
        
        if('label' in sample and self.task == 'segmentation'):
            label = sample['label']
            crop_max[0] = label.shape[0]
            label = crop_ND_volume_with_bounding_box(label, crop_min, crop_max)
            label = ndimage.interpolation.zoom(label, scale, order = 0)
            sample['label'] = label
        if('pixel_weight' in sample and self.task == 'segmentation'):
            weight = sample['pixel_weight']
            crop_max[0] = weight.shape[0]
            weight = crop_ND_volume_with_bounding_box(weight, crop_min, crop_max)
            weight = ndimage.interpolation.zoom(weight, scale, order = 1)
            sample['pixel_weight'] = weight
        return sample