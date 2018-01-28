# -*- coding: utf-8 -*-
"""
Author: Duino
Github: github.com/duinodu
Description: Most used data augmentation methods, including many fields,
             such as classification, object detection, segmentation
             and so on. Use PIL, numpy, pytorch and torchvision mainly.
"""
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from numpy import random
from torchvision import transforms as T
import cv2
from PIL import Image
import common 


class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):

        self.t = T.Resize(size, interpolation)

    def __call__(self, im, gt=None):
        return self.t(im), self.t(gt)


class ResizeTarget(object):
    def __init__(self, size, mode='bilinear'):
        self.size = (size, size)
        self.mode = mode
        self.t = common.resize_tensor

    def __call__(self, im, gt=None):
        return im, self.t(gt, self.size, self.mode)


class Fixsize(object):
    def __init__(self, size, value=[0,0,0]):

        self.size = (size, size)
        self.value = tuple([int(x) for x in value])


    def __call__(self, im, gt=None):
        is_pil = False
        if common.is_pil_image(im):
            is_pil = True
            im = common.to_tensor(im) 

        im_bg = torch.zeros((im.size(0), self.size[0], self.size[1]))
        im_bg[0].fill_(self.value[0])
        im_bg[1].fill_(self.value[1])
        im_bg[2].fill_(self.value[2])

        factor = max(im.size(1), im.size(2)) / self.size[0]
        h_new = int(im.size(1) / factor)
        w_new = int(im.size(2) / factor)

        # image
        origin_x = origin_y = 0
        if w_new > h_new:
            origin_y = int((self.size[0] - h_new)/2)
        else:
            origin_x = int((self.size[1] - w_new)/2)
        im = common.resize_tensor(im, (h_new, w_new))
        im_bg[:, origin_y:origin_y+h_new, origin_x:origin_x+w_new] = im 

        # gt
        gt_bg = None
        if gt is not None:
            gt_bg = []
            for g in gt:
                g_bg = np.zeros((self.size[0], self.size[1])).astype(np.float)
                g = common.resize_numpy(g, (h_new, w_new))
                g_bg[origin_y:origin_y+h_new, origin_x:origin_x+w_new] = g 
                gt_bg.append(g_bg)

        if is_pil:
            im_bg = common.to_pil_image(im_bg)

        return im_bg, gt_bg


class RandomHorizontalFlip(object):
    def __call__(self, im, gt=None):
        if not common.is_pil_image(im):
            raise TypeError('image should be PIL Image. \
                    Got {}'.format(type(im)))
        if gt is not None and not common.is_pil_image(gt[0]):
            raise TypeError('gt should be PIL Image. Got {}'.format(type(gt[0])))
        if ramdom.random() > 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if gt is not None:
                gt = [common.flip_tensor(g, index=1) for g in gt]
        return im, gt


class ToTensor(object):
    def __init__(self):

        self.t = T.ToTensor()

    def __call__(self, im, gt=None):
        gt_tensor = None
        if gt is not None:
            gt_tensor = [torch.Tensor(g).unsqueeze_(0) for g in gt] 
        return self.t(im), torch.cat(gt_tensor, 0) 


class Normalize(object):
    def __init__(self, mean, std):

        self.t = T.Normalize(mean, std)
        pass

    def __call__(self, im, gt=None):
        return self.t(im), gt


class VOCSegAugmentation(object):
    def __init__(self, size=300):

        self.transforms = [
                Resize(size),
                RandomHorizontalFlip(),
                ToTensor(),
                ]

    def __call__(self, im, gt):
        for t in self.transforms:
            im, gt = t(im, gt)
        return im, gt


class SemContextAugmentation(object):
    def __init__(self):

        self.transforms = [
                Fixsize(512),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])]

    def __call__(self, im, gt=None):
        for t in self.transforms:
            im, gt = t(im, gt)
        return im, gt


