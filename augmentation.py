# -*- coding: utf-8 -*-
"""
Author: Duino
Github: github.com/duinodu
Description: Most used data augmentation methods, including many fields, such as 
             classification, object detection, segmentation and so on. Use PIL, 
             numpy, pytorch and torchvision mainly.
"""
from __future__ import division
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import warnings

def _is_pil_image(im):
    return isinstance(im, Image.Image)

##################
# classification #
##################



#####################
# objdect detection #
#####################



################
# segmentation #
################

class ToTensor(object):
    def __init__():
        self.t = T.ToTensor()
    def __call__(self, im, gt):
        return self.t(im), self.t(gt)

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.t = transfroms.Resize(size, interpolation)
    def __call__(self, im, gt):
        return self.t(im), self.t(gt)

class RandomHorizontalFlip(object):
    def __call__(self, im, gt):
        if not _is_pil_image(im):
            raise TypeError('image should be PIL Image. Got {}'.format(type(im)))
        if not _is_pil_image(gt):
            raise TypeError('gt should be PIL Image. Got {}'.format(type(gt)))
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT) 
            gt = gt.transpose(Image.FLIP_LEFT_RIGHT) 
        return im, gt

class SemContextAugmentation(object):
    def __init__(transfroms):
        self.transforms = [
                RandomHorizontalFlip(),
                Resize(300),
                ToTensor(),
                ]
    def __call__(self, im, gt):
        for t in self.transforms:
            im, gt = t(im, gt)
        return im, gt


############
# saliency #
############

class GenerateHeatmap(object):
    """
    Generate heatmap given heatmap types, such as gaussian.
    """
    def __init__(self, heatmap_type='gaussian'):
        self.fn = self._makeGaussian

    def _makeGaussian(self, size, center=None, fwhm=None):
        """
        https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
        """
        if center == None:
            x0 = y0 = size / 2
        else:
            x0 = center[0]
            y0 = center[1]
        if fwhm is None:
            fwhm = size / 2
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm ** 2)

    def __call__(self, im, bboxes):
        w, h = im.size 
        gt = np.zeros((h, w), np.float32)
        for bbox in bboxes:
            x0 = int((bbox[0] + bbox[2])/2)
            y0 = int((bbox[1] + bbox[3])/2)
            box_w = bbox[2]-bbox[0]
            box_h = bbox[3]-bbox[1]
            xmin = max(0, x0 - box_w)
            ymin = max(0, y0 - box_h)
            xmax = min(w, x0 + box_w)
            ymax = min(h, y0 + box_h)

            # use min(box_w, box_h)
            size = min(xmax-xmin, ymax-ymin)
            region = self.fn(size)
            if ymax-ymin < xmax-xmin:
                gt[ymin:ymax, ymin:ymax] = region
            else: 
                gt[xmin:xmax, xmin:xmax] = region

        return im, Image.fromarray(gt, mode='F')
