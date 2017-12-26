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
import random

def _is_pil_image(im):
    return isinstance(im, Image.Image)

def unnormalize(tensor, mean, std):
    if not (torch.is_tensor(img) and img.ndimension() == 3):
        raise TypeError('tensor is not a torch image.')
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor
        


##################
# classification #
##################

class Augmentation(object):
    def __init__(self, means=()):
        self.transforms = [
                ]
    def __call__(self, im):
        for t in self.transforms:
            im = t(im)
        return im


#####################
# objdect detection #
#####################


################
# segmentation #
################

class ToTensor(object):
    def __init__(self):
        self.t = T.ToTensor()
    def __call__(self, im, gt=None):
        if gt != None:
            if gt.mode == 'F' and np.max(np.array(gt)) <= 1.0:
                gt_tensor = torch.FloatTensor(np.array(gt, dtype=np.float32))
            else:
                gt_tensor = self.t(gt)
        else:
            gt_tensor = None
        return self.t(im), gt_tensor 

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.t = T.Resize(size, interpolation)
    def __call__(self, im, gt=None):
        return self.t(im), self.t(gt)

class Fixsize(object):
    def __init__(self, size):
        self.size = (size, size)
    def __call__(self, im, gt=None):
        im_bg = Image.new(im.mode, self.size)
        factor = max(im.size) / self.size[0] 
        w_new = int(im.size[0] / factor)
        h_new = int(im.size[1] / factor)
        im = im.resize((w_new, h_new))
        
        origin_x = origin_y = 0
        if w_new > h_new:
            origin_y = int((self.size[1] - h_new)/2)
        else:
            origin_x = int((self.size[0] - w_new)/2)
        im_bg.paste(im, (origin_x, origin_y, origin_x + w_new, origin_y + h_new))

        gt_bg = None
        if gt != None:
            gt_bg = Image.new(gt.mode, self.size)
            gt = gt.resize((w_new, h_new))
            gt_bg.paste(gt, (origin_x, origin_y, origin_x + w_new, origin_y + h_new))

        return im_bg, gt_bg

class Normalize(object):
    def __init__(self, mean, std):
        self.t = T.Normalize(mean, std)
        pass
    def __call__(self, im, gt=None):
        return self.t(im), gt

class RandomHorizontalFlip(object):
    def __call__(self, im, gt=None):
        if not _is_pil_image(im):
            raise TypeError('image should be PIL Image. Got {}'.format(type(im)))
        if gt != None and not _is_pil_image(gt):
            raise TypeError('gt should be PIL Image. Got {}'.format(type(gt)))
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT) 
            if gt != None:
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT) 
        return im, gt

class SemContextAugmentation(object):
    def __init__(self):
        self.transforms = [
                Fixsize(300),
                RandomHorizontalFlip(),
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])]

    def __call__(self, im, gt=None):
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
            X = xmax - xmin
            Y = ymax - ymin
            if X > Y:
                gt[ymin:ymax, xmin:xmin+Y] = region
            else: 
                gt[ymin:ymin+X, xmin:xmax] = region

        return im, Image.fromarray(gt, mode='F')
