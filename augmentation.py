# -*- coding: utf-8 -*-
"""
Author: Duino
Github: github.com/duinodu
Description: Most used data augmentation methods, including many fields,
             such as classification, object detection, segmentation
             and so on. Use PIL, numpy, pytorch and torchvision mainly.
"""
from __future__ import division
import torch
import numpy as np
from PIL import Image
from torchvision import transforms as T
import warnings
import random


"""
Some most-used points:
1. PIL mode:
    l:      1-bit per pixel
    L:      8-bit per pixel
    P:      palette encoding
    RGB:    red-green-blue color, 3 bytes per pixel
    "I":    32-bit int pixels
    "F":    32-bit float pixels
    RGBA:   plus A [0,255], 4 bytes per pixel
    CMYK/YCbCr

"""


def _is_pil_image(im):
    return isinstance(im, Image.Image)


def unnormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    is_tensor = False
    if isinstance(x, torch.Tensor):
        x = x.numpy()
        is_tensor = True

    if not isinstance(x, np.ndarray):
        raise TypeError('input x is not a numpy.ndarray or torch.Tensor.')

    if x.ndim == 4:
        for i in range(x.shape[0]):
            x[i] = unnormalize(x[i], mean, std)
    elif x.ndim == 3:
        if x.shape[0] != len(mean):
            raise ValueError('input should have same channels. x is %d, \
                    mean is %d' % v(x.shape[0], len(mean)))
        for i in range(x.shape[0]):
            x[i] = x[i] * std[i] + mean[i]

    ret = torch.FloatTensor(x) if is_tensor else x
    return ret


def to_pil(tensor, mode):
    """Convert a tensor to PIL Image.

    Args:
        tensor (Tensor): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth
                                  of input data (optional).
        PIL.Image mode:
http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError('tensor should be Tensor. \
                Got {}.'.format(type(tensor)))

    tensor = tensor.mul(255).byte()
    npimg = np.transpose(tensor.numpy(), (1, 2, 0))
    return Image.fromarray(npimg, mode=mode)


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

class Resize(object):
    def __init__(self, size, interpolation=Image.BILINEAR):

        self.t = T.Resize(size, interpolation)

    def __call__(self, im, gt=None):
        return self.t(im), self.t(gt)


class Fixsize(object):
    def __init__(self, size):

        self.size = (size, size)

    def _fixsize_PIL(self, im, gt):
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
        if gt is not None:
            gt_bg = Image.new(gt.mode, self.size)
            gt = gt.resize((w_new, h_new))
            gt_bg.paste(gt, (origin_x, origin_y, origin_x + w_new, origin_y + h_new))

        return im_bg, gt_bg

    def _fixsize_tensor(self, im, gt):
        im_bg = torch.zeros((im.size(0), self.size[0], self.size[1]))
        factor = max(im.size(1), im.size(2)) / self.size[0]
        h_new = int(im.size(1) / factor)
        w_new = int(im.size(2) / factor)

        resize = lambda x : torch.nn.functional.upsample(x.unsqueeze_(0), \
                                size=(h_new, w_new), mode='bilinear').data.squeeze_(0)
        origin_x = origin_y = 0
        if w_new > h_new:
            origin_y = int((self.size[0] - h_new)/2)
        else:
            origin_x = int((self.size[1] - w_new)/2)

        im = resize(im)
        im_bg[:, origin_y:origin_y+h_new, origin_x:origin_x+w_new] = im 

        gt_bg = None
        if gt is not None:
            gt_bg = torch.zeros((self.size[0], self.size[1]))
            gt = resize(gt.unsqueeze_(0)).squeeze_(0)
            gt_bg[origin_y:origin_y+h_new, origin_x:origin_x+w_new] = gt 

        return im_bg, gt_bg


    def __call__(self, im, gt=None):

        if isinstance(im, torch.Tensor):
            # im: [3, h, w]
            # gt: [h, w]
            return self._fixsize_tensor(im, gt)
        else:
            return self._fixsize_PIL(im, gt)


class RandomHorizontalFlip(object):
    def __call__(self, im, gt=None):
        if not _is_pil_image(im):
            raise TypeError('image should be PIL Image. \
                    Got {}'.format(type(im)))
        if gt is not None and not _is_pil_image(gt):
            raise TypeError('gt should be PIL Image. Got {}'.format(type(gt)))
        if random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            if gt is not None:
                gt = gt.transpose(Image.FLIP_LEFT_RIGHT)
        return im, gt


class ToTensor(object):
    def __init__(self):

        self.t = T.ToTensor()

    def __call__(self, im, gt=None):
        if gt is not None:
            if gt.mode == 'F' and np.max(np.array(gt)) <= 1.0:
                gt_tensor = torch.FloatTensor(np.array(gt, dtype=np.float32))
            else:
                gt_tensor = self.t(gt)
        else:
            gt_tensor = None
        return self.t(im), gt_tensor


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


###############
# predict pos #
###############

class Resize_pos(object):
    """Resize image and centers 

        centers (dict): {'cls1':[[], [],..], 'cls2':[[], []..]}
                                 [cx, cy]
    """
    def __init__(self, size, interpolation=Image.BILINEAR):

        self.t = T.Resize(size, interpolation)

    def __call__(self, im, centers=None):
        # resize image
        im_r = self.t(im)
        # resize centers
        if centers is not None:
            if not isinstance(centers, dict):
                raise ValueError("centers should be dict, but get {}".format(type(centers)))
            factor = im_r.width / im.width
            for key in centers.keys():
                for ind, c in enumerate(centers[key]):
                    centers[key][ind][0] = int(centers[key][ind][0] * factor)
                    centers[key][ind][1] = int(centers[key][ind][1] * factor)
        return im_r, centers


class GenerateHeatmap(object):
    """
    Generate heatmap given heatmap types, such as gaussian.
    """
    def __init__(self, heatmap_type='gaussian', scale=1, size=100):
        self.fn = self._makeGaussian
        self.scale = scale
        self.size = size


    def _makeGaussian(self, size, center=None, fwhm=None):
        """
        https://stackoverflow.com/questions/7687679/how-to-generate-2d-gaussian-with-python
        """
        if center is None:
            x0 = y0 = size / 2
        else:
            x0 = center[0]
            y0 = center[1]
        if fwhm is None:
            fwhm = size / 2
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm ** 2)


    def __call__(self, im, centers):
        w, h = im.size
        gt = np.zeros((h, w), np.float32)

        # change dict to list
        centers_list = [] 
        for v in centers.values():
            centers_list += v

        for c in centers_list:
            x0 = c[0] 
            y0 = c[1]

            self.size *= self.scale
            region = self.fn(self.size)

            # pos for region in gt
            r_xmin = r_ymin = 0
            r_xmax = r_ymax = self.size
            r_center = int(self.size / 2)
            x_offset = x0 - r_center
            y_offset = y0 - r_center
            r_xmin += x_offset
            r_xmax += x_offset
            r_ymin += y_offset
            r_ymax += y_offset

            # pos in gt
            gt_xmin = max(0, r_xmin)
            gt_ymin = max(0, r_ymin)
            gt_xmax = min(w, r_xmax)
            gt_ymax = min(h, r_ymax)

            # pos in region
            rr_xmin = gt_xmin - r_xmin
            rr_ymin = gt_ymin - r_ymin
            rr_xmax = gt_xmax - r_xmax + self.size
            rr_ymax = gt_ymax - r_ymax + self.size

            gt[gt_ymin:gt_ymax, gt_xmin:gt_xmax] = region[rr_ymin:rr_ymax, rr_xmin:rr_xmax]

        return im, Image.fromarray(gt, mode='F')


class Fixsize_pos(object):
    def __init__(self, size):

        self.size = (size, size)

    def __call__(self, im, centers=None):
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

        if centers is not None:
            for key in centers.keys():
                for ind, c in enumerate(centers[key]):
                    centers[key][ind][0] = int(centers[key][ind][0] / factor) + origin_x
                    centers[key][ind][1] = int(centers[key][ind][1] / factor) + origin_y

        return im_bg, centers 


class PPAugmentation(object):
    def __init__(self):

        self.transforms = [
                Resize_pos(256),
                #RandomHorizontalFlip_pos(), # not implemented
                GenerateHeatmap(heatmap_type='gaussian', size=150), # Image, dict -> Image, Image
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
                Fixsize(256),
                ]

    def __call__(self, im, gt=None):
        for t in self.transforms:
            im, gt = t(im, gt)
        return im, gt
