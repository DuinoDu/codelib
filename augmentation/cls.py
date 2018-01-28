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
import types
import warnings


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


