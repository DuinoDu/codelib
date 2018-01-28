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

try:
    import accimage
except ImportError:
    accimage = None

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

def is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(is_pil_image(pic) or is_numpy_image(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        # backward compatibility
        return img.float().div(255)

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float().div(255)
    else:
        return img


def to_pil_image(pic, mode=None):
    """Convert a tensor or an ndarray to PIL Image.

    See :class:`~torchvision.transforms.ToPIlImage` for more details.

    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(is_numpy_image(pic) or is_tensor_image(pic)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    npimg = pic
    if isinstance(pic, torch.FloatTensor):
        pic = pic.mul(255).byte()
    if torch.is_tensor(pic):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a torch.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        if npimg.dtype == np.int16:
            expected_mode = 'I;16'
        if npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)


def read2tensor(imgpath, ndim=3):
    img = Image.open(imgpath)
    img = T.ToTensor()(img)
    if ndim == 4:
        img.unsqueeze_(0)
    elif ndim > 4:
        raise ValueError, 'Unknown ndim: {}'.format(ndim)
    return img


def show_tensor(x):
    if x.dim()!= 3:
        raise ValueError, 'x.ndimensions() should be 3'
    im = T.ToPILImage()(x)
    im.show()


def flip_tensor(x, index=0):
    """flip a tensor at specific index dim.
    https://github.com/pytorch/pytorch/issues/229

    Args:
        x: input tensor

    Kwargs:
        index (int): TODO

    Returns: 
        flipped tensor

    """
    __import__('ipdb').set_trace() 
    inv_idx = torch.arange(x.size(0)-1, -1, -1).long()
    # or equivalently torch.range(tensor.size(0)-1, 0, -1).long()
    inv_tensor = x.index_select(0, inv_idx)
    # or equivalently
    inv_tensor = x[inv_idx]


def resize_tensor(x, size, mode='bilinear'):
    """Resize given tensor, using mode method

    Args:
        x (tensor): input  
        size (tuple): (h, w) 

    Kwargs:
        mode (str): upsample method 

    Returns: 
        resized tensor

    """
    if len(size) != 2:
        raise ValueError, 'size should be tuple of (height, width), but got {}'.format(size)
    if x.dim() == 2:
        xx = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        xx = x.unsqueeze(0)

    xx = torch.nn.functional.upsample(xx, size=size, mode=mode).data

    if x.dim() == 2:
        xx.squeeze_(0).squeeze_(1)
    elif x.dim() == 3:
        xx.squeeze_(0)
    return xx


def unsqueeze(arr, axis=0):
    return np.expand_dims(arr, axis)


def resize_numpy(x, size):
    h, w = size
    xx = cv2.resize(x, (w, h))
    if x.ndim == 3:
        xx = unsqueeze(xx, 0)
    elif x.ndim == 4:
        xx = unsqueeze(xx, 0)
        xx = unsqueeze(xx, 0)
    return xx
