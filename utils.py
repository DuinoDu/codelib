# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch
from matplotlib import pyplot as plt
from augmentation import unnormalize

#############
# Visualize #
#############

def vis_float(x):
    """[0,1] -> [0,255], (1, H, W) -> (3, H, W)

    Args:
        x (numpy.ndarray): single-channel 2d array 

    Returns:
        (numpy.adarray)
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = (x * 255).astype(np.uint8)
    return np.concatenate((x, x, x), axis=0)

def imshow(x, y, y_pred, title='images'):
    """show x, y and y_pred for debug.

    Args:
        x (numpy.ndarray): input x, [B, C, H, W], normalized, [-1, 1]
        y (numpy.ndarray): target y, [B, 1, H, W], [0,1]
        y_pred (numpy.ndarray): predict y, [B, 1, H, W], [0,1]

    Returns: None

    """
    x = unnormalize(x[0]) 
    y = vis_float(y[0])
    y_pred = vis_float(y_pred[0])

    x = np.transpose((x*255), (1,2,0)).astype(np.uint8)
    y = np.transpose((y*255), (1,2,0)).astype(np.uint8)
    y_pred = np.transpose((y_pred*255), (1,2,0)).astype(np.uint8)

    plt.subplot(131)
    plt.imshow(x)
    plt.subplot(132)
    plt.imshow(y)
    plt.subplot(133)
    plt.imshow(y_pred)
    plt.suptitle(title)
    plt.show()

def merge(input, mode='CHW'):
    """merge input for debug.
        1. if C=1, change to 3
        2. change order if necessary
        3. concat along width 

    Args:
        input (list[np.ndarray | tensor]): input x, [B, C|1, H, W]
        mode (str): CHW | HWC

    Returns:
        merge (numpy.ndarray): [H, W*3, C] or [C, H, W*3]
    """
    using_tensor = False
    for ind, array in enumerate(input):
        arr = array[0] # batch-index = 0
        if isinstance(arr, torch.Tensor):
            arr = arr.numpy()
            using_tensor = True
        if arr.ndim == 2:
            arr = np.array([arr])
        if arr.shape[0] == 1:
            arr = np.concatenate((arr, arr, arr), axis=0)
        assert arr.shape[0] == 3, "Unknown channel nums: {}".format(arr.shape[0])
        if mode == 'HWC':
            arr = np.transpose(arr, (1,2,0))
        input[ind] = arr

    axis = 2 if mode == 'CHW' else 1
    ret = np.concatenate(input, axis=axis)
    if using_tensor:
        ret = torch.FloatTensor(ret.astype(float))
    return ret
