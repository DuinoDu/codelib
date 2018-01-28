# -*- coding: utf-8 -*-
from __future__ import division
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image


#########################
# simple data transform #
#########################

def unnormalize(x, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Unnormalize a tensor or ndarray.

    Args:
        x (tensor | ndarray): [B, C, H, W]

    Kwargs:
        mean (list): mean 
        std (list): std

    Returns: 
        unnormalized tensor or ndarray

    """

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


def intersect(box_a, box_b):
    """Compute inter area between box_a and box_b

    Args:
        box_a (np.ndarray): [N, 4]
        box_b (np.ndarray): [4]

    Returns:
        inter area (np.ndarray): [N]

    """
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: Multiple bounding boxes, Shape: [num_boxes,4]
        box_b: Single bounding box, Shape: [4]
    Return:
        jaccard overlap: Shape: [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]



#############
# Visualize #
#############

def vis_float(x, order='HWC'):
    """[0,1] -> [0,255], (1, H, W) -> (3, H, W)

    Args:
        x (numpy.ndarray): single-channel 2d array 

    Returns:
        (numpy.adarray)
    """
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    x = (x * 255).astype(np.uint8)
    if x.ndim == 2:
        x = np.array([x])
    x = np.concatenate((x, x, x), axis=0)
    if order == 'HWC':
        x = np.transpose(x, (1,2,0))
    return x

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



#########################
# save voc-foramt dataset 
#########################
Annotation = """<annotation>
	<folder>BBoxLabel</folder>
	<filename>{}</filename>
	<source>
		<database>The VOC2007 Database</database>
		<annotation>PASCAL VOC2007</annotation>
		<image>flickr</image>
		<flickrid>341012865</flickrid>
	</source>
	<size>
		<width>{}</width>
		<height>{}</height>
		<depth>3</depth>
	</size>
	<segmented>0</segmented>
	{}
</annotation>
"""
Object = """
	<object>
		<name>{}</name>
		<pose>Left</pose>
		<truncated>0</truncated>
		<difficult>0</difficult>
		<bndbox>
			<xmin>{}</xmin>
			<ymin>{}</ymin>
			<xmax>{}</xmax>
			<ymax>{}</ymax>
		</bndbox>
	</object>
"""

