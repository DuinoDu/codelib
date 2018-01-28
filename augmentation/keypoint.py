# -*- coding: utf-8 -*-
"""
Author: Duino
Github: github.com/duinodu
Description: 
    Most used data augmentation method for keypoint.
    Use PIL, numpy, pytorch and torchvision mainly.
"""
from __future__ import division
import torch
import numpy as np
from numpy import random
from torchvision import transforms as T
import cv2
from PIL import Image

import common
import seg


class ToTensor(object):
    def __init__():
        self.t = T.ToTensor()

    def __call__(self, im, centers_dict):
        return self.t(im), centers_dict 


class BBox2Point(object):
    """Convert bbox to point 

    Args:
        bboxes (list): [[xmin, ymin, xmax, ymax, cls_id], [..], ...]

    Returns: 
        center_dict (dict): {'id1':[[], [],..], 'id2':[[], []..]}
                                     [cx, cy]
    """
    def __call__(self, im, bboxes):
        center_dict = {}
        for bbox in bboxes:
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            cls_id = int(bbox[-1])
            if cls_id not in center_dict.keys():
                center_dict[cls_id] = [[cx, cy]] 
            else:
                center_dict[cls_id].append([cx, cy])
        return im, center_dict


class Resize(object):
    """Resize image and centers 

        centers (dict): {'cls1':[[], [],..], 'cls2':[[], []..]}
                                 [cx, cy]
    """
    def __init__(self, size, interpolation=Image.BILINEAR):

        self.size = size
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


class RandomHorizontalFlip(object):
    def __call__(self, img, centers_dict):
        if not common.is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        width = img.width
        if random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            for key in centers_dict.keys():
                for ind, pts in enumerate(centers_dict[key]):
                    pts[0] = min(width, max(0, width-pts[0]))
                    centers_dict[key][ind] = pts
        return img, centers_dict


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
        gts = [np.zeros((h, w), np.float32) for x in range(len(centers.keys()))]

        for ind, key in enumerate(sorted(centers.keys())):
            for c in centers[key]:
                x0 = int(c[0])
                y0 = int(c[1])

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

                gts[ind][gt_ymin:gt_ymax, gt_xmin:gt_xmax] = region[rr_ymin:rr_ymax, rr_xmin:rr_xmax]

        return im, gts 


class Fixsize(object):
    def __init__(self, size, value=[0,0,0]):

        self.size = (size, size)
        self.value = [int(x) for x in value]

    def __call__(self, im, centers=None):
        im_bg = Image.new(im.mode, self.size, color=self.value)
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



