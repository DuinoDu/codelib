# -*- coding: utf-8 -*-

import torch.utils.data as data
from PIL import Image
import numpy as np
from itertools import izip


class VOCSeg(data.Dataset):
    """ pascal-voc segmentation dataset

    Arguments:
        root (str): root of dataset.

        image_sets(str): 'train' | 'val' | 'trainval'

        transform(torchvision.transforms.Compose): data-augmentation for x

        target_transform(torchvision.transforms.Compose): data-augmentation for both x and groundtruth

    """

    def __init__(self, root, image_sets, transform=None, target_transform=None):
        self.root = root
        self.image_set = image_sets
        self.transform = transform # data-aug
        self.target_transform = target_transform
        self.name = 'voc-segmentation'
        self._imgfile = os.path.join(root, 'JPEGImages/%s.jpg') 
        annopath = os.path.join(root, 'SegmentationClassAug')
        if not os.path.exists(annopath):
            print("SegmentationClassAug not found")
            print("You can download from https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0")
            print("Use SegmentationClass instead.")
            annopath = os.path.join(root, 'SegmentationClass')
        self._annofile = os.path.join(annopath, '%s.png')

        splitfile = os.path.join(root, 'ImageSets/Segmentation/%s.txt' % image_sets) 
        self.ids = [x.rstrip() for x in open(splitfile).readlines()] 


    def __len__(self):
        return len(self.ids)


    def __getitem__(self, index):
        filename = self.ids[index]
        im = Image.open(self._imgfile % filename)
        gt = Image.open(self._annofile % filename).convert('RGB')
        gt = self._convert2label(gt)
        if self.transform != None:
            im = self.transform(im)
        if self.target_transform != None:
            im, gt = self.target_transform(im, gt)

        return im, gt


    def _get_pascal_labels(self):
        """
        https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loader/pascal_voc_loader.py
        """
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                           [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                           [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                           [0,192,0], [128,192,0], [0,64,128]])


    def _convert2label(self, gt):
        """convert gt from RGB to single-channel, label starts from 0,1,2...

        Args:
            gt (Image): input RGB image

        Returns:
            (Image): single-channle image

        """
        pass
        gt = np.array(gt)
        gt_label = np.zeros((gt.shape[0], gt.shape[1]), dtype=np.int16)
        for i, label in enumerate(self._get_pascal_labels()):
            gt_label[np.where(np.all(gt == label, axis=-1))[:2]] = i
        return Image.fromarray(gt_label.astype(np.uint8), mode='L')


    def _convert2vis(self, gt):
        gt = np.array(gt)
        gt_vis = np.zeros((gt.shape[0], gt.shape[1], 3), dtype=np.int16)
        for i, label in enumerate(self._get_pascal_labels()):
            for y,x in izip(np.where(gt==i)[0], np.where(gt==i)[1]):
                gt_vis[y, x] = label
        return Image.fromarray(gt_vis.astype(np.uint8), mode='RGB')


if __name__ == '__main__':
    import os
    root = os.path.join(os.environ['HOME'], 'data/VOCdevkit/VOC2012')
    ds = VOCSeg(root, 'train')
    print("dataset len :", len(ds))
    im, gt = ds[0]
    im.show()
    ds._convert2vis(gt).show()

    # display
    print('histogram:')
    print(gt.histogram())

    import IPython
    IPython.embed()
