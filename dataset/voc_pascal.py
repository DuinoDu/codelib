# -*- coding: utf-8 -*-

import torch.utils.data as data
from PIL import Image

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
        gt = Image.open(self._annofile % filename)
        return im, gt


if __name__ == '__main__':
    import os
    root = os.path.join(os.environ['HOME'], 'data/VOCdevkit/VOC2012')
    ds = VOCSeg(root, 'train')
    print("dataset len :", len(ds))
    im, gt = ds[0]
    im.show()
    gt.show()

    import IPython
    IPython.embed()
