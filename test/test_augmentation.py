# -*- coding: utf-8 -*-
from __future__ import division
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms as T
import warnings
import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from augmentation import *

class Tester(unittest.TestCase):

    def test_ToTensor(self):
        pass

    def test_Resize(self):
        pass

    def test_RandomHorizontalFlip(self):
        pass

    def test_SemContextAugmentation(self):
        pass

    def test_GenerateHeatmap(self):
        im = Image.open('test/img/test.jpg') 
        print(im.size)
        bboxes = [[10, 10, 60, 30], [50,50, 100,100]]
        im, gt = GenerateHeatmap()(im, bboxes)
        # show gt
        gt = np.array(gt,dtype='float')
        gt = (gt * 255).astype(np.uint8)
        gt = gt[:,:,np.newaxis]
        gt2 = np.concatenate((gt, gt, gt), axis=2)
        plt.imshow(gt2)
        plt.show()

    def test__is_pil_image(self):
        pass

if __name__ == "__main__":
    unittest.main()
