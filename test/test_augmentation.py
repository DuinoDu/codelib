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
from matplotlib import pyplot as plt

class Tester(unittest.TestCase):

    def test_Augmentation(self):
        augment = Augmentation()
        pass

    def test_ToTensor(self):
        im = Image.open('img/test.jpg') 
        bboxes = [[10, 10, 60, 30], [50,50, 100,100]]
        im, gt = GenerateHeatmap()(im, bboxes)
        f = ToTensor()
        y, yy = f(im, gt) 
        assert isinstance(y, torch.Tensor) and isinstance(yy, torch.Tensor)

    def test_Resize(self):
        x = Image.open('img/test.jpg')
        f = Resize(100)
        y, yy = f(x, x)

        plt.subplot(121)
        plt.imshow(x)
        plt.subplot(122)
        plt.imshow(y)
        plt.title('Resize')
        #plt.show()

    def test_Fixsize(self):
        x = Image.open('img/test.jpg')
        f = Fixsize(120)
        y, yy = f(x, x)

        plt.subplot(221)
        plt.imshow(x)
        plt.subplot(222)
        plt.imshow(y)
        plt.title('Fixsize')

        x = x.transpose(Image.ROTATE_90)
        y, yy = f(x, x)

        plt.subplot(223)
        plt.imshow(x)
        plt.subplot(224)
        plt.imshow(y)
        plt.title('Fixsize')
        plt.show()

    def test_Normalize(self):
        x = Image.new('RGB', (600,600))
        f1 = ToTensor()
        f2 = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        x, xx = f1(x, x)
        y, yy = f2(x, x)

    def test_RandomHorizontalFlip(self):
        x = Image.open('img/test.jpg')
        f = RandomHorizontalFlip()
        y, yy = f(x, x)
        plt.subplot(121)
        plt.imshow(x)
        plt.subplot(122)
        plt.imshow(y)
        plt.title('Random Horizontal Flip')
        #plt.show()

    def test_SemContextAugmentation(self):
        x = Image.open('img/test.jpg')
        f = SemContextAugmentation()
        y = f(x)[0]
        print("tensor size:", y.size())

    def test_GenerateHeatmap(self):
        im = Image.open('img/test.jpg') 
        bboxes = [[10, 10, 60, 30], [50,50, 100,100]]
        im, gt = GenerateHeatmap()(im, bboxes)
        plt.subplot(121)
        plt.imshow(im)
        # show gt
        gt = np.array(gt,dtype='float')
        gt = (gt * 255).astype(np.uint8)
        gt = gt[:,:,np.newaxis]
        gt2 = np.concatenate((gt, gt, gt), axis=2)
        plt.subplot(121)
        plt.imshow(gt2)
        plt.title('Generate gaussian')
        #plt.show()

    def test__is_pil_image(self):
        pass

if __name__ == "__main__":
    unittest.main()
