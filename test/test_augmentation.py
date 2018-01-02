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

    @unittest.skip("tested")
    def test_Augmentation(self):
        augment = Augmentation()
        pass

    @unittest.skip("tested")
    def test_ToTensor(self):
        im = Image.open('img/test.jpg') 
        bboxes = [[10, 10, 60, 30], [50,50, 100,100]]
        im, gt = GenerateHeatmap()(im, bboxes)
        f = ToTensor()
        y, yy = f(im, gt) 
        assert isinstance(y, torch.Tensor) and isinstance(yy, torch.Tensor)

    @unittest.skip("tested")
    def test_Resize(self):
        x = Image.open('img/test.jpg')
        f = Resize(100)
        y, yy = f(x, x)

        plt.subplot(121)
        plt.imshow(x)
        plt.subplot(122)
        plt.imshow(y)
        plt.suptitle('Resize')
        plt.show()

    @unittest.skip("tested")
    def test_Fixsize(self):
        x = Image.open('img/test.jpg')
        f = Fixsize(120)
        y, yy = f(x, x)

        plt.subplot(221)
        plt.imshow(x)
        plt.subplot(222)
        plt.imshow(y)
        plt.suptitle('Fixsize')

        x = x.transpose(Image.ROTATE_90)
        y, yy = f(x, x)

        plt.subplot(223)
        plt.imshow(x)
        plt.subplot(224)
        plt.imshow(y)
        plt.suptitle('Fixsize')
        plt.show()

    @unittest.skip("tested")
    def test_Normalize(self):
        x = Image.new('RGB', (600,600))
        f1 = ToTensor()
        f2 = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        x, xx = f1(x, x)
        y, yy = f2(x, x)

    @unittest.skip("tested")
    def test_RandomHorizontalFlip(self):
        x = Image.open('img/test.jpg')
        f = RandomHorizontalFlip()
        y, yy = f(x, x)
        plt.subplot(121)
        plt.imshow(x)
        plt.subplot(122)
        plt.imshow(y)
        plt.suptitle('Random Horizontal Flip')
        plt.show()

    @unittest.skip("tested")
    def test_SemContextAugmentation(self):
        x = Image.open('img/test.jpg')
        f = SemContextAugmentation()
        y = f(x)[0]
        print("tensor size:", y.size())

    @unittest.skip("tested")
    def test_GenerateHeatmap(self):
        im = Image.open('img/test.jpg') 
        bboxes = [[60, 30]]
        im, gt = GenerateHeatmap(size=200, scale=2)(im, bboxes)
        plt.subplot(121)
        plt.imshow(im)
        # show gt
        gt = np.array(gt,dtype='float')
        gt = (gt * 255).astype(np.uint8)
        gt = gt[:,:,np.newaxis]
        gt2 = np.concatenate((gt, gt, gt), axis=2)
        plt.subplot(122)
        plt.imshow(gt2)
        plt.suptitle('Generate gaussian')
        plt.show()

    @unittest.skip("tested")
    def test__is_pil_image(self):
        pass

    @unittest.skip("tested")
    def test_unnormalize(self):
        # ndim = 3
        im = Image.open('img/test.jpg') 
        f1 = ToTensor()
        f2 = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        y = f2(f1(im)[0])[0].numpy().astype(float)
        
        maxmin_scaler = lambda x : 255 * (x - np.min(x)) / (np.max(x) - np.min(x))
        z = unnormalize(y)
        z = maxmin_scaler(z).astype(np.uint8)
        z = np.transpose(z, (1,2,0))

        plt.subplot(121)
        plt.imshow(np.array(im))
        plt.subplot(122)
        plt.imshow(z)
        plt.suptitle('unnormalize, ndim=3')
        plt.show()

        # ndim = 4
        im = Image.open('img/test.jpg') 
        f1 = ToTensor()
        f2 = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
        y = f2(f1(im)[0])[0].unsqueeze(0).numpy().astype(float)
        
        maxmin_scaler = lambda x : 255 * (x - np.min(x)) / (np.max(x) - np.min(x))
        z = unnormalize(y)[0]
        z = maxmin_scaler(z).astype(np.uint8)
        z = np.transpose(z, (1,2,0))

        plt.subplot(121)
        plt.imshow(np.array(im))
        plt.subplot(122)
        plt.imshow(z)
        plt.suptitle('unnormalize, ndim=4')
        plt.show()

    @unittest.skip("tested")
    def test_Resize_pos(self):
        im = Image.open('img/test.jpg') 
        centers = {'cls1': [[60, 30]]}
        print('old centers:', centers)

        im2, centers2 = Resize_pos(100)(im, centers)
        print('new centers:', centers2)
        plt.subplot(121)
        plt.imshow(np.array(im))
        plt.subplot(122)
        plt.imshow(np.array(im2))
        plt.suptitle('Resize')
        plt.show()

    @unittest.skip("tested")
    def test_Fixsize_pos(self):
        im = Image.open('img/test.jpg') 
        centers = {'cls1': [[60, 30]]}
        print('old centers:', centers)

        im2, centers2 = Fixsize_pos(400)(im, centers)
        print('new centers:', centers2)
        plt.subplot(121)
        plt.imshow(np.array(im))
        plt.subplot(122)
        plt.imshow(np.array(im2))
        plt.suptitle('Fixsize')
        plt.show()
        pass


    def test_PPAugmentation(self):
        im = Image.open('img/test.jpg')
        gt = {'cls1': [[60, 30], [100, 40]]}
        f = PPAugmentation()
        x, y_target = f(im, gt)
        
        if isinstance(x, torch.Tensor):
            x = unnormalize(x)
            x = x.mul_(255).numpy().astype(np.uint8)
            x = np.transpose(x, (1,2,0))
            y_target = y_target.mul_(255).numpy().astype(np.uint8)

        plt.subplot(131)
        plt.imshow(np.array(im))
        plt.subplot(132)
        plt.imshow(np.array(x))
        plt.subplot(133)
        plt.imshow(np.array(y_target))
        plt.suptitle('PPAugmentation')
        plt.show()


if __name__ == "__main__":
    unittest.main()
