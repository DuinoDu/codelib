# -*- coding: utf-8 -*-
from __future__ import division
from __future__ import print_function
import torch
import numpy as np
from numpy import random
from torchvision import transforms as T
import cv2
from PIL import Image
import common 
import unittest
import sys, os
from seg import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

class Tester(unittest.TestCase):

    def test_Resize(self):
        pass

    def test_ResizeTarget(self):
        pass

    def test_Fixsize(self):
        im = Image.open('/home/duino/Pictures/0.jpg')
        gt = [Image.new('L', im.size)]
        f = Fixsize(256, value=[100,100,0])

        im, gt = f(im, gt)
        im.show()
        gt[0].show()

    def test_RandomHorizontalFlip(self):
        pass

    def test_ToTensor(self):
        pass

    def test_Normalize(self):
        pass

    def test_VOCSegAugmentation(self):
        pass

    def test_SemContextAugmentation(self):
        pass


if __name__ == "__main__":
    unittest.main()
