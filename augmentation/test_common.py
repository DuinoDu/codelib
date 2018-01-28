# -*- coding: utf-8 -*-
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
import unittest
import sys, os
from common import *
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


class Tester(unittest.TestCase):

    def test_is_pil_image(self):
        pass


    def test_is_tensor_image(self):
        pass


    def test_is_numpy_image(self):
        pass


    def test_to_tensor(self):
        pass


    def test_to_pil_image(self):
        pass


    def test_flip_tensor(self):
        pass


    def test_resize_tensor(self):
        a = read2tensor('/home/duino/Pictures/0.jpg')
        show_tensor(a)
        size = (int(a.size(1)/3), int(a.size(2)/3))
        a2 = resize_tensor(a, size)
        show_tensor(a2)



if __name__ == "__main__":
    unittest.main()
