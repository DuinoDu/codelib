# -*- coding: utf-8 -*-
from __future__ import division
import torch
import numpy as np
from numpy import random
from torchvision import transforms as T
import cv2
from PIL import Image
import common
import unittest
import sys, os
from keypoint import *

class Tester(unittest.TestCase):

    def test_ToTensor(self):
        pass

    def test_BBox2Point(self):
        pass

    def test_Resize(self):
        pass

    def test_RandomHorizontalFlip(self):
        pass

    def test_GenerateHeatmap(self):
        pass

    def test_Fixsize(self):
        pass

    def test_PPAugmentation(self):
        pass


if __name__ == "__main__":
    unittest.main()
