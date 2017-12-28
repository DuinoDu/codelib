# -*- coding: utf-8 -*-
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from matplotlib import pyplot as plt
import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from augmentation import Normalize, ToTensor, unnormalize

from utils import *

class Tester(unittest.TestCase):
    
    @unittest.skip("tested")
    def test_vis_float(self):
        a = np.random.rand(12).reshape(4,3)
        print("a")
        print(a)
        print("f(a)")
        print(vis_float(a))

    @unittest.skip("tested")
    def test_imshow(self):
        im = Image.open('img/test.jpg')
        f1 = ToTensor()
        f2 = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        x = f2(f1(im)[0])[0].unsqueeze(0).numpy()
        y = torch.rand((1, 1, x.shape[2], x.shape[3])).numpy()
        y_pred = torch.rand((1, 1, x.shape[2], x.shape[3])).numpy() 
        imshow(x, y, y_pred)


    def test_merge(self):

        im = Image.open('img/test.jpg')
        f1 = ToTensor()
        f2 = Normalize(mean=[0.485, 0.456, 0.406],
                       std=[0.229, 0.224, 0.225])
        x = f2(f1(im)[0])[0].unsqueeze(0)

        x = unnormalize(x).numpy()
        y = torch.rand((1, 1, x.shape[2], x.shape[3])).numpy()
        y_pred = torch.rand((1, 1, x.shape[2], x.shape[3])).numpy() 
        merged = merge([x, y, x, y_pred], 'HWC')
        __import__('ipdb').set_trace()
        plt.imshow(merged)
        plt.suptitle('merge')
        plt.show()

        ## test with tensorboard
        #from tensorboardX import SummaryWriter
        #writer = SummaryWriter()
        #x = unnormalize(x)
        #for i in range(10):
        #    y = torch.rand((1, 1, x.shape[2], x.shape[3]))
        #    y_pred = torch.rand((1, 1, x.shape[2], x.shape[3])) 
        #    merged = merge([x, y, y_pred], 'CHW')
        #    writer.add_image('image', merged, i)
        #writer.close()


if __name__ == "__main__":
    unittest.main()
