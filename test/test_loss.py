# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import unittest
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from loss import *

class Tester(unittest.TestCase):

    def test_RoiLoss(self):
        target = torch.Tensor([1,2,3,4,5,0,0,0,0,0])
        pred = torch.Tensor([1 for x in range(10)])
        target = Variable(target)
        pred = Variable(pred)

        criterion = nn.L1Loss()
        loss = criterion(pred, target)
        criterion2 = RoiLoss(criterion)
        loss2 = criterion2(pred, target)

        print('L1Loss:', loss.data, 'RoiLoss:', loss2.data)

if __name__ == "__main__":
    unittest.main()
