# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable

class RoiLoss(nn.Module):

    def __init__(self, loss_type=nn.L1Loss()):
        super(RoiLoss, self).__init__()
        self.loss = loss_type
        self.ratio = 0.1

    def forward(self, output, target):
        pos = (target > 0)
        neg = (target == 0)
        ret_loss = self.loss(output[pos], target[pos]) + self.ratio * self.loss(output[neg], target[neg])
        return ret_loss 
