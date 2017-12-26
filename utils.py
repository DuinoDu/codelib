# -*- coding: utf-8 -*-

import numpy as np
import torch
from matplotlib import pyplot as plt


def imshow(x, y, y_pred):
    """TODO: Docstring for imshow.

    Args:
        x (Variable): input x  
        y (Variable): target y
        y_pred (Variable): predict y

    Returns: None

    """
    x = x.cpu().data[0]
    y = y.cpu().data[0]
    y_pred = y_pred.cpu().data[0]
