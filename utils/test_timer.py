# -*- coding: utf-8 -*-
from __future__ import division
import time
import matplotlib.pyplot as plt
import unittest
import sys, os
from timer import *

import time

def foo(interval=1):
    print('run...')
    time.sleep(interval)

class Tester(unittest.TestCase):

    def test_Timer(self):
        timer = Timer()
        timer.start()
        for i in range(1):
            foo(0.5)
            timer.tictoc('p1')
            foo(0.1)
            timer.tictoc('p2')
            foo(0.4)
            timer.tictoc('p3')
        timer.end()
        timer.log(show=True)



if __name__ == "__main__":
    unittest.main()
