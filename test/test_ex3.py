#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.io as sio

data_file = "resource/ex3data1.mat"


class test_ex3(unittest.TestCase):

    def test_(self):
        data = sio.loadmat(data_file)
        print(data)
