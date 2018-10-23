#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.optimize as op

data = "ex2_logistic_regression/test/resource/ex2data2.txt"

class test_reg(unittest.TestCase):

    def test_map_feature(self):
        from ex2_logistic_regression.mapFeature import mapFeature
        ret = mapFeature(1, 2)
        self.assertEqual(len(ret), 28)
