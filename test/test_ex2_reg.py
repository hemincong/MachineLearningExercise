#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

data_file = "resource/ex2data2.txt"


class test_ex2_reg(unittest.TestCase):

    def test_map_feature(self):
        from ex2_logistic_regression.mapFeature import mapFeature
        ret = mapFeature(1, 2)
        self.assertEqual(len(ret), 28)

    def test_cost_function_reg(self):
        from ex2_logistic_regression.costFunctionReg import costFunctionReg
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file)
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col)
        _lambda = 1
        cost = costFunctionReg(theta, x, y, _lambda)
        self.assertAlmostEqual(cost, 0.693, delta=0.001)
