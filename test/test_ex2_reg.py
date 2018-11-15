#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

data_file_path = "resource/ex2data2.txt"


class test_ex2_reg(unittest.TestCase):

    def test_map_feature(self):
        from ex2_logistic_regression.mapFeature import mapFeature
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        ret = mapFeature(x[:, 1], x[:, 2])
        self.assertEqual(len(x), len(ret))
        self.assertEqual(len(ret[0]), 28)

    def test_cost_function_reg(self):
        from ex2_logistic_regression.costFunctionReg import costFunctionReg
        from ex2_logistic_regression.mapFeature import mapFeature
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(29)
        _lambda = 1
        cost = costFunctionReg(theta, x, y, _lambda)
        self.assertAlmostEqual(cost, 0.693, delta=0.001)

    def test_compute_grad_reg(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.costFunctionReg import compute_grad_reg
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col)
        _lambda = 1
        grad = compute_grad_reg(theta, x, y, _lambda)
        print("grad: {grad}".format(grad=grad))
        # self.assertAlmostEqual(grad[0], -0.1, delta=0.1)
        # self.assertAlmostEqual(grad[1], -12.00, delta=0.01)
        # self.assertAlmostEqual(grad[2], -11.262, delta=0.01)
