#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


class test_cost(unittest.TestCase):

    def test_cost(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col("test/resource/ex2data1.txt")
        from ex2_logistic_regression.costFunction import costFunction
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col + 1)
        ret = costFunction(theta, x, y)
        self.assertAlmostEqual(ret, 0.693, delta=0.01)

