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

    def test_compute_theta(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col("test/resource/ex2data1.txt")
        from ex2_logistic_regression.costFunction import compute_grad
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col + 1)
        grad = compute_grad(theta, x, y)
        self.assertAlmostEqual(grad[0], -0.1, delta=0.1)
        self.assertAlmostEqual(grad[1], -12.00, delta=0.01)
        self.assertAlmostEqual(grad[2], -11.262, delta=0.01)

    def test_compute_theta_2(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col("test/resource/ex2data1.txt")
        from ex2_logistic_regression.costFunction import compute_grad_2
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col + 1)
        grad = compute_grad_2(theta, x, y)
        self.assertAlmostEqual(grad[0], -0.1, delta=0.1)
        self.assertAlmostEqual(grad[1], -12.00, delta=0.01)
        self.assertAlmostEqual(grad[2], -11.262, delta=0.01)
