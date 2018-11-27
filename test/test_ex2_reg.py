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
        mapped = mapFeature(x[:, 1], x[:, 2])
        import numpy as np
        initial_theta = np.zeros(len(mapped[0]))
        _lambda = 1
        cost = costFunctionReg(initial_theta, mapped, y, _lambda)
        self.assertAlmostEqual(cost, 0.693, delta=0.001)

    def test_compute_grad_reg(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)

        from ex2_logistic_regression.costFunctionReg import compute_grad_reg
        from ex2_logistic_regression.mapFeature import mapFeature
        import numpy as np
        mapped = mapFeature(x[:, 1], x[:, 2])
        _, n = np.shape(mapped)
        theta = np.zeros(28)
        _lambda = 1
        grad = compute_grad_reg(theta, mapped, y, _lambda)
        self.assertTrue(len(grad), 28)

    def test_feature_mapped_logistic_regression(self):
        from ex2_logistic_regression.mapFeature import mapFeature
        from utils import file_utils
        import numpy as np
        from ex2_logistic_regression.ex2_reg import line_regression_reg_by_fmin

        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)

        X = np.asarray(mapFeature(x[:, 1], x[:, 2]))
        theta = np.zeros(X.shape[1])

        res = line_regression_reg_by_fmin(theta, X, y, 1)
        print(res)

    def test_feature_mapped_logistic_regression_2(self):
        from ex2_logistic_regression.mapFeature import mapFeature
        from utils import file_utils
        import numpy as np
        from ex2_logistic_regression.ex2_reg import line_regression_reg_by_fmin_2
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        X = np.asarray(mapFeature(x[:, 1], x[:, 2]))
        theta = np.zeros(X.shape[1])
        res = line_regression_reg_by_fmin_2(theta, X, y, 1)
        print(res)
