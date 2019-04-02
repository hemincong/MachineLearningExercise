#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

data_file_path = "resource/ex2data1.txt"


class test_ex2_cost(unittest.TestCase):

    def test_cost(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.costFunction import costFunction
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col)
        ret = costFunction(theta, x, y)
        self.assertAlmostEqual(ret, 0.693, delta=0.01)

    def test_cost_2(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.costFunction import costFunction_2
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col)
        ret = costFunction_2(theta, x, y)
        self.assertAlmostEqual(ret, 0.693, delta=0.01)

    def test_compute_theta(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.costFunction import compute_grad
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col)
        grad = compute_grad(theta, x, y)
        self.assertAlmostEqual(grad[0], -0.1, delta=0.1)
        self.assertAlmostEqual(grad[1], -12.00, delta=0.01)
        self.assertAlmostEqual(grad[2], -11.262, delta=0.01)

    def test_compute_theta_2(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.costFunction import compute_grad_2
        import numpy as np
        x_row, x_col = np.shape(x)
        theta = np.zeros(x_col)
        grad = compute_grad_2(theta, x, y)
        self.assertAlmostEqual(grad[0], -0.1, delta=0.1)
        self.assertAlmostEqual(grad[1], -12.00, delta=0.01)
        self.assertAlmostEqual(grad[2], -11.262, delta=0.01)

    def test_fmin(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.ex2 import line_regression_by_fmin
        ret = line_regression_by_fmin(x, y)
        self.assertAlmostEqual(ret.fun, 0.203, delta=0.01)

    def test_predict(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.ex2 import line_regression_by_fmin
        ret = line_regression_by_fmin(x, y)
        self.assertAlmostEqual(ret.fun, 0.203, delta=0.01)
        from ex2_logistic_regression.sigmoid import sigmoid
        ret_p = sigmoid(ret.x.dot([1, 45, 85]))
        self.assertAlmostEqual(ret_p, 0.776, delta=0.01)
        from ex2_logistic_regression.predict import predict
        p = predict(ret.x, x)
        count = 0
        for r in range(len(p)):
            if p[r] == y[r]:
                count = count + 1

        self.assertEqual(count, 89)

    def test_plotData(self):
        from ex2_logistic_regression.plotData import plotData
        from utils import file_utils
        import matplotlib.pyplot as plt

        x, y = file_utils.read_csv_split_last_col(data_file_path)
        plotData(x, y)
        plt.title('Scatter plot of training data')
        plt.xlabel('Exam 1 score')
        plt.ylabel('Exam 2 score')
        plt.show()

    def test_plotDecisionBoundary(self):
        from utils import file_utils
        x, y = file_utils.read_csv_split_last_col_and_add_one(data_file_path)
        from ex2_logistic_regression.ex2 import line_regression_by_fmin
        ret = line_regression_by_fmin(x, y)
        print(ret)
        from ex2_logistic_regression.plotDecisionBoundary import plotDecisionBoundary
        ret = line_regression_by_fmin(x, y)
        plotDecisionBoundary(ret.x, x, y)
