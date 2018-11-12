#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunctionReg(theta, x, y, _lambda):
    import numpy as np
    x_row, x_col = np.shape(x)
    sum_l = 0.0
    from ex2_logistic_regression.sigmoid import sigmoid
    for r in range(x_row):
        cost_row = sigmoid(theta.dot(x[r]))
        y_row = y[r]
        sum_l += (-y_row) * np.log(cost_row) - (1 - y_row) * np.log(1 - cost_row)
    cost_without_reg = sum_l / x_row
    theta_without_1 = theta[1:len(theta)]
    cost = cost_without_reg + sum(np.power(theta_without_1, 2)) * _lambda / 2 / x_row
    return cost

