#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunctionReg(theta, x, y, _lambda):
    import numpy as np
    x_row, x_col = np.shape(x)
    from ex2_logistic_regression.mapFeature import mapFeature
    sum_l = 0.0
    from ex2_logistic_regression.sigmoid import sigmoid
    for r in range(x_row):
        reg_x = [1.0] + mapFeature(x[r][1], x[r][2])
        cost_row = sigmoid(theta.dot(np.asarray(reg_x)))
        y_row = y[r]
        sum_l += (-y_row) * np.log(cost_row) - (1 - y_row) * np.log(1 - cost_row)
    cost_without_reg = sum_l / x_row
    theta_without_1 = theta[1:len(theta)]
    cost = cost_without_reg + sum(np.power(theta_without_1, 2)) * _lambda / 2 / x_row
    return cost

def compute_grad_reg(theta, x, y, _lambda):
    import numpy as np
    x_row, x_col = np.shape(x)
    from ex2_logistic_regression.sigmoid import sigmoid
    grad = [0] * x_col
    for r in range(x_row):
        cost_row = sigmoid(theta.dot(x[r]))
        y_row = y[r]
        for c in range(x_col):
            grad[c] += ((cost_row - y_row) * x[r][c])[0]

    for c in range(x_col):
        grad[c] = grad[c] / x_row

    for c in range(1, x_col):
        grad[c] = grad[c] + _lambda / x_row * theta[c]
    return grad
