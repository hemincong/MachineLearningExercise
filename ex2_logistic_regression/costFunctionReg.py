#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunctionReg(theta, x, y, _lambda):
    import numpy as np
    from ex2_logistic_regression.sigmoid import sigmoid
    h = sigmoid(x * theta.T)

    m, _ = np.shape(x)
    cost = (sum((-y) * np.log(h) - (1 - y) * (np.log(1 - h))))[0] / m
    reg_param = sum([np.power(t, 2) for t in theta[1:]]) * _lambda / 2 / m
    return cost + reg_param


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
