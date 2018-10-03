#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunction(theta, x, y):
    import numpy as np
    x_row, x_col = np.shape(x)
    one_col = np.ones((x_row, 1))
    whole_x = np.c_[one_col, x]
    sum_l = 0.0
    from ex2_logistic_regression.sigmoid import sigmoid
    for r in range(x_row):
        cost_row = sigmoid(theta.dot(whole_x[r]))
        y_row = y[r]
        sum_l += (-y_row) * np.log(cost_row) - (1 - y_row) * np.log(1 - cost_row)
    return sum_l / x_row


def compute_grad(theta, x, y):
    import numpy as np
    x_row, x_col = np.shape(x)
    one_col = np.ones((x_row, 1))
    whole_x = np.c_[one_col, x]
    from ex2_logistic_regression.sigmoid import sigmoid
    grad = [0] * (x_col + 1)
    for r in range(x_row):
        cost_row = sigmoid(theta.dot(whole_x[r]))
        y_row = y[r]
        for c in range(x_col + 1):
            grad[c] += ((cost_row - y_row) * whole_x[r][c])[0]

    for c in range(x_col + 1):
        grad[c] = grad[c] / x_row
    return grad


def compute_grad_2(theta, x, y):
    import numpy as np
    x_row, x_col = np.shape(x)
    one_col = np.ones((x_row, 1))
    whole_x = np.c_[one_col, x]
    wx_row, wx_col = whole_x.shape
    theta = theta.reshape((wx_col, 1))
    y = y.reshape((wx_row, 1))
    from ex2_logistic_regression.sigmoid import sigmoid
    sigmoid_x_theta = sigmoid(whole_x.dot(theta))
    grad = (whole_x.T.dot(sigmoid_x_theta - y)) / wx_row
    return grad.flatten()
