#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunction(theta, x, y):
    import numpy as np
    x_row, x_col = np.shape(x)
    b = np.ones((x_row, 1))
    xx = np.c_[b, x]
    sum_l = 0.0
    from ex2_logistic_regression.sigmoid import sigmoid
    for r in range(x_row):
        cost_row = sigmoid(theta.dot(xx[r]))
        y_row = y[r]
        sum_l += (-y_row) * np.log(cost_row) - (1 - y_row) * np.log(1 - cost_row)
    return sum_l / x_row
