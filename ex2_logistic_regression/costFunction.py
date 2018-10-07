#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunction(theta, x, y):
    import numpy as np
    x_row, x_col = np.shape(x)
    sum_l = 0.0
    from ex2_logistic_regression.sigmoid import sigmoid
    for r in range(x_row):
        cost_row = sigmoid(theta.dot(x[r]))
        y_row = y[r]
        sum_l += (-y_row) * np.log(cost_row) - (1 - y_row) * np.log(1 - cost_row)
    return sum_l / x_row


def costFunction_2(theta, x, y):
    import numpy as np
    from ex2_logistic_regression.sigmoid import sigmoid
    m, n = x.shape
    theta = theta.reshape((n, 1))
    y = y.reshape((m, 1))
    term1 = np.log(sigmoid(x.dot(theta)))
    term2 = np.log(1 - sigmoid(x.dot(theta)))
    term1 = term1.reshape((m, 1))
    term2 = term2.reshape((m, 1))
    term = y * term1 + (1 - y) * term2
    return -((np.sum(term)) / m)


def compute_grad(theta, x, y):
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
    return grad


def compute_grad_2(theta, x, y):
    import numpy as np
    x_row, x_col = np.shape(x)
    theta = theta.reshape((x_col, 1))
    y = y.reshape((x_row, 1))
    from ex2_logistic_regression.sigmoid import sigmoid
    sigmoid_x_theta = sigmoid(x.dot(theta))
    grad = (x.T.dot(sigmoid_x_theta - y)) / x_row
    return grad.flatten()
