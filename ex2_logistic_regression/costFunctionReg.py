#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunctionReg(theta, x, y, _lambda):
    import numpy as np
    from ex2_logistic_regression.sigmoid import sigmoid
    h = sigmoid(x * theta.T)

    m, _ = np.shape(x)
    cost = (sum((-y) * np.log(h) - (1 - y) * (np.log(1 - h))))[0] / m
    reg_param = sum(np.power(theta[1:], 2)) * _lambda / 2 / m
    return cost + reg_param


def compute_grad_reg(theta, x, y, _lambda):
    import numpy as np
    x_row, x_col = np.shape(x)
    from ex2_logistic_regression.sigmoid import sigmoid

    h = sigmoid(x * theta.T)[0]
    t = h - y
    grad = sum(t * x) / x_row

    for c in range(1, x_col):
        grad[c] = grad[c] + _lambda / x_row * theta[c]
    return grad


def compute_grad_reg_2(theta, X, y, _lambda):
    import numpy as np
    theta = np.matrix(theta)
    X = np.matrix(X)
    y = np.matrix(y)

    parameters = int(theta.ravel().shape[1])
    grad = np.zeros(parameters)

    from ex2_logistic_regression.sigmoid import sigmoid
    error = sigmoid(X * theta.T) - y

    for i in range(parameters):
        term = np.multiply(error, X[:, i])

        if i == 0:
            grad[i] = np.sum(term) / len(X)
        else:
            grad[i] = (np.sum(term) / len(X)) + ((_lambda / len(X)) * theta[:, i])

    print(len(grad))
    return grad
