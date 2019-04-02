#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunctionReg(theta, x, y, _lambda):
    import numpy as np
    from ex2_logistic_regression.sigmoid import sigmoid
    h = sigmoid(np.dot(x, theta.T))

    m, _ = np.shape(x)
    cost = (np.dot(-y, np.log(h)) - np.dot(1 - y, (np.log(1 - h))))
    reg_param = sum(np.power(theta[1:], 2)) * _lambda / 2 / m
    return (cost + reg_param) / m

def compute_grad_reg(theta, x, y, _lambda):
    import numpy as np
    m, n = np.shape(x)
    from ex2_logistic_regression.sigmoid import sigmoid

    h = sigmoid(np.dot(x, theta.T))
    t = h - y
    grad = [sum(np.dot(t, x)) / m] * n

    for c in range(1, n):
        grad[c] = grad[c] + _lambda / m * theta[c]
    return grad
