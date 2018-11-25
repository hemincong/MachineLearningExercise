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

    return grad
