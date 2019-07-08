#!/usr/bin/env python

import numpy as np


def linearRegCostFunction(X, y, theta, _lambda):
    m = len(y)

    theta_tmp = theta[1:]
    J = np.sum(np.square(np.dot(X, theta) - y)) / (2 * m) + _lambda / (2 * m) * np.square(theta_tmp)
    grad_no_reg = np.dot(X.T, np.dot(X, theta) - y) / m
    grad = grad_no_reg + theta * _lambda / m
    grad[0] = grad_no_reg[0]
    return J.flatten()[0], grad.flatten()
