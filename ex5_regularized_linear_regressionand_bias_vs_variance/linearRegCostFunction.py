#!/usr/bin/env python

import numpy as np


def linearRegCostFunction(X, y, theta, _lambda):
    theta = theta.reshape(np.shape(X)[1], 1)
    m = np.shape(X)[0]

    theta_tmp = theta[1:]
    delta = np.dot(X, theta) - y
    J = np.dot(delta.T, delta) / (2 * m) + _lambda / (2 * m) * np.dot(theta_tmp.T, theta_tmp)

    grad_no_reg = np.dot(X.T, np.dot(X, theta) - y) / m
    grad = grad_no_reg + theta * _lambda / m
    grad[0] = grad_no_reg[0]
    return J.flatten()[0], grad.flatten()
