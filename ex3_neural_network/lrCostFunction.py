#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def compute_cost(theta, X, y, _lambda):
    m, n = X.shape
    from utils.sigmoid import sigmoid
    theta = theta.reshape(n, 1)
    term1 = np.log(sigmoid(X.dot(theta)))
    term2 = np.log(1 - sigmoid(X.dot(theta)))
    term = y * term1.T + (1 - y) * term2.T
    reg = np.sum(np.power(theta, 2)) / m / 2 * _lambda
    return -np.sum(term) / m + reg


def gradient(theta, X, y, _lambda):
    m, n = X.shape

    from utils.sigmoid import sigmoid
    h = sigmoid(np.dot(X, theta))
    t = h.T - y
    tmp = np.dot(t, X)
    grad_reg_without_reg = tmp / m
    grad_reg = tmp / m + (float(_lambda) / m) * theta
    grad_reg[0] = grad_reg_without_reg[0]
    return grad_reg.flatten()
