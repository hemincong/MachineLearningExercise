#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def lrCostFunction(theta, X, y, _lambda):
    m, n = X.shape
    from utils.sigmoid import sigmoid
    theta = theta.reshape(n, 1)
    term1 = np.log(sigmoid(X.dot(theta)))
    term2 = np.log(1 - sigmoid(X.dot(theta)))
    term1 = term1.reshape((m, 1))
    term2 = term2.reshape((m, 1))
    term = y * term1 + (1 - y) * term2
    return -((np.sum(term)) / m)


def compute_grad(theta, X, y):
    m, n = X.shape
    from utils.sigmoid import sigmoid
    print(sigmoid(X.dot(theta)))
    t = sigmoid(X.dot(theta)) - y
    return X.T.dot(t) / m
