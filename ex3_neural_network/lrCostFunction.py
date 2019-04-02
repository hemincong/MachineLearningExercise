#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def costFunction(theta, X, y, _lambda):
    m, n = X.shape
    from utils.sigmoid import sigmoid
    item1 = y * (np.log(sigmoid(np.dot(X, theta))))
    item2 = (1 - y) * (np.log(1 - sigmoid(np.dot(X, theta))))
    return np.sum(-item1 - item2) / m


def gradient(theta, X, y, _lambda):
    m, n = X.shape

    from utils.sigmoid import sigmoid
    h = sigmoid(np.dot(X, theta))
    t = h.T - y
    tmp = np.dot(t, X)
    grad_reg_without_reg = tmp / m
    return grad_reg_without_reg.flatten()


def costFunctionReg(theta, X, y, _lambda):
    m, n = X.shape
    return costFunction(theta, X, y, _lambda) + (np.dot(theta[1:n], theta[1:n]) / m / 2 * _lambda)


def gradReg(theta, X, y, _lambda):
    m, n = X.shape

    grad = gradient(theta, X, y, _lambda)
    grad[1:] += _lambda / m * theta[1:]
    return grad


def lrCostFunction(theta, X, y, _lambda):
    cost = costFunctionReg(theta, X, y, _lambda)
    grad = gradReg(theta, X, y, _lambda)
    return cost, grad
