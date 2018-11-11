#!/usr/bin/env python
# -*- coding: utf-8 -*-


def costFunctionReg(theta, x, y, _lambda):
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
    cost_not_reg = -((np.sum(term)) / m)
    print("theta: {0}, term: {1}".format(theta, term))
    xx = _lambda / (2 * m)
    return xx

