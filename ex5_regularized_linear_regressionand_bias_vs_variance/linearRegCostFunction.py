#!/usr/bin/env python

import numpy as np


def linearRegCostFunction(X, y, theta, _lambda):
    m = len(y)

    theta_tmp = theta[1:]
    J = np.sum(np.square(np.dot(X, theta) - y)) / (2 * m) + _lambda / (2 * m) * np.square(theta_tmp)
    return J, 0
