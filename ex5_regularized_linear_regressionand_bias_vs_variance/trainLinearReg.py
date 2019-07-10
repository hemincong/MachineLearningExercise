#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
# TRAINLINEARREG Trains linear regression given a dataset (X, y) and a
# regularization parameter lambda
#   [theta] = TRAINLINEARREG (X, y, lambda) trains linear regression using
#   the dataset (X, y) and regularization parameter lambda. Returns the
#   trained parameters theta.
#
from scipy.optimize import fmin_cg

from ex5_regularized_linear_regressionand_bias_vs_variance.linearRegCostFunction import linearRegCostFunction


def trainLinearReg(X, y, _lambda):
    # Initialize Theta
    initial_theta = np.zeros((np.shape(X)[1], 1))

    def costFunc(theta, _X, _y, __lambda):
        cost, _ = linearRegCostFunction(_X, _y, theta, __lambda)
        return cost

    def grad(theta, _X, _y, __lambda):
        _, _grad = linearRegCostFunction(_X, _y, theta, __lambda)
        return _grad

    result = fmin_cg(costFunc,
                     fprime=grad,
                     x0=initial_theta,
                     args=(X, y, _lambda),
                     maxiter=400,
                     disp=True,
                     full_output=True)
    return result[1], result[0]
