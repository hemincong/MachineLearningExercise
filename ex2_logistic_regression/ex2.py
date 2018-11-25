#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.optimize as op


def line_regression_by_fmin(x, y):
    row, col = x.shape

    import numpy as np
    initial_theta = np.zeros(col)

    from ex2_logistic_regression.costFunction import costFunction, compute_grad
    return op.minimize(fun=costFunction,
                       x0=initial_theta,
                       args=(x, y),
                       method='TNC',
                       jac=compute_grad,
                       options={"maxiter": 400},
                       )


def line_regression_reg_by_fmin_2(theta, X, y, _lambda):
    from ex2_logistic_regression.costFunctionReg import costFunctionReg
    _lambda = 1
    options = {'full_output': True, 'retall': True}
    theta, cost, _, _, _, _, _, allvecs = op.fmin_bfgs(
        lambda t: costFunctionReg(t, X, y, _lambda), theta, maxiter=400, **options)
    return theta, cost


def line_regression_reg_by_fmin(theta, X, y, lamda):
    import scipy.optimize as op
    from ex2_logistic_regression.costFunctionReg import costFunctionReg
    result = op.minimize(costFunctionReg,
                         theta,
                         args=(X, y, lamda),
                         method='BFGS',
                         options={"maxiter": 500, "disp": True})
    return result.x, result.fun
