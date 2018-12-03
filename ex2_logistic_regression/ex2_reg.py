#!/usr/bin/env python
# -*- coding: utf-8 -*-

import scipy.optimize as op


def line_regression_reg_by_fmin_2(theta, X, y, _lambda):
    from ex2_logistic_regression.costFunctionReg import costFunctionReg
    _lambda = 1
    options = {'full_output': True, 'retall': True}
    theta, cost, _, _, _, _, _, allvecs = op.fmin_bfgs(
        lambda t: costFunctionReg(t, X, y, _lambda), theta, maxiter=400, **options)
    return theta, cost


def line_regression_reg_by_fmin(theta, X, y, _lambda):
    import scipy.optimize as op
    from ex2_logistic_regression.costFunctionReg import costFunctionReg
    result = op.minimize(costFunctionReg,
                         theta,
                         args=(X, y, _lambda),
                         method='BFGS',
                         options={"maxiter": 500, "disp": True})
    return result.x, result.fun
