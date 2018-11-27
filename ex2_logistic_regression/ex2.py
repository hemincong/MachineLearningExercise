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



