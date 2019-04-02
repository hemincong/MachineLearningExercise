#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def oneVsAll(X, y, num_labels, _lambda):
    m, n = X.shape

    all_theta = np.zeros((num_labels, n + 1))
    X = np.insert(X, 0, 1, axis=1)
    initial_theta = np.zeros(n + 1)
    options = {'disp': False, 'maxiter': 400}
    from ex3_neural_network.lrCostFunction import costFunctionReg, gradReg
    fun = lambda theta, y: costFunctionReg(theta, X, y, _lambda)
    jac = lambda theta, y: gradReg(theta, X, y, _lambda)

    import scipy.optimize as op
    for c in range(num_labels):
        args = ((y == c + 1).astype(np.int),)
        ret = op.minimize(fun=fun,
                          jac=jac,
                          x0=initial_theta,
                          method='CG',
                          args=args,
                          options=options
                          )
        all_theta[c, :] = ret.x
    return all_theta
