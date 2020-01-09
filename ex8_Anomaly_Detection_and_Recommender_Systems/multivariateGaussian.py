#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def multivariateGaussian(X, mu, Sigma2):
    # MULTIVARIATEGAUSSIAN Computes the probability density function of the
    # multivariate gaussian distribution.
    #    p = MULTIVARIATEGAUSSIAN(X, mu, Sigma2) Computes the probability
    #    density function of the examples X under the multivariate gaussian
    #    distribution with parameters mu and Sigma2. If Sigma2 is a matrix, it is
    #    treated as the covariance matrix. If Sigma2 is a vector, it is treated
    #    as the \sigma^2 values of the variances in each dimension (a diagonal
    #    covariance matrix)
    #

    k = len(mu)

    m, n = np.shape(X)

    temp_sigma2 = np.diag(Sigma2)
    temp_X = X - mu

    p = (2.0 * np.pi) ** (-k / 2.0) * np.linalg.det(temp_sigma2) ** -0.5 \
        * np.exp(-0.5 * np.sum((temp_X * np.diagonal(np.linalg.pinv(temp_sigma2))) * temp_X, axis=1))

    return p
