#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def estimateGaussian(X):
    # estimateGaussian This function estimates the parameters of a
    # Gaussian distribution using the data in X
    #    [mu sigma2] = estimateGaussian(X),
    #    The input X is the dataset with each n-dimensional data point in one row
    #    The output is an n - dimensional vector mu, the mean of the data set
    #    and the variances sigma ^ 2, an n x 1 vector
    #

    # Useful variables
    m, n = np.shape(X)

    # You should return these values correctly
    mu = np.zeros((n, 1))
    sigma2 = np.zeros((n, 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the mean of the data and the variances
    # In particular, mu(i) should contain the mean of
    # the data for the i - th feature and sigma2(i)
    # should contain variance of the i - th feature.
    #
    mu = np.sum(X, axis=0) / m

    tmp = X - mu
    sigma2 = np.sum(np.power(tmp, 2), axis=0) / m

    return mu, sigma2
