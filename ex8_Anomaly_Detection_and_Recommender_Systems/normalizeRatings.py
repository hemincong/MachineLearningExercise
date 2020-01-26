#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def normalizeRatings(Y, R):
    # NORMALIZERATINGS Preprocess data by subtracting mean rating for every
    # movie(every row)
    # [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
    # has a rating of 0 on average, and returns the mean rating in Ymean.
    #
    m, n = np.shape(Y)
    Ymean = np.zeros((m, 1))
    Ynorm = np.zeros((m, n))

    for i in range(m):
        idx = R[i, :] == 1
        Ymean[i] = np.mean(Y[i, idx])
        Ynorm[i, idx] = Y[i, idx] - Ymean[i]

    return Ynorm, Ymean
