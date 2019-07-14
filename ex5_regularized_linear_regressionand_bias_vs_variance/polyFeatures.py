#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def polyFeatures(X, p):
    # POLYFEATURES Maps X (1D vector) into the p-th power
    #   [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    #   maps each example into its polynomial features where
    #   X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    #

    X_poly = X

    # ====================== YOUR CODE HERE ======================
    # Instructions: Given a vector X, return a matrix X_poly where the p-th
    #               column of X contains the values of X to the p-th power.
    #
    #
    if p < 2:
        return None

    for i in range(1, p):
        np.column_stack((X_poly, np.power(X_poly, i + 1)))

    return X_poly
