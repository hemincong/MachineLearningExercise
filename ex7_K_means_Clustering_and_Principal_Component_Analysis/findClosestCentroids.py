#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# FINDCLOSESTCENTROIDS computes the centroid memberships for every example
#   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
#   in idx for a dataset X where each row is a single example. idx = m x 1
#   vector of centroid assignments (i.e. each entry in range [1..K])
#
def findClosestCentroids(X, centroids):
    # Set K
    K, _ = np.shape(centroids)

    # You need to return the following variables correctly. idx = zeros(size(X, 1), 1);
    idx = np.zeros((np.shape(X)[0], 1))

    # ====================== YOUR CODE HERE ======================
    # Instructions: Go over every example, find its closest centroid, and store
    # the index inside idx at the appropriate location.
    # Concretely, idx(i) should contain the index of the centroid
    # closest to example i.Hence, it should be a value in the
    # range 1..K
    #
    # Note: You can use a for-loop over the examples to compute this.
    #
    m = np.shape(X)[0]
    min_centoid = 0
    for i in range(m):
        min_distance = 100000000
        for j in range(K):
            x, y = X[i]
            x_c, y_c = centroids[j]
            # no need square
            distance = np.power(x - x_c, 2) + np.power(y - y_c, 2)

            if min_distance > distance:
                min_distance = distance
                min_centoid = j

        idx[i] = min_centoid
    return idx
