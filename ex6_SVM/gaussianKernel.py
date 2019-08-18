#!/usr/bin/env python
# -*- coding: utf-8 -*-

def gaussianKernel(x1, x2, sigma=0.1):
    # RBFKERNEL returns a radial basis function kernel between x1 and x2
    #   sim = gaussianKernel(x1, x2) returns a gaussian kernel between x1 and x2
    #   and returns the value in sim

    # Ensure that x1 and x2 are column vectors
    x1 = x1.flatten()
    x2 = x2.flatten()

    # You need to return the following variables correctly.
    sim = 0

    # ====================== YOUR CODE HERE ======================
    # Instructions: Fill in this function to return the similarity between x1
    #               and x2 computed using a Gaussian kernel with bandwidth
    #               sigma
    #
    #
    import numpy as np
    delta = x1 - x2
    ret = np.exp(-(np.dot(delta, delta.T) / 2 / (sigma * sigma)))
    return ret
