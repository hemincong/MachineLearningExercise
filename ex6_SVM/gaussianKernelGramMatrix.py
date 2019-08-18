#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from ex6_SVM.gaussianKernel import gaussianKernel


def gaussianKernelGramMatrix(X1, X2, K_function=gaussianKernel, sigma=0.1):
    gram_matrix = np.zeros((X1.shape[0], X2.shape[0]))
    for i, x1 in enumerate(X1):
        for j, x2 in enumerate(X2):
            gram_matrix[i, j] = K_function(x1, x2, sigma)

    return gram_matrix
