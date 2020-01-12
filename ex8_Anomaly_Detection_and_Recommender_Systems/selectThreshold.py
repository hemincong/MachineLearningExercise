#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# SELECTTHRESHOLD Find the best threshold(epsilon) to use for selecting
# outliers
# [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
# threshold to use for selecting outliers based on the results from a
# validation set(pval) and the ground truth(yval).
#
def selectThreshold(yval, pval):
    bestEpsilon = 0
    bestF1 = 0
    F1 = 0

    stepsize = (max(pval) - min(pval)) / 1000

    for epsilon in np.arange(min(pval), max(pval), stepsize):
        # ====================== YOUR CODE HERE ======================
        # Instructions: Compute the F1 score of choosing epsilon as the
        #               threshold and place the value in F1. The code at the
        #               end of the loop will compare the F1 score for this
        #               choice of epsilon and set it to be the best epsilon if
        #               it is better than the current choice of epsilon.
        #
        # Note: You can use predictions = (pval < epsilon) to get a binary vector
        #       of 0's and 1's of the outlier predictions

        predictions = (pval < epsilon).reshape(np.shape(yval)[0], 1)

        tp = np.sum((predictions == 1) & (yval == 1))
        fp = np.sum((predictions == 1) & (yval == 0))
        fn = np.sum((predictions == 0) & (yval == 1))

        prec = tp * 1.0 / (tp + fp)
        rec = tp * 1.0 / (tp + fn)

        F1 = 2 * prec * rec / (prec + rec)

        if F1 > bestF1:
            bestF1 = F1
            bestEpsilon = epsilon

    return bestEpsilon, bestF1
