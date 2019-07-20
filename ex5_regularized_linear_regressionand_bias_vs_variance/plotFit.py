#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plotFit(min_x, max_x, mu, sigma, theta, p):
    # PLOTFIT Plots a learned polynomial regression fit over an existing figure.
    # Also works with linear regression.
    #   PLOTFIT(min_x, max_x, mu, sigma, theta, p) plots the learned polynomial
    #   fit with power p and feature normalization (mu, sigma).

    # We plot a range slightly bigger than the min and max values to get
    # an idea of how the fit will vary outside the range of the data points
    x = np.arange(min_x - 15, max_x + 25, 0.05)  # 1D vector

    from ex5_regularized_linear_regressionand_bias_vs_variance.polyFeatures import polyFeatures
    # Map the X values
    X_poly = polyFeatures(x, p)
    X_poly = (X_poly - mu) / sigma

    # Add ones
    X_poly = np.column_stack((np.ones((x.shape[0], 1)), X_poly))

    # Plot
    plt.plot(x, X_poly.dot(theta), '--', linewidth=2)
