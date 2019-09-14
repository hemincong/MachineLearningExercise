#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def plotDataPoints(X, idx, K):
    # Create palette (see hsv.py)
    from ex7_K_means_Clustering_and_Principal_Component_Analysis.hsv import hsv
    palette = hsv(K)
    colors = np.array([palette[int(i)] for i in idx])

    # Plot the data
    plt.scatter(X[:, 0], X[:, 1], s=75, facecolors='none', edgecolors=colors)
