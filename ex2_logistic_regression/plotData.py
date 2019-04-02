#!/usr/bin/env python
# -*- coding: utf-8 -*-


def plotData(X, y):
    import matplotlib.pyplot as plt
    import numpy as np
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(X[pos, 0], X[pos, 1], color='g', marker='+')
    plt.scatter(X[neg, 0], X[neg, 1], color='y', marker='o')

    plt.grid()

