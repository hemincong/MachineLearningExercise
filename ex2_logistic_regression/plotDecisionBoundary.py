#!/usr/bin/env python
# -*- coding: utf-8 -*-


def f_y(theta, x):
    return (theta[0] + theta[1] * x) / - theta[2]


def plotDecisionBoundary(theta, X, y):
    import numpy as np
    import matplotlib.pyplot as plt
    from ex2_logistic_regression.plotData import plotData
    plotData(X[:, 1:], y)

    m, n = np.shape(X)
    if n <= 3:
        min_x = min(X[:,1])
        max_x = max(X[:,1])
        x = np.array([X[:, 1].min(), X[:, 1].max()])
        y = [f_y(theta, min_x), f_y(theta, max_x)]

        plt.figure(1)
        plt.title('Linear regression With GCD')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.scatter(x, y, marker='o', color='k', s=10, label='point')
        plt.legend(loc='lower right')
        plt.plot(x, y)
        plt.show()
    else:
        # TODO:
        pass
