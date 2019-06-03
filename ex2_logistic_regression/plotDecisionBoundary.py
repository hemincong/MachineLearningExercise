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
        min_x = min(X[:, 1])
        max_x = max(X[:, 1])
        x = np.array([X[:, 1].min(), X[:, 1].max()])
        if n < 2:
            theta[3] = 1
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
        from ex2_logistic_regression.mapFeature import mapFeature
        x = np.linspace(-1, 1.5, 50)
        y = np.linspace(-1, 1.5, 50)
        z = np.zeros(shape=(len(x), len(y)))
        for i in range(len(x)):
            for j in range(len(y)):
                z[i, j] = (mapFeature([x[i]], [y[j]]).dot(theta))
        z = z.T
        c = plt.contour(x, y, z, 0, origin='upper')
        c.collections[0].set_label('Decision Boundary')
