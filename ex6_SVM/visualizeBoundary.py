import matplotlib.pyplot as plt
import numpy as np

import ex6_SVM.gaussianKernelGramMatrix as gkgm


def visualizeBoundary(X, y, model, varargin=0):
    # VISUALIZEBOUNDARY plots a non-linear decision boundary learned by the SVM
    #   VISUALIZEBOUNDARYLINEAR(X, y, model) plots a non-linear decision 
    #   boundary learned by the SVM and overlays the data on it

    # Plot the training data on top of the boundary
    y = y.flatten()
    pos = y == 1
    neg = y == 0

    # Plot Examples
    plt.plot(X[:, 0][pos], X[:, 1][pos], "k+", markersize=10)
    plt.plot(X[:, 0][neg], X[:, 1][neg], "yo", markersize=10)

    # Make classification predictions over a grid of values
    x1plot = np.linspace(X[:, 0].min(), X[:, 0].max(), 100).T
    x2plot = np.linspace(X[:, 1].min(), X[:, 1].max(), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.column_stack((X1[:, i], X2[:, i]))
        vals[:, i] = model.predict(gkgm.gaussianKernelGramMatrix(this_X, X))

    # Plot the SVM boundary
    plt.contour(X1, X2, vals, colors="blue", levels=[0, 0])
    plt.show(block=False)
