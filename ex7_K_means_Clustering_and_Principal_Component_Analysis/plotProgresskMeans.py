#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def plotProgresskMeans(X, centroids, previous, idx, K, i):
    # PLOTPROGRESSKMEANS is a helper function that displays the progress of
    # k-Means as it is running. It is intended for use only with 2D data.
    #   PLOTPROGRESSKMEANS(X, centroids, previous, idx, K, i) plots the data
    #   points with colors assigned to each centroid. With the previous
    #   centroids, it also plots a line between the previous locations and
    #   current locations of the centroids.
    #
    from ex7_K_means_Clustering_and_Principal_Component_Analysis.plotDataPoints import plotDataPoints

    # Plot the example
    plotDataPoints(X, idx, K)

    # Plot the centroids as black x's
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=400, c='k', linewidth=1)

    from ex7_K_means_Clustering_and_Principal_Component_Analysis.drawLine import drawLine
    # Plot the history of the centroids with lines
    for j in range(centroids.shape[0]):
        drawLine(centroids[j, :], previous[j, :], c='b')

    # Title
    plt.title('Iteration number {i:d}'.format(i=i + 1))
