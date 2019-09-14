#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def runkMeans(X, initial_centroids, max_iters, plot_progress):
    # RUNKMEANS runs the K-Means algorithm on data matrix X, where each row of X
    # is a single example
    #   [centroids, idx] = RUNKMEANS(X, initial_centroids, max_iters, ...
    #   plot_progress) runs the K-Means algorithm on data matrix X, where each
    #   row of X is a single example. It uses initial_centroids used as the
    #   initial centroids. max_iters specifies the total number of interactions
    #   of K-Means to execute. plot_progress is a true/false flag that
    #   indicates if the function should also plot its progress as the
    #   learning happens. This is set to false by default. runkMeans returns
    #   centroids, a Kxn matrix of the computed centroids and idx, a m x 1
    #   vector of centroid assignments (i.e. each entry in range [1..K])
    #

    # Set default value for plot progress
    # (commented out due to pythonic default parameter assignment above)
    # if not plot_progress:
    #     plot_progress = False

    # Plot the data if we are plotting progress
    # if plot_progress:
    #     plt.hold(True)

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros((m, 1))

    import matplotlib.pyplot as plt
    if plot_progress:
        plt.close()
        plt.ion()

    from ex7_K_means_Clustering_and_Principal_Component_Analysis.findClosestCentroids import findClosestCentroids
    from ex7_K_means_Clustering_and_Principal_Component_Analysis.computeCentroids import computeCentroids
    from ex7_K_means_Clustering_and_Principal_Component_Analysis.plotProgresskMeans import plotProgresskMeans
    # Run K-Means
    for i in range(max_iters):
        print("K-Means iteration {i:d}/{max_iters:d}".format(i=i, max_iters=max_iters))

        idx = findClosestCentroids(X, centroids)

        # 图去哪了？
        if plot_progress:
            plotProgresskMeans(X, centroids, previous_centroids, idx, K, i)
            previous_centroids = centroids

        # Given the memberships, compute new centroids
        centroids = computeCentroids(X, idx, K)

    # Hold off if we are plotting progress
    print("")

    return centroids, idx
