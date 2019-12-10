#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import scipy.io


#  Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

class test_ex7_pca(unittest.TestCase):
    @classmethod
    def setUp(cls):
        # Load Training Data
        data_file = "resource/ex7data1.mat"
        mat = scipy.io.loadmat(data_file)
        cls.X = mat["X"]

    # ================== Part 1: Load Example Dataset ===================
    # We start this exercise by using a small dataset that is easily to
    # visualize
    #
    # The following command loads the dataset.You should now have the
    # variable X in your environment
    def test_Visualizing_example_dataset_for_PCA(self):
        print('Visualizing example dataset for PCA.')

        import matplotlib.pyplot as plt
        plt.close()

        plt.scatter(self.X[:, 0], self.X[:, 1], s=75, facecolors='none', edgecolors='b')
        plt.axis([0.5, 6.5, 2, 8])
        plt.gca().set_aspect('equal', adjustable='box')

        # =============== Part 2: Principal Component Analysis ===============
        #  You should now implement PCA, a dimension reduction technique. You
        #  should complete the code in pca.m
        #
        # def test_Principal_Component_Analysis(self):
        print('Running PCA on example dataset.')

        # Before running PCA, it is important to first normalize X
        import utils.featureNormalize
        X_norm, mu, sigma = utils.featureNormalize.featureNormalize(self.X)

        # Run PCA
        import ex7_K_means_Clustering_and_Principal_Component_Analysis.pca
        U, S = ex7_K_means_Clustering_and_Principal_Component_Analysis.pca.pca(X_norm)

        # Compute mu, the mean of the each feature

        #  Draw the eigenvectors centered at mean of data. These lines show the
        #  directions of maximum variations in the dataset.
        p1 = mu + 1.5 * S[0, 0] * U[:, 0].T
        p2 = mu + 1.5 * S[1, 1] * U[:, 1].T
        plt.plot([mu[0], p1[0]], [mu[1], p1[1]], c='k', linewidth=2)
        plt.plot([mu[0], p2[0]], [mu[1], p2[1]], c='k', linewidth=2)
        plt.show(block=False)
        print('Top eigenvector:')
        print(' U[:,1] = {:f} {:f}'.format(U[0, 0], U[1, 0]))
        self.assertAlmostEqual(U[0, 0], -0.707107, delta=0.0001)
        self.assertAlmostEqual(U[1, 0], -0.707107, delta=0.0001)
