#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
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
#     pca.py
#     projectData.py
#     recoverData.py
#     computeCentroids.py
#     findClosestCentroids.py
#     kMeansInitCentroids.py
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
class test_ex6_svm(unittest.TestCase):

    @classmethod
    def setUp(cls):
        # Load Training Data
        data_file = "resource/ex7data2.mat"
        mat = scipy.io.loadmat(data_file)
        cls.X = mat["X"]

    #  ================= Part 1: Find Closest Centroids ====================
    #  To help you implement K-Means, we have divided the learning algorithm
    #  into two functions -- findClosestCentroids and computeCentroids. In this
    #  part, you shoudl complete the code in the findClosestCentroids function.
    #
    def test_finding_closet_centroids(self):
        print('Finding closest centroids.')
        # Select an initial set of centroids
        K = 3  # 3 Centroids
        initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

        # Find the closest centroids for the examples using the
        # initial_centroids
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.findClosestCentroids import findClosestCentroids
        idx = findClosestCentroids(self.X, initial_centroids)

        print('Closest centroids for the first 3 examples:')
        print(' {idx}'.format(idx=idx))
        # adjusted next string for python's 0-indexing
        print('(the closest centroids should be 0, 2, 1 respectively)')

        self.assertEqual(idx.shape[0], self.X.shape[0])
        self.assertEqual(idx.flatten()[0], 0)
        self.assertEqual(idx.flatten()[1], 2)
        self.assertEqual(idx.flatten()[2], 1)
