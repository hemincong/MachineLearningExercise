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
    # ================== Part 1: Load Example Dataset ===================
    # We start this exercise by using a small dataset that is easily to
    # visualize
    #
    # The following command loads the dataset.You should now have the
    # variable X in your environment
    @staticmethod
    def Visualizing_example_dataset_for_PCA():
        print('Visualizing example dataset for PCA.')
        data_file = "resource/ex7data1.mat"
        mat = scipy.io.loadmat(data_file)
        X = np.array(mat["X"])

        import matplotlib.pyplot as plt
        plt.close()

        plt.scatter(X[:, 0], X[:, 1], s=75, facecolors='none', edgecolors='b')
        plt.axis([0.5, 6.5, 2, 8])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=False)
