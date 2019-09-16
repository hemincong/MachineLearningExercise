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
        # print(' {idx}'.format(idx=idx))
        # adjusted next string for python's 0-indexing
        print('(the closest centroids should be 0, 2, 1 respectively)')

        self.assertEqual(idx.shape[0], self.X.shape[0])
        self.assertEqual(idx.flatten()[0], 0)
        self.assertEqual(idx.flatten()[1], 2)
        self.assertEqual(idx.flatten()[2], 1)
        # ===================== Part 2: Compute Means =========================
        #  After implementing the closest centroids function, you should now
        #  complete the computeCentroids function.
        #
        print('Computing centroids means.')

        #  Compute means based on the closest centroids found in the previous part.
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.computeCentroids import computeCentroids
        centroids = computeCentroids(self.X, idx, K)

        print('Centroids computed after initial finding of closest centroids: ')
        print(' {centroids} '.format(centroids=centroids))
        print('(the centroids should be')
        print('   [ 2.428301 3.157924 ]')
        print('   [ 5.813503 2.633656 ]')
        print('   [ 7.119387 3.616684 ]')
        self.assertAlmostEqual(centroids[0, :][0], 2.429301, delta=0.001)
        self.assertAlmostEqual(centroids[0, :][1], 3.157924, delta=0.001)
        self.assertAlmostEqual(centroids[1, :][0], 5.813503, delta=0.001)
        self.assertAlmostEqual(centroids[1, :][1], 2.633656, delta=0.001)
        self.assertAlmostEqual(centroids[2, :][0], 7.119387, delta=0.001)
        self.assertAlmostEqual(centroids[2, :][1], 3.616684, delta=0.001)

    # =================== Part 3: K-Means Clustering ======================
    #  After you have completed the two functions computeCentroids and
    #  findClosestCentroids, you have all the necessary pieces to run the
    #  kMeans algorithm. In this part, you will run the K-Means algorithm on
    #  the example dataset we have provided.
    #
    def test_k_means_clustering(self):
        print('Running K-Means clustering on example dataset.')
        K = 3
        max_iters = 10

        # For consistency, here we set centroids to specific values
        # but in practice you want to generate them automatically, such as by
        # settings them to be random examples(as can be seen in
        # kMeansInitCentroids).
        initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

        from ex7_K_means_Clustering_and_Principal_Component_Analysis.runkMeans import runkMeans
        runkMeans(self.X, initial_centroids, max_iters, True)

        print("K-Means Done")

    # ============== Part 4: K-Means Clustering on Pixels ===============
    #  In this exercise, you will use K-Means to compress an image. To do this,
    #  you will first run K-Means on the colors of the pixels in the image and
    #  then you will map each pixel on to it's closest centroid.
    #
    #  You should now complete the code in kMeansInitCentroids.m
    #
    def test_k_means_clustering_on_pixels(self):
        print('Running K-Means clustering on pixels from an image.')

        # Load an image of a bird
        # If imread does not work for you, you can try instead
        image_file = "resource/bird_small.mat"
        image_mat = scipy.io.loadmat(image_file)
        A = image_mat["A"]
        A = A / 255.0  # Divide by 255 so that all values are in the range 0 - 1

        # Size of the image
        m, n, c = np.shape(A)

        # Reshape the image into an Nx3 matrix where N = number of pixels.
        # Each row will contain the Red, Green and Blue pixel values
        # This gives us our dataset matrix X that we will use K - Means on.
        X = np.reshape(A, ((m * n), c), order='F').copy()

        # Run your K-Means algorithm on this data
        # You should try different values of K and max_iters here
        K = 16
        max_iters = 10

        # When using K-Means, it is important the initialize the centroids
        # randomly.
        # You should complete the code in kMeansInitCentroids.m before proceeding
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.kMeansInitCentroids import kMeansInitCentroids
        rand_centroid = kMeansInitCentroids(X, K)

        from ex7_K_means_Clustering_and_Principal_Component_Analysis.runkMeans import runkMeans
        centroids, idx = runkMeans(X, rand_centroid, max_iters, True)

        # ================= Part 5: Image Compression ======================
        #  In this part of the exercise, you will use the clusters of K-Means to
        #  compress an image. To do this, we first find the closest clusters for
        #  each example. After that, we

        print('Applying K-Means to compress an image.')

        # Find closest cluster members
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.findClosestCentroids import findClosestCentroids
        # Reshape the image into an Nx3 matrix where N = number of pixels.
        # Each row will contain the Red, Green and Blue pixel values
        # This gives us our dataset matrix X that we will use K - Means on.
        idx = findClosestCentroids(X, centroids)

        # Essentially, now we have represented the image X as in terms of the
        # indices in idx.

        # We can now recover the image from the indices (idx) by mapping each pixel
        # (specified by it's index in idx) to the centroid value
        X_recovered = centroids[idx]

        # Reshape the recovered image into proper dimensions
        X_recovered = X_recovered.reshape(m, n, 3, order='F')

        import matplotlib.pyplot as plt
        # Display the original image
        plt.close()
        plt.subplot(1, 2, 1)
        plt.imshow(A)
        plt.title('Original')

        # Display compressed image side by side
        plt.subplot(1, 2, 2)
        plt.imshow(X_recovered)
        plt.title('Compressed, with {:d} colors.'.format(K))
        plt.show(block=False)
