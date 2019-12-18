#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import matplotlib.pyplot as plt
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

        plt.ion()
        plt.close()

        plt.scatter(self.X[:, 0], self.X[:, 1], s=75, facecolors='none', edgecolors='b')
        plt.axis([0.5, 6.5, 2, 8])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show(block=False)

    # =============== Part 2: Principal Component Analysis ===============
    #  You should now implement PCA, a dimension reduction technique. You
    #  should complete the code in pca.m
    #
    def test_Principal_Component_Analysis(self):
        print('Running PCA on example dataset.')

        # Before running PCA, it is important to first normalize X
        import utils.featureNormalize
        X_norm, mu, _ = utils.featureNormalize.featureNormalize(self.X)

        plt.scatter(self.X[:, 0], self.X[:, 1], s=75, facecolors='none', edgecolors='b')
        plt.axis([0.5, 6.5, 2, 8])
        plt.gca().set_aspect('equal', adjustable='box')

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

        print('Top eigenvector:')
        print(' U[:,1] = {:f} {:f}'.format(U[0, 0], U[1, 0]))
        self.assertAlmostEqual(U[0, 0], -0.707107, delta=0.0001)
        self.assertAlmostEqual(U[1, 0], -0.707107, delta=0.0001)
        plt.show(block=False)

    # =================== Part 3: Dimension Reduction ===================
    # You should now implement the projection step to map the data onto the
    # first k eigenvectors.The code will then plot the data in this reduced
    # dimensional space.This will show you what the data looks like when
    # using only the corresponding eigenvectors to reconstruct it.
    #
    # You should complete the code in projectData.m
    #
    def test_Dimension_Reduction(self):
        print("Dimension reduction on example dataset.")

        import utils.featureNormalize
        X_norm, mu, _ = utils.featureNormalize.featureNormalize(self.X)

        import ex7_K_means_Clustering_and_Principal_Component_Analysis.pca
        U, S = ex7_K_means_Clustering_and_Principal_Component_Analysis.pca.pca(X_norm)

        #  Plot the normalized dataset (returned from pca)
        plt.close()
        plt.scatter(X_norm[:, 0], X_norm[:, 1], s=75, facecolors='none', edgecolors='b')
        plt.axis([-4, 3, -4, 3])
        plt.gca().set_aspect('equal', adjustable='box')

        from ex7_K_means_Clustering_and_Principal_Component_Analysis.projectData import projectData
        #  Project the data onto K = 1 dimension
        K = 1
        Z = projectData(X_norm, U, K)
        print("Projection of the first example: {Z}".format(Z=Z[0]))
        print("(this value should be about 1.481274)")
        self.assertAlmostEqual(Z[0], 1.481, delta=0.1)

        from ex7_K_means_Clustering_and_Principal_Component_Analysis.recoverData import recoverData
        X_rec = recoverData(Z, U, K)
        print("Approximation of the first example: {num1} {num2}".format(num1=X_rec[0, 0], num2=X_rec[0, 1]))
        print("(this value should be about  -1.047419 -1.047419)")
        self.assertAlmostEqual(X_rec[0, 0], -1.047419, delta=0.1)
        self.assertAlmostEqual(X_rec[0, 1], -1.047419, delta=0.1)

        plt.scatter(X_rec[:, 0], X_rec[:, 1], s=75, facecolors='none', edgecolors='r')

        for i in range(X_norm.shape[0]):
            plt.plot([X_norm[i, :][0], X_rec[i, :][0]], [X_norm[i, :][1], X_rec[i, :][1]], linestyle='--', color='k',
                     linewidth=1)

        plt.show(block=False)

    # =============== Part 4: Loading and Visualizing Face Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  The following code will load the dataset into your environment
    #
    def test_Loading_and_Visualizing_Face_Data(self):
        print('Loading face dataset.')

        #  Load Face dataset
        mat = scipy.io.loadmat('resource/ex7faces.mat')
        X = np.array(mat["X"])

        #  Display the first 100 faces in the dataset
        from utils.displayData import displayData
        displayData(X[:100, :])

    #  =========== Part 5: PCA on Face Data: Eigenfaces  ===================
    #  Run PCA and visualize the eigenvectors which are in this case eigenfaces
    #  We display the first 36 eigenfaces.
    #
    def test_PCA_on_Face_Data_Eignfaces(self):
        print("Running PCA on face dataset.")
        print("this mght take a minute or two ...")

        #  Load Face dataset
        mat = scipy.io.loadmat('resource/ex7faces.mat')
        X = np.array(mat["X"])

        #  Before running PCA, it is important to first normalize X by subtracting
        #  the mean value from each feature
        from utils.featureNormalize import featureNormalize
        X_norm, _, _ = featureNormalize(X)

        #  Run PCA
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.pca import pca
        U, S = pca(X_norm)

        #  Visualize the top 36 eigenvectors found
        from utils.displayData import displayData
        displayData(U[:, :36].T)

        #  ============= Part 6: Dimension Reduction for Faces =================
        #  Project images to the eigen space using the top k eigenvectors
        #  If you are applying a machine learning algorithm
        print("Dimension reduction for face dataset.")
        K = 100
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.projectData import projectData
        Z = projectData(X_norm, U, K)

        print("The projected data Z has a size of: {z}".format(z=np.shape(Z)))

        #  ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
        #  Project images to the eigen space using the top K eigen vectors and
        #  visualize only using those K dimensions
        #  Compare to the original input, which is also displayed

        print("Visualizing the projected (reduced dimension) faces.")
        K = 100
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.recoverData import recoverData
        X_rec = recoverData(Z, U, K)

        # Display normalized data
        plt.close()
        plt.subplot(1, 2, 1)
        displayData(X_norm[:100, :])
        plt.title('Original faces')
        plt.gca().set_aspect('equal', adjustable='box')

        # Display reconstructed data from only k eigenfaces
        plt.subplot(1, 2, 2)
        displayData(X_rec[:100, :])
        plt.title('Recovered faces')
        plt.gca().set_aspect('equal', adjustable='box')

    #  === Part 8(a): Optional (ungraded) Exercise: PCA for Visualization ===
    #  One useful application of PCA is to use it to visualize high-dimensional
    #  data. In the last K-Means exercise you ran K-Means on 3-dimensional
    #  pixel colors of an image. We first visualize this output in 3D, and then
    #  apply PCA to obtain a visualization in 2D.
    def test_PCA_for_Visualization(self):
        plt.close()
        # Re-load the image from the previous exercise and run K-Means on it
        # For this to work, you need to complete the K-Means assignment first

        # A = double(imread('bird_small.png'));
        # If imread does not work for you, you can try instead
        mat = scipy.io.loadmat('resource/bird_small.mat')
        A = mat["A"]

        A = A / 255
        image_size = np.shape(A)
        X = A.reshape(image_size[0] * image_size[1], 3)
        K = 16
        max_iters = 10
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.kMeansInitCentroids import kMeansInitCentroids
        initial_centroids = kMeansInitCentroids(X, K)
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.runkMeans import runkMeans
        centorids, idx = runkMeans(X, initial_centroids, max_iters, True)

        # Sample 1000 random indexes(since working with all the data is
        # too expensive.If you have a fast computer, you may increase this.
        sel = np.floor(np.random.rand(1000, 1) * X.shape[0]).astype(int).flatten()

        # Setup Color Palette
        from utils.hsv import hsv
        palette = hsv(K)
        colors = np.array([palette[int(i)] for i in idx[sel]])

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(X[sel, 0], X[sel, 1], X[sel, 2], s=100, c=colors)
        # plt.title('Pixel dataset plotted in 3D. Color shows centroid memberships')
        # plt.show(block=False)

        # === Part 8(b): Optional (ungraded) Exercise: PCA for Visualization ===
        # Use PCA to project this cloud to 2D for visualization

        from ex5_regularized_linear_regressionand_bias_vs_variance.featureNormalize import featureNormalize
        # Subtract the mean to use PCA
        X_norm, _, _ = featureNormalize(X)

        # PCA and project the data to 2D
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.pca import pca
        U, S = pca(X_norm)

        from ex7_K_means_Clustering_and_Principal_Component_Analysis.projectData import projectData
        Z = projectData(X_norm, U, 2)

        # Plot in 2D
        plt.figure(2)
        from ex7_K_means_Clustering_and_Principal_Component_Analysis.plotDataPoints import plotDataPoints
        plotDataPoints(Z[sel, :], idx[sel], K)
        plt.title('Pixel dataset plotted in 2D, using PCA for dimensionality reduction')
        plt.show(block=False)
