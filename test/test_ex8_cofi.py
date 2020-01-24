#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import matplotlib.pyplot as plt
import numpy as np
import scipy.io


#  Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
class test_ex8_cofi(unittest.TestCase):

    @classmethod
    def setUp(cls):
        #  Load Training Data
        data_file = "resource/ex8_movies.mat"
        mat = scipy.io.loadmat(data_file)
        cls.movies_Y = mat["Y"]
        cls.movies_R = mat["R"]

    #  =============== Part 1: Loading movie ratings dataset ================
    #  You will start by loading the movie ratings dataset to understand the
    #  structure of the data.
    #
    def test_Load_movie_ratings_dataset(self):
        # Y is a 1682 x943 matrix, containing ratings(1 - 5) of 1682 movies on
        # 943 users # # R is a 1682 x943 matrix, where R(i, j) = 1 if and only if user j gave a
        # rating to movie i

        # From the matrix, we can compute statistics like average rating.
        #  We can "visualize" the ratings matrix by plotting it with imagesc
        # need aspect='auto' for a ~16:9 (vs almost vertical) aspect
        plt.imshow(self.movies_Y, aspect='auto')
        plt.ylabel('Movies')
        plt.xlabel('Users')
        plt.show(block=False)

    #  ============ Part 2: Collaborative Filtering Cost Function ===========
    #  You will now implement the cost function for collaborative filtering.
    #  To help you debug your cost function, we have included set of weights
    #  that we trained on that. Specifically, you should complete the code in
    #  cofiCostFunc.m to return J.

    #  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
    def test_Collaborative_Filtering_Cost_Function(self):
        data_file = "resource/ex8_movieParams.mat"
        #  Reduce the data set size so that this runs faster
        mat = scipy.io.loadmat(data_file)
        X = mat["X"]
        Theta = mat["Theta"]

        num_users = 4
        num_movies = 5
        num_features = 3

        X = X[:num_movies, :num_features]
        Theta = Theta[:num_users, :num_features]
        Y = self.movies_Y[:num_movies, :num_users]
        R = self.movies_R[:num_movies, :num_users]
        from ex8_Anomaly_Detection_and_Recommender_Systems.cofiCostFunc import cofiCostFunc

        # Evaluate cost function
        params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
        J, _ = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 0)
        print("Cost at loaded parameters: {cost}".format(cost=J))
        print("(this value should be about 22.22)")
        self.assertAlmostEqual(J, 22.22, delta=0.01)

        #  ============== Part 3: Collaborative Filtering Gradient ==============
        #  Once your cost function matches up with ours, you should now implement
        #  the collaborative filtering gradient function. Specifically, you should
        #  complete the code in cofiCostFunc.m to return the grad argument.
        #
        print('Checking Gradients (without regularization) ... ')
        from ex8_Anomaly_Detection_and_Recommender_Systems.checkCostFunction import checkCostFunction
        checkCostFunction()

        #  ========= Part 4: Collaborative Filtering Cost Regularization ========
        #  Now, you should implement regularization for the cost function for
        #  collaborative filtering. You can implement it by adding the cost of
        #  regularization to the original cost computation.
        #

        #  Evaluate cost function
        J, _ = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, 1.5)
        print("'Cost at loaded parameters (lambda = 1.5): {cost}".format(cost=J))
        print("(this value should be about 31.34)")
        self.assertAlmostEqual(J, 31.34, delta=0.01)

        #  ======= Part 5: Collaborative Filtering Gradient Regularization ======
        #  Once your cost matches up with ours, you should proceed to implement
        #  regularization for the gradient.
        #
        print('Checking Gradients (with regularization) ... ')

        #  Check gradients by running checkNNGradients
        checkCostFunction(1.5)


if __name__ == '__main__':
    unittest.main()
