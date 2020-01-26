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

    #  ============== Part 6: Entering ratings for a new user ===============
    #  Before we will train the collaborative filtering model, we will first
    #  add ratings that correspond to a new user that we just observed. This
    #  part of the code will also allow you to put in your own ratings for the
    #  movies in our dataset!
    #
    def test_Entering_ratings_for_a_new_user(self):
        from ex8_Anomaly_Detection_and_Recommender_Systems.loadMovieList import loadMovieList
        movieList = loadMovieList()

        #  Initialize my ratings
        my_ratings = np.zeros((1682, 1))

        # Check the file movie_idx.txt for id of each movie in our dataset
        # For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
        my_ratings[0] = 4

        # Or suppose did not enjoy Silence of the Lambs (1991), you can set
        my_ratings[97] = 2

        # We have selected a few movies we liked / did not like and the ratings we
        # gave are as follows:
        my_ratings[6] = 3

        # We have selected a few movies we liked / did not like and the ratings we
        # gave are as follows:
        my_ratings[6] = 3
        my_ratings[11] = 5
        my_ratings[53] = 4
        my_ratings[63] = 5
        my_ratings[65] = 3
        my_ratings[68] = 5
        my_ratings[182] = 4
        my_ratings[225] = 5
        my_ratings[354] = 5
        print('New user ratings:')
        for i in range(my_ratings.size):
            if my_ratings[i] > 0:
                print("Rated {rate} for {movie}".format(rate=my_ratings[i], movie=movieList[i]))

        #  ================== Part 7: Learning Movie Ratings ====================
        #  Now, you will train the collaborative filtering model on a movie rating
        #  dataset of 1682 movies and 943 users
        #

        print('Training collaborative filtering...')

        data_file = "resource/ex8_movies.mat"
        #  Reduce the data set size so that this runs faster
        #  Load data
        mat = scipy.io.loadmat(data_file)

        Y = mat["Y"]
        R = mat["R"]
        #  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by
        #  943 users
        #
        #  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
        #  rating to movie i

        #  Add our own ratings to the data matrix
        Y = np.column_stack((my_ratings, Y))
        R = np.column_stack(((my_ratings != 0).astype(int), R))

        from ex8_Anomaly_Detection_and_Recommender_Systems.normalizeRatings import normalizeRatings
        Ynorm, Ymean = normalizeRatings(Y, R)

        # Userful Values
        num_users = Y.shape[1]
        num_movies = Y.shape[0]
        num_features = 10

        # Set Initial Parameters (Theta, X)
        X = np.random.randn(num_movies, num_features)
        Theta = np.random.randn(num_users, num_features)

        params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

        # Set Regularization
        _lambda = 10
        # Set options
        maxiter = 100
        options = {'disp': True, 'maxiter': maxiter}

        from ex8_Anomaly_Detection_and_Recommender_Systems.cofiCostFunc import cofiCostFunc

        def cf(_p):
            return cofiCostFunc(_p, Y, R, num_users, num_movies, num_features, _lambda)[0]

        def gf(_p):
            return cofiCostFunc(_p, Y, R, num_users, num_movies, num_features, _lambda)[1]

        from scipy.optimize import fmin_l_bfgs_b
        result2 = fmin_l_bfgs_b(cf, fprime=gf, x0=params,
                                maxiter=100, disp=True)
        print(result2)
        from scipy.optimize import minimize
        result = minimize(lambda _p: cofiCostFunc(_p, Y, R, num_users, num_movies, num_features, _lambda), x0=params,
                          options=options, method='L-BFGS-B', jac=True)
        print(result)
        # r = result["x"]
        r = result2[0]

        X = np.reshape(r[:num_movies * num_features], (num_movies, num_features), order='F')
        Theta = np.reshape(r[num_movies * num_features:], (num_users, num_features), order='F')
        print('Recommender system learning completed.')

        #  ================== Part 8: Recommendation for you ====================
        #  After training the model, you can now make recommendations by computing
        #  the predictions matrix.
        #

        p = np.dot(X, Theta.T)
        my_predictions = p[:, 0:1] + Ymean
        # reverse sorting by index
        idx = my_predictions.argsort(axis=0)[::-1]
        my_predictions = my_predictions[idx]

        print('Top recommendations for you:')
        for i in range(10):
            j = idx[i, 0]
            print('Predicting rating {p} for movie {name}'.format(p=my_predictions[j], name=movieList[j]))

        print('Original ratings provided:')
        for i in range(len(my_ratings)):
            if my_ratings[i] > 0:
                print('Rated {:d} for {:s}'.format(int(my_ratings[i]), movieList[i]))


if __name__ == '__main__':
    unittest.main()
