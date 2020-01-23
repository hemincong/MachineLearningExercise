#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def checkCostFunction(_lambda=0):
    # CHECKCOSTFUNCTION Creates a collaborative filering problem
    # to check your cost function and gradients
    # CHECKCOSTFUNCTION(lambda ) Creates a collaborative filering problem
    # to check your cost function and gradients, it will output the

    # analytical gradients produced by your code and the numerical gradients
    # (computed using computeNumericalGradient).These two gradient
    # computations should result in very similar values.

    # Create small problem
    X_t = np.random.rand(4, 3)
    Theta_t = np.random.rand(5, 3)

    # Zap out most entries
    Y = np.dot(X_t, Theta_t.T)
    Y[np.random.rand(Y.shape[0], Y.shape[1]) > 0.5] = 0
    R = np.zeros(np.shape(Y))
    R[Y != 0] = 1

    # Run Gradient Checking
    X_t_shape = np.shape(X_t)
    X = np.random.randn(X_t_shape[0], X_t_shape[1])
    Theta_t_shape = np.shape(Theta_t)
    Theta = np.random.randn(Theta_t_shape[0], Theta_t_shape[1])
    num_movies, num_users = np.shape(Y)
    num_features = Theta_t_shape[1]

    params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))

    # Short hand for cost function
    from ex8_Anomaly_Detection_and_Recommender_Systems.computerNumericalGradient import computeNumericalGradient
    from ex8_Anomaly_Detection_and_Recommender_Systems.cofiCostFunc import cofiCostFunc
    numgrad = computeNumericalGradient(lambda p: cofiCostFunc(p, Y, R, num_users, num_movies, num_features, _lambda),
                                       params)

    cost, grad = cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda)
    print(np.column_stack((numgrad, grad)))
    print("The above two columns you get should be very similar.")
    print("(Left-Your Numerical Gradient, Right-Analytical Gradient)")

    diff = np.linalg.norm(numgrad - grad) / np.linalg.norm(numgrad + grad)
    print("If your backpropagation implementation is correct, then")
    print("the relative difference will be small (less than 1e-9). ")
    print("Relative Difference: {diff}".format(diff=diff))
