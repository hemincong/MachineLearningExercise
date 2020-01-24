#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def cofiCostFunc(params, Y, R, num_users, num_movies, num_features, _lambda):
    # COFICOSTFUNC Collaborative filtering cost function
    #   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
    #   num_features, lambda ) returns the cost and gradient for the
    #   collaborative filtering problem.
    #

    # Unfold the U and W matrices from params
    # params = np.concatenate((X.reshape(X.size, order='F'), Theta.reshape(Theta.size, order='F')))
    X = np.reshape(params[:num_movies * num_features], (num_movies, num_features), order='F')
    Theta = np.reshape(params[num_movies * num_features:], (num_users, num_features), order='F')

    # You need to return the following values correctly
    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: Compute the cost function and gradient for collaborative
    #               filtering. Concretely, you should first implement the cost
    #               function (without regularization) and make sure it is
    #               matches our costs. After that, you should implement the
    #               gradient and use the checkCostFunction routine to check
    #               that the gradient is correct. Finally, you should implement
    #               regularization.
    #
    # Notes: X - num_movies  x num_features matrix of movie features
    #        Theta - num_users  x num_features matrix of user features
    #        Y - num_movies x num_users matrix of user ratings of movies
    #        R - num_movies x num_users matrix, where R(i, j) = 1 if the
    #            i-th movie was rated by the j-th user
    #
    # You should set the following variables correctly:
    #
    #        X_grad - num_movies x num_features matrix, containing the
    #                 partial derivatives w.r.t. to each element of X
    #        Theta_grad - num_users x num_features matrix, containing the
    #                     partial derivatives w.r.t. to each element of Theta
    #

    ### COST FUNCTION, NO REGULARIZATION
    J = (1 / 2) * np.sum(np.power(np.dot(X, Theta.T) - Y, 2) * R) \
        + (1 / 2) * _lambda * np.sum(np.sum(np.power(Theta, 2))) \
        + (1 / 2) * _lambda * np.sum(np.sum(np.power(X, 2)))

    X_grad = np.dot((np.dot(X, Theta.T) - Y) * R, Theta) + _lambda * X
    Theta_grad = np.dot(((np.dot(X, Theta.T) - Y) * R).T, X) + _lambda * Theta

    grad = np.concatenate((X_grad.reshape(X_grad.size, order='F'), Theta_grad.reshape(Theta_grad.size, order='F')))
    return J, grad
