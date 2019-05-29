#!/usr/bin/env python

from utils.sigmoid import sigmoid
import numpy as np


def nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda):
    # NNCOSTFUNCTION Implements the neural network cost function for a two layer
    # neural network which performs classification
    # [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
    # X, y, lambda ) computes the cost and gradient of the neural network.The
    # parameters for the neural network are "unrolled" into the vector
    # nn_params and need to be converted back into the weight matrices.
    #
    # The returned parameter grad should be a "unrolled" vector of the
    # partial derivatives of the neural network.
    #

    # Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
    # for our 2 layer neural network
    Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)],
                        (hidden_layer_size, input_layer_size + 1), order='F')

    Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):],
                        (num_labels, hidden_layer_size + 1), order='F')

    # Setup some useful vaiables
    m = len(X)

    # # You need to return the following variables correctly
    J = 0
    Theta1_grad = np.zeros(Theta1.shape)
    Theta2_grad = np.zeros(Theta2.shape)

    # ====================== YOUR CODE HERE ======================
    # Instructions: You should complete the code by working through the
    #               following parts.
    #
    # Part 1: Feedforward the neural network and return the cost in the
    #         variable J. After implementing Part 1, you can verify that your
    #         cost function computation is correct by verifying the cost
    #         computed in ex4.m
    #
    # Part 2: Implement the backpropagation algorithm to compute the gradients
    #         Theta1_grad and Theta2_grad. You should return the partial derivatives of
    #         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
    #         Theta2_grad, respectively. After implementing Part 2, you can check
    #         that your implementation is correct by running checkNNGradients
    #
    #         Note: The vector y passed into the function is a vector of labels
    #               containing values from 1..K. You need to map this vector into a
    #               binary vector of 1's and 0's to be used with the neural network
    #               cost function.
    #
    #         Hint: We recommend implementing backpropagation using a for-loop
    #               over the training examples if you are implementing it for the
    #               first time.
    #
    # Part 3: Implement regularization with the cost function and gradients.
    #
    #         Hint: You can implement this around the code for
    #               backpropagation. That is, you can compute the gradients for
    #               the regularization separately and then add them to Theta1_grad
    #               and Theta2_grad from Part 2.
    #

    # add bias 1
    X = np.column_stack((np.ones((m, 1)), X))

    # a2 = X . dot Theta1.T
    a2 = sigmoid(np.dot(X, Theta1.T))

    # a2 add bias
    a2 = np.column_stack((np.ones((a2.shape[0], 1)), a2))

    a3 = sigmoid(np.dot(a2, Theta2.T))

    # Also, recall that whereas the original labels (in the variable y) were 1, 2, ..., 10, for the purpose of
    # training a neural network, we need to recode the labels as vectors containing only values 0 or 1, so that For
    # example, if x(i) is an image of the digit 5, then the corresponding y(i) (that you should use with the cost
    # function) should be a 10-dimensional vector with y5 = 1, and the other elements equal to 0
    labels = y
    y = np.zeros((m, num_labels))
    for i in range(m):
        y[i, labels[i] - 1] = 1

    # You should implement the feedforward computation that computes hθ(x(i)) for every example i and sum the cost
    # over all examples. Your code should also work for a dataset of any size, with any number of labels (you can
    # assume that there are always at least K ≥ 3 labels).
    cost = 0
    for i in range(m):
        cost += np.sum(y[i] * np.log(a3[i]) + (1 - y[i]) * np.log(1 - a3[i]))

    J = -(1.0 / m) * cost

    theta1_square = np.sum(np.square(Theta1[:, 1:]))
    theta2_square = np.sum(np.square(Theta2[:, 1:]))

    J = J + _lambda * (theta1_square + theta2_square) / 2 / m
    return J, 0
