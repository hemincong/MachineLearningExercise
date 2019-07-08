#!/usr/bin/env python

import unittest

import numpy as np
import scipy.io

data_file = "resource/ex5data1.mat"


#  Exercise 5 | Regularized Linear Regression and Bias-Variance
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     linearRegCostFunction.m
#     learningCurve.m
#     validationCurve.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
class test_ex5_regularized_linear_regressionand_bias_vs_variance(unittest.TestCase):

    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset.
    # The following code will load the dataset into your environment and plot
    # the data.
    #
    def test_load_and_visualzing_data(self):

        # Load Training Data
        print('Loading and Visualizing Data ...')

        # Load
        # You will have X, y, Xval, yval, Xtest, ytest in your environment
        mat = scipy.io.loadmat(data_file)

        # m = Number of examples
        X = mat["X"]
        y = mat["y"]

        import matplotlib.pyplot as plt
        # print("point_end_y: {max_y}".format(max_y = point_end_y))
        plt.figure(1)
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')
        plt.scatter(X, y, marker='o', color='k', s=10)
        plt.legend(loc='lower right')
        plt.show()

        # Plot training data
        print('Program paused. Press enter to continue.')

    # =========== Part 2: Regularized Linear Regression Cost =============
    # You should now implement the cost function for regularized linear
    # regression.
    def (self):
        mat = scipy.io.loadmat(data_file)

        # m = Number of examples
        X = mat["X"]
        y = mat["y"]
        m = np.shape(X)[0]
        theta = np.array([[1], [1]])
        X_padded = np.column_stack((np.ones((m, 1)), X))
        from ex5_regularized_linear_regressionand_bias_vs_variance.linearRegCostFunction import linearRegCostFunction
        J = linearRegCostFunction(X_padded, y, theta, 1)
        print(J)

