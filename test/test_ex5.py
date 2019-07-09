#!/usr/bin/env python

import unittest

import numpy as np
import scipy.io


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
    @classmethod
    def setUp(cls):
        # Load Training Data
        print('Loading and Visualizing Data ...')
        data_file = "resource/ex5data1.mat"

        # Load
        # You will have X, y, Xval, yval, Xtest, ytest in your environment
        mat = scipy.io.loadmat(data_file)

        cls.X = mat["X"]
        cls.y = mat["y"]

    def test_load_and_visualzing_data(self):
        import matplotlib.pyplot as plt
        # print("point_end_y: {max_y}".format(max_y = point_end_y))
        plt.figure(1)
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')
        plt.scatter(self.X, self.y, marker='o', color='k', s=10)
        plt.legend(loc='lower right')
        plt.show()

        # Plot training data
        print('Program paused. Press enter to continue.')

    # =========== Part 2: Regularized Linear Regression Cost =============
    # You should now implement the cost function for regularized linear
    # regression.
    def test_regularized_linear_regression_cost_and_grad(self):
        # m = Number of examples
        m = np.shape(self.X)[0]
        theta = np.array([[1],[1]])
        X_padded = np.column_stack((np.ones((m, 1)), self.X))
        from ex5_regularized_linear_regressionand_bias_vs_variance.linearRegCostFunction import linearRegCostFunction
        J, grad = linearRegCostFunction(X_padded, self.y, theta, 1)
        self.assertAlmostEqual(J, 303.993, delta=0.001)
        print('Cost at theta = [1 ; 1]: {cost} \n'
              '(this value should be about 303.993192)'.format(cost=J))

        # =========== Part 3: Regularized Linear Regression Gradient =============
        # You should now implement the gradient for regularized linear
        # regression.
        self.assertAlmostEqual(grad[0], -15.303016, delta=0.0001)
        self.assertAlmostEqual(grad[1], 598.250744, delta=0.0001)
        print('Gradient at theta = [1 ; 1]:  [{grad_0}; {grad_1}] \n'
              '(this value should be about [-15.303016; 598.250744])\n'.format(grad_0=grad[0], grad_1=grad[1]))

    # =========== Part 4: Train Linear Regression =============
    # Once you have implemented the cost and gradient correctly, the
    # trainLinearReg function will use your cost function to train
    # regularized linear regression.
    #
    # Write Up Note: The data is non - linear, so this will not give a great
    # fit.
    #

    def test_train_linear_reg(self):
        from ex5_regularized_linear_regressionand_bias_vs_variance.trainLinearReg import trainLinearReg
        # Train linear regression with lambda = 0
        _lambda = 0
        ret = trainLinearReg(self.X, self.y, _lambda)
        print(ret)
