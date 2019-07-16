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
        cls.Xval = mat["Xval"]
        cls.yval = mat["yval"]
        cls.Xtest = mat["Xtest"]
        cls.ytest = mat["ytest"]
        cls.m = np.shape(cls.X)[0]

    def test_load_and_visualzing_data(self):
        import matplotlib.pyplot as plt
        # print("point_end_y: {max_y}".format(max_y = point_end_y))
        plt.figure(1)
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')
        plt.scatter(self.X, self.y, marker='o', color='k', s=10)
        plt.show()

        # Plot training data
        print('Program paused. Press enter to continue.')

    # =========== Part 2: Regularized Linear Regression Cost =============
    # You should now implement the cost function for regularized linear
    # regression.
    def test_regularized_linear_regression_cost_and_grad(self):
        # m = Number of examples
        theta = np.array([[1], [1]])
        X_padded = np.column_stack((np.ones((self.m, 1)), self.X))
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
        x_with_bias = np.column_stack((np.ones(self.m), self.X))
        cost, theta = trainLinearReg(x_with_bias, self.y, _lambda)
        ret = x_with_bias.dot(theta)

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')
        plt.scatter(self.X, self.y, marker='x', c='r', s=30, linewidth=2)
        plt.plot(self.X, ret, linewidth=2)
        plt.show()

    #  =========== Part 5: Learning Curve for Linear Regression =============
    #  Next, you should implement the learningCurve function.
    #
    #  Write Up Note: Since the model is underfitting the data, we expect to
    #                 see a graph with "high bias" -- slide 8 in ML-advice.pdf
    #
    def test_learning_curve_for_linear_regression(self):
        _lambda = 0
        from ex5_regularized_linear_regressionand_bias_vs_variance.learningCurve import learningCurve
        x_with_bias = np.column_stack((np.ones(self.m), self.X))
        x_val_with_bias = np.column_stack((np.ones(np.shape(self.Xval)[0]), self.Xval))
        error_train, error_val = learningCurve(x_with_bias, self.y, x_val_with_bias, self.yval, 0)

        print('# Training Examples\tTrain Error\tCross Validation Error')

        for i in range(self.m):
            print('  \t{index}\t\t{error_train}\t{error_val}\n'.format(index=i,
                                                                       error_train=error_train[i],
                                                                       error_val=error_val[i]))

        import matplotlib.pyplot as plt
        temp = np.array([x for x in range(1, self.m + 1)])
        # plt.plot(1:m, error_train, 1:m, error_val);
        plt.title('Learning curve for linear regression')
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.plot(temp, np.array(error_train), color='b', linewidth=2, label='Train')
        plt.plot(temp, np.array(error_val), color='y', linewidth=2, label='Cross Validation')
        plt.legend()
        plt.show(block=True)

    # =========== Part 6: Feature Mapping for Polynomial Regression =============
    # One solution to this is to use polynomial regression.You should now
    # complete polyFeatures to map each example into its powers
    #
    def test_feature_mapping_for_polynomial_regression(self):
        p = 8
        # Map X onto Polynomial Features and Normalize
        from ex5_regularized_linear_regressionand_bias_vs_variance.polyFeatures import polyFeatures
        X_poly = polyFeatures(self.X, p)
        X_poly_m, X_poly_n = np.shape(X_poly)
        self.assertEqual(X_poly_m, self.m)
        self.assertEqual(X_poly_n, p)

        from ex5_regularized_linear_regressionand_bias_vs_variance.featureNormalize import featureNormalize
        X_norm, mu, sigma = featureNormalize(X_poly)
        print(X_norm)
        print(mu)
        print(sigma)
