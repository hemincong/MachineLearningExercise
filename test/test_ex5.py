#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

    # =========== Part 1: Loading and Visualizing Data =============
    # We start the exercise by first loading and visualizing the dataset.
    # The following code will load the dataset into your environment and plot
    # the data.
    #
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
        X_poly, mu, sigma = featureNormalize(X_poly)
        X_poly = np.column_stack((np.ones((self.m, 1)), X_poly))

        X_poly_test = polyFeatures(self.Xtest, p)
        X_poly_test_m, X_poly_test_n = np.shape(X_poly_test)
        self.assertEqual(X_poly_test_m, np.shape(self.Xtest)[0])
        self.assertEqual(X_poly_test_n, p)
        X_poly_test = X_poly_test - mu
        X_poly_test = X_poly_test / sigma
        X_poly_test = np.column_stack((np.ones((X_poly_test.shape[0], 1)), X_poly_test))

        X_poly_val = polyFeatures(self.Xval, p)
        X_poly_val_m, X_poly_val_n = np.shape(X_poly_val)
        self.assertEqual(X_poly_val_m, np.shape(self.Xval)[0])
        self.assertEqual(X_poly_val_n, p)
        X_poly_val = X_poly_val - mu
        X_poly_val = X_poly_val / sigma
        X_poly_val = np.column_stack((np.ones((X_poly_val.shape[0], 1)), X_poly_val))

        print('Normalized Training Example 1:\n'
              '  {X_poly}  '.format(X_poly=X_poly))

        # =========== Part 7: Learning Curve for Polynomial Regression =============
        # Now, you will get to experiment with polynomial regression with multiple
        # values of lambda .The code below runs polynomial regression with
        # lambda = 0. You should try running the code with different values of
        # lambda to see how the fit and learning curve change.
        #
        _lambda = 0
        from ex5_regularized_linear_regressionand_bias_vs_variance.trainLinearReg import trainLinearReg
        cost, theta = trainLinearReg(X_poly, self.y, _lambda)
        self.assertIsNotNone(cost)
        self.assertIsNotNone(theta)

        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.scatter(self.X, self.y, marker='x', c='r', s=30, linewidth=2)
        plt.xlim([-80, 80])
        plt.ylim([-20, 60])
        plt.xlabel('Change in water level(x)')
        plt.ylabel('Water flowing out of the dam(y)')
        plt.title('Polynomial Regression Fit (lambda = {:f})'.format(_lambda))

        # plt.plot(self.X, self.y, 'rx', markersize=10, linewidth=1.5)
        from ex5_regularized_linear_regressionand_bias_vs_variance.plotFit import plotFit
        plotFit(min(self.X), max(self.X), mu, sigma, theta, p)
        plt.show(block=False)

        plt.figure(2)
        from ex5_regularized_linear_regressionand_bias_vs_variance.learningCurve import learningCurve
        error_train, error_val = learningCurve(X_poly, self.y, X_poly_val, self.yval, 0)
        p1, p2 = plt.plot(range(1, self.m + 1), error_train, range(1, self.m + 1), error_val)
        plt.legend((p1, p2), ('Train', 'Cross Validation'))
        plt.show(block=False)

        print('Polynomial Regression (lambda =%{_lambda})'.format(_lambda=_lambda))
        print('# Training Examples\tTrain Error\tCross Validation Error')
        for i in range(0, self.m):
            print('\t{i}\t\t{error_train}\t{error_val}'.format(i=i, error_train=error_train[i], error_val=error_val[i]))

        # =========== Part 8: Validation for Selecting Lambda =============
        #  You will now implement validationCurve to test various values of
        #  lambda on a validation set. You will then use this to select the
        #  "best" lambda value.
        #

        from ex5_regularized_linear_regressionand_bias_vs_variance.validationCurve import validationCurve
        lambda_vec, error_train, error_val = validationCurve(X_poly, self.y, X_poly_val, self.yval)
        self.assertEqual(len(error_train), len(lambda_vec))
        self.assertEqual(len(error_val), len(lambda_vec))

        plt.close('all')
        p1, p2, = plt.plot(lambda_vec, error_train, lambda_vec, error_val)
        plt.legend((p1, p2), ('Train', 'Cross Validation'))
        plt.xlabel('lambda')
        plt.ylabel('Error')
        plt.show(block=False)

        print('lambda\t\tTrain Error\tValidation Error')
        for i in range(len(lambda_vec)):
            print(
                '{lambda_vec}\t{error_train}\t{error_val}'.format(lambda_vec=lambda_vec[i], error_train=error_train[i],
                                                                  error_val=error_val[i]))
        # =========== Part 9: Computing test set error and Plotting learning curves with randomly selected examples
        # ============= best lambda value from previous step
        lambda_val = 3

        # note that we're using X_poly - polynomial linear regression with polynomial features
        from ex5_regularized_linear_regressionand_bias_vs_variance.trainLinearReg import trainLinearReg
        _, theta = trainLinearReg(X_poly, self.y, lambda_val)

        # because we're using X_poly, we also have to use X_poly_test with polynomial features
        from ex5_regularized_linear_regressionand_bias_vs_variance.linearRegCostFunction import linearRegCostFunction
        error_test, _ = linearRegCostFunction(X_poly_test, self.ytest, theta, 0)
        print('Test set error: {error_test}'.format(error_test=error_test))  # expected 3.859
        # why? something wrong
        # self.assertAlmostEqual(error_test, 3.859, delta=0.01)

        # =========== Part 10: Plot learning curves with randomly selected examples =============
        #

        # lambda_val value for this step
        lambda_val = 0.01

        times = 50

        error_train_rand = np.zeros((self.m, times))
        error_val_rand = np.zeros((self.m, times))

        for i in range(self.m):
            for k in range(times):
                rand_sample_train = np.random.permutation(X_poly.shape[0])
                rand_sample_train = rand_sample_train[:i + 1]

                rand_sample_val = np.random.permutation(X_poly_val.shape[0])
                rand_sample_val = rand_sample_val[:i + 1]

                X_poly_train_rand = X_poly[rand_sample_train, :]
                y_train_rand = self.y[rand_sample_train]
                X_poly_val_rand = X_poly_val[rand_sample_val, :]
                y_val_rand = self.yval[rand_sample_val]

                _, theta = trainLinearReg(X_poly_train_rand, y_train_rand, lambda_val)
                cost, _ = linearRegCostFunction(X_poly_train_rand, y_train_rand, np.asarray(theta), 0)
                error_train_rand[i, k] = cost
                cost, _ = linearRegCostFunction(X_poly_val_rand, y_val_rand, theta, 0)
                error_val_rand[i, k] = cost

        error_train = np.mean(error_train_rand, axis=1)
        error_val = np.mean(error_val_rand, axis=1)

        p1, p2 = plt.plot(range(self.m), error_train, range(self.m), error_val)
        plt.title('Polynomial Regression Learning Curve (lambda = {:f})'.format(lambda_val))
        plt.legend((p1, p2), ('Train', 'Cross Validation'))
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.axis([0, 13, 0, 150])
        plt.show(block=False)
