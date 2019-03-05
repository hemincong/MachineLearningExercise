#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from ex3_neural_network.displayData import plot_100_image, plot_an_image

data_file = "resource/ex3data1.mat"


class test_ex3(unittest.TestCase):

    def test_displayAnImage(self):
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])
        plot_an_image(X[0])
        plt.show()

    def test_displayData(self):
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])
        plot_100_image(X)
        plt.show()

    def test_compute_cost_2(self):
        theta_t = np.array([-2, -1, 1, 2])
        X_t = np.column_stack([np.ones(5), np.arange(0.1, 1.6, 0.1).reshape((5, 3), order='F')])
        y_t = np.array([1, 0, 1, 0, 1]) >= 0.5
        lambda_t = 3
        from ex3_neural_network.lrCostFunction import lrCostFunction
        J, grad = lrCostFunction(theta_t, X_t, y_t, lambda_t)
        np.testing.assert_almost_equal(grad.tolist(), [0.146561, -0.548558, 0.724722, 1.398003], 5)
        np.testing.assert_almost_equal(J, 2.534819, 5)

    def test_compute_cost(self):
        from ex3_neural_network.lrCostFunction import costFunction
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
        y = data.get('y')
        m, n = X.shape
        theta = np.zeros(n)
        ret,_ = costFunction(theta, X, y, 1)
        self.assertAlmostEqual(ret, 801971.28, delta=0.01)

    def test_compute_grad(self):
        from ex3_neural_network.lrCostFunction import gradient, lrCostFunction
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
        y = data.get('y')
        m, n = X.shape
        theta = np.zeros(n)
        grad = gradient(theta, X, y, 1)
        self.assertEqual(grad.shape[0], 400 * 5000)

    def test_one_vs_all(self):
        from ex3_neural_network.oneVsAll import oneVsAll
        data = sio.loadmat(data_file)
        lambda_ = 1
        num_labels = 10
        X = data.get('X')
        y = data.get('y').reshape(-1)
        all_theta = oneVsAll(X, y, num_labels, lambda_)
        print(all_theta)


