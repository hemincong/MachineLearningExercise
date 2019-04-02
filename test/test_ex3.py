#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.io as sio
import numpy as np

data_file = "resource/ex3data1.mat"


class test_ex3(unittest.TestCase):

    def test_displayData(self):
        import utils.displayData as dd
        # Load Training Data
        print('Loading and Visualizing Data ...')
        mat = sio.loadmat(data_file)
        X = mat["X"]
        m = X.shape[0]
        rand_indices = np.random.permutation(m)
        sel = X[rand_indices[:100], :]
        dd.displayData(sel)

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
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        X = np.array([im.reshape(400) for im in X])
        y = data.get('y')
        m, n = X.shape
        theta = np.zeros(n)
        from ex3_neural_network.lrCostFunction import costFunction
        ret = costFunction(theta, X, y, 1)
        self.assertAlmostEqual(ret, 801971.28, delta=0.01)

    def test_compute_grad(self):
        from ex3_neural_network.lrCostFunction import gradient
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

    def test_predict_one_vs_all(self):
        from ex3_neural_network.oneVsAll import oneVsAll
        from ex3_neural_network.predictOneVsAll import predictOneVsAll
        data = sio.loadmat(data_file)
        lambda_ = 0.5
        num_labels = 10
        X = data.get('X')
        y = data.get('y').reshape(-1)
        all_theta = oneVsAll(X, y, num_labels, lambda_)
        ret = predictOneVsAll(all_theta, X) + 1
        radio = np.mean((ret == y))
        self.assertGreater(radio, 0.90)


