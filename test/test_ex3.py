#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from ex3_neural_network.displayData import plot_100_image

data_file = "resource/ex3data1.mat"


class test_ex3(unittest.TestCase):

    def test_displayData(self):
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])
        plot_100_image(X)
        plt.show()

    def test_lrCostFunction(self):
        from ex3_neural_network.lrCostFunction import lrCostFunction
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])
        y = data.get('y')
        theta = np.zeros(X.shape[1])
        ret = lrCostFunction(theta, X, y, 1)
        #print(ret)
        from ex3_neural_network.lrCostFunction import compute_grad
        grad = compute_grad(theta, X, y)
        print(grad)
