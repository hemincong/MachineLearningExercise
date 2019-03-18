#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from ex3_neural_network.displayData import plot_100_image

data_file = "resource/ex3data1.mat"
weight_file = "resource/ex3weights.mat"
input_layer_size = 400
hidden_layer_size = 25
num_labels = 10


class test_ex3_nn(unittest.TestCase):

    def test_displayData(self):
        data = sio.loadmat(data_file)
        X = data.get('X')
        X = np.array([im.reshape((20, 20)).T for im in X])
        # and I flat the image again to preserve the vector presentation
        X = np.array([im.reshape(400) for im in X])
        plot_100_image(X)
        plt.show()

    def test_predict(self):
        from ex3_neural_network.predict import predict

        data = sio.loadmat(data_file)
        X = data.get('X')

        weight_data = sio.loadmat(weight_file)
        Theta1, Theta2 = weight_data['Theta1'], weight_data['Theta2']
        pred = predict(Theta1, Theta2, X)
        y = data.get('y').reshape(-1)
        radio = np.mean((pred == y))
        print("radio : {radio}".format(radio=radio))
        self.assertGreater(radio, 0.975)

