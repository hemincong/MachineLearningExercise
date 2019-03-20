#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

from ex3_neural_network.displayData import plot_100_image, plot_an_image

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
        m, n = X.shape

        weight_data = sio.loadmat(weight_file)
        Theta1, Theta2 = weight_data['Theta1'], weight_data['Theta2']
        pred = predict(Theta1, Theta2, X)
        y = data.get('y').reshape(-1)
        radio = np.mean((pred == y))
        self.assertGreater(radio, 0.975)
        print("Training set accuracy: {p}".format(p=radio))

        # check incorrect
        n_correct = 0
        incorrect_indices = []
        for irow in range(m):
            if predict(Theta1, Theta2, X[irow]) == int(y[irow]):
                n_correct += 1
            else:
                incorrect_indices.append(irow)

        rp = np.random.permutation(incorrect_indices)

        for i in rp:
            # Display
            print('Displaying Example Image')
            pred = predict(Theta1, Theta2, X[i])
            plot_an_image(X[i])
            plt.show()

            print('Neural Network Prediction: {:d} (digit {:d})'.format(pred[0], y[i]))
