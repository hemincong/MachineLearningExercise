#!/usr/bin/env python

import scipy.io
import numpy as np
import unittest

# Setup the parameters you will use for this exercise
input_layer_size = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25  # 25 hidden units
num_labels = 10  # 10 labels, from 1 to 10
# (note that we have mapped "0" to label 10)


data_file = "resource/ex4data1.mat"
weight_file = "resource/ex4weights.mat"


class test_ex4_nn_back_propagation(unittest.TestCase):

    # =========== Part 1: Loading and Visualizing Data =============
    #  We start the exercise by first loading and visualizing the dataset.
    #  You will be working with a dataset that contains handwritten digits.
    #
    def test_displayData(self):
        import utils.displayData as dd
        # Load Training Data
        print('Loading and Visualizing Data ...')
        mat = scipy.io.loadmat(data_file)
        X = mat["X"]
        m = X.shape[0]

        # Randomly select 100 data points to display
        rand_indices = np.random.permutation(m)
        sel = X[rand_indices[:100], :]

        dd.displayData(sel)

    def test_nnCostFunction(self):
        # ================ Part 2: Loading Parameters ================
        # In this part of the exercise, we load some pre-initialized
        # neural network parameters.
        print('Loading Saved Neural Network Parameters ...')

        # Load the weights into variables Theta1 and Theta2
        weight = scipy.io.loadmat(weight_file)

        # Unroll parameters
        Theta1 = weight["Theta1"]
        Theta2 = weight["Theta2"]
        self.assertIsNotNone(Theta1)
        self.assertIsNotNone(Theta2)

        # Unroll parameters
