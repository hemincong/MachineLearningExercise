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

        # Load Training Data
        print('Loading and Visualizing Data ...')
        mat = scipy.io.loadmat(data_file)
        X = mat["X"]
        y = mat["y"]
        y = y.flatten()
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
        nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))
        self.assertIsNotNone(nn_params)

        # ================ Part 3: Compute Cost (Feedforward) ================
        #  To the neural network, you should first start by implementing the
        #  feedforward part of the neural network that returns the cost only. You
        #  should complete the code in nnCostFunction.m to return cost. After
        #  implementing the feedforward to compute the cost, you can verify that
        #  your implementation is correct by verifying that you get the same cost
        #  as us for the fixed debugging parameters.
        #
        #  We suggest implementing the feedforward cost *without* regularization
        #  first so that it will be easier for you to debug. Later, in part 4, you
        #  will get to implement the regularized cost.
        print('Feedforward Using Neural Network ...')

        # Weight regularization parameter (we set this to 0 here).
        _lambda = 0

        from ex4_NN_back_propagation.nnCostFunction import nnCostFunction
        J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

        print("Cost at parameters (loaded from ex4weights): {f}".format(f=J))
        print('(this value should be about 0.287629)')
        self.assertAlmostEqual(J, 0.287529, delta=0.001)

        # =============== Part 4: Implement Regularization ===============
        #  Once your cost function implementation is correct, you should now
        #  continue to implement the regularization with the cost.
        #
        print('Checking Cost Function (Regularization) ... \n')

        # Weight regularization parameter (we set this to 1 here).
        _lambda = 1
        J, _ = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda)

        print("Cost at parameters (loaded from ex4weights): {f}".format(f=J))
        print('(this value should be about 0.383770))')
        self.assertAlmostEqual(J, 0.383770, delta=0.001)



