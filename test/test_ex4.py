#!/usr/bin/env python

import unittest

import numpy as np
import scipy.io

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

    def test_sigmoid_gradient(self):
        # ================ Part 5: Sigmoid Gradient ================
        # Before you start implementing the neural network, you will first
        # implement the gradient for the sigmoid function.You should complete the
        # code in the sigmoidGradient.py file.
        print('Evaluating sigmoid gradient...')

        from ex4_NN_back_propagation.sigmoidGradient import sigmoidGradient
        g = sigmoidGradient(np.asarray([1, -0.5, 0, 0.5, 1]))
        print('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]:  ')
        print(g)
        self.assertAlmostEqual(g[0], 0.196, delta=0.01)
        self.assertAlmostEqual(g[1], 0.235, delta=0.01)

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

    def test_grad_check(self):
        # CHECKNNGRADIENTS Creates a small neural network to check the
        # backpropagation gradients
        #   CHECKNNGRADIENTS(lambda_reg) Creates a small neural network to check the
        #   backpropagation gradients, it will output the analytical gradients
        #   produced by your backprop code and the numerical gradients (computed
        #   using computeNumericalGradient). These two gradient computations should
        #   result in very similar values.
        #

        def debugInitializeWeights(fan_out, fan_in):
            # DEBUGINITIALIZEWEIGHTS Initialize the weights of a layer with fan_in
            # incoming connections and fan_out outgoing connections using a fixed
            # strategy, this will help you later in debugging
            #   W = DEBUGINITIALIZEWEIGHTS(fan_in, fan_out) initializes the weights
            #   of a layer with fan_in incoming connections and fan_out outgoing
            #   connections using a fix set of values
            #
            #   Note that W should be set to a matrix of size(1 + fan_in, fan_out) as
            #   the first row of W handles the "bias" terms
            #

            # Set W to zeros
            W = np.zeros((fan_out, 1 + fan_in))

            # Initialize W using "sin", this ensures that W is always of the same
            # values and will be useful for debugging
            W = np.reshape(np.sin(range(np.size(W))), W.shape) / 10
            return W

        _input_layer_size = 3
        _hidden_layer_size = 5
        _num_labels = 3
        _m = 5
        _lambda_reg = 0

        # We generate some 'random' test data
        Theta1 = debugInitializeWeights(_hidden_layer_size, _input_layer_size)
        Theta2 = debugInitializeWeights(_num_labels, _hidden_layer_size)
        # Reusing debugInitializeWeights to generate X
        X = debugInitializeWeights(_m, _input_layer_size - 1)
        y = 1 + np.mod(range(_m), _num_labels).T

        # Unroll parameters
        nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))

        from ex4_NN_back_propagation.nnCostFunction import nnCostFunction
        # Short hand for cost function
        cost_func = lambda p: nnCostFunction(p, _input_layer_size, _hidden_layer_size, _num_labels, X, y, _lambda_reg)

        _, grad = cost_func(nn_params)

        from ex4_NN_back_propagation.computeNumericalGradient import computeNumericalGradient
        numgrad = computeNumericalGradient(cost_func, nn_params)

        print('Numerical Gradient', 'Analytical Gradient')
        for n in range(np.size(grad)):
            print("{ng} {g}".format(ng=numgrad[n], g=grad[n]))

        print('The above two columns you get should be very similar.')
        print('(Left-Your Numerical Gradient, Right-Analytical Gradient)')

        # Evaluate the norm of the difference between two solutions.
        # If you have a correct implementation, and assuming you used
        # in computeNumericalGradient.m, then diff below should be less than 1e-9
        # diff = np.norm(numgrad - grad) / np.norm(numgrad + grad)
        from decimal import Decimal
        diff = Decimal(np.linalg.norm(numgrad - grad)) / Decimal(np.linalg.norm(numgrad + grad))

        print('If your backpropagation implementation is correct, then ')
        print('the relative difference will be small (less than 1e-9).')
        print('Relative Difference: {diff}'.format(diff=diff))
        self.assertLess(diff, 1e-9)

    def test_rand_init(self):
        # ================ Part 6: Initializing Pameters ================
        #  In this part of the exercise, you will be starting to implment a two
        #  layer neural network that classifies digits. You will start by
        #  implementing a function to initialize the weights of the neural network
        #  (randInitializeWeights.m)

        print('Initializing Neural Network Parameters...')

        from ex4_NN_back_propagation.randInitializeWeights import randInitializeWeights
        initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
        initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

        # Unroll parameters
        initial_nn_params = np.concatenate((initial_Theta1.reshape(initial_Theta1.size, order='F'),
                                            initial_Theta2.reshape(initial_Theta2.size, order='F')))
        m = initial_nn_params.shape[0]
        self.assertAlmostEqual(m, (input_layer_size + 1) * hidden_layer_size + (1 + hidden_layer_size) * num_labels)
        self.assertIsNotNone(initial_nn_params)
        print(initial_nn_params)

    def test_implement_regularization(self):
        # =============== Part 8: Implement Regularization ===============
        # Once your backpropagation implementation is correct, you should now
        # continue to implement the regularization with the cost and gradient.
        #

        print('Checking Backpropagation (Regularization) ... ')
        # Check gradients by running checkNNGradients
        _lambda_reg = 3
        weight = scipy.io.loadmat(weight_file)
        Theta1 = weight["Theta1"]
        Theta2 = weight["Theta2"]

        mat = scipy.io.loadmat(data_file)
        X = mat["X"]
        y = mat["y"]

        from ex4_NN_back_propagation.nnCostFunction import nnCostFunction
        nn_params = np.concatenate((Theta1.reshape(Theta1.size, order='F'), Theta2.reshape(Theta2.size, order='F')))
        debug_J, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, _lambda_reg)

        # Also output the costFunction debugging values
        print(
            'nCost at (fixed) debugging parameters (lambda = 3): {cost} (this value should be about 0.576051)'.format(
                cost=debug_J))
        self.assertAlmostEqual(debug_J, 0.576, delta=0.001)
