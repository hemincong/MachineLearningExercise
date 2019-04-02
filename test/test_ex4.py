#!/usr/bin/env python

# python adaptation of solved ex4.m
#
# Neural network learning
#
# depends on
#
#     displayData.py
#     sigmoidGradient.py
#     randInitializeWeights.py
#     nnCostFunction.py
#

import scipy.io
import numpy as np
import unittest
#import nnCostFunction as nncf
#import sigmoidGradient as sg
#import randInitializeWeights as riw
#import checkNNGradients as cnng
#from scipy.optimize import minimize
#import predict as pr

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  You will be working with a dataset that contains handwritten digits.
#

data_file = "resource/ex4data1.mat"

class test_ex4_nn_back_propagation(unittest.TestCase):

    def test_displayData(self):
        import utils.displayData as dd
        # Load Training Data
        print('Loading and Visualizing Data ...')
        mat = scipy.io.loadmat(data_file)
        X = mat["X"]
        y = mat["y"]
        m = X.shape[0]

        # crucial step in getting good performance!
        # changes the dimension from (m,1) to (m,)
        # otherwise the minimization isn't very effective...
        y=y.flatten()

        # Randomly select 100 data points to display
        rand_indices = np.random.permutation(m)
        sel = X[rand_indices[:100],:]

        dd.displayData(sel)

