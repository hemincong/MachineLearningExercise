#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import numpy as np
import scipy.io


#  Exercise 6 | Support Vector Machines
#
class test_ex5_svm(unittest.TestCase):

    @classmethod
    def setUp(cls):
        # Load Training Data
        print('Loading and Visualizing Data ...')
        data_file = "resource/ex6data1.mat"

        # Load
        # You will have X, y, Xval, yval, Xtest, ytest in your environment
        mat = scipy.io.loadmat(data_file)

        cls.X = mat["X"]
        cls.y = mat["y"]
        cls.m = np.shape(cls.X)[0]

    # =============== Part 1: Loading and Visualizing Data ================
    #  We start the exercise by first loading and visualizing the dataset.
    #  The following code will load the dataset into your environment and plot
    #  the data.
    #
    def test_load_and_visualzing_data(self):
        import matplotlib.pyplot as plt

        y = self.y.flatten()
        pos = y == 1
        neg = y == 0
        print(self.X[:, 1][pos])

        plt.figure(1)
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')
        plt.plot(self.X[:, 0][pos], self.X[:, 1][pos], 'k+', markersize=10)
        plt.plot(self.X[:, 0][neg], self.X[:, 1][neg], 'yo', markersize=10)
        plt.show(block=False)

        # Plot training data
        print('Program paused. Press enter to continue.')

    # ==================== Part 2: Training Linear SVM ====================
    #  The following code will train a linear SVM on the dataset and plot the
    #  decision boundary learned.
    #

    # Load from ex6data1:
    # You will have X, y in your environment
    def test_train_liner_svm(self):
        C = 1
        from ex6_SVM.svmTrain import svmTrain
        model = svmTrain(self.X, self.y, C, "linear", 1e-3, 20)

        import matplotlib.pyplot as plt
        plt.close()
        from ex6_SVM.visualizeBoundaryLinear import visualizeBoundaryLinear
        visualizeBoundaryLinear(self.X, self.y, model)

    # =============== Part 3: Implementing Gaussian Kernel ===============
    #  You will now implement the Gaussian kernel to use
    #  with the SVM. You should complete the code in gaussianKernel.m
    #
    def test_gaussian_kernel(self):
        print("Evaluating the Gaussian Kernel ...")

        x1 = np.asarray([1, 2, 1])
        x2 = np.asarray([0, 4, -1])
        sigma = 2
        from ex6_SVM.gaussianKernel import gaussianKernel
        sim = gaussianKernel(x1, x2, sigma)
        print(sim)

        print("Gaussian Kernel between x1 = [1; 2; 1], x2 = [0; 4; -1], sigma = 0.5 :{sim}".format(sim=sim))
        print("this value should be about 0.324652")

        self.assertAlmostEqual(sim, 0.324652, delta=0.00001)
