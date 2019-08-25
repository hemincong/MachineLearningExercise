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
        data_file_2 = "resource/ex6data2.mat"
        data_file_3 = "resource/ex6data3.mat"

        # Load
        # You will have X, y, Xval, yval, Xtest, ytest in your environment
        mat = scipy.io.loadmat(data_file)
        mat_2 = scipy.io.loadmat(data_file_2)
        mat_3 = scipy.io.loadmat(data_file_3)

        cls.X = mat["X"]
        cls.y = mat["y"]
        cls.m = np.shape(cls.X)[0]

        cls.X_2 = mat_2["X"]
        cls.y_2 = mat_2["y"]
        cls.m_2 = np.shape(cls.X_2)[0]

        cls.X_3 = mat_3["X"]
        cls.y_3 = mat_3["y"]
        cls.m_3 = np.shape(cls.X_3)[0]

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

    # =============== Part 4: Visualizing Dataset 2 ================
    #  The following code will load the next dataset into your environment and
    #  plot the data.
    #
    def test_load_and_visualizing_data_2(self):
        print('Loading and Visualizing Data ...')

        # Load from ex6data2:
        # You will have X, y in your environment

        y = self.y_2.flatten()
        pos = y == 1
        neg = y == 0

        import matplotlib.pyplot as plt
        # Plot training data
        plt.close()
        plt.figure(1)
        plt.xlabel('Change in water level (x)')
        plt.ylabel('Water flowing out of the dam (y)')
        plt.plot(self.X_2[:, 0][pos], self.X_2[:, 1][pos], 'k+', markersize=10)
        plt.plot(self.X_2[:, 0][neg], self.X_2[:, 1][neg], 'yo', markersize=10)
        plt.show(block=False)

    # ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
    #  After you have implemented the kernel, we can now use it to train the
    #  SVM classifier.
    #
    def test_training_svm_with_rbf_kernel(self):
        # Load from ex6data2:
        # You will have X, y in your environment

        # SVM Parameters
        C = 1
        sigma = 0.1

        # We set the tolerance and max_passes lower here so that the code will run
        # faster.However, in practice, you will want to run the training to
        # convergence.
        from ex6_SVM.svmTrain import svmTrain
        model = svmTrain(self.X_2, self.y_2, C, "gaussian")
        import matplotlib.pyplot as plt
        plt.close()
        from ex6_SVM.visualizeBoundary import visualizeBoundary
        visualizeBoundary(self.X_2, self.y_2, model)

    # =============== Part 6: Visualizing Dataset 3 ================
    #  The following code will load the next dataset into your environment and
    #  plot the data.
    #
    def test_load_and_visualizing_data_3(self):
        # Load from ex6data2:
        # You will have X, y in your environment

        # Plot training data
        from ex6_SVM.plotData import plotData
        plotData(self.X_3, self.y_3)

    # ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

    # This is a different dataset that you can use to experiment with.Try
    # different values of C and sigma here.
    #
    def test_training_svm_with_rbf_kernel_data_3(self):
        # Load from ex6data2:
        # You will have X, y in your environment

        # SVM Parameters
        C = 1
        sigma = 0.1

        # We set the tolerance and max_passes lower here so that the code will run
        # faster.However, in practice, you will want to run the training to
        # convergence.
        from ex6_SVM.svmTrain import svmTrain
        model = svmTrain(self.X_3, self.y_3, C, "gaussian")
        import matplotlib.pyplot as plt
        plt.close()
        from ex6_SVM.visualizeBoundary import visualizeBoundary
        visualizeBoundary(self.X_3, self.y_3, model)
