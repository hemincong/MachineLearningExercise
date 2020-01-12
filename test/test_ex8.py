#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

import matplotlib.pyplot as plt
import scipy.io


#  Machine Learning Online Class
#  Exercise 8 | Anomaly Detection and Collaborative Filtering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     estimateGaussian.m
#     selectThreshold.m
#     cofiCostFunc.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#
class test_ex8(unittest.TestCase):
    # Initialization
    @classmethod
    def setUp(cls):
        #  Load Training Data
        #  The following command loads the dataset. You should now have the
        #  variables X, Xval, yval in your environment
        data_file = "resource/ex8data1.mat"
        mat = scipy.io.loadmat(data_file)
        cls.X = mat["X"]
        cls.yVal = mat["yval"]
        cls.xVal = mat["Xval"]

    #  ================== Part 1: Load Example Dataset  ===================
    #  We start this exercise by using a small dataset that is easy to
    #  visualize.
    #
    #  Our example case consists of 2 network server statistics across
    #  several machines: the latency and throughput of each machine.
    #  This exercise will help us find possibly faulty (or very fast) machines.
    #
    def test_Load_Example_Dataset(self):
        #  Visualize the example dataset
        print('Visualizing example dataset for outlier detection.')
        plt.plot(self.X[:, 0], self.X[:, 1], 'bx')
        plt.ylim([0, 30])
        plt.xlim([0, 30])
        plt.xlabel('Latency (ms)')
        plt.ylabel('Throughput (ms/s)')
        plt.show(block=True)

    #  ================== Part 2: Estimate the dataset statistics ===================
    #  For this exercise, we assume a Gaussian distribution for the dataset.
    #
    #  We first estimate the parameters of our assumed Gaussian distribution,
    #  then compute the probabilities for each of the points and then visualize
    #  both the overall distribution and where each of the points falls in
    #  terms of that distribution.
    #
    def test_Estimate_the_dataset_statistics(self):
        print('Visualizing Gaussian fit.')

        from ex8_Anomaly_Detection_and_Recommender_Systems.estimateGaussian import estimateGaussian
        mu, sigma2 = estimateGaussian(self.X)

        self.assertAlmostEqual(mu[0], 14.1122578, delta=0.001)
        self.assertAlmostEqual(mu[1], 14.99771051, delta=0.001)
        self.assertAlmostEqual(sigma2[0], 1.83263141, delta=0.001)
        self.assertAlmostEqual(sigma2[1], 1.70974533, delta=0.001)

        from ex8_Anomaly_Detection_and_Recommender_Systems.visualizeFit import visualizeFit
        visualizeFit(self.X, mu, sigma2)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Throughput (mb/s)')
        plt.show(block=False)

    #  ================== Part 3: Find Outliers ===================
    #  Now you will find a good epsilon threshold using a cross-validation set
    #  probabilities given the estimated Gaussian distribution
    #
    def test_Find_Outliers(self):
        from ex8_Anomaly_Detection_and_Recommender_Systems.estimateGaussian import estimateGaussian
        mu, sigma2 = estimateGaussian(self.X)
        from ex8_Anomaly_Detection_and_Recommender_Systems.multivariateGaussian import multivariateGaussian
        p = multivariateGaussian(self.X, mu, sigma2)
        pval = multivariateGaussian(self.xVal, mu, sigma2)
        from ex8_Anomaly_Detection_and_Recommender_Systems.selectThreshold import selectThreshold
        epsilon, F1 = selectThreshold(self.yVal, pval)
        print("Best epsilon found using cross-validation: {epsilon}".format(epsilon=epsilon))
        print("Best F1 on Cross Validation Set:  {F1}".format(F1=F1))
        print("   (you should see a value epsilon of about 8.99e-05)")
        self.assertAlmostEqual(epsilon, 8.99e-05, delta=0.00001)
        outliers = p < epsilon

        #  Draw a red circle around those outliers
        from ex8_Anomaly_Detection_and_Recommender_Systems.visualizeFit import visualizeFit
        visualizeFit(self.X, mu, sigma2)
        plt.plot(self.X[outliers, 0], self.X[outliers, 1], 'ro', linewidth=2, markersize=18, fillstyle='none',
                 markeredgewidth=1)
        plt.xlabel('Latency (ms)')
        plt.ylabel('Throughput (mb/s)')
        plt.show(block=False)

    #  ================== Part 4: Multidimensional Outliers ===================
    #  We will now use the code from the previous part and apply it to a
    #  harder problem in which more features describe each datapoint and only
    #  some features indicate whether a point is an outlier.
    #
    def test_Multidimensional_Outliers(self):
        #  Loads the second dataset. You should now have the
        #  variables X, Xval, yval in your environment
        data_file = "resource/ex8data2.mat"
        mat2 = scipy.io.loadmat(data_file)
        X_2 = mat2["X"]
        yVal_2 = mat2["yval"]
        xVal_2 = mat2["Xval"]

        #  Apply the same steps to the larger dataset
        from ex8_Anomaly_Detection_and_Recommender_Systems.estimateGaussian import estimateGaussian
        mu, sigma2 = estimateGaussian(X_2)

        #  Training set
        from ex8_Anomaly_Detection_and_Recommender_Systems.multivariateGaussian import multivariateGaussian
        p = multivariateGaussian(X_2, mu, sigma2);

        #  Cross-validation set
        pval = multivariateGaussian(xVal_2, mu, sigma2)

        from ex8_Anomaly_Detection_and_Recommender_Systems.selectThreshold import selectThreshold
        #  Find the best threshold
        epsilon, F1 = selectThreshold(yVal_2, pval)

        outliers = p < epsilon
        outliers_count = sum(outliers)

        print("Best epsilon found using cross-validation: {epsilon}".format(epsilon=epsilon))
        print("Best F1 on Cross Validation Set:  {F1}".format(F1=F1))
        print("# Outliers found: {outliers}".format(outliers=outliers_count))
        print("    (you should see a value epsilon of about 1.38e-18)")
        self.assertAlmostEqual(epsilon, 1.38e-18)


if __name__ == '__main__':
    unittest.main()
