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
