#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


class test_data_normal(unittest.TestCase):

    def test_normal(self):
        from ex1_linear_regression.normal import norm_list
        normed_list = norm_list([1, 100, 101])
        filter_great_than = list(filter(lambda x: abs(x) >= 1.0, normed_list))
        self.assertEqual(0, len(filter_great_than))

    def test_normal_matrix(self):
        from utils.file_utils import read_csv
        from ex1_linear_regression.normal import norm_matrix
        import numpy
        m = read_csv("resource/ex1data2.txt")
        m2 = numpy.asarray(m)
        normed_matrix = norm_matrix(m2)
        self.assertIsNotNone(normed_matrix)


class test_line_regression(unittest.TestCase):

    def test_ex1_drew1(self):
        from ex1_linear_regression.ex1 import drew_1
        drew_1()

    def test_ex1_drew_j_theta(self):
        from ex1_linear_regression.ex1 import drew_J_theta
        drew_J_theta()

    def test_gcd_multi_vars_1(self):
        from ex1_linear_regression.file_utils import read_csv
        import numpy
        m = read_csv("resource/ex1data1.txt")
        m2 = numpy.asarray(m)
        self.assertIsNotNone(m2)
        from ex1_linear_regression.multi_vars import gcd_m
        alpha = 0.01
        ret = gcd_m(m2, alpha)
        self.assertAlmostEqual(ret[0], -3.84, delta=alpha)
        self.assertAlmostEqual(ret[1], 1.18, delta=alpha)

    def test_gcd_multi_vars_2(self):
        from utils.file_utils import read_csv
        from ex1_linear_regression.normal import norm_matrix
        import numpy
        m = read_csv("test/resource/ex1data2.txt")
        m2 = numpy.asarray(m)
        normed_matrix = norm_matrix(m2)
        self.assertIsNotNone(normed_matrix)
        from ex1_linear_regression.multi_vars import gcd_m
        ret = gcd_m(normed_matrix, 0.001)
        self.assertAlmostEqual(ret[0], 89597, delta=1)
        self.assertAlmostEqual(ret[1], 139, delta=1)
        self.assertAlmostEqual(ret[2], -8738, delta=1)

    def test_normal_eq_1(self):
        from utils.file_utils import read_csv
        from ex1_linear_regression.normalEqn import normal_eq_n
        import numpy
        m = read_csv("test/resource/ex1data1.txt")
        m2 = numpy.asarray(m)
        ret = normal_eq_n(m2)
        alpha = 0.1
        self.assertAlmostEqual(ret[0], -3.84, delta=alpha)
        self.assertAlmostEqual(ret[1], 1.18, delta=alpha)

    def test_normal_eq_2(self):
        from utils.file_utils import read_csv
        from ex1_linear_regression.normalEqn import normal_eq_n
        import numpy
        m = read_csv("test/resource/ex1data2.txt")
        m2 = numpy.asarray(m)
        ret = normal_eq_n(m2)
        self.assertAlmostEqual(ret[0], 89597, delta=1)
        self.assertAlmostEqual(ret[1], 139, delta=1)
        self.assertAlmostEqual(ret[2], -8738, delta=1)
