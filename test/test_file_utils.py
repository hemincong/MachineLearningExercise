#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


class test_ex1_linear_regression(unittest.TestCase):

    def test_read_csv(self):
        from utils.file_utils import read_csv
        import numpy
        m = read_csv("resource/ex1data2.txt")
        m2 = numpy.asarray(m)
        self.assertGreater(len(m), 0)
        self.assertIsNotNone(m2)