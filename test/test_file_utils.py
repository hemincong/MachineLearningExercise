#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest

data_file_path = "resource/ex1data2.txt"


class test_ex1_linear_regression(unittest.TestCase):

    def test_read_csv(self):
        from utils.file_utils import read_csv
        import numpy
        m = read_csv(data_file_path)
        m2 = numpy.asarray(m)
        self.assertGreater(len(m), 0)
        self.assertIsNotNone(m2)

    def test_read_csv_split_last_col(self):
        from utils.file_utils import read_csv_split_last_col
        x1, y1 = read_csv_split_last_col(data_file_path)
        import numpy as np
        data = np.loadtxt(data_file_path, delimiter=',')
        x2 = data[:, :2]
        y2 = data[:, 2]
        self.assertTrue(np.array_equal(x1, x2))
        self.assertTrue(np.array_equal(y1, y2))
