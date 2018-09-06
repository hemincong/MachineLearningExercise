#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


class file_utils(unittest.TestCase):

    def test_upper(self):
        from file_utils import read_csv
        import numpy
        m = read_csv("test/resource/ex1data2.txt")
        m2 = numpy.asarray(m)
        self.assertGreater(len(m), 0)
        self.assertIsNotNone(m2)

    def test_normal(self):
        from normal import norm_list
        normed_list = norm_list([1, 100, 101])
        filter_great_than = list(filter(lambda x: abs(x) >= 1.0, normed_list))
        self.assertEqual(0, len(filter_great_than))

    def test_normal_martix(self):
        from file_utils import read_csv
        from normal import norm_matrix
        import numpy
        m = read_csv("test/resource/ex1data2.txt")
        m2 = numpy.asarray(m)
        normed_matrix = norm_matrix(m2)

        self.assertIsNotNone(normed_matrix)


if __name__ == '__main__':
    unittest.main()
