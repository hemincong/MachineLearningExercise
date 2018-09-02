#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


class file_utils(unittest.TestCase):

    def test_upper(self):
        from file_utils import read_csv
        import numpy
        m = read_csv("../resource/ex1data2.txt")
        m2 = numpy.asmatrix(m)
        self.assertGreater(len(m), 0)
        self.assertIsNotNone(m2)

    def test_normal(self):
        from normal import norm_list
        normed_list = norm_list([1, 100, 101])
        print (normed_list)
        filter_great_than = list(filter(lambda x: abs(x) >= 1.0, normed_list))
        self.assertEqual(0, len(filter_great_than))


if __name__ == '__main__':
    unittest.main()
