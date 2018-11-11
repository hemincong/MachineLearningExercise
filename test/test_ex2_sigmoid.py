#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest


class test_ex2_sigmoid(unittest.TestCase):

    def test_sigmoid(self):
        from ex2_logistic_regression.sigmoid import sigmoid
        ret = sigmoid(-1)
        self.assertAlmostEqual(ret, 0.268, delta=0.01)
        ret = sigmoid(0)
        self.assertAlmostEqual(ret, 0.5, delta=0.01)
