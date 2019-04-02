#!/usr/bin/env python
# -*- coding: utf-8 -*-


def sigmoid(z):
    from numpy import exp
    return 1 / (1 + exp(-z))
