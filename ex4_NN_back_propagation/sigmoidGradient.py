#!/usr/bin/env python

import numpy as np


def sigmoidGradient(z):
    g = 1 / (1 + np.exp(-z))
    return g * (1 - g)
