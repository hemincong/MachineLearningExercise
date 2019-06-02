#!/usr/bin/env python

import numpy as np


def sigmoidGradient(z):
    return 1 / 1 + np.exp(-z)
