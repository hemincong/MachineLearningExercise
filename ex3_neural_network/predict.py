#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def predict(Theta1, Theta2, X):
    from utils.sigmoid import sigmoid
    X = np.insert(X, 0, 1, axis=1)
    a2 = sigmoid(np.dot(X, Theta1.T))
    a2 = np.insert(a2, 0, 1, axis=1)
    a3 = sigmoid(np.dot(a2, Theta2.T))
    p = a3.argmax(axis=1) + 1

    return p
