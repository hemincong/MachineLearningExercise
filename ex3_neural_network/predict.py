#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


def predict(Theta1, Theta2, X):
    from utils.sigmoid import sigmoid
    if X.ndim == 1:
        X = np.reshape(X, (-1, X.shape[0]))

    m = X.shape[0]

    X = np.column_stack((np.ones((m, 1)), X))
    a2 = sigmoid(np.dot(X, Theta1.T))

    a2 = np.column_stack((np.ones((m, 1)), a2))
    a3 = sigmoid(np.dot(a2, Theta2.T))

    p = a3.argmax(axis=1) + 1

    return p
