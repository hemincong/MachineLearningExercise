#!/usr/bin/env python
# -*- coding: utf-8 -*-


def predict(theta, X):
    import numpy as np
    from ex2_logistic_regression.sigmoid import sigmoid
    sigValue = sigmoid(np.dot(X, theta))
    p = sigValue >= 0.5

    return p
