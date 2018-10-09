#!/usr/bin/env python
# -*- coding: utf-8 -*-


def predict(theta, X):
    from ex2_logistic_regression.sigmoid import sigmoid
    import numpy as np
    ret = sigmoid(np.asarray(X).dot(theta))
    return ret >= 0.5
