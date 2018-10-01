#!/usr/bin/env python
# -*- coding: utf-8 -*-


def normal_eq_n(m):
    import numpy as np
    row, col = np.shape(m)

    s = m[0:row, 0:col - 1]
    y = m[0:row, col - 1:col]
    X = np.c_[np.ones(row), s]
    from numpy.linalg import inv
    return list((inv(X.T.dot(X)).dot(X.T).dot(y)).flat)
