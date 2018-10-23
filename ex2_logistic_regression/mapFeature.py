#!/usr/bin/env python
# -*- coding: utf-8 -*-

def mapFeature(X1, X2):
    degree = 6
    out = []
    from math import pow
    for i in range(degree + 2):
        for j in range(i):
            out.append(pow(X1, i-j) * pow(X2, j))
    return out

