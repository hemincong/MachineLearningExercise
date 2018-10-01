#!/usr/bin/env python
# -*- coding: utf-8 -*-


def read_csv(file_name):
    m = []
    with open(file_name, "r") as infile:
        for line in infile:
            pos = line.strip().split(',')
            tmp = list(map(float, pos))
            m.append(tmp)
    return m


def read_csv_split_last_col(file_name):
    m = []
    with open(file_name, "r") as infile:
        for line in infile:
            pos = line.strip().split(',')
            tmp = list(map(float, pos))
            m.append(tmp)

    import numpy as np
    mm = np.asarray(m)
    row, col = np.shape(mm)
    x = mm[0:row, 0:col - 1]
    y = mm[0:row, col - 1:col]
    return x, y
