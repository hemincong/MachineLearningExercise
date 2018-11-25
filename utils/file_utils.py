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
    m = read_csv(file_name)

    import numpy as np
    mm = np.asarray(m)
    row, col = np.shape(mm)
    x = mm[0:row, 0:col - 1]
    y = mm[0:row, col - 1:col]
    return x, y.flatten()


def read_csv_split_last_col_and_add_one(file_name):
    x, y = read_csv_split_last_col(file_name)
    import numpy as np
    x_row, x_col = np.shape(x)
    one_col = np.ones((x_row, 1))
    x = np.c_[one_col, x]
    return x, y
