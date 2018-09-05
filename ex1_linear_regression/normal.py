#!/usr/bin/env python
# -*- coding: utf-8 -*-


def norm_list(l):
    assert (len(l) != 0)
    ave = sum(l) / len(l)
    range_lst = max(l) - min(l)
    return list(map(lambda x: (x - ave) / range_lst, l))


def norm_matrix(m):
    import numpy as np
    row, col = np.shape(m)

    new_m = []
    for c in range(1, col, 1):
        new_m.append(norm_list(list(m[1:row, c-1: c].flat)))
    return np.asarray(new_m)
