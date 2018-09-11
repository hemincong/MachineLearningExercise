#!/usr/bin/env python
# -*- coding: utf-8 -*-


def norm_list(l):
    assert (len(l) != 0)
    range_lst = max(l) - min(l)
    import numpy as np
    return list(map(lambda x: (x - np.mean(l)) / range_lst, l))


def norm_matrix(m):
    import numpy as np
    row, col = np.shape(m)

    new_m = []
    for c in range(1, col + 1, 1):
        new_m.append(norm_list(list(m[0:row, c - 1: c].flat)))
    return np.asarray(new_m).T
