#!/usr/bin/env python
# -*- coding: utf-8 -*-


def norm_list(l):
    ave = sum(l) / len(l)
    print(ave)
    range_lst = max(l) - min(l)
    print(range_lst)
    return list(map(lambda x: (x - ave) / range_lst, l))

