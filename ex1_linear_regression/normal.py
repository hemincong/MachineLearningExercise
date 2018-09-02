#!/usr/bin/env python
# -*- coding: utf-8 -*-


def norm_list(l):
    ave = sum(l) / len(l)
    range_lst = max(l) - min(l)
    return list(map(lambda x: (x - ave) / range_lst, l))

