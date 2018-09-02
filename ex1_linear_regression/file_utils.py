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
