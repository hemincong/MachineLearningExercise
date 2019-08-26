#!/usr/bin/env python
# -*- coding: utf-8 -*-


def getVocabList(file_path):
    vocabList = {}
    with open(file_path, "r") as fp:
        for line in fp.readlines():
            ls = line.split()
            vocabList[ls[1]] = int(ls[0])

    return vocabList
