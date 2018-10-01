#!/usr/bin/env python
# -*- coding: utf-8 -*-

if __name__ == "__main__":
    from utils.file_utils import read_csv

    data = read_csv("ex2_logistic_regression/test/resource/ex2data1.txt")

    import numpy as np

    m = np.asarray(data)
    row, col = m.shape
    x = m[0:row, 0]
    y = m[0:row, 1]
    z = m[0:row, 2]

    x0 = []
    x1 = []
    y0 = []
    y1 = []
    for i in range(row):
        if z[i] == 0:
            x0.append(x[i])
            y0.append(y[i])
        else:
            x1.append(x[i])
            y1.append(y[i])

    import matplotlib.pyplot as plt

    plt.figure(1)
    plt.title('Scatter plot of training data')
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.scatter(x0, y0, marker='o', color='y', s=10, label='Admitted')
    plt.scatter(x1, y1, marker='X', color='b', s=10, label='Not admitted')
    plt.legend(loc='upper right')
    plt.show()
