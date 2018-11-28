#!/usr/bin/env python
# -*- coding: utf-8 -*-


def plotDecisionBoundary(data_file_path):
    from utils.file_utils import read_csv

    data = read_csv(data_file_path)

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
    plt.title('Figure 3: Plot of traning data')
    plt.xlabel('Marcochip test 1')
    plt.ylabel('Marcochip test 2')
    plt.scatter(x0, y0, marker='o', color='y', s=10, label='y=0')
    plt.scatter(x1, y1, marker='+', color='b', s=10, label='y=1')
    plt.legend(loc='upper right')
    plt.show()

