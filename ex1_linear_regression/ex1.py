#!/usr/bin/env python
# -*- coding: utf-8 -*-

x = []
y = []
alpha = 0.01

with open("ex1data1.txt", "r") as infile:
    for line in infile:
        pos = line.strip().split(',')
        x.append(float(pos[0]))
        y.append(float(pos[1]))

import numpy as np

#print ("x:{x}, min:{min_x}, max:{max_x}".format(x=x, min_x=min(x), max_x=max(x)))
#print ("y:{y}, min:{min_y}, max:{max_y}".format(y=y, min_y=min(y), max_y=max(y)))

def compute_cost(list_of_x, list_of_real_y, theta_0, theta_1):
    m = len(list_of_x)
    total = 0.0
    for i in range(m):
        temp = list_of_x[i] * theta_1 + theta_0 - list_of_real_y[i]
        s_temp = temp * temp
        total += temp * temp
    return (total / 2 / m)

f_0 = lambda x, theta_0, theta_1, read_y: theta_0 + theta_1 * x - read_y
f_1 = lambda x, theta_0, theta_1, read_y: (theta_0 + theta_1 * x - read_y) * x

def gcd_line(list_of_x, list_of_real_y, theta_0_p, theta_1_p):
    total_0 = theta_0_p
    total_1 = theta_1_p
    theta_0 = theta_0_p
    theta_1 = theta_1_p

    while abs(total_0) > 0.01 or abs(total_1) > 0.01 :
        total_0 = 0.0
        total_1 = 0.0
        m = len(list_of_x)
        for i in range(m):
            total_0 += f_0(list_of_x[i], theta_0, theta_1, list_of_real_y[i]) 
            total_1 += f_1(list_of_x[i], theta_0, theta_1, list_of_real_y[i])
        total_0 = total_0 / m 
        total_1 = total_1 / m 
        theta_0 = theta_0 - alpha * total_0
        theta_1 = theta_1 - alpha * total_1
        cost = compute_cost(list_of_x, list_of_real_y, theta_0, theta_1)

        #print("total_0: {total_0}, total_1: {total_1}".format(total_0=total_0, total_1=total_1))
        #print("theta_0: {t0}, theta_1: {t1}, cost: {cost}".format(t0=theta_0, t1=theta_1, cost = cost))

    cost_min = compute_cost(list_of_x, list_of_real_y, theta_0, theta_1)
    print("theta_0: {t0}, theta_1: {t1}, cost: {cost}".format(t0=theta_0, t1=theta_1, cost = cost_min))
    return theta_0, theta_1

def drew_1():
    import matplotlib.pyplot as plt
    theta_0, theta_1 = gcd_line(x, y, 100, 100)

    point_start_y = theta_0 + min(x) * theta_1
    point_end_y = theta_0 + max(x) * theta_1
    #print("point_end_y: {max_y}".format(max_y = point_end_y))

    f1 = plt.figure(1)
    plt.title('Linear regression With GCD')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.scatter(x, y, marker = 'o', color = 'k', s = 10, label='point')
    plt.legend(loc='lower right')
    plt.plot([min(x), max(x)], [point_start_y, point_end_y])
    plt.show()

def drew_J_theta():
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator, FormatStrFormatter

    theta_0_list = np.arange(-10, 10, 0.1)
    theta_1_list = np.arange(-1, 4, 0.1)
    size_t0 = len(theta_0_list)
    size_t1 = len(theta_1_list)

    cost = np.zeros((size_t0, size_t1))
    for i in range(len(theta_0_list)):
        for j in range(len(theta_1_list)):
            cost[i, j] = compute_cost(x, y, theta_0_list[i], theta_1_list[j])
            print("theta_0: {t0}, theta_1: {t1}, cost: {c}".format(t0=theta_0_list[i], t1=theta_1_list[j], c=cost[i, j]))
        temp = []

    X = np.asarray(theta_0_list)
    Y = np.asarray(theta_1_list)
    X, Y = np.meshgrid(X, Y)
    Z = np.asmatrix(cost)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, np.transpose(Z), cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 300)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

drew_J_theta()
drew_1()
