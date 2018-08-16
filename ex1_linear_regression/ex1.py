#!/usr/bin/env python
# -*- coding: utf-8 -*-

x = []
y = []

with open("ex1data1.txt", "r") as infile:
    for line in infile:
        pos = line.strip().split(',')
        x.append(float(pos[0]))
        y.append(float(pos[1]))

import matplotlib.pyplot as plt
import numpy as np

print ("x:{x}, min:{min_x}, max:{max_x}".format(x=x, min_x=min(x), max_x=max(x)))
print ("y:{y}, min:{min_y}, max:{max_y}".format(y=y, min_y=min(y), max_y=max(y)))

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

alpha = 0.01

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
        print("total_0: {total_0}, total_1: {total_1}".format(total_0=total_0, total_1=total_1))
        print("theta_0: {t0}, theta_1: {t1}, cost: {cost}".format(t0=theta_0, t1=theta_1, cost = cost))

    return theta_0, theta_1

theta_0, theta_1 = gcd_line(x, y, 100, 100)

point_start_y = theta_0 + min(x) * theta_1
point_end_y = theta_0 + max(x) * theta_1
print("point_end_y: {max_y}".format(max_y = point_end_y))

f1 = plt.figure(1)
plt.title('Linear regression With GCD')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(x, y, marker = 'o', color = 'k', s = 10, label='point')
plt.legend(loc='lower right')
plt.plot([min(x), max(x)], [point_start_y, point_end_y])
plt.show()

