#!/usr/bin/env python
# -*- coding: utf-8 -*-


def any_above_alpha(l, alpha):
    filter_great_than = list(filter(lambda x: abs(x) >= alpha, l))
    return len(filter_great_than) > 0


def compute_theta(params, theta_list, y):
    ret = []
    import numpy as np
    # 补1
    s = [1] + list(params)
    t = np.asarray(s) * np.asarray(theta_list).T
    h = sum(t) - y
    for i in range(0, len(theta_list)):
        x = s[i]
        ret.append(h * x)
    return ret


def compute_cost(params, theta_list, y):
    import numpy as np
    s = [1] + list(params)
    t = np.asarray(s) * np.asarray(theta_list).T
    return sum(t) - y


def gcd_m(m, alpha):
    import numpy as np
    col, row = np.shape(m)
    # print("col:{col}, row:{row}".format(col=col, row=row))

    theta_list = [100] * row

    ret = [100.0] * len(theta_list)
    while any_above_alpha(ret, alpha):
        ret = [0.0] * len(theta_list)
        # 每每项数据
        for i in range(0, col):
            y = m.item(i, -1)
            s = m[i, 0:row - 1]
            ret_temp = compute_theta(s, theta_list, y)
            # cost = compute_cost(s, theta_list, y)
            for r in range(0, row):
                ret[r] += ret_temp[r]

        for i in range(0, row):
            ret[i] = ret[i] / col
        # print(ret)

        # 必须分开两个循环,更新和计算不能交叉做
        for i in range(0, row):
            theta_list[i] = theta_list[i] - alpha * ret[i]
        # print(theta_list)
    return theta_list
