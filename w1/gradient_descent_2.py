# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/9
Description:参考一元的梯度下降代码，完成二元函数极值的梯度下降法
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x, y):
    return (x + y - 3) ** 2 + (x + 2 * y - 5) ** 2 + 2


def d_f_1(x, y):
    return np.array([2.0 * (x + y - 3) + 2.0 * (x + 2 * y - 5), 2.0 * (x + y - 3) + 2.0 * (x + 2 * y - 5) * 2.0])


def d_f_2(f, x, delta=1e-4):
    return (f(x + delta) - f(x - delta)) / (2 * delta)


# xs = [np.arange(-10, 11), np.arange(-10, 11)]
# plt.plot(xs, f(xs[0], xs[1]))
# plt.show()
learning_rate = 0.1
max_loop = 30
tolerance = 0.001
x_init = 0.5
y_init = 1.5
x_y = [x_init, y_init]
fx_pre = 0
for i in range(max_loop):
    d_f_x = d_f_1(x_y[0], x_y[1])
    # d_f_x = d_f_2(f,x)
    x_y = x_y - learning_rate * d_f_x
    print(x_y)
    fx_cur = f(x_y[0], x_y[1])
    # if abs(fx_cur - fx_pre) < tolerance:
    #     break
    fx_pre = fx_cur
print('initial x = ', x_init, 'y = ', y_init)
print('arg min f(x) of x = ', x_y)
print('f(x) = ', f(x_y[0], x_y[1]))
