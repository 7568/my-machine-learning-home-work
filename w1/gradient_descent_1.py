# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/9
Description:参考一元的梯度下降代码，完成二元函数极值的梯度下降法
"""

import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return np.power(x, 2)


def d_f_1(x):
    return 2.0 * x


def d_f_2(f, x, delta=1e-4):
    return (f(x + delta) - f(x - delta)) / (2 * delta)


xs = np.arange(-10,11)
plt.plot(xs, f(xs))
plt.show()
learning_rate = 0.1
max_loop = 30
tolerance = 0.001
x_init = 10.0
x = x_init
fx_pre = 0
for i in range(max_loop):
    d_f_x = d_f_1(x)
    # d_f_x = d_f_2(f,x)
    x = x - learning_rate * d_f_x
    print(x)
    fx_cur = f(x)
    if abs(fx_cur - fx_pre) < tolerance:
        break
    fx_pre = fx_cur
print('initial x = ', x_init)
print('arg min f(x) of x = ', x)
print('f(x) = ', f(x))
