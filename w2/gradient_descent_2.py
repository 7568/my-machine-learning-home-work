# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/15
Description:
1.实现一个一元线性回归算法。

要求：

1) 要拟合的函数为y = 3.0x + 4.0

2) 由这个函数生成100个数据（x：-100~100），并加适当随机噪声,所以实际y = 3.0x + 4.0 +噪声

3)假设y' = ax+b

4)loss（a,b）=1/2 sigma(y-y')^2

5)用梯度下降法求出loss最小值，a,b初值取随机值，


"""

import numpy as np
import matplotlib.pyplot as plt
import random


def loss(x, y, a, b):
    """
    损失函数
    y:样本数据对应的函数
    a,b都是与x，y长度一样的数组
    """
    l = (1 / 2) * np.sum(np.power(3.0 * x + 4 - a * x - b, 2))
    # print('l = ', l)
    return l


def loss_d_f_a(x, y, a, b):
    """
    损失函数对a求倒
    """
    return a * np.sum(np.power(x, 2)) - np.sum((y - b) * x)


def loss_d_f_b(x, y, a, b):
    """
    损失函数对b求倒
    """
    return len(x) * b - np.sum(y - a * x)


def loss_d_f(x, y, a, b):
    """
    损失函数对a，b的倒数
    loss=1/2 sigma(y-y')^2
    dl/da = loss*(1-loss)*2*(y-y')*x
    dl/da = loss*(1-loss)*2*(y-y')
    """
    l_d = np.array([loss_d_f_a(x, y, a, b) / len(x), loss_d_f_b(x, y, a, b) / len(x)])
    # print('l_d = ', l_d)
    return l_d


# 产生原始数据，并添加干扰
def create_data():
    d = [np.arange(-100, 100, 2), 3.0 * np.arange(-100, 100, 2) + 4.0 + (np.random.sample(100) * 2 - 1)]
    return d


xs = create_data()

print(len(xs))
plt.scatter(xs[0], xs[1], s=2)

learning_rate = [0.0001, 0.01]
max_loop = 50000
tolerance = 0.001
a_init = random.random()
b_init = random.random()
a_b = [a_init, b_init]
fx_pre = 0
count = 0
for i in range(max_loop):
    count += 1
    d_f_x = loss_d_f(xs[0], xs[1], a_b[0], a_b[1])
    a_b -= learning_rate * d_f_x
    print(a_b)
    fx_cur = loss(xs[0], xs[1], a_b[0], a_b[1])
    if abs(fx_cur - fx_pre) < tolerance:
        break
    fx_pre = fx_cur
print('time of loop : ', count)
print('a = ', a_b[0], 'b = ', a_b[1])
print('loss = ', loss(xs[0], xs[1], a_b[0], a_b[1]))
plt.plot(xs[0], a_b[0] * xs[0] + a_b[1])
plt.show()
