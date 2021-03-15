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
import math


def f(x, y, a, b):
    """
    损失函数
    y:样本数据对应的函数
    a,b都是与x，y长度一样的数组
    """
    l = (1 / 2) * 1 / (1 + np.power(math.e, -(y - a * x - b) ** 2))
    # print('l = ', l)
    return l


def loss_d_f_a(x, y, a, b):
    """
    损失函数对a求倒
    """
    return f(x, y, a, b) * (1 - f(x, y, a, b)) * 2 * (y - a * x - b) * (-x)


def loss_d_f_b(x, y, a, b):
    """
    损失函数对b求倒
    """
    return f(x, y, a, b) * (1 - f(x, y, a, b)) * 2 * (y - a * x - b) * (-1)


def loss_d_f(x, y, a, b):
    """
    损失函数对a，b的倒数
    loss=1/2 sigma(y-y')^2
    dl/da = loss*(1-loss)*2*(y-y')*x
    dl/da = loss*(1-loss)*2*(y-y')
    """
    l_d = np.array([np.sum(loss_d_f_a(x, y, a, b)), np.sum(loss_d_f_b(x, y, a, b))])
    # print('l_d = ', l_d)
    return l_d


# 产生原始数据，并添加干扰
def create_data():
    d = [np.arange(-100, 100, 2), 3.0 * np.arange(-100, 100, 2) + 4.0 + (np.random.sample(100)*2-1)*30]
    return d


xs = create_data()

print(xs)
plt.scatter(xs[0], xs[1],s=2)

learning_rate = 0.00001
max_loop = 5000
tolerance = 0.001
a_init = 2.0
b_init = 15.0

fx_pre = 0
for i in range(max_loop):
    a_b = [np.repeat(a_init, 100), np.repeat(b_init, 100)]
    d_f_x = loss_d_f(xs[0], xs[1], a_b[0], a_b[1])
    # print('d_f_x = ', d_f_x)
    # d_f_x = d_f_2(f,x)
    [a_init, b_init] = [a_init, b_init] - learning_rate * d_f_x
    # print([a_init, b_init])
    # a_b = [np.repeat(a_init, 100), np.repeat(b_init, 100)]
    # fx_cur = np.sum(f(xs[0], xs[1], a_b[0], a_b[1]))
    # if abs(fx_cur - fx_pre) < tolerance:
        # break
    # fx_pre = fx_cur
print('initial a = ', a_init, 'b = ', b_init)
# print('arg min loss of a_b = ', a_b)
a_b = [np.repeat(a_init, 100), np.repeat(b_init, 100)]
print('loss = ', np.sum(f(xs[0], xs[1], a_b[0], a_b[1])))
plt.plot(xs[0], a_init*xs[0]+b_init)
plt.show()
