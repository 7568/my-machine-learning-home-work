# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/31
Description:
"""

import numpy as np


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 2, b'Iris-virginica': 1}
    return it[s]


def init_data(path):
    """
    初始化数据：
    1 过滤掉 Iris-virginica 对应的数据
    2 将x加一列全为1的列，将w*x+b变成beta*x
    :param path:
    :return:
    """
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    np.random.shuffle(data)
    # data = [d for d in data if d[4] != 2]
    data = data[data[:, 4] != 2]
    x, y = np.split(data, (4,), axis=1)
    x = x[:, :4]
    y = y[:, 0]

    x_hat = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    # init_beta = np.array([0.7, -2.0, 1.0, -0.2, 0.2])
    return x_hat, y


if __name__ == '__main__':
    path = './iris.data'  # 数据文件路径
    x, y = init_data(path)

    # beta = np.array([-1, 0, 2, 0, 0])
    beta = np.array([-1.27270067 , 0.22995894 , 1.86186498 , 0.00978654 , 0.29477177])
    p_0 = 1 / (1 + np.exp(beta @ x.T))
    print(np.where(p_0 < 0.5, 1, 0))
    y = y.astype('int32')
    print(y)
