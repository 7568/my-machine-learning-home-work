# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description:
"""
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from my_first_net import FirstNet, BasicModule
from data_iter import IrisDataIter
import random


def init_data(path):
    """
    初始化数据：
    1 过滤掉 Iris-virginica 对应的数据
    2 将x加一列全为1的列，将w*x+b变成beta*x
    :param path:
    :return:
    """
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})
    # data = [d for d in data if d[4] != 2]
    # data = data[data[:, 4] != 2]
    x, y = np.split(data, (4,), axis=1)
    x = x[:, :4]
    y = np.array(y[:, 0], dtype=int)
    y = np.eye(y.max() + 1)[y]

    # x = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    return x, y


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


def train(x, y):
    """
    开始训练
    :param x:
    :param y:
    :return:
    """

    num = len(x)
    val_set_indexs = np.random.choice(range(num), int(num / 7), replace=False)
    train_set_indexs = [k for k in range(num) if k not in val_set_indexs]
    _count = 1
    epoch = 5000
    bach = 30
    L_R = 0.001
    first_net = FirstNet(L_R)
    for i in range(epoch):
        random.shuffle(train_set_indexs)
        for j in IrisDataIter(train_set_indexs, bach):
            x_train = x[j]
            y_train = y[j]
            first_net.forward(x_train.T, y_train.T)
            first_net.backward(y_train.T)

        y_hat, loss = first_net.forward(x[val_set_indexs].T, y[val_set_indexs].T)
        y_test = np.argmax(y[val_set_indexs], axis=1)
        y_hat = np.argmax(y_hat.T, axis=1)
        test_accuracy = accuracy_score(y_hat, y_test)

        y_hat, loss = first_net.forward(x[train_set_indexs].T, y[train_set_indexs].T)
        y_test = np.argmax(y[train_set_indexs], axis=1)
        y_hat = np.argmax(y_hat.T, axis=1)
        train_accuracy = accuracy_score(y_hat, y_test)
        print(f'epich : {i} , 精度 in test ： {test_accuracy} , 精度 in train : {train_accuracy}')


if __name__ == '__main__':
    path = '../iris.data'  # 数据文件路径
    x, y = init_data(path)
    train(x, y)
