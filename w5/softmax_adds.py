# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/07
Description:
softmax回归对iris数据全部类别进行多分类，用交叉验证法输出评估结果
"""
import math

import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
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
    # data = [d for d in data if d[4] != 2]
    # data = data[data[:, 4] != 2]
    x, y = np.split(data, (4,), axis=1)
    x = x[:, :4]
    y = np.array(y[:, 0], dtype=int)
    y = np.eye(y.max() + 1)[y]

    x_hat = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    # init_beta = np.array([0.7, -2.0, 1.0, -0.2, 0.2])
    return x_hat, y, 3


def loss(x, y, omiga):
    """
    损失函数,根据公式3.27
    y:样本数据对应的函数
    """
    N = len(x)
    x = x.T  # 为了跟书上的公式一致
    y = y.T
    omiga = omiga.T
    y_n_hat = np.exp(omiga.T @ x) / (np.ones(C) @ np.exp(omiga.T @ x))
    l_n = y * np.log(y_n_hat)
    l = (-1 / N) * np.sum(l_n)
    return l


def loss_d_f_omiga(x, y, omiga):
    """
    损失函数对omiga求导
    """
    N = len(x)
    x = x.T  # 为了跟书上的公式一致
    y = y.T
    omiga = omiga.T
    y_n_hat = np.exp(omiga.T @ x) / (np.ones(C) @ np.exp(omiga.T @ x))
    l_n = x @ (y - y_n_hat).T
    l = (-1 / N) * l_n
    return l


def get_omiga(x_hat, y):
    """
    通过梯度下降法，计算参数
    :param x_hat:
    :param y:
    :return:
    """
    omiga = np.random.normal(0, 1, (3, 5))
    # learning_rate = np.array([0.00001, 0.00001, 0.00001, 0.00001, 0.00001])  # 学习率
    learning_rate = np.ones((3, 5)) * 0.001  # 学习率
    max_loop = 50000  # 最大迭代次数
    tolerance = 0.00001  # 容忍度
    fx_pre = 0  # 临时变量，用于存放上一次迭代的损失
    for i in range(max_loop):
        d_f_x = loss_d_f_omiga(x_hat, y, omiga)
        omiga -= learning_rate * d_f_x.T
        # print(omiga)
        fx_cur = loss(x_hat, y, omiga)
        if abs(fx_cur - fx_pre) < tolerance:
            break
        fx_pre = fx_cur
    print('omiga : ', omiga)
    return omiga


def predict(x_test, omiga):
    """
    进行预测，根据公式3.24
    :param x_test:
    :param omiga:
    :return:
    """
    x_test = x_test.T
    omiga = omiga.T
    y_n_hat = np.exp(omiga.T @ x_test) / (np.ones(C) @ np.exp(omiga.T @ x_test))
    y_n_hat = np.argmax(y_n_hat, axis=0)  # 找到最大值所在的位置
    return y_n_hat


def train(x, y):
    """
    开始训练
    :param x:
    :param y:
    :return:
    """
    my_n_splits = 5
    kf = KFold(n_splits=my_n_splits, shuffle=True)
    _count = 1
    all_measure_value = np.array([])
    # all_confusion_matrix = np.zeros((3, 3))
    for train_index, test_index in kf.split(x):
        print(f'==========第 {_count} 折数据训练的结果如下===========')
        _count += 1
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        _beta = get_omiga(x_train, y_train)
        y_hat = predict(x_test, _beta)
        y_test = np.argmax(y_test, axis=1)

        _accuracy = accuracy_score(y_test, y_hat)
        all_measure_value = np.append(all_measure_value, _accuracy)
        print('精度 ： ', _accuracy, ' , 错误率 ： ', 1 - _accuracy)
        all_measure_value = np.append(all_measure_value, 1 - _accuracy)

        p1 = precision_score(y_test, y_hat, average=None)
        r1 = recall_score(y_test, y_hat, average=None)
        f1 = f1_score(y_test, y_hat, average=None)
        for i in range(3):
            # c_num = len(np.where(np.argmax(y_test, axis=1) == i))
            # 第i类的查准率 : 第i类中被检测为第i类的数量/预测中被认为第i类的数量
            print(f'第{i}类的查准率 : {p1[i]}')
            all_measure_value = np.append(all_measure_value, p1[i])
            # 第i类的查全率 : 第i类中被检测为第i类的数量/总共第i类的数量
            print(f'第{i}类的查全率 : {r1[i]}')
            all_measure_value = np.append(all_measure_value, r1[i])
            print(f'第{i}类的F1 : {f1[i]}')
            all_measure_value = np.append(all_measure_value, f1[i])

    all_measure_value = all_measure_value.reshape((5, 11))
    mean_measure_value = np.mean(all_measure_value, axis=0)

    print('=====================================\n')
    print('平均精度 ： ', mean_measure_value[0], ' , 平均错误率 ： ', mean_measure_value[1])
    print('第一类的平均查准率 : ', mean_measure_value[2])
    print('第一类的平均查全率 : ', mean_measure_value[3])
    print('第一类的平均F1 : ', mean_measure_value[4])
    print('第二类的平均查准率 : ', mean_measure_value[5])
    print('第二类的平均查全率 : ', mean_measure_value[6])
    print('第二类的平均F1 : ', mean_measure_value[7])
    print('第三类的平均查准率 : ', mean_measure_value[8])
    print('第三类的平均查全率 : ', mean_measure_value[9])
    print('第三类的平均F1 : ', mean_measure_value[10])


if __name__ == '__main__':
    path = './iris.data'  # 数据文件路径
    x, y, C = init_data(path)
    train(x, y)
