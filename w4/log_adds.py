# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/30
Description:
1.完成书上公式3.10和3.27的推导。

2.用梯度下降的方法实现对数几率回归二分类算法，数据采用iris，数据只取前两个类别，并如上次实验一样进行模型评估（不需要ROC曲线）。
"""
import math

import numpy as np
from sklearn.model_selection import KFold


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
    data = data[data[:, 4] != 2]
    x, y = np.split(data, (4,), axis=1)
    x = x[:, :4]
    y = y[:, 0]
    init_beta = np.random.normal(0, 1, 5)
    x_hat = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    # init_beta = np.array([0.7, -2.0, 1.0, -0.2, 0.2])
    print('init_beta : ', init_beta)
    return x_hat, y, init_beta


def loss(x_hat, y, beta):
    """
    损失函数,根据公式3.27
    y:样本数据对应的函数
    """
    x_hat = x_hat.T  # 为了跟书上的公式一致
    y = y.T
    beta = beta.T
    l = -y @ (beta.T @ x_hat).T + np.sum(np.log(1 + np.exp(beta.T @ x_hat)).T)
    return l


def loss_d_f_beta(x_hat, y, beta):
    """
    损失函数对beta求导
    """
    x_hat = x_hat.T  # 为了跟书上的公式一致
    y = y.T
    beta = beta.T
    l = -x_hat @ (y - np.exp(beta.T @ x_hat) / (1 + np.exp(beta.T @ x_hat)))
    return l


def get_beta(x_hat, y, init_beta):
    """
    通过梯度下降法，计算参数
    :param x_hat:
    :param y:
    :param init_beta:
    :return:
    """
    beta = init_beta
    learning_rate = [0.00001, 0.00001, 0.00001, 0.00001, 0.00001]  # 学习率
    max_loop = 5000  # 最大迭代次数
    tolerance = 0.001  # 容忍度
    fx_pre = 0  # 临时变量，用于存放上一次迭代的损失
    for i in range(max_loop):
        d_f_x = loss_d_f_beta(x_hat, y, init_beta)
        beta -= learning_rate * d_f_x
        # print(beta)
        fx_cur = loss(x_hat, y, beta)
        if abs(fx_cur - fx_pre) < tolerance:
            break
        fx_pre = fx_cur
    print('beta : ', beta)
    return beta


def predict(x_test, beta):
    """
    进行预测，根据公式3.24
    :param x_test:
    :param beta:
    :return:
    """
    x_test = x_test.T
    beta = beta.T
    _y = 1 / (1 + np.exp(beta.T @ x_test))
    return np.where(_y < 0.5, 1, 0)  # l中大于0.5的为第0类，小于等于0.5的为第1类


def train(x, y, init_beta):
    """
    开始训练
    :param x:
    :param y:
    :param init_beta:
    :return:
    """
    my_n_splits = 10
    kf = KFold(n_splits=my_n_splits, shuffle=True)
    _count = 1
    all_measure_value = np.array([])
    all_confusion_matrix = np.zeros((3, 3))
    for train_index, test_index in kf.split(x):
        print(f'==========第 {_count} 折数据训练的结果如下===========')
        _count += 1
        x_train = x[train_index]
        x_test = x[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        _beta = get_beta(x_train, y_train, init_beta)
        y_hat = predict(x_test, _beta)
        _diff_y = y_hat + y_test * 2  # y_test 如果为0 y_hat 也为0的话，或者y_test 如果为1 y_hat 也为1的话就表示预测争取，否则错误
        _accuracy = len([_y for _y in _diff_y if (_y == 0 or _y == 3)]) / len(_diff_y)
        all_measure_value = np.append(all_measure_value, _accuracy)
        print('精度 ： ', _accuracy, ' , 错误率 ： ', 1 - _accuracy)
        all_measure_value = np.append(all_measure_value, 1 - _accuracy)
        tp = len([_y for _y in _diff_y if _y == 0])  # TP：被检索到第一类样本，实际也是第一类样本
        fp = len([_y for _y in _diff_y if _y == 1])  # FP：被检索到第一类样本，实际是第二类样本
        fn = len([_y for _y in _diff_y if _y == 2])  # FN：未被检索到第二类样本，实际是第一类样本
        tn = len([_y for _y in _diff_y if _y == 3])  # TN：未被检索到第二类样本，实际也是第二类样本

        # 第一类的查准率 : 第一类中被检测为第一类的数量/预测中被认为第一类的数量
        print('第一类的查准率 : ', tp / (tp + fp))
        all_measure_value = np.append(all_measure_value, tp / (tp + fp))
        # 第一类的查全率 : 第一类中被检测为第一类的数量/总共第一类的数量
        print('第一类的查全率 : ', tp / (tp + fn))
        all_measure_value = np.append(all_measure_value, tp / (tp + fn))
        print('第一类的F1 : ', 2 * tp / (len(y_test) + tp - tn))
        all_measure_value = np.append(all_measure_value, 2 * tp / (len(y_test) + tp - tn))

        print('第二类的查准率 : ', tn / (tn + fn))
        all_measure_value = np.append(all_measure_value, tn / (tn + fn))
        print('第二类的查全率 : ', tn / (tn + fp))
        all_measure_value = np.append(all_measure_value, tn / (tn + fp))
        print('第二类的F1 : ', 2 * tn / (len(y_test) + tn - tp))
        all_measure_value = np.append(all_measure_value, 2 * tn / (len(y_test) + tn - tp))
    all_measure_value = all_measure_value.reshape((10, 8))
    mean_measure_value = np.mean(all_measure_value, axis=0)

    print('=====================================\n')
    print('平均精度 ： ', mean_measure_value[0], ' , 平均错误率 ： ', mean_measure_value[1])
    print('第一类的平均查准率 : ', mean_measure_value[2])
    print('第一类的平均查全率 : ', mean_measure_value[3])
    print('第一类的平均F1 : ', mean_measure_value[4])
    print('第二类的平均查准率 : ', mean_measure_value[5])
    print('第二类的平均查全率 : ', mean_measure_value[6])
    print('第二类的平均F1 : ', mean_measure_value[7])


if __name__ == '__main__':
    path = './iris.data'  # 数据文件路径
    x, y, beta = init_data(path)
    train(x, y, beta)
