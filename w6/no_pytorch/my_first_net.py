# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description:
"""
import numpy as np


class BasicModule(object):
    l_r = 0

    def __init__(self):
        raise NotImplementedError

    def forward(self, x_i):
        raise NotImplementedError

    def backward(self, grad_H):
        raise NotImplementedError


class LinearLayer(BasicModule):

    def __init__(self, input_num, output_num):
        """
         线性层层，实现 X 和 W 的矩阵乘法运算
        :param input_num: 神经元的输入参数个数，即input的属性的维度
        :param output_num: 该层神经元的个数
        """
        self.X = None
        self.W = np.random.normal(0, 1, (input_num, output_num))

    def forward(self, x_i):
        self.X = x_i
        H = np.matmul(self.W.T, self.X)
        return H

    def backward(self, grad_H):
        '''
        param {
            grad_H: shape(m, d'), Loss 关于 H 的梯度
        }
        return {
            grad_X: shape(m, d), Loss 关于 X 的梯度
            grad_W: shape(d, d'), Loss 关于 W 的梯度
        }
        '''
        grad_X = np.matmul(self.W, grad_H)
        grad_W = np.matmul(self.X, grad_H.T)
        self.W -= BasicModule.l_r * grad_W
        return grad_X  # , grad_W


class Sigmoid1(BasicModule):
    def __init__(self):
        """
        # 实现 sigmoid 激活函数
        """
        self.mem = {}

    def forward(self, X):
        self.mem["X"] = X
        s = 1 / (1 + np.exp(-X))
        return s

    def backward(self, grad_y):
        X = self.mem["X"]
        return X * (1 - X) * grad_y


class Relu():
    def __init__(self):
        self.mem = {}

    def forward(self, X):
        self.mem["X"] = X
        return np.where(X > 0, X, np.zeros_like(X))

    def backward(self, grad_y):
        X = self.mem["X"]
        return (X > 0).astype(np.float32) * grad_y


class Softmax(BasicModule):

    def __init__(self, input_num=1):
        """实现 Softmax 激活函数"""
        self.mem = {}
        self.input_num = input_num
        self.epsilon = 1e-12  # 防止求导后分母为 0

    def forward(self, p):
        p_exp = np.exp(p)
        denominator = np.sum(p_exp, axis=1, keepdims=True)
        s = p_exp / (denominator + self.epsilon)
        self.mem["s"] = s
        self.mem["p_exp"] = p_exp
        return s

    def backward(self, grad_s):
        s = self.mem["s"]
        sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        tmp = np.matmul(np.expand_dims(grad_s, axis=1), sisj)
        tmp = np.squeeze(tmp, axis=1)
        grad_p = -tmp + grad_s * s
        return grad_p


# 实现交叉熵损失函数
class CrossEntropy(BasicModule):
    def __init__(self):
        self.mem = {}
        self.epsilon = 1e-12  # 防止求导后分母为 0

    def forward(self, p, y):
        self.mem['p'] = p
        log_p = np.log(p + self.epsilon)
        return np.mean(np.sum(-y.T * log_p, axis=1))

    def backward(self, y):
        p = self.mem['p']
        return -y.T * (1 / (p + self.epsilon))


# 搭建全连接神经网络模型
class FirstNet(BasicModule):
    def __init__(self, l_r):
        BasicModule.l_r = l_r
        self.hides = [LinearLayer(5, 64),
                      Relu(),
                      LinearLayer(64, 3),
                      Softmax()]
        self.cross_en = CrossEntropy()

    def forward(self, x, labels):
        for n in self.hides:
            x = n.forward(x)
        loss = self.cross_en.forward(x, labels)
        return x, loss

    def backward(self, labels):
        loss_grad = self.cross_en.backward(labels)
        for n in reversed(self.hides):
            loss_grad = n.backward(loss_grad)
