# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description:
https://blog.csdn.net/qq_44009891/article/details/110475333
http://cs231n.stanford.edu/vecDerivs.pdf
https://themaverickmeerkat.com/2019-10-23-Softmax/
https://towardsdatascience.com/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
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
        self.prev_data = None
        self.W = np.random.normal(0, 1, (input_num + 1, output_num)) / np.sqrt((input_num + 1) / 2)

    def forward(self, prev_data):
        self.prev_data = prev_data
        prev_data_new = np.concatenate((prev_data, np.ones((1, prev_data.shape[1]))), axis=0)
        H = self.W.T @ prev_data_new
        return H

    def backward(self, grad):
        new_grad = grad @ self.W.T
        # x = np.row_stack(self.prev_data,self.prev_data,self.prev_data)
        grad_w =  self.prev_data@grad
        grad_w = np.concatenate((grad_w, np.ones((1, grad_w.shape[1]))), axis=0)
        self.W -= BasicModule.l_r * grad_w
        return new_grad[:,0:-1]  # , grad_W


class Sigmoid(BasicModule):
    def __init__(self):
        """
        # 实现 sigmoid 激活函数
        """
        self.prev_data = None

    def forward(self, prev_data):
        self.prev_data = prev_data
        s = 1 / (1 + np.exp(-prev_data))
        return s

    def backward(self, grad_b):
        X = self.prev_data
        return X * (1 - X) * grad_b


class Relu(BasicModule):
    def __init__(self):
        self.prev_data = None

    def forward(self, prev_data):
        self.prev_data = prev_data
        return np.where(prev_data > 0, prev_data, np.zeros_like(prev_data))

    def backward(self, grad):
        X = self.prev_data
        return grad @ ((X > 0).astype(np.float32) * np.eye(X.shape[0]))


class Softmax(BasicModule):

    def __init__(self):
        """实现 Softmax 激活函数"""
        self.forward_output = None
        self.epsilon = 1e-12  # 防止求导后分母为 0

    def forward(self, prev_data):
        p_exp = np.exp(prev_data)
        denominator = np.sum(p_exp, axis=0, keepdims=True)
        self.forward_output = p_exp / (denominator + self.epsilon)
        return self.forward_output

    def backward(self, grad_b):
        """
        :param grad_b:
        :return:
        https://themaverickmeerkat.com/2019-10-23-Softmax/
        """
        forward_output = self.forward_output
        # n = forward_output.shape[1]
        # tensor1 = np.einsum('ij,ik->ijk', forward_output, forward_output)
        # tensor2 = np.einsum('ij,jk->ijk', forward_output, np.eye(n, n))
        # dSoftmax = tensor2 - tensor1
        # dz = np.einsum('ijk,ik->ij', dSoftmax, grad_b)
        #
        # s = self.forward_output
        # sisj = np.matmul(np.expand_dims(s, axis=2), np.expand_dims(s, axis=1))
        # tmp = np.matmul(np.expand_dims(grad_b, axis=1), sisj)
        # tmp = np.squeeze(tmp, axis=1)
        # grad_p = -tmp + grad_b * s

        # a = np.column_stack((forward_output, forward_output, forward_output))
        # b = np.eye(3, 3) - a.T
        # c = a * b
        # b = np.array([np.sum(i * forward_output, axis=0) for i in forward_output])
        # c = forward_output - b
        d_softmax = np.eye(3, 3) - forward_output @ forward_output.T
        input_grad = grad_b @ d_softmax
        # d = c * grad_b
        return input_grad


# 实现交叉熵损失函数
class CrossEntropy(BasicModule):
    def __init__(self):
        self.pre_data = None
        self.epsilon = 1e-12  # 防止求导后分母为 0

    def forward(self, prev_data, y):
        self.pre_data = prev_data
        log_p = np.log(prev_data + self.epsilon)
        return np.mean(np.sum(-y * log_p, axis=0))

    def backward(self, y):
        p = self.pre_data
        return -y * (1 / (p + self.epsilon))


# 实现均方误差损失函数
class MeanSquaredError(BasicModule):

    def __init__(self):
        self.mem = {}

    def forward(self, p, y):
        """

        :param p: 神经网络最后一层输出的结果
        :param y: 真实标签
        :return:
        """
        self.mem['p'] = p
        return (y - p) * (y - p) / 2

    def backward(self, y):
        p = self.mem['p']
        return (p - y).T  # 一个数对一个向量求导的输出是一个行向量


# 搭建全连接神经网络模型
class FirstNet(BasicModule):

    def __init__(self, l_r):
        self.pre_loss = 0
        BasicModule.l_r = l_r
        self.hides = [LinearLayer(4, 8),
                      Relu(),
                      LinearLayer(8, 3),
                      # Relu(),
                      # LinearLayer(16, 3),
                      # Relu(),
                      Softmax()]
        self.error_measure = MeanSquaredError()

    def forward(self, x, labels):
        # x2 = np.array(x, dtype=float)
        # x = (x2 - np.mean(x2, axis=1, keepdims=True)) / np.std(x2, axis=1, keepdims=True)  # 将x进行标准化操作
        for n in self.hides:
            x = n.forward(x)
        loss = self.error_measure.forward(x, labels)
        self.pre_loss = loss
        return x, loss

    def backward(self, labels):
        loss_grad = self.error_measure.backward(labels)
        for n in reversed(self.hides):
            loss_grad = n.backward(loss_grad)

# a = np.array([[2, 2, 2, 2, 2], [1, 2, 3, 4, 5], [1, 2, 3, 4, 5]])
# b = np.array([np.sum(i * a, axis=0) for i in a])
# print(b)
