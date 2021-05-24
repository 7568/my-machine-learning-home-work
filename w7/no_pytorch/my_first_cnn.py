# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description: 简单的卷积神经网络的实现
https://victorzhou.com/blog/intro-to-cnns-part-2/
http://web.eecs.utk.edu/~zzhang61/docs/reports/2016.10%20-%20Derivation%20of%20Backpropagation%20in%20Convolutional%20Neural%20Network%20(CNN).pdf
https://medium.com/@pavisj/convolutions-and-backpropagations-46026a8f5d2c
https://medium.com/@ngocson2vn/a-gentle-explanation-of-backpropagation-in-convolutional-neural-network-cnn-1a70abff508b
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
        self.input_num = input_num
        self.output_num = output_num
        self.W = np.random.normal(0, 1, (input_num + 1, output_num)) / np.sqrt((input_num + 1) / 2)

    def forward(self, prev_data):
        self.prev_data = prev_data
        prev_data_new = np.concatenate((prev_data, np.ones((1, prev_data.shape[1]))), axis=0)
        H = self.W.T @ prev_data_new
        return H

    def backward(self, grad):
        new_grad = self.W @ grad
        grad_w = self.prev_data @ grad.T
        grad_w = np.concatenate((grad_w, np.ones((1, grad_w.shape[1]))), axis=0)
        self.W -= BasicModule.l_r * (grad_w / self.prev_data.shape[1])
        return new_grad[0:-1, :]  # , grad_W


class Conv2d(BasicModule):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0):
        """
            为方便计算统一步长为1
        :param input_num:
        :param output_num:
        """
        self.last_input = None
        self.input = None
        self.output = None
        self.input_padded = None
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # filters is a 3d array with dimensions (num_filters, 3, 3)
        # We divide by (kernel_size*kernel_size) to reduce the variance of our initial values
        self.filters = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) / (kernel_size ** 2)
        self.filters = self.filters.flatten()
        self.filters = self.filters.reshape((in_channels * kernel_size * kernel_size, out_channels))
        self.bias = np.random.randn(self.filters.shape[1], 1)

    def forward(self, input, filters=None):
        """
        input 为 b,n,h,w ，当input与filters进行卷积的时候，将卷积操作转换成向量的乘法操作
        :param input:
        :param filters:
        :return:
        """

        self.input = input
        input_padded = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                              'constant')
        self.input_padded = input_padded
        b0, n0, h0, w0 = input_padded.shape
        out_size = (np.array([h0, w0]) - self.kernel_size + 1) / self.stride
        output = np.zeros((b0, self.out_channels, int(out_size[0]), int(out_size[1])))

        flatted_input = np.zeros(
            (b0, int(out_size[0]) * int(out_size[1]), n0 * self.kernel_size * self.kernel_size))

        for bb in range(b0):
            count = 0
            for h in range(int(out_size[0])):
                for w in range(int(out_size[1])):
                    x_slice = input_padded[bb, :, h * self.stride:h * self.stride + self.kernel_size,
                              w * self.stride:w * self.stride + self.kernel_size]
                    x_slice = x_slice.flatten()
                    flatted_input[bb, count, :] = x_slice
                    count += 1

        self.last_input = flatted_input
        _output = np.zeros((b0, self.out_channels, flatted_input.shape[1]))

        for bb in range(b0):
            _output[bb, :, :] = (flatted_input[bb, :, :] @ self.filters).T + self.bias

        # output = _output.reshape(b0, self.out_channels, int(out_size[0]), int(out_size[1]))
        for bb in range(b0):
            output[bb, :, :, :] = _output[bb, :, :].reshape(self.out_channels, int(out_size[0]),
                                                                int(out_size[1]))

        return output

    def backward(self, grad_pre):
        '''

        :param grad_pre:
        :return:
        '''

        # xb, xn, xh, xw = self.input.shape
        # 由于forward里面的卷积是用的矩阵乘法的方式，所以先将grad_pre转化成为flatted_input的结构
        o, c, h, w = grad_pre.shape
        grad_pre = grad_pre.reshape((o, c, h * w))
        dF = np.zeros(self.filters.shape)
        db = np.zeros(self.bias.shape)
        for bb in range(o):
            dF += (grad_pre[bb, :, :] @ self.last_input[bb, :, :]).T
            db += np.sum(grad_pre[bb, :, :], axis=1).reshape(c, 1)

        _dX = np.zeros(self.last_input.shape)
        for bb in range(o):
            _dX[bb, :, :] = (self.filters @ grad_pre[bb, :, :]).T

        self.filters -= BasicModule.l_r * dF / o
        self.bias -= BasicModule.l_r * db / o
        # 剔除掉_dX中重复数据，方法是跟卷积操作形式一样，原来是一步一步的取小方格，现在是从_dX中获得小方格一步一步的放回去
        padding_x = np.zeros(self.input_padded.shape)
        pb, pn, ph, pw = padding_x.shape
        dX = np.zeros(self.input.shape)
        out_size = (np.array([ph, pw]) - self.kernel_size + 1) / self.stride
        for bb in range(pb):
            count = 0
            for h in range(int(out_size[0])):
                for w in range(int(out_size[1])):
                    k = _dX[bb, count, :].reshape(self.in_channels, self.kernel_size, self.kernel_size)
                    padding_x[bb, :, h * self.stride:h * self.stride + self.kernel_size,
                    w * self.stride:w * self.stride + self.kernel_size] = k
                    count += 1
        # 去掉padding的部分
        # dX[:, :, :, :] = padding_x[:, :, self.padding:ph - self.padding, self.padding:pw - self.padding]
        for bb in range(pb):
            dX[bb, :, :, :] = padding_x[bb, :, self.padding:ph - self.padding, self.padding:pw - self.padding]

        return dX


class Flatting(BasicModule):
    def __init__(self):
        """
        # 实现 sigmoid 激活函数
        """
        self.prev_data = None

    def forward(self, prev_data):
        self.prev_data = prev_data
        flated = prev_data.flatten()
        flated = flated.reshape(prev_data.shape[0], int(flated.shape[0] / prev_data.shape[0]))
        return flated.T

    def backward(self, grad_b):
        grad_b = grad_b.reshape(self.prev_data.shape)
        return grad_b


class Relu2d(BasicModule):
    def __init__(self):
        self.prev_data = None

    def forward(self, prev_data):
        self.prev_data = prev_data
        result = np.where(prev_data > 0, prev_data, np.zeros_like(prev_data))
        return result

    def backward(self, grad_b):
        X = self.prev_data
        result = (X > 0).astype(np.float32) * grad_b
        return result


class MaxPool2d(BasicModule):

    def __init__(self, kernel_size):
        self.kernel_size = kernel_size
        self.output = None
        self.max_pool_for_back = None

    def forward(self, prev_data):

        self.prev_data = prev_data

        bach, num_filters, h, w = prev_data.shape
        k_s = self.kernel_size
        if h % k_s != 0 or w % k_s != 0:
            print('MaxPool2d 参数错误！')
        new_h = h // k_s
        new_w = w // k_s
        output = np.zeros((bach, num_filters, new_h, new_w))
        max_pool_for_back = np.zeros(prev_data.shape)
        for bb in range(bach):
            for n in range(num_filters):
                for i in range(new_h):
                    for j in range(new_w):
                        im_region = prev_data[bb, n, (i * k_s):(i * k_s + k_s), (j * k_s):(j * k_s + k_s)]
                        max_index = np.argmax(im_region)
                        output[bb, n, i, j] = prev_data[bb, n, i * k_s + max_index // k_s, j * k_s + max_index % k_s]
                        max_pool_for_back[bb, n, i * k_s + max_index // k_s, j * k_s + max_index % k_s] = 1

        self.max_pool_for_back = max_pool_for_back
        self.output = output
        return output

    def backward(self, grad_p):
        """
        新建一个与原输入一样大的tensor，然后将愿输入中对应区域最大值的位置，设置为1，然后将该tensor乘以grad_p
        一个简单的做法就是，然后将愿输入中对应区域最大值的位置，设置为grad_p对应的位置的值
        :param grad_b:
        :return:
        """
        grad_p = np.repeat(grad_p, self.kernel_size, axis=2)
        grad_p = np.repeat(grad_p, self.kernel_size, axis=3)
        new_grad_p = self.max_pool_for_back * grad_p
        return new_grad_p


class Softmax(BasicModule):

    def __init__(self):
        """实现 Softmax 激活函数"""
        self.forward_output = None
        self.epsilon = 1e-12  # 防止求导后分母为 0

    def forward(self, prev_data):
        p_exp = np.exp(prev_data - np.max(prev_data, axis=0))
        # p_exp = np.exp(prev_data)
        denominator = np.sum(p_exp, axis=0, keepdims=True)
        self.forward_output = p_exp / (denominator + self.epsilon)
        return self.forward_output

    def backward(self, grad_b):
        """
        :param grad_b:
        :return:
        https://themaverickmeerkat.com/2019-10-23-Softmax/
        """

        return grad_b


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
        # return -y * (1 / (p + self.epsilon))
        return p - y


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
        return y - p


# 搭建全连接神经网络模型
class FirstNet(BasicModule):

    def __init__(self, l_r):
        self.pre_loss = 0
        BasicModule.l_r = l_r
        self.head = [
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
            [Conv2d(1, 1, 3, padding=1),
             Relu2d()],
        ]
        self.hides = [
            Conv2d(1, 6, 5),
            Relu2d(),
            MaxPool2d(2),
            Conv2d(6, 12, 5, padding=4),
            Relu2d(),
            MaxPool2d(2),
            Flatting(),
            LinearLayer(768, 10),
            Softmax()]
        self.error_measure = CrossEntropy()

    def forward(self, x, labels):
        # x2 = np.array(x, dtype=float)
        # x = (x - np.mean(x, axis=(2, 3), keepdims=True)) / np.std(x, axis=(2, 3), keepdims=True)  # 将x进行标准化操作

        # b0, n0, h0, w0 = x.shape
        # head_layels = np.zeros((b0, len(self.head) + 1, h0, w0))
        # head_layels[:, 0:1, :, :] = x[:, :, :, :]
        # count = 1
        # for n in self.head:
        #     x = n[0].forward(x)
        #     x = n[1].forward(x)
        #     head_layels[:, count:count + 1, :, :] = x
        #     count += 1
        # x = head_layels.reshape(b0, len(self.head) + 1, 28, 28)

        for n in self.hides:
            x = n.forward(x)
        loss = self.error_measure.forward(x, labels)
        self.pre_loss = loss
        return x, loss

    def backward(self, labels):
        loss_grad = self.error_measure.backward(labels)
        for n in reversed(self.hides):
            loss_grad = n.backward(loss_grad)

        # for l in range(1, loss_grad.shape[1]):
        #     _loss_grad = loss_grad[:, l:l + 1, :, :]
        #     for n in reversed(self.head[0:l]):
        #         _loss_grad = n[1].backward(_loss_grad)
        #         _loss_grad = n[0].backward(_loss_grad)
