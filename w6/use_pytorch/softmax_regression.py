# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/13
Description:
"""
import torch
from IPython import display
import d2l as d2l
from torch.utils import data
from sklearn.model_selection import train_test_split
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

src = torch.tensor(np.arange(1, 10, 0.1))
trg = torch.tensor(np.arange(1, 10, 0.1))


def _load_data(x_train, x_test, y_train, y_test, batch_size):
    """Download the Fashion-MNIST dataset and then load it into memory."""
    train_data = data.TensorDataset(x_train, y_train)
    test_data = data.TensorDataset(x_test, y_test)
    return (data.DataLoader(train_data, batch_size, shuffle=True),
            data.DataLoader(test_data, batch_size, shuffle=False))


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


def init_data():
    path = '../iris.data'  # 数据文件路径
    data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

    x, y = np.split(data, (4,), axis=1)
    # x = x[:, :2]x_hat = np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.9)
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train[:, 0]
    y_train = y_train.astype(np.float32)
    y_test = y_test[:, 0]
    y_test = y_test.astype(np.float32)
    return torch.tensor(x_train), torch.tensor(x_test), torch.tensor(y_train), torch.tensor(y_test)


x_train, x_test, y_train, y_test = init_data()
batch_size = 32
train_iter, test_iter = _load_data(x_train, x_test, y_train, y_test, batch_size)

num_inputs = 4
num_outputs = 3
num_hide = 4

W = torch.normal(0, 0.01, size=(num_hide, num_inputs, num_outputs), requires_grad=True)
b = torch.zeros((num_hide, num_outputs), requires_grad=True)


# X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
# X.sum(0, keepdim=True), X.sum(1, keepdim=True)


def softmax(X):
    X_exp = torch.exp(X)
    partition = X_exp.sum(1, keepdim=True)
    return X_exp / partition  # The broadcasting mechanism is applied here


# X = torch.normal(0, 1, (2, 5))
# X_prob = softmax(X)
# X_prob, X_prob.sum(1)


def net(X):
    X = X.double()
    hide = []
    for i in range(len(W)):
        _w = W[i]
        hide.append(torch.matmul(X.reshape((-1, _w.shape[0])), _w.double()) + b[i])
    return softmax(torch.cat(hide, dim=1))


# y = torch.tensor([0, 2])
# y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
# y_hat[[0, 1], y]


def cross_entropy(y_hat, y):
    # y=y[:,0]
    y = y.long()
    return -torch.log(y_hat[:, y])


# cross_entropy(y_hat, y)


def accuracy(y_hat, y):  # @save
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())


# accuracy(y_hat, y) / len(y)


def evaluate_accuracy(net, data_iter):  # @save
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    metric = Accumulator(2)  # No. of correct predictions, no. of predictions
    for X, y in data_iter:
        metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]


class Accumulator:  # @save
    """For accumulating sums over `n` variables."""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# evaluate_accuracy(net, test_iter)


def train_epoch_ch3(net, train_iter, loss, updater):  # @save
    """The training loop defined in Chapter 3."""
    # Set the model to training mode
    if isinstance(net, torch.nn.Module):
        net.train()
    # Sum of training loss, sum of training accuracy, no. of examples
    metric = Accumulator(3)
    for X, y in train_iter:
        # Compute gradients and update parameters
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # Using PyTorch in-built optimizer & loss criterion
            updater.zero_grad()
            l.sum().backward()
            updater.step()
            metric.add(float(l.sum()) * len(y), accuracy(y_hat, y), y.numel())
        else:
            # Using custom built optimizer & loss criterion
            l.sum().backward()
            updater(X.shape[0])
            metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # Return training loss and training accuracy
    # print(b)
    return metric[0] / metric[2], metric[1] / metric[2]


def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  # @save
    """Train a model (defined in Chapter 3)."""
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        print(f'epoch :  {epoch}, train_acc : {train_metrics[1]} , test_acc : {test_acc}')
        # animator.add(epoch + 1, train_metrics + (test_acc,))
    # train_loss, train_acc = train_metrics
    # print(train_acc)


lr = 0.002


def updater(batch_size):

    return d2l.sgd([W, b], lr, batch_size)


num_epochs = 200
opt = torch.optim.Adam([W, b],lr)
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, opt)
