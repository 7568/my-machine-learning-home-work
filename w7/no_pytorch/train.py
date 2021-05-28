# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description:
"""
import numpy as np
from sklearn.metrics import accuracy_score
from my_first_cnn_2 import FirstNet
import time
import pickle
import mnist
from data_iter import IrisDataIter


# from tqdm import tqdm


def train():
    """
    开始训练
    :param x:
    :param y:
    :return:
    """

    _count = 1
    epoch = 200
    L_R = 0.002
    bach = 16
    first_net = FirstNet(L_R)
    # with open("net_save/first_net.pkl", 'rb') as file:
    #     first_net = pickle.loads(file.read())
    for i in range(epoch):
        if (epoch+1)%3==0:
            L_R *= 0.8
            first_net.l_r = L_R
        now = time.time()
        for j in IrisDataIter(np.arange(len(train_images)), bach):
            x_train = train_images[j]
            formated_x_train = x_train.reshape(len(j), 1, 28, 28)
            y_train = train_labels[j]
            one_hot_y_train = np.eye(10)[y_train].T
            y_hat, loss = first_net.forward(formated_x_train, one_hot_y_train)
            first_net.backward(one_hot_y_train)
            # print(f'loss : {loss}')

        with open("net_save/first_net.pkl", "wb") as f:
            pickle.dump(first_net, f)
        print((time.time() - now))

        one_hot_test_labels = np.eye(10)[test_labels].T
        formated_test_images = test_images.reshape(len(test_images), 1, 28, 28)
        y_hat, loss = first_net.forward(formated_test_images, one_hot_test_labels)
        y_hat = np.argmax(y_hat, axis=0)
        test_accuracy = accuracy_score(y_hat, test_labels)
        print(f'epoch : {i} , 精度 in test ： {test_accuracy} , 平均loss ： {np.mean(loss)}')


if __name__ == '__main__':
    mnist_root = '../data/'
    train_images = mnist.download_and_parse_mnist_file(fname='train-images-idx3-ubyte.gz', target_dir=mnist_root)
    # train_images = (train_images / 255) - 0.5
    train_labels = mnist.download_and_parse_mnist_file(fname='train-labels-idx1-ubyte.gz', target_dir=mnist_root)
    test_images = mnist.download_and_parse_mnist_file(fname='t10k-images-idx3-ubyte.gz', target_dir=mnist_root)
    # test_images = (test_images / 255) - 0.5
    test_labels = mnist.download_and_parse_mnist_file(fname='t10k-labels-idx1-ubyte.gz', target_dir=mnist_root)


    # train_images = train_images[:3000]
    # train_labels = train_labels[:3000]
    # test_images = test_images[:1000]
    # test_labels = test_labels[:1000]


    # train_images1 = np.rot90(train_images, 1, (1, 2))
    # # train_images2 = np.rot90(train_images, 3, (1, 2))
    # train_images = np.concatenate((train_images, train_images1), axis=0)
    # # train_images = np.concatenate((train_images, train_images2), axis=0)
    # train_labels = np.concatenate((train_labels, train_labels), axis=0)
    # # train_labels = np.concatenate((train_labels, train_labels), axis=0)

    train()
