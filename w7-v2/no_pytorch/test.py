# -*- coding: UTF-8 -*-
"""
Created by root at 2021/5/31
Description:
"""

import numpy as np
from sklearn.metrics import accuracy_score
from my_first_cnn import FirstNet
import time
import pickle
import mnist
from data_iter import IrisDataIter


# from tqdm import tqdm


def do_test():
    """
    开始训练
    :param x:
    :param y:
    :return:
    """

    with open("net_save/first_net3.pkl", 'rb') as file:
        first_net = pickle.loads(file.read())
    bach=500
    test_accuracy_list = list()
    loss_list = list()
    for j in IrisDataIter(np.arange(len(test_images)), bach):
        _test_images = test_images[j]
        _test_labels = test_labels[j]
        one_hot_test_labels = np.eye(10)[_test_labels].T
        formated_test_images = _test_images.reshape(len(_test_images), 1, 28, 28)
        y_hat, loss = first_net.predict(formated_test_images, one_hot_test_labels)
        y_hat = np.argmax(y_hat, axis=0)
        test_accuracy2 = accuracy_score(y_hat, _test_labels)
        test_accuracy_list.append(test_accuracy2)
        loss_list.append(loss)

    print(f' test精度 in test ： {np.mean(np.asarray(test_accuracy_list))} , test平均loss ： {np.mean(np.asarray(loss_list))}')


if __name__ == '__main__':
    mnist_root = '../data/'
    # train_images = mnist.download_and_parse_mnist_file(fname='train-images-idx3-ubyte.gz', target_dir=mnist_root)
    # train_images = (train_images / 255) - 0.5
    # train_labels = mnist.download_and_parse_mnist_file(fname='train-labels-idx1-ubyte.gz', target_dir=mnist_root)
    test_images = mnist.download_and_parse_mnist_file(fname='t10k-images-idx3-ubyte.gz', target_dir=mnist_root)
    # test_images = (test_images / 255) - 0.5
    test_labels = mnist.download_and_parse_mnist_file(fname='t10k-labels-idx1-ubyte.gz', target_dir=mnist_root)


    # train_images = train_images[:3000]
    # train_labels = train_labels[:3000]
    test_images = test_images[3000:6000]
    test_labels = test_labels[3000:6000]


    # train_images1 = np.rot90(train_images, 1, (1, 2))
    # # train_images2 = np.rot90(train_images, 3, (1, 2))
    # train_images = np.concatenate((train_images, train_images1), axis=0)
    # # train_images = np.concatenate((train_images, train_images2), axis=0)
    # train_labels = np.concatenate((train_labels, train_labels), axis=0)
    # # train_labels = np.concatenate((train_labels, train_labels), axis=0)

    do_test()
