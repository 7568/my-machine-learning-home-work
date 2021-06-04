# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/4/14
Description:
"""
import numpy as np
from sklearn.metrics import accuracy_score
from my_first_cnn import FirstNet
import time
import pickle
import mnist
from data_iter import IrisDataIter
import logging
LOG_FORMA = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(filename='log.txt',level=logging.DEBUG,format=LOG_FORMA)

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
    bach = 128
    first_net = FirstNet(L_R)
    # with open("net_save/first_net2.pkl", 'rb') as file:
    #     first_net = pickle.loads(file.read())
    first_net.l_r = L_R
    test_accuracy=0.9999
    for i in range(epoch):
        if i==20:
            first_net.l_r = 0.001
        elif i==40:
            first_net.l_r = 0.0005
        elif i == 60:
            first_net.l_r = 0.0002
        elif i == 100:
            first_net.l_r = 0.0001

        now = time.time()
        train_accu = list()
        train_loss = list()
        for j in IrisDataIter(np.arange(len(train_images)), bach):
            x_train = train_images[j]
            formated_x_train = x_train.reshape(len(j), 1, 28, 28)
            y_train = train_labels[j]
            one_hot_y_train = np.eye(10)[y_train].T
            y_hat, loss = first_net.forward(formated_x_train, one_hot_y_train)
            first_net.backward(one_hot_y_train)
            y_hat = np.argmax(y_hat, axis=0)
            train_accu.append(accuracy_score(y_hat,y_train))
            train_loss.append(loss)
            # print(f'loss : {loss}')

        # with open("net_save/first_net3.pkl", "wb") as f:
        #     pickle.dump(first_net, f)
        logging.info((time.time() - now))

        test_accu = list()
        test_loss = list()
        for j in IrisDataIter(np.arange(len(test_images)), bach):
            _test_images = test_images[j]
            _test_labels = test_labels[j]
            one_hot_test_labels = np.eye(10)[_test_labels].T
            formated_test_images = _test_images.reshape(len(_test_images), 1, 28, 28)
            y_hat, loss = first_net.predict(formated_test_images, one_hot_test_labels)
            y_hat = np.argmax(y_hat, axis=0)
            test_accu.append(accuracy_score(y_hat, _test_labels))
            test_loss.append(loss)


        train_accu_array = np.asarray(train_accu)
        train_loss_array = np.asarray(train_loss)
        test_accu_array = np.asarray(test_accu)
        test_loss_array = np.asarray(test_loss)
        logging.info(f'epoch : {i} , train精度 in test ： {np.mean(train_accu_array)} , train平均loss ： {np.mean(train_loss_array)}')
        logging.info(f'epoch : {i} , test精度 in test ： {np.mean(test_accu_array)}  , test平均loss ： {np.mean(test_loss_array)}')

        test_accuracy2 = np.mean(test_accu_array)
        if test_accuracy2>test_accuracy:
            test_accuracy = test_accuracy2
            with open("net_save/first_net.pkl", "wb") as f:
                pickle.dump(first_net, f)


if __name__ == '__main__':
    mnist_root = '../data/'
    train_images = mnist.download_and_parse_mnist_file(fname='train-images-idx3-ubyte.gz', target_dir=mnist_root)
    # train_images = (train_images.astype('float32') / 255)
    train_labels = mnist.download_and_parse_mnist_file(fname='train-labels-idx1-ubyte.gz', target_dir=mnist_root)
    test_images = mnist.download_and_parse_mnist_file(fname='t10k-images-idx3-ubyte.gz', target_dir=mnist_root)
    # test_images = (test_images.astype('float32') / 255)
    test_labels = mnist.download_and_parse_mnist_file(fname='t10k-labels-idx1-ubyte.gz', target_dir=mnist_root)


    # train_images = train_images[:300]
    # train_labels = train_labels[:300]
    # test_images = test_images[:100]
    # test_labels = test_labels[:100]


    # train_images1 = np.rot90(train_images, 1, (1, 2))
    # # train_images2 = np.rot90(train_images, 3, (1, 2))
    # train_images = np.concatenate((train_images, train_images1), axis=0)
    # # train_images = np.concatenate((train_images, train_images2), axis=0)
    # train_labels = np.concatenate((train_labels, train_labels), axis=0)
    # # train_labels = np.concatenate((train_labels, train_labels), axis=0)

    train()
