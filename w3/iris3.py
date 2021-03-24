# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/23
Description:SVM算法的线性核、高斯核函数 数据集划分方法采用10折交叉验证法,
性能度量分别采用计算错误率、精度、第一类的查准率和查全率、F1、ROC绘制
"""
from sklearn import svm
import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


path = './iris.data'  # 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

x, y = np.split(data, (4,), axis=1)
x = x[:, :2]
y = y[:, 0]

# clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr', probability=True)
clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')

my_n_splits = 10
kf = KFold(n_splits=my_n_splits, shuffle=True)
_count = 1
all_mean_accuracy = np.array([])
all_confusion_matrix = np.zeros((3, 3))
for train_index, test_index in kf.split(x):
    print(f'==========第 {_count} 折数据训练的结果如下===========')
    _count += 1
    x_train = x[train_index]
    x_test = x[test_index]
    y_train = y[train_index]
    y_test = y[test_index]
    clf.fit(x_train, y_train.ravel())
    y_hat = clf.predict(x_train)
    mean_accuracy = clf.score(x_test, y_test)
    all_mean_accuracy = np.append(all_mean_accuracy, mean_accuracy)
    print('精度 ： ', mean_accuracy, ' , 错误率 ： ', 1 - mean_accuracy)
    y_hat2 = clf.predict(x_test)
    print('该三分类的混淆矩阵为：')
    confusion_matrix = metrics.confusion_matrix(y_test, y_hat2)
    print(confusion_matrix)
    all_confusion_matrix = np.add(all_confusion_matrix, confusion_matrix)
    print('第一类的查准率 : ', confusion_matrix[0][0] / np.sum(confusion_matrix[:, 0]))
    print('第一类的查全率 : ', confusion_matrix[0][0] / np.sum(confusion_matrix[1:3, 1:3]))
    print('F1 : ',
          2 * confusion_matrix[0][0] / (len(y_test) + confusion_matrix[0][0] - np.sum(confusion_matrix[1:3, 1:3])))
print('=============== end =================\n')
print('平均精度 ： ', np.mean(all_mean_accuracy), ' , 平均错误率 ： ', 1 - np.mean(all_mean_accuracy))
print('该三分类的平均混淆矩阵为：')
all_confusion_matrix /= my_n_splits
print(all_confusion_matrix)
print('第一类的平均查准率 : ', all_confusion_matrix[0][0] / np.sum(all_confusion_matrix[:, 0]))
print('第一类的平均查全率 : ', all_confusion_matrix[0][0] / np.sum(all_confusion_matrix[1:3, 1:3]))
print('平均F1 : ',
          2 * confusion_matrix[0][0] / (len(y_test) + confusion_matrix[0][0] - np.sum(confusion_matrix[1:3, 1:3])))