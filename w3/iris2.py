# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/23
Description:SVM算法的线性核、高斯核函数 数据集划分方法采用自助法,
性能度量分别采用计算错误率、精度
"""
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.utils import resample


def iris_type(s):
    it = {b'Iris-setosa': 0, b'Iris-versicolor': 1, b'Iris-virginica': 2}
    return it[s]


def bootstrapping(x, y):
    """
    先使用resample对数据的索引进行采样
    然后取出对应索引的数据为训练集，其余的为测试集
    :param x:
    :param y:
    :return:
    """
    data_index = np.array(range(len(x)))
    bootstrapSamples = resample(data_index, n_samples=len(data_index), replace=1)
    x_train = x[bootstrapSamples]
    x_test = x[np.delete(data_index, bootstrapSamples)]
    y_train = y[bootstrapSamples]
    y_test = y[np.delete(data_index, bootstrapSamples)]

    return x_train, x_test, y_train, y_test


path = './iris.data'  # 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter=',', converters={4: iris_type})

x, y = np.split(data, (4,), axis=1)
x = x[:, :2]
x_train, x_test, y_train, y_test = bootstrapping(x, y)

clf = svm.SVC(C=0.1, kernel='linear', decision_function_shape='ovr',probability=True)
# clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
clf.fit(x_train, y_train.ravel())

# print(clf.score(x_train, y_train))  # 精度
y_hat = clf.predict(x_train)
mean_accuracy = clf.score(x_test, y_test)
print('精度 ： ', mean_accuracy,' , 错误率 ： ', 1 - mean_accuracy)
y_hat2 = clf.predict(x_test)
y_test_ = y_test[:, 0]
print('该三分类的混淆矩阵为：')
confusion_matrix = metrics.confusion_matrix(y_test_, y_hat2)
print(confusion_matrix)
print('第一类的查准率 : ', confusion_matrix[0][0] / np.sum(confusion_matrix[:, 0]))
print('第一类的查全率 : ', confusion_matrix[0][0] / np.sum(confusion_matrix[1:3, 1:3]))
print('F1 : ',
      2 * confusion_matrix[0][0] / (len(y_test_) + confusion_matrix[0][0] - np.sum(confusion_matrix[1:3, 1:3])))

# 获得第一类的ROC数据
class_num = 0
y_test_ = np.where(y_test_ == class_num, 1, 0)
y_decision_score = clf.decision_function(x_test)[:, class_num]
fpr, tpr, _ = metrics.roc_curve(y_test_, y_decision_score)
roc_auc = metrics.auc(fpr, tpr)

x1_min, x1_max = x[:, 0].min(), x[:, 0].max()  # 第0列的范围
x2_min, x2_max = x[:, 1].min(), x[:, 1].max()  # 第1列的范围
x1, x2 = np.mgrid[x1_min:x1_max:200j, x2_min:x2_max:200j]  # 生成网格采样点
grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
mpl.rcParams['axes.unicode_minus'] = False

cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
# cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])

grid_hat = clf.predict(grid_test)  # 预测分类值
grid_hat = grid_hat.reshape(x1.shape)  # 使之与输入的形状相同

alpha = 0.5
plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)  # 预测值的显示
plt.plot(x[:, 0], x[:, 1], 'o', alpha=alpha, color='blue', markeredgecolor='k')
plt.scatter(x_test[:, 0], x_test[:, 1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
plt.xlabel(u'花萼长度', fontsize=13)
plt.ylabel(u'花萼宽度', fontsize=13)
plt.xlim(x1_min, x1_max)
plt.ylim(x2_min, x2_max)
plt.title(u'SVM分类', fontsize=15)
plt.show()

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('第一类的ROC曲线')
plt.legend(loc="lower right")
plt.show()
