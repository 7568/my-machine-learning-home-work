import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets,metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier

plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
# 导入鸢尾花数据集
iris = datasets.load_iris()
X = iris.data  # X.shape==(150, 4)
y = iris.target  # y.shape==(150, )

# 二进制化输出
y = label_binarize(y, classes=[0, 1, 2])  # shape==(150, 3)
n_classes = y.shape[1]  # n_classes==3

# 添加噪音特征，使问题更困难
random_state = np.random.RandomState(0)
# n_samples, n_features = X.shape  # n_samples==150, n_features==4
# X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]  # shape==(150, 84)

# 打乱数据集并切分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                    random_state=0)
# X_train.shape==(75, 804), X_test.shape==(75, 804), y_train.shape==(75, 3), y_test.shape==(75, 3)

# 学习区分某个类与其他的类
classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                 random_state=random_state))
y_score = classifier.fit(X_train, y_train).decision_function(X_test)
# y_score.shape==(75, 3)

# 为每个类别计算ROC曲线和AUC
# fpr = dict()
# tpr = dict()
# roc_auc = dict()
# for i in range(n_classes):
#     fpr[i], tpr[i], _ = metrics.roc_curve(y_test[:, i], y_score[:, i])
#     roc_auc[i] = metrics.auc(fpr[i], tpr[i])
# fpr[0].shape==tpr[0].shape==(21, ), fpr[1].shape==tpr[1].shape==(35, ), fpr[2].shape==tpr[2].shape==(33, ) 
# roc_auc {0: 0.9118165784832452, 1: 0.6029629629629629, 2: 0.7859477124183007}

class_num = 1
y_test_ = y_test[:, class_num]
y_score_ = y_score[:, class_num]
fpr, tpr, _ = metrics.roc_curve(y_test_, y_score_)
roc_auc = metrics.auc(fpr, tpr)


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