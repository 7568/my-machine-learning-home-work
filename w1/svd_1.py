# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/3/9
Description: this is a svd python program
if m > n U is the eigenvector of AA' , sigma is the eignmatrix of AA' , VT = sigma^ - 1 * U' * A
if m <= n V is the eigenvector of A'A , sigma is the eignmatrix of A'A , U = A * V * sigma^ - 1
"""

import numpy as np  ##引入numpy模块

# A = np.array([[0, 1], [1, 1], [1, 0]])
A = np.array([[4, 0, 3], [3, -5, 4]])

# 得到矩阵的行，列
m = A.shape[0]
n = A.shape[1]

# 转置
AT = np.transpose(A)

# m,n大小不同，处理方法不一样
if m > n:
    # U 是AA' 的特征向量构成的矩阵，eigen_value是AA'的特征值
    AAT = A @ AT
    eigen_value, U = np.linalg.eigh(AAT)
else:
    # V 是A'A 的特征向量构成的矩阵，eigen_value是A'A的特征值
    ATA = AT @ A
    eigen_value, V = np.linalg.eigh(ATA)

# 降序排序后 ， 逆序输出
evall_sort_idx = np.argsort(eigen_value)[::-1]
# 特征值排序放好
eigen_value = np.sort(eigen_value)[::-1]
# 讲特征值对应的特征向量也对应排好序
if m > n:
    U = U[:, evall_sort_idx]
else:
    V = V[:, evall_sort_idx]

# 构建由特征值平方根构成的对角矩阵及其逆矩阵
sigma = np.zeros([m, n])
inverse_sigma = np.zeros([n, m])
i = 0
for eig in eigen_value:
    if eig > 0.000001:
        eig_sqrt = np.sqrt(eig)
        sigma[i, i] = eig_sqrt
        inverse_sigma[i, i] = 1 / eig_sqrt
        i = i + 1
if m > n:
    UT = np.transpose(U)
    # 下一行代表的数学公式为VT = sigma^ -1 * U' * A
    VT = inverse_sigma @ UT @ A
else:
    VT = np.transpose(V)
    # 下一行代表的数学公式为U = A * V * sigma^-1
    U = A @ V @ inverse_sigma

# 恢复A
A_recover = U @ sigma @ VT
print(" -- U --")
print(U)
print(" -- VT --")
print(VT)
print("-- sigma --")
print(sigma)
print("-- A_recover--")
print(A_recover)

# A = np.array([[0, 1], [1, 1], [1, 0]])
U1, sigma1, VT1 = np.linalg.svd(A)
print("-- U1 --")
print(U1)
print("--VT1 --")
print(VT1)
print("--sigma1-- ")
print(sigma1)
#  请根据上一行的SVD求出的U1 ， sigma1 ， VT1 ， 恢复A的值
# 注： 上一上求出的sigma1为向量，要把他变成对角矩阵

# sigma1 里面缺少0
sigma1 = np.diag(sigma1)
# sigma1 行上补0，如果U1的列数大于sigma1的行数，则需要在sigma1中补为0的行
if U1.shape[1] - sigma1.shape[0] > 0:
    sigma1 = np.concatenate((sigma1, np.zeros((U1.shape[1] - sigma1.shape[0], sigma1.shape[1]))), axis=0)

# sigma1 列上补0，如果VT1的行数大于sigma1的列数，则需要在sigma1中补为0的列
if VT1.shape[0] - sigma1.shape[1] > 0:
    sigma1 = np.concatenate((sigma1, np.zeros((sigma1.shape[0], VT1.shape[1] - sigma1.shape[1]))), axis=1)
print("--sigma1-- ")
print(sigma1)
A_recover1 = U1 @ sigma1 @ VT1

print("-- A_recover1--")
print(A_recover1)
