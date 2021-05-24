# -*- coding: UTF-8 -*-
"""
Created by louis at 2021/5/19
Description:
"""
import numpy as np
import pickle

class A:
    def __init__(self):
        self.a = 1
        self.b = 2
        self.c = 3

    def __str__(self):
        return f'a:{self.a} , b:{self.b} , c:{self.c}'

# a = A()
# a.c=4
# print(a)
# out_put = open("test_a.pkl", 'wb')
# tree_str = pickle.dumps(a)
# out_put.write(tree_str)
# out_put.close()


a = A()
with open("test_a.pkl", 'rb') as file:
    a = pickle.loads(file.read())
print(a)
# for i in range(1,10):
#     print(i)
# m = np.arange(18).reshape((1,18))
# print(m)
# for mm in reversed(m):
#     print(mm)
# print(np.rot90(m, 2,(1,2)))
# pd = np.pad(m, ((0, 0), (1, 1), (1, 1)), 'constant')
# print(pd)
# print(np.random.randn(0, 0, 0))
# print(m.flatten())

# m = np.arange(27).reshape(3,3,3)
# print(m)
# mm = np.repeat(m,2,axis=1)
# mm = np.repeat(mm,2,axis=2)
# print(mm)


# m = np.arange(27).reshape((3, 3, 3)) // 2
# max_pool = np.zeros((3,3,3))
# max_pool_for_back = np.zeros((3,3,3))
# print(np.argmax(m))
# for i in range(3):
#     max_index = np.argmax(m[i, :, :])
#     max_pool[i, max_index // 3, max_index % 3] = m[i, max_index // 3, max_index % 3]
#     max_pool_for_back[i, max_index // 3, max_index % 3] = 1
#     # max_indexs = np.append(max_indexs,[i, max_index // 3, max_index % 3])
#
# # print(max_indexs.reshape(3,3))
# print(max_pool)
# print(max_pool_for_back)
# import random
# a = [1,2,3,4,5]
# random.shuffle(a)
# print(a)

# a = np.arange(9).reshape((3,3))
# b = np.arange(9).reshape((3,3))
# a = np.append(a,b)
# print(a.reshape(2,3,3))

