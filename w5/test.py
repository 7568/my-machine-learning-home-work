import numpy as np

# an_array = np.array([[1, 1, 3], [1, 2, 1]])
#
# max_index_col = np.argmax(an_array, axis=0)
#
# print(max_index_col)
#
# max_index_row = np.argmax(an_array, axis=1)
#
# print(max_index_row)

a = np.arange(90).reshape(3, 30)
b = np.arange(5)
c = np.arange(6).reshape(2, 3)

# print(a)
# print(b)
# print(c)
# print(np.einsum('ij->i', a))
# print(np.einsum('ii', a))
# print(np.einsum(a, [0, 0]))
# print('============')
# print(np.einsum('ii->i', a))
# print(np.einsum('ji', c))
# print(np.einsum('i,i', b, b))


# print(np.sum(b*b))
def softmax(z):
    e = np.exp(z - np.max(z))
    s = np.sum(e, axis=1, keepdims=True)
    return e / s


# print(np.array([1, 2, 3]).repeat(5, axis=0))
# print(np.tile([[1],[2],[3],[4],[5]], (1, 10)))
# m, n = a.shape
# p = softmax(a)
# tensor1 = np.einsum('ij,ik->ijk', p, p)
# tensor2 = np.einsum('ij,jk->ijk', p, np.eye(n, n))
# dSoftmax = tensor2 - tensor1
# dz = np.einsum('ijk,ik->ij', dSoftmax, a)

# a = np.tile([[1],[2],[3],[4],[5]], (1, 10)) + [[1],[1],[1],[1],[1]]
# b = np.array([1,2,3,4,5])
# print(b[-1])

a = np.array(range(16)).reshape(4,4)
print(a)