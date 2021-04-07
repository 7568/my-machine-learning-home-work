import numpy as np

an_array = np.array([[1, 1, 3], [1, 2, 1]])

max_index_col = np.argmax(an_array, axis=0)

print(max_index_col)

max_index_row = np.argmax(an_array, axis=1)

print(max_index_row)