import numpy as np

A = np.array([[1, 2],
              [3, 4]])

B = np.array([[2, 0],
              [1, 2]])

# TODO: matris çarpımı yap
C = A * B   #element-wise çarğım
D = A @ B  # matrix multiplication
print(C)
print(D)