import numpy as np
import math

v = np.array([6, 8])
A = np.array([[1, 2, 3],
              [4, 5, 6]])

# TODO: norm hesapla
norm = np.linalg.norm(v)
normA = np.linalg.norm(A)
#norm2 = math.sqrt(v @ v.T) only for 1D
print(norm)
print(normA)
