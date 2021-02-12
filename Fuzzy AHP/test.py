import numpy as np


a = np.array([
  [1, 1, 3, 5], 
  [3, 3, 1, 1], 
  [3, 7, 3, 1]
], dtype="float")

def normalize(comparisonMatrix):
    matrix = comparisonMatrix.copy()
    return matrix/np.sum(matrix, axis=0)

def getWeightVector(matrix):
    w = np.sum(matrix, axis=1)
    print("----------")
    print(w)
    print(np.sum(w, axis=0))
    print("----------")

    return normalize(w)


def geometrix_mean(arr):
  temp = np.copy(arr)
  print(temp.shape)
  for j in range(arr.shape[-1]):
      arr[:, j] /= sum(temp[:, j])
  return arr

def calculate_weight(matrix):
  geometrix = geometrix_mean(np.copy(matrix))
  return np.mean(geometrix, axis=1)
x1 = normalize(a)
print("x1")
print(x1)
x11 = geometrix_mean(a)
print("x11")
print(x11)


x2 = getWeightVector(x1)
print("x2")
print(x2)


x22 = calculate_weight(a)
print("x22")
print(x22)
