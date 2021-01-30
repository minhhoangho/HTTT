
import numpy as np

triangular_membership_function = {
  1:[1,1,1], 
  2:[1,2,3], 
  3:[2,3,4], 
  4:[3,4,5], 
  5:[4,5,6], 
  6: [5,6,7], 
  7:[6,7,8],
  8:[7,8,9],
  9:[9,9,9]
}

fuzzy_scale = {
  1: [1,3,3],
  3: [1,3,5],
  5: [3,5,7],
  7: [5,7,9],
  9: [7,9,9]
}


sample_maxtrix1 = np.array(
[
    [float('nan'), 5, 2, 4],
    [float('nan'), float('nan'), 1/2, 1/2],
    [float('nan'), float('nan'), float('nan'), 2],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])
sample_maxtrix2 = np.array(
[
    [float('nan'), 5, 1, 7],
    [float('nan'), float('nan'), 1/5, 1/7],
    [float('nan'), float('nan'), float('nan'), 3],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])

sample_maxtrix3 = np.array(
[
    [float('nan'), 6, 2, 6],
    [float('nan'), float('nan'), 1/4, 1/6],
    [float('nan'), float('nan'), float('nan'), 4],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])

# Ma trận so sánh
def comparision_matrix(arr):
    for i in range(arr.shape[0]):
        arr[i][i] = 1
        for j in range(arr.shape[1]):
            if i > j:
                arr[i][j] = 1 / arr[j][i]
    return arr


def myFuzzyAHP(list_of_matrix):
  n = len(list_of_matrix)
  if n == 0:
    return
  list_comparision_matrix = []
  # Generate comparision matrix
  for matrix in list_of_matrix:
    list_comparision_matrix.append(comparision_matrix(matrix))

  L_matrix = list_comparision_matrix[0]
  U_matrix = list_comparision_matrix[0]
  for m in list_comparision_matrix[1:]:
    L_matrix = np.minimum(L_matrix, m)
    U_matrix = np.maximum(U_matrix, m)
  M_matrix = sum(list_comparision_matrix)/n
  print(L_matrix)
  print(U_matrix)
  print(M_matrix)


def fuzzy_AHP(AHP_matrix):
	#print(triangular_membership_function)
	test_data = np.array(AHP_matrix).copy()
	n = len(test_data)
	fuzzified_test_data = numpy.zeros((n,n,3))

	for x in range(n):
		for y in range(n):
			if(test_data[x][y] >= 1):
				fuzzified_test_data[x][y] = triangular_membership_function[test_data[x][y]]
			else:
				index = round(1/test_data[x][y])
				#print(index)
				temp = triangular_membership_function[index]
				for i in range(3):
					fuzzified_test_data[x][y][i] = 1.0/temp[2-i]
	#print(fuzzified_test_data)

	fuzzy_geometric_mean = [[1 for x in range(3)] for y in range(n)]
	#print(fuzzy_geometric_mean)

	for i in range(n):
		for j in range(3):
			for k in range(n):
				fuzzy_geometric_mean[i][j] *= fuzzified_test_data[i][k][j]
			fuzzy_geometric_mean[i][j] = fuzzy_geometric_mean[i][j]**(1/float(n))
	#print(fuzzy_geometric_mean)

	sum_fuzzy_gm = [0 for x in range(3)]
	inv_sum_fuzzy_gm = [0 for x in range(3)]

	for i in range(3):
		for j in range(n):
			sum_fuzzy_gm[i] += fuzzy_geometric_mean[j][i]

	for i in range(3):
		inv_sum_fuzzy_gm[i] = (1.0/sum_fuzzy_gm[2-i])
	#print(sum_fuzzy_gm)

	fuzzy_weights = [[1 for x in range(3)] for y in range(n)]

	for i in range(n):
		for j in range(3):
			fuzzy_weights[i][j] = fuzzy_geometric_mean[i][j]*inv_sum_fuzzy_gm[j]
	#print(fuzzy_weights)

	weights = [0 for i in range(n)]
	normalized_weights = [0 for i in range(n)]
	sum_weights = 0

	for i in range(n):
		for j in range(3):
			weights[i] += fuzzy_weights[i][j]
		weights[i] /= 3
		sum_weights += weights[i]
	#print(weights)
	#print(sum_weights)

	for i in range(n):
		normalized_weights[i] = (1.0*weights[i])/(1.0*sum_weights)
	#print(normalized_weights)

	return normalized_weights





myFuzzyAHP([sample_maxtrix1, sample_maxtrix2, sample_maxtrix3])
