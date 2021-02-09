
import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})
"""
  Bài toán:
  Muốn lựa chọn 3 options theo 4 tiêu chí
  Tham khảo ý kiến 5 chuyên gia
"""

triangular_membership_function = {
  1:[1,1,1], 
  2:[1,2,3], 
  3:[2,3,4], 
  4:[3,4,5], 
  5:[4,5,6], 
  6:[5,6,7], 
  7:[6,7,8],
  8:[7,8,9],
  9:[9,9,9]
}

fuzzy_scale = {
  1: [1,1,3],
  3: [1,3,5],
  5: [3,5,7],
  7: [5,7,9],
  9: [7,9,9]
}

# Ý kiến các chuyên gia về độ quan trọng của từng tiêu chí (5 matrix 4x4)
person_1 = np.array(
[
    [float('nan'), 5, 2, 4],
    [float('nan'), float('nan'), 1/2, 1/2],
    [float('nan'), float('nan'), float('nan'), 2],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])
person_2 = np.array(
[
    [float('nan'), 5, 1, 7],
    [float('nan'), float('nan'), 1/5, 1/7],
    [float('nan'), float('nan'), float('nan'), 3],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])
person_3 = np.array(
[
    [float('nan'), 6, 2, 6],
    [float('nan'), float('nan'), 1/4, 1/6],
    [float('nan'), float('nan'), float('nan'), 4],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])
person_4 = np.array(
[
    [float('nan'), 2, 1, 6],
    [float('nan'), float('nan'), 1, 1/3],
    [float('nan'), float('nan'), float('nan'), 5],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])
person_5 = np.array(Í
[
    [float('nan'), 8, 2, 7],
    [float('nan'), float('nan'), 1/4, 1/4],
    [float('nan'), float('nan'), float('nan'), 1],
    [float('nan'), float('nan'), float('nan'), float('nan')]
])


# Đánh gía điểm về 3 option theo từng tiêu chí của 5 chuyên gia (cột: tiêu chí, hàng: opt)
person_1_op = np.array([
  [1, 1, 3, 5], 
  [3, 3, 1, 1], 
  [3, 7, 3, 1]
])
person_2_op = np.array([
  [1, 1, 3, 5], 
  [7, 3, 1, 1], 
  [3, 3, 3, 3]
])
person_3_op = np.array([
  [3, 1, 1, 4], 
  [5, 9, 1, 3], 
  [3, 7, 1, 3]
])
person_4_op = np.array([
  [3, 1, 3, 3], 
  [7, 3, 1, 3], 
  [7, 7, 3, 3]
])
person_5_op = np.array([
  [9, 1, 3, 7], 
  [4, 3, 1, 5], 
  [9, 1, 1, 1]
])



def geometrix_mean(arr):
  temp = np.copy(arr)
  for j in range(arr.shape[1]):
      arr[:, j] /= sum(temp[:, j])
  return arr




"""
  Xây dựng vector trọng số cho matrix (vector trọng số mờ của từng tiêu chí)
"""

"""
  input : n x n matrix
  output: n x 1 matrix
"""
def calculate_weight(matrix):
  geometrix = geometrix_mean(np.copy(matrix))
  return np.mean(geometrix, axis=1)



# Ma trận so sánh
def comparision_matrix(arr):
  for i in range(arr.shape[0]):
      arr[i][i] = 1
      for j in range(arr.shape[1]):
          if i > j:
              arr[i][j] = 1 / arr[j][i]
  return arr


"""
  input:   list of n x n matrices
  output: 3 x n x n matrix
"""
def fuzzification1(matrices):
  matrices = np.array(matrices)
  n = len(matrices)
  if n == 0:
    print("No matrices provided, exit")
    return
  L = np.min(matrices, axis = 0)
  M = np.average(matrices, axis = 0)
  U = np.max(matrices, axis = 0)
  return np.dstack((L, M, U))


"""
  input:   list of n x n matrices
  output: 3 x n x n matrix
"""
def fuzzification2(matrics):
  matrices = np.array(matrices)
  n = len(matrices)
  if n == 0:
    print("No matrices provided, exit")
    return


def myFuzzyAHP(list_comparision, list_eval):
  if len(list_comparision) == 0 or len(list_eval):
    return
  list_comparision_matrix = []
  list_eval_matrix = []

  # Generate comparision matrix
  for matrix in list_comparision:
    list_comparision_matrix.append(comparision_matrix(matrix))


  fuzzied_matrix = fuzzification1(list_comparision_matrix)
  print("fuzzied_matrix")
  print(fuzzied_matrix)
  criteria_weight_fuzzy_vector = calculate_weight(fuzzied_matrix)
  print("criteria_weight_fuzzy_vector")
  print(criteria_weight_fuzzy_vector)
  # print(L_matrix)
  # print(U_matrix)
  # print(M_matrix)


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




list_comparision = [person_1, person_2, person_3, person_4, person_5]
list_eval = [person_1_op, person_2_op, person_3_op, person_4_op, person_5_op]
myFuzzyAHP(list_comparision, list_eval)
