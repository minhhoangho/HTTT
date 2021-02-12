
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
person_5 = np.array(
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
def fuzzification2(matrices):
  M = np.array(matrices).copy()
  n = len(matrices)
  if n == 0:
    print("No matrices provided, exit")
    return
  s = (M.shape)
  L = np.maximum(M-2, np.full(s, 1))
  U = np.minimum(M+2, np.full(s, 9))
  # print("@@@@@@")
  # print(M.shape)
  # print("----")
  # print(L)
  # print("----")
  # print(M)
  # print("----")
  # print(U)
  # print("@@@@")
  return np.stack((L, M, U), axis=3)

def defuzzification(fuzzy_matrix, alpha = 0.5, beta = 0.5):
  alpha_l = fuzzy_matrix[:,:,0] + alpha*(fuzzy_matrix[:,:,1] - fuzzy_matrix[:,:,0])
  alpha_r = fuzzy_matrix[:,:,2] - alpha*(fuzzy_matrix[:,:,2] - fuzzy_matrix[:,:,1])
  alpha_matrix = np.dstack((alpha_l, alpha_r))

  alpha_beta_matrix = np.add(np.multiply(alpha_matrix[:,:,0],beta),np.multiply(alpha_matrix[:,:,1],(1-beta)))
  
  return alpha_beta_matrix



def myFuzzyAHP(list_comparision, list_eval):
  if len(list_comparision) == 0 or len(list_eval) == 0:
    return
  list_comparision_matrix = []
  list_eval_matrix = []

  # Generate comparision matrix
  for matrix in list_comparision:
    list_comparision_matrix.append(comparision_matrix(matrix))

  # print("comparision_matrix")
  # print(list_comparision_matrix)
  fuzzied_matrix = fuzzification1(list_comparision_matrix)
  # print("fuzzied_matrix")
  # print(fuzzied_matrix)
  criteria_weight_fuzzy_vector = calculate_weight(fuzzied_matrix)
  print("criteria_weight_fuzzy_vector")
  print(criteria_weight_fuzzy_vector) # done buoc 1
  print(criteria_weight_fuzzy_vector.shape) # done buoc 1


  fuzzied_matrix2 = fuzzification2(list_eval)
  print("fuzzied_matrix2")
  # print(fuzzied_matrix2)
  print(fuzzied_matrix2.shape)
  G_L = np.min(fuzzied_matrix2[:,:,:,0], axis=0)
  G_M = np.average(fuzzied_matrix2[:,:,:,1], axis=0)
  G_U = np.max(fuzzied_matrix2[:,:,:,2], axis=0)
  G_matrix = np.stack((G_L, G_M, G_U), axis=2)
  # print(G_matrix) # G fuzzy matrix
  print("G_matrix")
  print(G_matrix.shape)

  A_matrix = np.empty(G_matrix.shape)
  for j in range(0, len(A_matrix[0])):
      sum_column_normalize = np.sqrt(np.sum(G_matrix[:,j]**2, axis=0))
      for i in range(0, len(A_matrix)):
          A_matrix[i][j] = G_matrix[i][j]/sum_column_normalize

  print("A_matrix")
  # print(A_matrix)
  print(A_matrix.shape)

  # Tong hop H = A x W (multiple wise)
  H = np.multiply(A_matrix,criteria_weight_fuzzy_vector)
  print("H")
  # print(H)
  print(H.shape)

  defuzzied_matrix = defuzzification(H)
  print("defuzzied_matrix")
  print(defuzzied_matrix)

  h_max = np.max(defuzzied_matrix, axis=0) # max tung cot
  h_min = np.min(defuzzied_matrix, axis=0) # min tung cot
  h_beta1 = defuzzied_matrix.copy()
  h_beta2 = defuzzied_matrix.copy()
  for i in range(0, defuzzied_matrix.shape[0]): # lap tung options
      h_beta1[i] = h_beta1[i] - h_max
      h_beta2[i] = h_beta2[i] - h_min
  S_max = np.sqrt(np.sum(h_beta1**2, axis=1))
  S_min = np.sqrt(np.sum(h_beta2**2, axis=1))
  print("log s_max, s_min")
  S = np.concatenate((S_max.reshape((len(S_max), 1)), S_min.reshape((len(S_max), 1))), axis=1)
  print(S)
  # PA xep hang
  R = S[:,1]/(S[:,1]+S[:,0])
  print(R)
  




list_comparision = [person_1, person_2, person_3, person_4, person_5]
list_eval = [person_1_op, person_2_op, person_3_op, person_4_op, person_5_op]

myFuzzyAHP(list_comparision, list_eval)
