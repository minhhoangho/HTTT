import numpy as np
np.set_printoptions(formatter={'float':lambda x:"{0:0.3f}".format(x)})
"""
  Bài toán:
  Muốn lựa chọn 2 options theo 3 tiêu chí
  Tham khảo ý kiến 15 chuyên gia
"""

# Khảo sát độ quan trọng của mỗi tiêu chí, khảo sát 15 người
criteria_investigation_matrix = np.array(
  #C1, C2, C1C2
  [6   , 4    ,5],
  [6/15, 4/15,5/15]
  [6/15, 4/15, 1]
  [11/15, 9/15, 1]
  )

# Với mỗi tiêu chí, khảo sát 15 người -> xác định độ yêu thích tiêu chí

investigation_matrix1 = np.array([
  #A1, A2, A3, A1A2, A1A3 A2A3, A1A2A3
  [5,  2,  3,   4,   0,    0,    1  ],
  [5/15, 2/15, 3/15, 4/15, 0, 0, 1/15],
  [5/15, 2/15, 3/15, 7/15, 8/15, 5/15, 1],
  [10/15, 7/15, 4/15, 12/15, 13/15, 10/15, 1],
])


investigation_matrix2 = np.array([
  #A1, A2, A3, A1A2, A1A3 A2A3, A1A2A3
  [3,  1,  2,   3,   3,    1,    2  ],
  [3/15, 1/15, 2/15, 3/15, 3/15, 1/15, 2/15],
  [3/15, 1/15, 2/15, 7/15, 8/15, 4/15, 1],
  [11/15, 7/15, 8/15, 13/15, 14/15, 12/15, 1],
])



# hàm tần suât
# input nxm
def frequency(matrices):
  result = []
  for row in matrices:
    total = np.sum(row)
    result.append(row/total)
  return np.array(result)



print(frequency(investigation_matrix))
