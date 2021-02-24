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
  [6,4,5]
  )

# Với mỗi tiêu chí, khảo sát 15 người -> xác định độ yêu thích tiêu chí

investigation_matrix = np.array([
  #A1, A2, A3, A1A2, A1A3 A2A3, A1A2A3
  [5,  2,  3,   4,   0,    0,    1  ],
  [2,  1,  2,   3,   3,    1,    2  ]
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
