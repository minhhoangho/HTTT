import numpy as np
RI = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41] # Random Consistency Index


# Ma trận so sánh
def comparision_matrix(arr):
    for i in range(arr.shape[0]):
        arr[i][i] = 1
        for j in range(arr.shape[1]):
            if i > j:
                arr[i][j] = 1 / arr[j][i]
    return arr

def geometrix_mean(arr):
    temp = np.copy(arr)
    for j in range(arr.shape[1]):
        arr[:, j] /= sum(temp[:, j]) # từng phần tử chia sum từng cột
    return arr

def AHP(arr):
    n = np.shape(arr)[0]
    arr = comparision_matrix(arr)
    geometrix = geometrix_mean(np.copy(arr)) # matrix table 4
    row_avgs = np.mean(geometrix, axis=1) # trung bình từng hàng (matrix table 5)
    weighted_sum = np.sum(row_avgs * arr, axis=1) # sum của element wise multiple
    consistency = weighted_sum / row_avgs # chia từng phần tử
    lamb = np.mean(consistency) # lambda = trung bình cộng consistency
    CI = (lamb - n) / (n - 1) # Chỉ số nhất quán
    CR = CI / RI[n - 1] # Chỉ số thích hợp CR (điều kiện CR < 10% thì OK)
    return CR

if __name__ == "__main__":
    arr = np.array(
        [
            [float('nan'), 5, 2, 4],
            [float('nan'), float('nan'), 1/2, 1/2],
            [float('nan'), float('nan'), float('nan'), 2],
            [float('nan'), float('nan'), float('nan'), float('nan')]
        ])
    CR = AHP(arr, arr.shape[0])
    if CR < 0.1:
        print('Chỉ số CR = {} thỏa mãn điều kiện'.format(CR))
    else:
        print('Chỉ số CR = {} không thỏa mãn điều kiện'.format(CR))