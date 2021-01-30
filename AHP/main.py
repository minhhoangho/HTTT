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
        arr[:, j] /= sum(temp[:, j])
    return arr

def AHP(arr):
    n = np.shape(arr)[0]
    arr = comparision_matrix(arr)
    geometrix = geometrix_mean(np.copy(arr))
    row_avgs = np.mean(geometrix, axis=1)
    weighted_sum = np.sum(row_avgs * arr, axis=1)
    consistency = weighted_sum / row_avgs
    lamb = np.mean(consistency)
    CI = (lamb - n) / (n - 1)
    CR = CI / RI[n - 1]
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