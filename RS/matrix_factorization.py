import numpy as np
import random


def cost_function(Y, H, W):
    cost = 0
    for idxRow, row in enumerate(Y):
        for idxCol, val in enumerate(row):
            if val != 0:
                cost += 0.5*- (val - np.dot(H[idxRow, :], W[:, idxCol])) ** 2
    return cost

def update_matrix(Y, H, W, learning_rate):
    for idxRow, row in enumerate(Y):
        for idxCol, val in enumerate(row):
            if val != 0:
                # Thực hiện cập nhật 2 ma trận H, W
                H[idxRow, :] = H[idxRow, :] + learning_rate * (val - np.dot(H[idxRow, :], W[:, idxCol])) * W[:, idxCol]
                W[:, idxCol] = W[:, idxCol] + learning_rate * (val - np.dot(H[idxRow, :], W[:, idxCol])) * H[idxRow, :]
    return H, W

def train(A, learning_rate, iter, k):
    num_of_row = A.shape[0]
    num_of_col = A.shape[1]
    H = np.random.rand(num_of_row, k)
    W = np.random.rand(k, num_of_col)
    cost_history = list()
    for i in range(iter):
        H, W = update_matrix(A, H, W, learning_rate)
        cost = cost_function(A, H, W)
        cost_history.append(cost)
    return H, W, cost_history


if __name__ == '__main__':
    B = np.random.rand(10, 5)
    for i in range(10):
        for j in range(5):
            B[i][j] = random.randint(1, 5)
    B = B.astype('int')
    print(B)
    print("---------------------------------------------")
    for i in range(5):
        col_num = random.randint(0, 4)
        row_num = random.randint(0, 9)
        B[row_num][col_num] = 0
    print(B)
    print("---------------------------------------------")

    H, W, cost_history = train(B, learning_rate=0.01, iter=3000, k=7)
    print(cost_history[-1])
    result = np.dot(H, W)
    print(result)
