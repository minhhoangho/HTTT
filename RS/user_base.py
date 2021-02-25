import numpy as np

def cosine_sim(u1, u2):
    return np.dot(u1, u2)/\
        np.sqrt(np.sum(np.square(u1))*np.sum(np.square(u2)))

def peason_sim(u1, u2):
    u1_mean = sum(u1)/np.count_nonzero(u1)
    u2_mean = sum(u2)/np.count_nonzero(u2)
    
    numerator = sum1 = sum2 = 0

    for i in range(len(u1)):
        if u1[i] > 0 and u2[i] > 0:
            numerator +=(u1[i] - u1_mean)*(u2[i] - u2_mean)
            sum1 +=np.square(u1[i] - u1_mean)
            sum2 +=np.square(u2[i] - u2_mean)
    return numerator / np.sqrt(sum1 * sum2)

# Lấy ví dụ là Steve sẽ rating cho phim Titanic
def knn(rating_arr, mean_ratring_arr, k, pos):
    '''
    rating_arr: Tập bộ dữ liệu tranining
    mean_rating_arr: Vector rating trung bình của các user 
    k: Số user lân cận được xem xét 
    pos: Vị trí cần dự đoán rating
    '''
    # Vị trí cần điền ở bảng rating, trong ví dụ tại hàng 3 cột 0

    length = rating_arr.shape[0]
    # Khai báo giá trị sim
    cosine = np.zeros((length, length))
    peason = np.zeros((length, length))

    for i in range(length):
        for j in range(i, length):
            if i == j:
                cosine[i][j] = 1.0
            else:
                # Cosine Similarity 
                cosine[i][j] = \
                    cosine_sim(rating_arr[i, :], rating_arr[j, :])
                
                # Peason Similarity 
                peason[i][j] = \
                    peason_sim(rating_arr[i, :], rating_arr[j, :])

    # Sử dụng mảng để lưu vị trí các tham số cần thiết khi sắp xếp giá trị sim 
    # Có thể sử dụng cấu trúc dữ liệu dictionary: {key: value}
    temp = np.zeros((rating_arr.shape[0], 3))
    temp[:, 0] = mean_ratring_arr[:]
    temp[:, 1] = rating_arr[:, pos[1]]
    temp[:, 2] = peason[:, pos[0]]
    r_mean = temp[pos[0], 0]
    # Sắp xếp giá trị Sim
    temp = temp[temp[:,2].argsort()]

    numerator = denominator = 0
    for i in range(k):
        numerator += temp[3-i][2]*(temp[3-i][1] - temp[3-i][0])
        denominator += temp[3-i][2]

    # Tính Rui
    return r_mean + numerator/denominator

def main():
    # Khởi tạo dữ liệu train 
    arr = np.array([a
    [1, 4, 5, None, 3],
    [5, 1, None, 5, 2],
    [4, 1, 2, 5, 0],
    [0, 3, 4, 0, 4]])  

    # Tính giá trị trung bình rating của các user 
    mean_rating_arr = [ 
        sum(filter(None, arr[0]))/len(list(filter(lambda e: e is not None, arr[0]))),
        sum(filter(None, arr[1]))/len(list(filter(lambda e: e is not None, arr[1]))),
        sum(filter(None, arr[2]))/len(list(filter(lambda e: e is not None, arr[2]))),
        sum(filter(None, arr[3]))/len(list(filter(lambda e: e is not None, arr[3]))),

    print(knn(arr, mean_rating_arr, 2, [3, 0]))

if __name__ == '__main__':
    main()
    