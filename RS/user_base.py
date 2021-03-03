import numpy as np

def cosine_sim(u1, u2):
 
    return np.dot(u1, u2)/np.sqrt(np.sum(np.square(u1))*np.sum(np.square(u2)))

def peason_sim(u1, u2):
    u1_mean = sum(u1)/np.count_nonzero(u1) # Calculate mean of voting
    u2_mean = sum(u2)/np.count_nonzero(u2) # Calculate mean of voting
    
    numerator = sum1 = sum2 = 0
    # num_features (num of movies)
    num_features = len(u1)

    normalized_u1 = np.zeros_like(u1)
    normalized_u2 = np.zeros_like(u2)
    for i in range(num_features):
        normalized_u1[i] = (u1[i] - u1_mean)
        normalized_u2[i] = (u2[i] - u2_mean)

    normalized_u1_square = np.square(normalized_u1)
    normalized_u2_square = np.square(normalized_u2)

    return np.dot(normalized_u1, normalized_u2)/np.sqrt(np.sum(normalized_u1_square)*np.sum(normalized_u2_square))

# Example Steve is rating for Titanic
def knn(rating_arr, mean_ratring_arr, k, pos):
    """
    rating_arr: Training data
    mean_rating_arr:  Mean vector training of users
    k: total user
    pos: position to predict
    """

    # Replace None with zero
    for idr, row in enumerate(rating_arr):
        for idc, item in enumerate(row):
            if item == None:
                rating_arr[idr][idc] = 0
    
    rating_arr = np.array(rating_arr)
    length = rating_arr.shape[0]
    # Declare sim value
    cosine = np.zeros((length, length))
    peason = np.zeros((length, length))

    for i in range(length):
        for j in range(i, length):
            if i == j:
                cosine[i][j] = 1.0
            else:
                # Cosine Similarity (between user i and user j)
                cosine[i][j] = cosine_sim(rating_arr[i, :], rating_arr[j, :]) 
                # user i with all movie and user j with all movie

                # Peason Similarity (between user i and user j)
                peason[i][j] = peason_sim(rating_arr[i, :], rating_arr[j, :])
                # user i with all movie and user j with all movie


    print("Cosine similarity: ")
    print(cosine)
    print("Peason similarity: ")
    print(peason)

    print("-----------")
    print()
    # We can use array to store params when ordering sim val
    # Dictionary: {key: value} is posible
    temp = np.zeros((rating_arr.shape[0], 3))
    temp[:, 0] = mean_ratring_arr[:]
    temp[:, 1] = rating_arr[:, pos[1]]
    temp[:, 2] = peason[:, pos[0]]
    r_mean = temp[pos[0], 0]
    # Order sim val
    temp = temp[temp[:,2].argsort()]

    numerator = denominator = 0
    for i in range(k):
        numerator += temp[3-i][2]*(temp[3-i][1] - temp[3-i][0])
        denominator += temp[3-i][2]
        # Calculate Rui index
    return r_mean + numerator/denominator

def main():
    # Init train data
    # Row: users
    # Col: movie
    arr = np.array([
    [1, 4, 5, None, 3],
    [5, 1, None, 5, 2],
    [4, 1, 2, 5, None],
    [None, 3, 4, None, 4]
    ])  
    

    # Find mean rating of users
    mean_rating_arr = [ 
        sum(filter(None, arr[0]))/len(list(filter(lambda e: e is not None, arr[0]))),
        sum(filter(None, arr[1]))/len(list(filter(lambda e: e is not None, arr[1]))),
        sum(filter(None, arr[2]))/len(list(filter(lambda e: e is not None, arr[2]))),
        sum(filter(None, arr[3]))/len(list(filter(lambda e: e is not None, arr[3]))),
    ]
    print(knn(arr, mean_rating_arr, 2, [3, 0]))

if __name__ == '__main__':
    main()
    