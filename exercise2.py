import numpy as np
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

def distance_matrix(X1, X2, metric="euclidean"):
    return np.triu(pairwise_distances(X1, X2, metric=metric))

def sammon_stress(in_X, out_X):
    S = distance_matrix(in_X, in_X)
    d = distance_matrix(out_X, out_X)
    return (1 / np.sum(S)) * np.sum(np.divide(np.square(d - S), S, out=np.zeros_like(S), where=S!=0))

def gradient(c, in_dist, Y):
    out_dist = distance_matrix(Y, Y)
    out_dist_city = distance_matrix(Y, Y, "cityblock")
    
    num1 = in_dist - out_dist
    den1 = out_dist*in_dist
    first = (-2/c) * np.sum(np.divide(num1, den1, out=np.zeros_like(num1), where=den1!=0) * out_dist_city)
    den2 = in_dist*out_dist
    second = (-2/c) * np.sum(
        np.divide(1, den2, out=np.zeros_like(den2), where=den2!=0)
        * (in_dist - out_dist)
        - np.divide(np.square(out_dist_city), out_dist, out=np.zeros_like(out_dist_city), where=out_dist!=0)
        *(1 + np.divide(num1, out_dist, out=np.zeros_like(num1), where=out_dist!=0)))
    return first / second


def gradi(i, X, Y):
    S = distance_matrix(X, X)
    d = distance_matrix(Y, Y)
    c = np.sum(S)
    first = np.array([0, 0], dtype=np.float64)
    second = np.array([0, 0], dtype=np.float64)
    for j in range(len(Y)):
        if j == i:
            continue

        # print(first)
        calc = (Y[i] - Y[j]) * (S[i, j] - d[i, j]) / (d[i, j] * S[i, j])

        calc2 = (1/(S[i, j]*d[i, j])) * ((S[i, j] - d[i, j]) - (np.square(Y[i] - Y[j]) / d[i, j]) * (1 + (S[i, j] - d[i, j])/d[i, j]) )
        # print(calc)
        first += calc
        second += calc2
        # print(first)
    print("=======================")
    print(first)
    print((-2/c)*first)
    print("=======================")
    print(second)
    print((-2/c)*second)
    return (first / second)


def sammon(X, max_iter=100, epsilon=0.05, alpha=0.5):
    """Sammon Mapping"""
    n = X.shape[0]
    Y = make_blobs(n_samples=n, n_features=2, centers=1, random_state=1337)[0]
    in_dist = distance_matrix(X, X)
    c = np.sum(in_dist)
    for it in range(max_iter):
        out_dist = distance_matrix(Y, Y)
        # first = np.array()
        for i in range(Y.shape[0]):
            # Y[i] = Y[i] - alpha * gradi(i, X, Y)
            print(gradi(i, X, Y))
            # FIX DIVIDE BY ZERO!


                # print( ((in_dist[i, j] - out_dist[i, j]) / (out_dist[i, j] * in_dist[i, j])) * (Y[i] - Y[j]) )
                # print((in_dist[i, j] - out_dist[i, j]) * (Y[i] - Y[j]))
        exit()

        # Y = Y - alpha*gradient(c, in_dist, Y)
        # E = sammon_stress(X, Y)
        # print("E:", E)
        # if E < epsilon:
        #     break
    print(E)


X, y = make_blobs(n_samples=10, n_features=10, random_state=1)
Y = PCA(n_components=2).fit_transform(X)
print(X)
print(Y)
print(sammon_stress(X, Y))
exit()

X = make_blobs(n_samples=3, n_features=3, centers=1, random_state=2)[0]
Y = np.array([[3, 10], [4, 7], [1, 4]])

# (3*(3-1))/2

# print(Y[0] - Y[1])

# print(Y)
# print(pairwise_distances(Y, Y, metric="euclidean"))

# print("-----------------------")
# for y1 in Y:
#     for y2 in Y:
#         if np.array_equal(y1, y2):
#             continue
#         print(y1 - y2)


# print( np.array([]) )

# [[[0, 0, 0], [-1, 3], []] 
#  [0.         0.         4.24264069] 
#  [0.         0.         0.        ]]

# exit()


# print(distance_matrix(X, X, metric=lambda i, j: i))

# print("------------------------")
# print(distance_matrix(X, X, "cityblock"))
# print("------------------------")

# print(np.subtract(X, X))
# exit()

print(distance_matrix(X, X))
print(np.sum(distance_matrix(X, X)))
print("-----------")
sammon(X)