from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.datasets import make_blobs

def bkmeans(X:np.ndarray, k:int, i:int) -> np.ndarray:
    """Bisecting k-Means"""
    res = np.zeros(X.shape[0], dtype=int)
    for kk in range(1, k):
        values, counts = np.unique(res, return_counts=True)
        most_common_k = values[counts.argmax()]
        most_common_indices = res == most_common_k
        new_clusters = k_means(X[most_common_indices], 2, n_init=i)[1]
        res[most_common_indices] = np.where(new_clusters == 0, most_common_k, kk)
       
        fig = plt.figure()
        # ax = fig.add_subplot(projection="3d")
        # ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=res, cmap="Dark2")

        ax = fig.add_subplot()
        ax.scatter(X[:, 0], X[:, 1], c=res, cmap="Dark2")
    return res


X, y = make_blobs(n_samples=1000, centers=8, n_features=2)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap="Dark2")

c = bkmeans(X, 8, 10)

# plt.figure()
# plt.scatter(X[:, 0], X[:, 1], c=c, cmap="Dark2")


plt.show()