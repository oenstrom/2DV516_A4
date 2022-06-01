import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.io import arff
from exercise2 import sammon
import pandas as pd

def vehicle():
    """"""
    data = np.array(arff.loadarff("data/vehicle.arff")[0].tolist())
    X, labels = np.array(data[:, :-1], dtype=np.float64), np.array(data[:, -1], dtype=str)
    y = np.unique(labels, return_inverse=True)[1]

    Y_p = PCA(n_components=2).fit_transform(X)
    plt.figure()
    plt.title("PCA")
    plt.scatter(Y_p[:,0], Y_p[:, 1], c=y, cmap="Dark2", marker=".")
    # plt.show()

    # for p in [10, 20, 30, 40, 50]:
    Y_t = TSNE(n_components=2, learning_rate=500).fit_transform(X)
    plt.figure()
    plt.title("t-SNE")
    plt.scatter(Y_t[:,0], Y_t[:, 1], c=y, cmap="Dark2")

    # Y = sammon(X, max_iter=50, epsilon=0.01, alpha=1, verbose=True)
    # plt.figure()
    # plt.title("Sammon")
    # plt.scatter(Y[:,0], Y[:,1], c=y, cmap="Dark2", marker=".")
    plt.show()


vehicle()
# data = np.array(arff.loadarff("data/diabetes.arff")[0].tolist())
# X, y = np.array(data[:, :-1], dtype=np.float64), np.array(data[:, -1], dtype=str)
# y = np.where(y=="tested_positive", 1, 0)


# Y_p = PCA(n_components=2).fit_transform(X)
# plt.figure()
# plt.title("PCA")
# plt.scatter(Y_p[:,0], Y_p[:, 1], c=y, cmap=ListedColormap(["#990000", "#009900"]), marker=".")


# for p in [10, 20, 30, 40, 50]:
#     Y_t = TSNE(n_components=2, perplexity=30).fit_transform(X)
#     plt.figure()
#     plt.title("t-SNE")
#     plt.scatter(Y_t[:,0], Y_t[:, 1], c=y, cmap=ListedColormap(["#990000", "#009900"]))

# Y = sammon(X, max_iter=50, epsilon=0.01, alpha=0.3, verbose=True)
# plt.figure()
# plt.title("Sammon")
# plt.scatter(Y[:,0], Y[:,1], c=y, cmap=ListedColormap(["#990000", "#009900"]), marker=".")


# data = pd.read_csv("data/covid.csv", keep_default_na=True).to_numpy(na_value="nan")

# print(data.shape)
# data = data[~np.any(np.equal(data, "nan"), axis=1), 1:]
# print(data)
# print(data.shape)

# data = data[np.char.endswith(np.array(data[:, 2], dtype='<U3'), '21')]
# print(data)
# print(data.shape)

# plt.show()
