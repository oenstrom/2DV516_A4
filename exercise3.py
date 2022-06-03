import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.cm import get_cmap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
from scipy.io import arff
from exercise2 import sammon
from exercise1 import bkmeans
import pandas as pd

def vehicle():
    """"""
    data = np.array(arff.loadarff("data/vehicle.arff")[0].tolist())
    X, classes = np.array(data[:, :-1], dtype=np.float64), np.array(data[:, -1], dtype=str)
    y = np.unique(classes, return_inverse=True)[1]
    labels = ["Opel", "Saab", "Bus","Van"]
    cmap = get_cmap("Set2")
    norm = Normalize(vmin=0, vmax=3)

    Y_p = PCA(n_components=2).fit_transform(X)
    # plt.figure("Vehicle, PCA")
    plt.subplot(3, 3, 1)
    plt.title("Vehicle, PCA")
    for i, l in enumerate(labels):
        plt.scatter(Y_p[y==i, 0], Y_p[y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
    plt.legend()

    Y_t = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X)
    # plt.figure("Vehicle, t-SNE")
    plt.subplot(3, 3, 2)
    plt.title("Vehicle, t-SNE")
    for i, l in enumerate(labels):
        plt.scatter(Y_t[y==i, 0], Y_t[y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
    plt.legend()

    # Y_s = sammon(X, max_iter=50, epsilon=0.01, alpha=1, verbose=True)
    # # plt.figure("Vehicle, Sammon")
    # plt.subplot(3, 3, 3)
    # plt.title("Vehicle, Sammon")
    # for i, l in enumerate(labels):
    #     plt.scatter(Y_s[y==i, 0], Y_s[y==i, 1], c=cmap(norm(y[y==i])), label=l)
    # plt.legend()


def diabetes():
    data = np.array(arff.loadarff("data/diabetes.arff")[0].tolist())
    X, classes = np.array(data[:, :-1], dtype=np.float64), np.array(data[:, -1], dtype=str)
    y = np.unique(classes, return_inverse=True)[1]
    labels = ["Tested Negative", "Tested Positive"]
    cmap = get_cmap("Set2")
    norm = Normalize(vmin=0, vmax=1)


    # Y_p = PCA(n_components=2).fit_transform(X)
    # # plt.figure("Diabetes, PCA")
    # plt.subplot(3, 3, 4)
    # plt.title("Diabetes, PCA")
    # for i, l in enumerate(labels):
    #     plt.scatter(Y_p[y==i, 0], Y_p[y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
    # plt.legend()

    # Y_t = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X)
    # # plt.figure("Diabetes, t-SNE")
    # plt.subplot(3, 3, 5)
    # plt.title("Diabetes, t-SNE")
    # for i, l in enumerate(labels):
    #     plt.scatter(Y_t[y==i, 0], Y_t[y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
    # plt.legend()

    Y_s = sammon(X, max_iter=50, epsilon=0.01, alpha=0.3, verbose=True)
    plt.figure("Diabetes, Sammon")
    # plt.subplot(3, 3, 6)
    plt.title("Diabetes, Sammon")
    for i, l in enumerate(labels):
        plt.scatter(Y_s[y==i, 0], Y_s[y==i, 1], c=cmap(norm(y[y==i])), label=l)
    plt.legend()



def vowel():
    data = np.array(arff.loadarff("data/vowel.arff")[0].tolist())
    X, classes = np.array(data[:, 2:-1], dtype=np.float64), np.array(data[:, -1], dtype=str)
    labels, y = np.unique(classes, return_inverse=True)

    # labels = ["Andrew", "Bill", "David", "Mark", "Jo", "Kate", "Penny", "Rose", "Mike", "Nick", "Rich", "Tim", "Sarah", "Sue", "Wendy"]
    cmap = get_cmap("tab20")
    norm = Normalize(vmin=0, vmax=14)

    # Y_p = PCA(n_components=2).fit_transform(X)
    # # plt.figure("Vowel, PCA")
    # plt.subplot(3, 3, 7)
    # plt.title("Vowel, PCA")
    # for i, l in enumerate(labels):
    #     plt.scatter(Y_p[y==i, 0], Y_p[y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
    # plt.legend()

    Y_t = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X)

    Y = bkmeans(Y_t, k=len(labels), i=10)
    plt.figure()
    plt.title("Bisecting k-Means")
    plt.scatter(Y_t[:, 0], Y_t[:, 1], c=cmap(norm(Y)), marker=".")

    from sklearn.cluster import k_means
    Y = k_means(Y_t, n_clusters=len(labels), n_init=10)[1]
    plt.figure()
    plt.title("Classic k-Means")
    plt.scatter(Y_t[:, 0], Y_t[:, 1], c=cmap(norm(Y)), marker=".")

    from sklearn.cluster import AgglomerativeClustering
    Y = AgglomerativeClustering(n_clusters=len(labels)).fit_predict(Y_t)
    plt.figure()
    plt.title("Agglomerative Clustering")
    plt.scatter(Y_t[:, 0], Y_t[:, 1], c=cmap(norm(Y)), marker=".")



    # plt.figure("Vowel, t-SNE")
    # # plt.subplot(3, 3, 8)
    # plt.title("Vowel, t-SNE")
    # for i, l in enumerate(labels):
    #     plt.scatter(Y_t[y==i, 0], Y_t[y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
    # plt.legend()

    # Y_s = sammon(X, max_iter=50, epsilon=0.01, alpha=1, verbose=True)
    # # plt.figure("Vowel, Sammon")
    # plt.subplot(3, 3, 9)
    # plt.title("Vowel, Sammon")
    # for i, l in enumerate(labels):
    #     plt.scatter(Y_s[y==i, 0], Y_s[y==i, 1], c=cmap(norm(y[y==i])), label=l)
    # plt.legend()


# vehicle()
# diabetes()
vowel()
# plt.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=0.97, wspace=0.1, hspace=0.2)
plt.show()



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
