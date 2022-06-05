import numpy as np
import matplotlib.pyplot as plt
from sammon import sammon as ss
from sklearn.datasets import make_blobs, make_s_curve
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

def sammons_stress(in_X, out_X):
    """Calculate Sammon's Stress"""
    S = np.triu(in_X)
    d = np.triu(out_X)
    return (1 / np.sum(S)) * np.sum(np.divide(np.square(d - S), S, out=np.zeros_like(S), where=S!=0))

def sammon(X, max_iter=100, epsilon=0.01, alpha=0.3, verbose=False):
    """Sammon Mapping"""
    S = pairwise_distances(X)
    S = np.where(S==0, 1e-100, S)
    c = np.sum(np.triu(S))
    y_indices = range(X.shape[0])

    Y = make_blobs(n_samples=X.shape[0], n_features=2, centers=1, random_state=1337)[0]
    # Y = PCA(n_components=2, random_state=1).fit_transform(X)

    for t in range(max_iter):
        d = pairwise_distances(Y)
        d = np.where(d==0, 1e-100, d)
        E = sammons_stress(S, d)
        if verbose:
            print(f"Iter: {t}, E = {E}")
        if E < epsilon:
            print(f"Error threshold of {epsilon}, reached at iter {t}. E = {E}")
            break

        for i in y_indices:
            first = np.array([0, 0], dtype=np.float64)
            second = np.array([0, 0], dtype=np.float64)
            for j in y_indices:
                if j == i: continue
                first += ((S[i,j] - d[i,j]) / (d[i,j] * S[i,j])) * (Y[i] - Y[j])
                second += (1 / (S[i,j] * d[i,j])) * ( (S[i,j] - d[i,j]) - ((np.square(Y[i] - Y[j]) / d[i,j]) * (1 + ( (S[i,j] - d[i,j]) / d[i,j] ))) )
            
            Y[i] = Y[i] - alpha * ((-2/c)*first)/np.abs((-2/c)*second)
    return Y


def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

def fast_sammon(X, max_iter=100, epsilon=0.001, alpha=0.3, init="random", verbose=False):
    if init == "random":
        Y = make_blobs(n_samples=X.shape[0], n_features=2, centers=1, random_state=1337)[0]
    else:
        Y = PCA(n_components=2).fit_transform(X)

    S = pairwise_distances(X)
    S = np.where(S==0, 1e-150, S)

    c = np.sum(np.triu(S))
    l = S.shape[0]

    for t in range(max_iter):
        d = pairwise_distances(Y)
        d = np.where(d==0, 1e-150, d)

        E = sammons_stress(S, d)
        if verbose:
            print(f"Iter: {t}, E = {E}")
        if E < epsilon:
            print(f"Error threshold of {epsilon}, reached at iter {t}. E = {E}")
            break

        first_1_1d  = ((S-d)/(d*S)).reshape(l*l)
        second_1_1d, second_2_1d = (1/(S*d)).reshape(l*l), (S-d).reshape(l*l)
        second_3_1d, second_4_1d = d.reshape(l*l), (1 + ((S-d)/d)).reshape(l*l)
        for i in range(Y.shape[0]):
            start, end = i*l, (i*l)+l

            first_f1 = first_1_1d[start:end]
            first = (-2/c)*np.sum(np.c_[first_f1, first_f1] * (Y[i] - Y), axis=0)
            
            second_1, second_2 = second_1_1d[start:end], second_2_1d[start:end]
            second_3, second_4 = second_3_1d[start:end], second_4_1d[start:end]
            second = (-2/c)*np.sum(np.c_[second_1, second_1] * (np.c_[second_2, second_2] - ((np.square(Y[i] - Y)/second_3[:, None]) * second_4[:, None])), axis=0)

            Y[i] = Y[i] - (alpha * (first/np.abs(second)))
    return Y


def main():
    X, y = make_s_curve(1000, random_state=1)
    # Y = sammon(X, max_iter=50, epsilon=0.023, alpha=1, verbose=True)

    # X, y = make_blobs(2, n_features=3, centers=1, random_state=1)
    # print(X)
    # Y = sammon(X, max_iter=1, epsilon=0.001, alpha=0.3, verbose=False)
    # print(Y)
    # print("-------------------------")
    Y = fast_sammon(X, max_iter=200, epsilon=0.001, alpha=0.1, verbose=True)
    # print(Y_new)
    # exit()

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(X[:,0], X[:,1], X[:,2])

    plt.figure()
    plt.scatter(Y[:,0], Y[:,1])

    # X_sammon = ss(X, 2)
    # X_sammon = X_sammon[0]
    # plt.figure()
    # plt.scatter(X_sammon[:,0], X_sammon[:,1])

    plt.show()

def load_data(file_path, x_lower, x_upper, y_pos):
    data = np.array(arff.loadarff(file_path)[0].tolist())
    X, classes = np.array(data[:, x_lower:x_upper], dtype=np.float64), np.array(data[:, y_pos], dtype=str)
    labels, y = np.unique(classes, return_inverse=True)
    return (X, y, labels)

def plot(X_pca, X_tsne, X_sammon, y, plt_i, labels, title):
    min_y, max_y = min(y), max(y)
    norm = Normalize(vmin=min_y, vmax=max_y)
    cmap = get_cmap("Set3")
    for r, X in enumerate([[X_pca, "PCA"], [X_tsne, "t-SNE"], [X_sammon, "Sammon"]]):
        plt.subplot(3, 3, plt_i + r)
        plt.title(f"{title} | {X[1]}")
        for i, l in enumerate(labels):
            plt.scatter(X[0][y==i, 0], X[0][y==i, 1], c=cmap(norm(y[y==i])), label=l, marker=".")
        plt.legend()

if __name__ == "__main__":
    # main()
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    import warnings
    from matplotlib.cm import get_cmap
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np
    from scipy.io import arff
    warnings.filterwarnings("ignore")

    X_v, y_v, labels_v    = load_data("data/vehicle.arff", 0, -1, -1)
    X_d, y_d, labels_d    = load_data("data/diabetes.arff", 0, -1, -1)
    X_vo, y_vo, labels_vo = load_data("data/vowel.arff", 2, -1, -1)


    Y_v_p = PCA(n_components=2).fit_transform(X_v)
    Y_v_t = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X_v)
    Y_v_s = fast_sammon(X_v, max_iter=200, epsilon=0.005, alpha=1, verbose=True)

    Y_d_p = PCA(n_components=2).fit_transform(X_d)
    Y_d_t = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X_d)
    Y_d_s = fast_sammon(X_d, max_iter=200, epsilon=0.013, alpha=1, verbose=True)

    Y_vo_p = PCA(n_components=2).fit_transform(X_vo)
    Y_vo_t = TSNE(n_components=2, init="pca", learning_rate="auto").fit_transform(X_vo)
    Y_vo_s = fast_sammon(X_vo, max_iter=200, epsilon=0.065, alpha=0.9, verbose=True)

    plt.figure(figsize=(24, 12))
    plot(Y_v_p, Y_v_t, Y_v_s, y_v, 1, labels_v, "Vehicle")
    plot(Y_d_p, Y_d_t, Y_d_s, y_d, 4, labels_d, "Diabetes")
    plot(Y_vo_p, Y_vo_t, Y_vo_s, y_vo, 7, labels_vo, "Vowel")
    plt.subplots_adjust(left=0.025, bottom=0.025, right=0.99, top=0.97, wspace=0.1, hspace=0.2)
    plt.show()