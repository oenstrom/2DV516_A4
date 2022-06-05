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

def fast_sammon(X, max_iter=100, epsilon=0.001, alpha=0.3, verbose=False):
    Y = make_blobs(n_samples=X.shape[0], n_features=2, centers=1, random_state=1337)[0]
    Y = PCA(n_components=2, random_state=1).fit_transform(X)
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

        d_1d = d.reshape(l*l)
        first_first_1d = ((S-d)/(d*S)).reshape(l*l)
        second_last_1d = (1 + ((S-d)/d))
        # np.fill_diagonal(second_last_1d, 0)
        second_last_1d = second_last_1d.reshape(l*l)
        second_first_1d = (1/(S*d)).reshape(l*l)
        second_mid_1d = (S-d).reshape(l*l)
        for i in range(Y.shape[0]):
            first_first_M = first_first_1d[(i*l):(i*l)+l]
            d_temp = d_1d[(i*l):(i*l)+l]
            first_M = second_first_1d[(i*l):(i*l)+l]
            mid_M = second_mid_1d[(i*l):(i*l)+l]

            first = (-2/c)*np.sum(np.c_[first_first_M, first_first_M] * (Y[i] - Y), axis=0)
            second = (-2/c)*np.sum(np.c_[first_M, first_M] * (np.c_[mid_M, mid_M] - ((np.square(Y[i] - Y)/d_temp[:, None]) * second_last_1d[(i*l):(i*l)+l][:, None])), axis=0)

            # (1/(S*d))     (S-d) - (np.square(Y[i] - Y)/d_temp[:, None]) * second_last_1d[(i*l):(i*l)+l][:, None]    # Maybe remove 1 from the diagonal!!!!
            Y[i] = Y[i] - (alpha * (first/np.abs(second)))
    return Y

def gradient_descent(X, y, a = 0.01, n = 1000):
    """Perform gradient descent on the given X"""
    w = np.zeros(X.shape[1])
    for _ in range(n):
        j = (X.T).dot(X.dot(w) - y)
        w = w - (a * j) / X.shape[0]
    return w

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

if __name__ == "__main__":
    main()