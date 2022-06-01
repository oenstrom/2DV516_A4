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

def main():
    X, y = make_s_curve(300, random_state=1)
    Y = sammon(X, max_iter=50, epsilon=0.023, alpha=1, verbose=True)

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