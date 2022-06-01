import numpy as np
import matplotlib.pyplot as plt
from sammon import sammon as ss
from sklearn.datasets import make_blobs, make_s_curve
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA

def distance_matrix(X1, X2, metric="euclidean"):
    return pairwise_distances(X1, X2, metric=metric)

def sammon_stress(in_X, out_X):
    """Calculate Sammon's Stress"""
    S = np.triu(in_X)
    d = np.triu(out_X)
    return (1 / np.sum(S)) * np.sum(np.divide(np.square(d - S), S, out=np.zeros_like(S), where=S!=0))


def gradient(i, Y, S, d):
    """Gradient for Sammon Mapping"""
    c = np.sum(np.triu(S))
    first = np.array([0, 0], dtype=np.float64)
    second = np.array([0, 0], dtype=np.float64)
    y_indices = list(range(Y.shape[0]))
    y_indices.remove(i)
    for j in y_indices:
        first += ((S[i,j] - d[i,j]) / (d[i,j] * S[i,j])) * (Y[i] - Y[j])
        second += (1 / (S[i,j] * d[i,j])) * ( (S[i,j] - d[i,j]) - ((np.square(Y[i] - Y[j]) / d[i,j]) * (1 + ( (S[i,j] - d[i,j]) / d[i,j] ))) )
    return ((-2/c)*first)/np.abs((-2/c)*second)

def sammon(X, max_iter=100, epsilon=0.01, alpha=0.3, verbose=False):
    """Sammon Mapping"""
    S = distance_matrix(X, X)
    S = np.where(S==0, 1e-100, S)
    Y = make_blobs(n_samples=X.shape[0], n_features=2, centers=1, random_state=1337)[0]
    # Y = PCA(n_components=2, random_state=1).fit_transform(X)

    for t in range(max_iter):
        d = distance_matrix(Y, Y)
        d = np.where(d==0, 1e-100, d)
        E = sammon_stress(S, d)
        if verbose:
            print(f"Iter: {t}, E = {E}")
        if E < epsilon:
            print(f"Error threshold of {epsilon}, reached at iter {t}. E = {E}")
            break

        for i in range(Y.shape[0]):
            Y[i] = Y[i] - alpha * gradient(i, Y, S, d)
    return Y


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