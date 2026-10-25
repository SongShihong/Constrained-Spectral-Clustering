# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict

from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, lobpcg
import networkx as nx

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

import tqdm
from scipy.sparse.linalg import norm

from utils import *

# %%

k_nn = 3
n_clusters = 10
random_state = 81


# %%
import keras.datasets.mnist
print("Loading MNIST subset...")
(X, y), (test_x, _) = keras.datasets.mnist.load_data()


# %%

print("Building graphs...")
W_G = build_knn_graph(np.reshape(X, (len(X), -1)), k=3, use_gaussian=False, sigma=2)
G = nx.from_scipy_sparse_matrix(W_G)

# %% [markdown]
# # Spectral Clustering

# %%

deg_g = np.array(W_G.sum(axis=1)).ravel()
normalized_Lap_G = normalized_laplacian_from_D_sp(W_G, (deg_g))
sc_labels, _, _ = spectral_clustering(
    normalized_Lap_G, 
    n_clusters=n_clusters,
    random_state=random_state
)
sc_ari = adjusted_rand_score(y, sc_labels)
sc_nmi = normalized_mutual_info_score(y, sc_labels)
sc_cut = max(conductance_all_clusters(G, sc_labels).values())
print("SC ARI between clustering and true labels:", sc_ari)
print("SC NMI between clustering and true labels:", sc_nmi)
print("Spectral clustering cut: ", sc_cut)

# %% [markdown]
# # ICSC

# %%
aris = []
nmis = []
aris.append(sc_ari)
nmis.append(sc_nmi)
cut_gh = []
cut_g = []
cut_g.append(sc_cut)
cut_gh.append(None)
pred_labels = sc_labels
n_neighbors = 5
for i in range(10):
    W_H = build_cannot_link_graph(pred_labels, n_cl=n_neighbors + i, random_state=random_state)
    H = nx.from_scipy_sparse_matrix(W_H)
    pred_labels, vals, vecs = iterative_clustering(W_G, W_H, k = n_clusters, exact=False, eps = 0.95, random_seed=random_state)
    generalized_spectral_clustering_ari = adjusted_rand_score(y, pred_labels)
    generalized_spectral_clustering_nmi = normalized_mutual_info_score(y, pred_labels)
    print(f"{i+1}th iteration: Constrained Spectral Clustering ARI: {generalized_spectral_clustering_ari:.4f}")
    print(f"{i+1}th iteration: Constrained Spectral Clustering NMI: {generalized_spectral_clustering_nmi:.4f}")
    generalized_cutratio = max(ratiocut_all_clusters(G, H, pred_labels).values())
    print("cutratio ", generalized_cutratio)
    generalized_spectral_clustering_cut = max(conductance_all_clusters(G, pred_labels).values())
    print("Conductance ", generalized_spectral_clustering_cut)
    aris.append(generalized_spectral_clustering_ari)
    nmis.append(generalized_spectral_clustering_nmi)
    cut_g.append(generalized_spectral_clustering_cut)
    cut_gh.append(generalized_cutratio)

# %% [markdown]
# # Plot Results

# %%
import matplotlib.pyplot as plt

plt.plot(aris, label = "ARI")
plt.plot(nmis, label = "NMI")
plt.plot(cut_g, label = "conductance")
plt.plot(cut_gh, label = "cutratio")
plt.xlabel('Iteration')
plt.ylabel('ARI')
plt.title('ARI over Iterations')
plt.legend()
plt.grid()
plt.show()


