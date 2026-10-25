# %%
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict

from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.linalg import eigh
import networkx as nx

from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score

import time

import tqdm
from scipy.sparse.linalg import norm

from utils import *

# %%


# Parameters
n_samples = 60000        # keep this modest for spectral methods
k_nn = 3
n_cl = 20
n_clusters = 10
random_state = int((time.time_ns() / 100000) % 100)



# %%
import keras.datasets.mnist

print("Loading MNIST subset...")
# X, y = load_mnist_subset(n_samples=n_samples, random_state=random_state)
# X, y = load_mnist_subset(n_samples=n_samples, random_state=random_state)

(X, y), (test_x, _) = keras.datasets.mnist.load_data()


# %%

print("Building graphs...")
W_G = build_knn_graph(np.reshape(X, (len(X), -1)), k=3, use_gaussian=False, sigma=2)
G = nx.from_scipy_sparse_matrix(W_G)

# %%
# found_clusters = sgtl.clustering.spectral_clustering(graph,n_clusters, 10)
# sc_ari = pysc.evaluation.adjusted_rand_index(y, found_clusters)
# sc_nmi = pysc.evaluation.normalized_mutual_info(y, found_clusters)
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
print("Spectral clustering ARI between clustering and true labels:", sc_ari)
print("Spectral clustering NMI between clustering and true labels:", sc_nmi)
print("Spectral clustering cut: ", sc_cut)

# %%

W_H = build_cannot_link_graph(y, n_cl=n_cl, random_state=random_state)
H = nx.from_scipy_sparse_matrix(W_H)


# %%
pred_labels, vals, vecs = iterative_clustering(W_G, W_H, k = n_clusters, exact=False, eps = 0.7, random_seed=random_state)
# print("vals   ", vals)


# %%
generalized_spectral_clustering_ari = adjusted_rand_score(y, pred_labels)
generalized_spectral_clustering_nmi = normalized_mutual_info_score(y, pred_labels)
print(f"Constrained Spectral Clustering ARI: {generalized_spectral_clustering_ari:.4f}")
print(f"Constrained Spectral Clustering NMI: {generalized_spectral_clustering_nmi:.4f}")
generalized_cutratio = max(ratiocut_all_clusters(G, H, pred_labels).values())
print("cutratio ", generalized_cutratio)
generalized_spectral_clustering_cut = max(conductance_all_clusters(G, pred_labels).values())
print("Conductance ", generalized_spectral_clustering_cut)
    # print(pred_labels)


