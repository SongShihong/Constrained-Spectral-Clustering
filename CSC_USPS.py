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

import matplotlib.pyplot as plt

from utils import *

# %%


k_nn = 3
n_clusters = 10
n_cl = 5
eps = 1
## when n_cl = 20, eps = 0.5 works well; otherwise try eps = 0.9

# %%
import h5py
with h5py.File("./datasets/usps/usps.h5", 'r') as hf:
    train = hf.get('train')
    X = train.get("data")[:]
    y = train.get('target')[:]


# %%

random_state = 63
print("random state = ",random_state)
W_G = build_knn_graph(np.reshape(X, (len(X), -1)), k=5, use_gaussian=False, sigma=2)
G = nx.from_scipy_sparse_matrix(W_G)


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
print("Spectral Clustering ARI between clustering and true labels:", sc_ari)
print("Spectral Clustering NMI between clustering and true labels:", sc_nmi)
print("Spectral clustering cut: ", sc_cut)


# %%

W_H = build_cannot_link_graph(y, n_cl=n_cl, random_state=random_state)
H = nx.from_scipy_sparse_matrix(W_H)
pred_labels, vals, vecs = iterative_clustering(W_G, W_H, k = n_clusters, exact=False, eps = eps, random_seed=random_state)
# print("vals   ", vals)
generalized_spectral_clustering_ari = adjusted_rand_score(y, pred_labels)
generalized_spectral_clustering_nmi = normalized_mutual_info_score(y, pred_labels)
print(f"Constrained Spectral Clustering ARI: {generalized_spectral_clustering_ari:.4f}")
print(f"Constrained Spectral Clustering NMI: {generalized_spectral_clustering_nmi:.4f}")
# generalized_cutratio = max(ratiocut_all_clusters(G, H, pred_labels).values())
# print("cutratio ", generalized_cutratio)
# generalized_spectral_clustering_cut = max(conductance_all_clusters(G, pred_labels).values())
# print("Conductance ", generalized_spectral_clustering_cut)

    # print(pred_labels)
    


