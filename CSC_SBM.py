# %%
import os
# os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import scipy as sp
import tqdm
import torch

import warnings
warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, lobpcg
from collections import defaultdict
from sklearn.metrics import normalized_mutual_info_score

import time

from utils import *

# %%
num_block = 10
block_size = 500
p = 0.2
q = 0.155

# %%

random_state = int((time.time_ns() / 100000) % 100)
G, H = SBM(block_size=block_size, k=num_block, p=p, q=q)

y = [G.nodes[i]['block'] for i in G.nodes()]
print("random state = ",random_state)
# W_G = build_knn_graph(np.reshape(X, (len(X), -1)), k=3, use_gaussian=False, sigma=2)
W_G = nx.to_scipy_sparse_array(G)
W_H = nx.to_scipy_sparse_array(H)
# G = nx.from_scipy_sparse_matrix(W_G)

deg_g = np.array(W_G.sum(axis=1)).ravel()
normalized_Lap_G = normalized_laplacian_from_D_sp(W_G, (deg_g))
sc_labels, _, _ = spectral_clustering(
    normalized_Lap_G, 
    n_clusters=num_block,
    random_state=random_state
)
sc_ari = adjusted_rand_score(y, sc_labels)
sc_nmi = normalized_mutual_info_score(y, sc_labels)
sc_cut = max(conductance_all_clusters(G, sc_labels).values())
print("Spectral Clustering Results:")
print("ARI between clustering and true labels:", sc_ari)
print("NMI between clustering and true labels:", sc_nmi)
print("Spectral clustering cut: ", sc_cut)

aris = []
nmis = []
aris.append(sc_ari)
nmis.append(sc_nmi)
cut_gh = []
cut_g = []
cut_g.append(sc_cut)
cut_gh.append(None)
pred_labels = sc_labels

    # W_H = build_cannot_link_graph(pred_labels, n_cl=min(n_neighbors + i, neighbor_max), random_state=random_state)
    # H = nx.from_scipy_sparse_matrix(W_H)
pred_labels, vals, vecs = iterative_clustering(W_G, W_H, k = num_block, exact=False, eps = 0.9 ,random_seed=random_state)
# print("vals   ", vals)
generalized_spectral_clustering_ari = adjusted_rand_score(y, pred_labels)
generalized_spectral_clustering_nmi = normalized_mutual_info_score(y, pred_labels)
print(f"Constrained Spectral Clustering ARI: {generalized_spectral_clustering_ari:.4f}")
print(f"Constrained Spectral Clustering NMI: {generalized_spectral_clustering_nmi:.4f}")
    



