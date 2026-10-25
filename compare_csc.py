"""
Comparison of Spectral Clustering methods on SBM, varying k from 2 to 10.

Methods:
  1. Unconstrained Spectral Clustering (SC)
  2. Wang et al. (2014) Algorithm 2 — Constrained Spectral Clustering
  3. Our Constrained SC (iterative_clustering / generalized eigenproblem)
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import networkx as nx
import scipy.linalg as sla
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils import (
    SBM,
    spectral_clustering,
    iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Wang et al. (2014) Algorithm 2 ──────────────────────────────────────────
def wang_algorithm2(A, Q, beta, K, n_extra=20, random_state=0, eps=1e-10):
    """
    Wang et al. (2014) Algorithm 2 — Constrained Spectral Clustering (K-way).
    Solves: L̄ v = λ (Q̄ − β/vol · I) v
    """
    N = A.shape[0]

    # Step 1
    vol        = float(A.sum())
    deg        = np.array(A.sum(axis=1)).ravel()
    inv_sqrt   = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = diags(inv_sqrt)

    # Step 2
    I_sp  = eye(N, format='csr')
    L_bar = I_sp - D_inv_sqrt @ A @ D_inv_sqrt
    Q_bar = D_inv_sqrt @ Q @ D_inv_sqrt

    # Step 3 — top eigenvalues of Q̄
    n_q = min(K - 1 + n_extra, N - 2)
    q_vals, q_vecs = eigsh(Q_bar, k=n_q, which='LM')
    order  = np.argsort(q_vals)[::-1]
    q_vals = q_vals[order];  q_vecs = q_vecs[:, order]

    lambda_K1 = q_vals[K - 2]   # (K−1)-th largest

    # Step 4 — feasibility
    if beta >= lambda_K1 * vol:
        raise ValueError(
            f"Infeasible: β={beta:.4f} >= λ_{{K-1}}·vol={lambda_K1*vol:.4f}"
        )

    # Steps 5-7 — Rayleigh-Ritz in Q̄'s positive eigenspace
    threshold = beta / vol
    pos_mask  = q_vals > threshold
    V         = q_vecs[:, pos_mask]
    m_diag    = q_vals[pos_mask] - threshold

    if V.shape[1] < K - 1:
        raise ValueError(f"Only {V.shape[1]} feasible Ritz vectors, need K-1={K-1}")

    L_sub = V.T @ (L_bar @ V)
    M_sub = np.diag(m_diag)
    sub_vals, sub_vecs = sla.eigh(L_sub, M_sub)

    pos_idx = np.where(sub_vals > 1e-8)[0]
    if len(pos_idx) < K - 1:
        pos_idx = np.argsort(sub_vals)[-(K - 1):]

    best_k1 = np.argsort(sub_vals[pos_idx])[:K - 1]
    sel_idx = pos_idx[best_k1]
    V_star  = V @ sub_vecs[:, sel_idx]

    norms  = np.linalg.norm(V_star, axis=0, keepdims=True)
    V_star = V_star / np.maximum(norms, eps) * np.sqrt(vol)

    # Step 8 — K-means on D^{-½} V*
    embedding = D_inv_sqrt @ V_star
    km     = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    labels = km.fit_predict(embedding)
    return labels


# ── Experiment parameters ────────────────────────────────────────────────────
BLOCK_SIZE   = 500
P_INTRA      = 0.20
Q_INTER      = 0.155
RANDOM_STATE = 42
K_RANGE      = range(2, 11)

print(f"SBM: block_size={BLOCK_SIZE}, p={P_INTRA}, q={Q_INTER}, seed={RANDOM_STATE}")
print(f"Varying k from {K_RANGE.start} to {K_RANGE.stop - 1}\n")

rows = []

for k in K_RANGE:
    G, H = SBM(block_size=BLOCK_SIZE, k=k, p=P_INTRA, q=Q_INTER)
    y    = [G.nodes[i]['block'] for i in G.nodes()]
    W_G  = nx.to_scipy_sparse_array(G)
    W_H  = nx.to_scipy_sparse_array(H)

    deg_g = np.array(W_G.sum(axis=1)).ravel()

    # 1. Unconstrained Spectral Clustering
    L_G = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_labels, _, _ = spectral_clustering(L_G, n_clusters=k, random_state=RANDOM_STATE)
    sc_ari = adjusted_rand_score(y, sc_labels)
    sc_nmi = normalized_mutual_info_score(y, sc_labels)

    # 2. Wang et al. (2014) Algorithm 2
    Q_wang = (W_G - W_H).astype(float)
    try:
        wang_labels = wang_algorithm2(
            A=W_G, Q=Q_wang, beta=0.0, K=k,
            n_extra=20, random_state=RANDOM_STATE
        )
        wang_ari = adjusted_rand_score(y, wang_labels)
        wang_nmi = normalized_mutual_info_score(y, wang_labels)
    except Exception as e:
        wang_ari = wang_nmi = float('nan')
        print(f"  [Wang k={k}] {e}")

    # 3. Our Constrained SC
    our_labels, _, _ = iterative_clustering(
        W_G, W_H, k=k, exact=False, eps=0.9, random_seed=RANDOM_STATE
    )
    our_ari = adjusted_rand_score(y, our_labels)
    our_nmi = normalized_mutual_info_score(y, our_labels)

    rows.append((k, sc_ari, sc_nmi, wang_ari, wang_nmi, our_ari, our_nmi))
    print(f"k={k:2d} done")

# ── Results table ─────────────────────────────────────────────────────────────
header_top = (
    f"{'':>3} | {'Spectral Clustering':^20} | {'Wang et al. (2014)':^20} | {'Our CSC':^20}"
)
header_sub = (
    f"{'k':>3} | {'ARI':>9} {'NMI':>9} | {'ARI':>9} {'NMI':>9} | {'ARI':>9} {'NMI':>9}"
)
sep = "-" * len(header_sub)

print("\n" + "=" * len(header_sub))
print(header_top)
print(header_sub)
print(sep)
for k, sc_ari, sc_nmi, wang_ari, wang_nmi, our_ari, our_nmi in rows:
    print(
        f"{k:>3} | {sc_ari:>9.4f} {sc_nmi:>9.4f} | "
        f"{wang_ari:>9.4f} {wang_nmi:>9.4f} | "
        f"{our_ari:>9.4f} {our_nmi:>9.4f}"
    )
print("=" * len(header_sub))
