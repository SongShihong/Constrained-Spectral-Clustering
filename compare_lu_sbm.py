"""
Comparison on SBM-generated graphs — all four methods, k varied from 2 to 10.

Graph construction
------------------
  W_G : within-block edges  (similarity graph,  prob p inside, q between)
  W_H : between-block edges (cannot-link graph, prob q inside, p between)
Both graphs are taken directly from the SBM; no k-NN or CL sampling needed.

Methods
-------
  1. Unconstrained Spectral Clustering (SC)
  2. Wang et al. (2014) Algorithm 2
  3. Lu & Carreira-Perpinan (2008) Algorithm A
  4. Our Constrained SC (iterative_clustering)

Notes on calibration
--------------------
  Wang Q : Q = W_G − scale·W_H, scale = W_G.sum()/W_H.sum().
            Balances the graphs before subtracting so that Q has a clear
            positive-eigenvalue structure for the Wang feasibility step.

  Lu eps_c: auto-calibrated per k so that M is comparable to K^{-1}.
            avg H-degree grows with k (more inter-block edges), so
            eps_c = sqrt(avg_deg_H / 7.4) is recomputed for each k.
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils import (
    SBM,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Fixed parameters ──────────────────────────────────────────────────────────
BLOCK_SIZE   = 200     # N = k * BLOCK_SIZE; kept fixed so only k changes
P_INTRA      = 0.2     # within-block edge probability  (from CSC_SBM.py)
Q_INTER      = 0.13   # between-block edge probability (from CSC_SBM.py)
RANDOM_STATE = 42
EPS_CSC      = 0.9     # eps for our method (from CSC_SBM.py)
K_RANGE      = range(2, 11)   # k = 2, 3, ..., 10


# ── Wang et al. (2014) — Algorithm 2 ──────────────────────────────────────────
def wang_algorithm2(A, Q, beta, K, n_extra=20, random_state=0, eps=1e-10):
    N = A.shape[0]
    vol        = float(A.sum())
    deg        = np.array(A.sum(axis=1)).ravel()
    inv_sqrt   = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = diags(inv_sqrt)

    I_sp  = eye(N, format='csr')
    L_bar = I_sp - D_inv_sqrt @ A @ D_inv_sqrt
    Q_bar = D_inv_sqrt @ Q @ D_inv_sqrt

    n_q = min(K - 1 + n_extra, N - 2)
    q_vals, q_vecs = eigsh(Q_bar, k=n_q, which='LM')
    order  = np.argsort(q_vals)[::-1]
    q_vals = q_vals[order];  q_vecs = q_vecs[:, order]

    lambda_K1 = q_vals[K - 2]
    if beta >= lambda_K1 * vol:
        raise ValueError(
            f"Infeasible: β={beta:.4f} >= λ_{{K-1}}·vol={lambda_K1*vol:.4f}"
        )

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

    norms   = np.linalg.norm(V_star, axis=0, keepdims=True)
    V_star  = V_star / np.maximum(norms, eps) * np.sqrt(vol)

    embedding = D_inv_sqrt @ V_star
    return KMeans(n_clusters=K, n_init=10,
                  random_state=random_state).fit_predict(embedding)


# ── Lu & Carreira-Perpinan (2008) — Algorithm A ───────────────────────────────
def lu_csc(W_G, W_H, n_clusters, t=1.0, k_eig=50, eps_c=3.0,
           random_state=0, eps=1e-12):
    N = W_G.shape[0]

    deg  = np.array(W_G.sum(axis=1)).ravel()
    isqd = 1.0 / np.sqrt(np.maximum(deg, eps))
    L_sp = eye(N, format='csr') - diags(isqd) @ W_G @ diags(isqd)

    k_eig = min(k_eig, N - 2)
    lam, V = eigsh(L_sp, k=k_eig, which='SM', tol=1e-3)
    scale  = np.exp(-t * lam) - np.exp(-2.0 * t)
    K      = (V * scale[None, :]) @ V.T
    K     += np.exp(-2.0 * t) * np.eye(N)

    w2    = 1.0 / (eps_c ** 2)
    deg_H = np.array(W_H.sum(axis=1)).ravel()
    M     = w2 * W_H.toarray().astype(np.float64)
    M    += w2 * np.diag(deg_H)

    IpMK  = np.eye(N) + M @ K
    Z     = np.linalg.solve(IpMK, np.eye(N))
    del IpMK
    K_bar = K @ Z
    del K, Z, M
    K_bar = (K_bar + K_bar.T) * 0.5

    A = np.maximum(0.0, K_bar);  del K_bar
    deg_a  = A.sum(axis=1)
    isqd_a = 1.0 / np.sqrt(np.maximum(deg_a, eps))
    L_sym  = np.eye(N) - (isqd_a[:, None] * A * isqd_a[None, :])
    del A

    _, evecs = sla.eigh(L_sym, subset_by_index=[0, n_clusters - 1])
    del L_sym

    return KMeans(n_clusters=n_clusters, n_init=10,
                  random_state=random_state).fit_predict(evecs)


# ── Sweep over k ──────────────────────────────────────────────────────────────
all_results = {}   # all_results[k] = {'SC': (ari, nmi), 'Wang': ..., ...}

for k in K_RANGE:
    N_CLUSTERS = k
    print(f"\n{'='*60}")
    print(f"k = {k}  (N = {k * BLOCK_SIZE})")
    print(f"{'='*60}")

    G, H = SBM(block_size=BLOCK_SIZE, k=k, p=P_INTRA, q=Q_INTER)
    y    = np.array([G.nodes[i]['block'] for i in G.nodes()])
    W_G  = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    W_H  = nx.to_scipy_sparse_array(H, format='csr').astype(float)
    N    = W_G.shape[0]

    avg_deg_H = float(W_H.sum()) / N
    # eps_c calibration: w2 * avg_deg_H ≈ K^{-1}_max ≈ e^{2t} ≈ 7.4 for t=1
    eps_c_lu = float(np.sqrt(avg_deg_H / 7.4))
    print(f"  W_G avg-deg={W_G.sum()/N:.1f}  W_H avg-deg={avg_deg_H:.1f}"
          f"  → eps_c(Lu)={eps_c_lu:.2f}")

    res = {}

    # [1] Unconstrained SC
    deg_g    = np.array(W_G.sum(axis=1)).ravel()
    L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_labels, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                           random_state=RANDOM_STATE)
    res['SC'] = (adjusted_rand_score(y, sc_labels),
                 normalized_mutual_info_score(y, sc_labels))
    print(f"  [SC]   ARI={res['SC'][0]:.4f}  NMI={res['SC'][1]:.4f}")

    # [2] Wang
    _scale = float(W_G.sum()) / float(W_H.sum())
    Q_wang = (W_G - _scale * W_H).astype(float)
    try:
        wl = wang_algorithm2(W_G, Q_wang, beta=0.0, K=N_CLUSTERS,
                              random_state=RANDOM_STATE)
        res['Wang'] = (adjusted_rand_score(y, wl),
                       normalized_mutual_info_score(y, wl))
    except Exception as e:
        res['Wang'] = (float('nan'), float('nan'))
        print(f"  [Wang] FAILED: {e}")
    print(f"  [Wang] ARI={res['Wang'][0]:.4f}  NMI={res['Wang'][1]:.4f}")

    # [3] Lu
    lu_labels = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS,
                       t=1.0, k_eig=50, eps_c=eps_c_lu, random_state=RANDOM_STATE)
    res['Lu'] = (adjusted_rand_score(y, lu_labels),
                 normalized_mutual_info_score(y, lu_labels))
    print(f"  [Lu]   ARI={res['Lu'][0]:.4f}  NMI={res['Lu'][1]:.4f}")

    # [4] Ours
    ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS, exact=False,
                                      eps=EPS_CSC, random_seed=RANDOM_STATE)
    res['Ours'] = (adjusted_rand_score(y, ol),
                   normalized_mutual_info_score(y, ol))
    print(f"  [Ours] ARI={res['Ours'][0]:.4f}  NMI={res['Ours'][1]:.4f}")

    all_results[k] = res

# ── Summary table (ARI) ───────────────────────────────────────────────────────
METHODS = [
    ("SC",   "SC (unconstrained)"),
    ("Wang", "Wang (2014)"),
    ("Lu",   "Lu (2008)"),
    ("Ours", "Ours"),
]
W = 72
print(f"\n\n{'='*W}")
print(f"SBM  block_size={BLOCK_SIZE}, p={P_INTRA}, q={Q_INTER}, seed={RANDOM_STATE}  — ARI")
print(f"{'='*W}")
hdr = f"{'k':>4}  {'N':>5}  " + "  ".join(f"{name:>16}" for _, name in METHODS)
print(hdr)
print("-" * W)
for k in K_RANGE:
    row = f"{k:>4}  {k*BLOCK_SIZE:>5}  "
    row += "  ".join(f"{all_results[k][key][0]:>16.4f}" for key, _ in METHODS)
    print(row)
print(f"{'='*W}")

print(f"\n{'='*W}")
print(f"SBM  block_size={BLOCK_SIZE}, p={P_INTRA}, q={Q_INTER}, seed={RANDOM_STATE}  — NMI")
print(f"{'='*W}")
print(hdr)
print("-" * W)
for k in K_RANGE:
    row = f"{k:>4}  {k*BLOCK_SIZE:>5}  "
    row += "  ".join(f"{all_results[k][key][1]:>16.4f}" for key, _ in METHODS)
    print(row)
print(f"{'='*W}")
