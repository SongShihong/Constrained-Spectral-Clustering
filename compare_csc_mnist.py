"""
Comparison of Constrained Spectral Clustering methods on MNIST.

Dataset : MNIST (full training set, 60 000 samples)
Graph   : k-NN (k=3, binary weights)
Cannot-link (CL) constraints: build_cannot_link_graph(y, n_cl=20)
Must-link  (ML) constraints : (none explicit — kNN graph edges serve as soft ML in Q)

Methods
-------
1. Unconstrained Spectral Clustering (SC)
2. Wang et al. (2014) Algorithm 2 — CSC
3. Our Constrained SC  (iterative_clustering, eps=0.7)
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils import (
    build_knn_graph,
    build_cannot_link_graph,
    spectral_clustering,
    iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Load MNIST ────────────────────────────────────────────────────────────────
print("Loading MNIST...")
try:
    import keras.datasets.mnist as keras_mnist
    (X, y), _ = keras_mnist.load_data()
except Exception:
    from sklearn.datasets import fetch_openml
    import numpy as _np
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)
    # use all 70000; keep full set
    X = X;  y = y

X = np.reshape(X, (len(X), -1)).astype(np.float32)
y = np.array(y, dtype=int)
print(f"  X: {X.shape}, classes: {np.unique(y)}")

# ── Parameters ────────────────────────────────────────────────────────────────
K_NN         = 3
N_CL         = 20
N_CLUSTERS   = 10
RANDOM_STATE = 42
EPS_OUR      = 0.7      # eps for our method (from CSC_MNIST.ipynb)

# ── Build graphs ──────────────────────────────────────────────────────────────
print("Building kNN similarity graph (W_G)...")
W_G = build_knn_graph(X, k=K_NN, use_gaussian=False)

print(f"Building cannot-link graph (W_H, n_cl={N_CL})...")
W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)

# ── Method 1: Unconstrained Spectral Clustering ───────────────────────────────
print("\n[1/3] Unconstrained Spectral Clustering...")
deg_g    = np.array(W_G.sum(axis=1)).ravel()
L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
sc_labels, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                       random_state=RANDOM_STATE)
sc_ari = adjusted_rand_score(y, sc_labels)
sc_nmi = normalized_mutual_info_score(y, sc_labels)
print(f"   ARI = {sc_ari:.4f}   NMI = {sc_nmi:.4f}")

# ── Method 2: Wang et al. (2014) Algorithm 2 ─────────────────────────────────
def wang_algorithm2(A, Q, beta, K, n_extra=20, random_state=0, eps=1e-10):
    """
    Wang et al. (2014) Algorithm 2 — Constrained Spectral Clustering (K-way).
    Solves: L̄ v = λ (Q̄ − β/vol · I) v  via Rayleigh-Ritz projection.

    A : sparse affinity matrix
    Q : sparse constraint matrix (+1 must-link, −1 cannot-link)
    """
    N = A.shape[0]

    # Step 1
    vol        = float(A.sum())
    deg        = np.array(A.sum(axis=1)).ravel()
    inv_sqrt   = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_inv_sqrt = diags(inv_sqrt)

    # Step 2
    I_sp  = eye(N, format='csr')
    L_bar = I_sp - D_inv_sqrt @ A @ D_inv_sqrt     # L̄ = I − D^{-½} A D^{-½}
    Q_bar = D_inv_sqrt @ Q @ D_inv_sqrt            # Q̄ = D^{-½} Q D^{-½}

    # Step 3 — top eigenvalues/vectors of Q̄
    n_q = min(K - 1 + n_extra, N - 2)
    q_vals, q_vecs = eigsh(Q_bar, k=n_q, which='LM')
    order  = np.argsort(q_vals)[::-1]
    q_vals = q_vals[order];  q_vecs = q_vecs[:, order]

    lambda_K1 = q_vals[K - 2]   # (K−1)-th largest eigenvalue of Q̄

    # Step 4 — feasibility
    if beta >= lambda_K1 * vol:
        raise ValueError(
            f"Infeasible: β={beta:.4f} >= λ_{{K-1}}·vol={lambda_K1*vol:.4f}"
        )

    # Steps 5-7 — Rayleigh-Ritz in Q̄'s positive eigenspace
    # M = Q̄ − (β/vol)·I  is PD where q_vals > β/vol
    threshold = beta / vol
    pos_mask  = q_vals > threshold
    V         = q_vecs[:, pos_mask]
    m_diag    = q_vals[pos_mask] - threshold

    if V.shape[1] < K - 1:
        raise ValueError(f"Only {V.shape[1]} feasible Ritz vectors, need K-1={K-1}")

    L_sub = V.T @ (L_bar @ V)          # Vᵀ L̄ V  (small dense m×m)
    M_sub = np.diag(m_diag)
    sub_vals, sub_vecs = sla.eigh(L_sub, M_sub)

    pos_idx = np.where(sub_vals > 1e-8)[0]
    if len(pos_idx) < K - 1:
        pos_idx = np.argsort(sub_vals)[-(K - 1):]

    best_k1 = np.argsort(sub_vals[pos_idx])[:K - 1]
    sel_idx = pos_idx[best_k1]
    V_star  = V @ sub_vecs[:, sel_idx]           # N × (K−1)

    norms  = np.linalg.norm(V_star, axis=0, keepdims=True)
    V_star = V_star / np.maximum(norms, eps) * np.sqrt(vol)

    # Step 8 — K-means on D^{-½} V*
    embedding = D_inv_sqrt @ V_star
    km     = KMeans(n_clusters=K, n_init=10, random_state=random_state)
    labels = km.fit_predict(embedding)
    return labels


print("\n[2/3] Wang et al. (2014) Algorithm 2...")
# Q: kNN edges as soft must-link (+1), CL edges as cannot-link (-1)
Q_wang = (W_G - W_H).astype(float)
try:
    wang_labels = wang_algorithm2(
        A=W_G, Q=Q_wang, beta=0.0, K=N_CLUSTERS,
        n_extra=20, random_state=RANDOM_STATE
    )
    wang_ari = adjusted_rand_score(y, wang_labels)
    wang_nmi = normalized_mutual_info_score(y, wang_labels)
    print(f"   ARI = {wang_ari:.4f}   NMI = {wang_nmi:.4f}")
except Exception as e:
    wang_ari = wang_nmi = float('nan')
    print(f"   FAILED: {e}")

# ── Method 3: Our Constrained SC ─────────────────────────────────────────────
print("\n[3/3] Our Constrained SC (eps=0.7)...")
our_labels, _, _ = iterative_clustering(
    W_G, W_H, k=N_CLUSTERS, exact=False, eps=EPS_OUR, random_seed=RANDOM_STATE
)
our_ari = adjusted_rand_score(y, our_labels)
our_nmi = normalized_mutual_info_score(y, our_labels)
print(f"   ARI = {our_ari:.4f}   NMI = {our_nmi:.4f}")

# ── Comparison table ──────────────────────────────────────────────────────────
print()
print("=" * 65)
print(f"Dataset : MNIST  (N={len(y)}, K={N_CLUSTERS}, k_nn={K_NN}, n_cl={N_CL})")
print("=" * 65)
print(f"{'Method':<40} {'ARI':>10} {'NMI':>10}")
print("-" * 65)
print(f"{'Spectral Clustering (unconstrained)':<40} {sc_ari:>10.4f} {sc_nmi:>10.4f}")
print(f"{'Wang et al. (2014) Algorithm 2':<40} {wang_ari:>10.4f} {wang_nmi:>10.4f}")
print(f"{'Our Constrained SC':<40} {our_ari:>10.4f} {our_nmi:>10.4f}")
print("=" * 65)
