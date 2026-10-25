"""
Comparison on USPS (N=5000 subset) — four methods.

USPS training set has 7291 samples (16×16 images, 256 features).
A subset of N=5000 is used so Lu's dense K^{-1} (O(N²) memory) stays
tractable (~200 MB per matrix).

Methods
-------
  1. Unconstrained Spectral Clustering (SC)
  2. Wang et al. (2014) Algorithm 2
  3. Lu & Carreira-Perpinan (2008) Algorithm A
  4. Our Constrained SC (iterative_clustering)

Key calibration (same fixes as compare_lu_mnist_v2.py)
-------------------------------------------------------
  Wang Q : Q = W_G − scale·W_H, scale = W_G.sum()/W_H.sum().
            With K_NN=5 and N_CL=20, avg deg(W_G)≈10 vs avg deg(W_H)≈40,
            so W_H is ~4x denser — normalization is essential.

  Lu eps_c=1.0: avg deg(W_H)≈40; w2=1 keeps M_ii≈40 comparable to K^{-1}
                (max eigenvalue ≈7.4 for t=1), avoiding kernel collapse.
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import h5py
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils import (
    build_knn_graph, build_cannot_link_graph,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Load USPS ─────────────────────────────────────────────────────────────────
print("Loading USPS...")
with h5py.File("./datasets/usps/usps.h5", 'r') as hf:
    train = hf.get('train')
    X_full = train.get('data')[:]
    y_full = train.get('target')[:]

X_full = np.reshape(X_full, (len(X_full), -1)).astype(np.float32)
y_full = np.array(y_full, dtype=int)
print(f"  Full training set: N={len(X_full)}, features={X_full.shape[1]}, "
      f"classes={np.unique(y_full)}")

# ── Subsample ─────────────────────────────────────────────────────────────────
N_SAMPLES    = 5000
K_NN         = 5      # matches CSC_USPS.py
N_CL         = 20
N_CLUSTERS   = 10
RANDOM_STATE = 63     # matches CSC_USPS.py
EPS_CSC      = 0.5    # from CSC_USPS.py: "when n_cl=20, eps=0.5 works well"

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X = X_full[idx];  y = y_full[idx]
print(f"  Subset N={N_SAMPLES}, classes={np.unique(y)}")

# ── Shared graphs ─────────────────────────────────────────────────────────────
print("Building W_G (kNN)...")
W_G = build_knn_graph(X, k=K_NN, use_gaussian=False)
print(f"Building W_H (CL, n_cl={N_CL})...")
W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
print(f"  W_G avg-deg={W_G.sum()/N_SAMPLES:.1f},  "
      f"W_H avg-deg={W_H.sum()/N_SAMPLES:.1f}")

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
def lu_csc(W_G, W_H, n_clusters, t=1.0, k_eig=150, eps_c=1.0,
           random_state=0, eps=1e-12):
    N = W_G.shape[0]

    print("    [Lu] Building heat diffusion kernel K...")
    deg  = np.array(W_G.sum(axis=1)).ravel()
    isqd = 1.0 / np.sqrt(np.maximum(deg, eps))
    L_sp = eye(N, format='csr') - diags(isqd) @ W_G @ diags(isqd)

    k_eig = min(k_eig, N - 2)
    lam, V = eigsh(L_sp, k=k_eig, which='SM', tol=1e-3)
    scale  = np.exp(-t * lam) - np.exp(-2.0 * t)
    K      = (V * scale[None, :]) @ V.T
    K     += np.exp(-2.0 * t) * np.eye(N)

    print("    [Lu] Building constraint matrix M...")
    w2    = 1.0 / (eps_c ** 2)
    deg_H = np.array(W_H.sum(axis=1)).ravel()
    M     = w2 * W_H.toarray().astype(np.float64)
    M    += w2 * np.diag(deg_H)

    print("    [Lu] Computing K̄  (dense N×N solve)...")
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


# ── Run all methods ───────────────────────────────────────────────────────────
results = {}

print("\n[1/4] Unconstrained Spectral Clustering...")
deg_g    = np.array(W_G.sum(axis=1)).ravel()
L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
sc_labels, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                       random_state=RANDOM_STATE)
results['SC'] = (adjusted_rand_score(y, sc_labels),
                 normalized_mutual_info_score(y, sc_labels))
print(f"   ARI={results['SC'][0]:.4f}  NMI={results['SC'][1]:.4f}")

print("\n[2/4] Wang et al. (2014) Algorithm 2...")
_scale = float(W_G.sum()) / float(W_H.sum())
Q_wang = (W_G - _scale * W_H).astype(float)
try:
    wl = wang_algorithm2(W_G, Q_wang, beta=0.0, K=N_CLUSTERS,
                          random_state=RANDOM_STATE)
    results['Wang'] = (adjusted_rand_score(y, wl),
                       normalized_mutual_info_score(y, wl))
except Exception as e:
    results['Wang'] = (float('nan'), float('nan'));  print(f"   FAILED: {e}")
print(f"   ARI={results['Wang'][0]:.4f}  NMI={results['Wang'][1]:.4f}")

print("\n[3/4] Lu & Carreira-Perpinan (2008)...")
lu_labels = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS,
                   t=1.0, k_eig=150, eps_c=1.0, random_state=RANDOM_STATE)
results['Lu'] = (adjusted_rand_score(y, lu_labels),
                 normalized_mutual_info_score(y, lu_labels))
print(f"   ARI={results['Lu'][0]:.4f}  NMI={results['Lu'][1]:.4f}")

print("\n[4/4] Our Constrained SC (eps=0.5)...")
ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS, exact=False,
                                  eps=EPS_CSC, random_seed=RANDOM_STATE)
results['Ours'] = (adjusted_rand_score(y, ol),
                   normalized_mutual_info_score(y, ol))
print(f"   ARI={results['Ours'][0]:.4f}  NMI={results['Ours'][1]:.4f}")

# ── Table ─────────────────────────────────────────────────────────────────────
print()
W = 68
print("=" * W)
print(f"USPS subset  N={N_SAMPLES}, K={N_CLUSTERS}, k_nn={K_NN}, n_cl={N_CL}, seed={RANDOM_STATE}")
print("=" * W)
print(f"{'Method':<45} {'ARI':>10} {'NMI':>10}")
print("-" * W)
for name, key in [
    ("Spectral Clustering (unconstrained)", "SC"),
    ("Wang et al. (2014) Algorithm 2",      "Wang"),
    ("Lu & Carreira-Perpinan (2008)",        "Lu"),
    ("Our Constrained SC",                  "Ours"),
]:
    a, n = results[key]
    print(f"{name:<45} {a:>10.4f} {n:>10.4f}")
print("=" * W)
