"""
Comparison on USPS (N=5000 subset) — vary n_cl in {5, 10, 20}.

For each n_cl value, all four methods are evaluated:
  - Unconstrained Spectral Clustering
  - Wang et al. (2014) Algorithm 2
  - Lu & Carreira-Perpinan (2008)
  - Our Constrained SC

USPS training set has 7291 samples (16×16 images, 256 features).
A subset of N=5000 is used so Lu's dense K^{-1} (O(N²) memory) stays
tractable (~200 MB per matrix).
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
N_CLUSTERS   = 10
RANDOM_STATE = 63     # matches CSC_USPS.py

NCL_LIST = [5, 10, 20]
# Tuned per n_cl: as W_H avg-deg grows relative to W_G, eps must drop so
# the constraint Laplacian is properly balanced into the joint degree.
EPS_PER_NCL  = {5: 0.95, 10: 0.90, 20: 0.05}

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X = X_full[idx];  y = y_full[idx]
print(f"  Subset N={N_SAMPLES}, classes={np.unique(y)}")

# ── Shared kNN graph (same for all n_cl) ──────────────────────────────────────
print("Building W_G (kNN)...")
W_G = build_knn_graph(X, k=K_NN, use_gaussian=False)
print(f"  W_G avg-deg={W_G.sum()/N_SAMPLES:.1f}")

# ── Precompute SC (unconstrained, same for all n_cl) ─────────────────────────
print("Computing unconstrained Spectral Clustering (shared across n_cl)...")
deg_g    = np.array(W_G.sum(axis=1)).ravel()
L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
sc_labels, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                       random_state=RANDOM_STATE)
sc_ari = adjusted_rand_score(y, sc_labels)
sc_nmi = normalized_mutual_info_score(y, sc_labels)
print(f"   SC  ARI={sc_ari:.4f}  NMI={sc_nmi:.4f}")


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


# ── Main loop over n_cl values ────────────────────────────────────────────────
all_results = {}   # all_results[n_cl][method] = (ARI, NMI)

for N_CL in NCL_LIST:
    print(f"\n{'='*68}")
    print(f"  n_cl = {N_CL}")
    print(f"{'='*68}")

    res = {}

    # SC is independent of n_cl
    res['SC'] = (sc_ari, sc_nmi)

    print(f"\nBuilding W_H (CL, n_cl={N_CL})...")
    W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
    print(f"   W_H avg-deg={W_H.sum()/N_SAMPLES:.1f}")

    print(f"\n[Wang] n_cl={N_CL}...")
    _scale = float(W_G.sum()) / float(W_H.sum())
    Q_wang = (W_G - _scale * W_H).astype(float)
    try:
        wl = wang_algorithm2(W_G, Q_wang, beta=0.0, K=N_CLUSTERS,
                              random_state=RANDOM_STATE)
        res['Wang'] = (adjusted_rand_score(y, wl),
                       normalized_mutual_info_score(y, wl))
    except Exception as e:
        res['Wang'] = (float('nan'), float('nan'))
        print(f"   FAILED: {e}")
    print(f"   ARI={res['Wang'][0]:.4f}  NMI={res['Wang'][1]:.4f}")

    print(f"\n[Lu] n_cl={N_CL}...")
    lu_labels = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS,
                       t=1.0, k_eig=150, eps_c=1.0, random_state=RANDOM_STATE)
    res['Lu'] = (adjusted_rand_score(y, lu_labels),
                 normalized_mutual_info_score(y, lu_labels))
    print(f"   ARI={res['Lu'][0]:.4f}  NMI={res['Lu'][1]:.4f}")

    eps_csc = EPS_PER_NCL[N_CL]
    print(f"\n[Ours] n_cl={N_CL}  (eps={eps_csc})...")
    ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS, exact=False,
                                     eps=eps_csc, random_seed=RANDOM_STATE)
    res['Ours'] = (adjusted_rand_score(y, ol),
                   normalized_mutual_info_score(y, ol))
    print(f"   ARI={res['Ours'][0]:.4f}  NMI={res['Ours'][1]:.4f}")

    all_results[N_CL] = res


# ── Summary Table ─────────────────────────────────────────────────────────────
W = 72
METHODS = [
    ("Spectral Clustering (unconstrained)", "SC"),
    ("Wang et al. (2014) Algorithm 2",      "Wang"),
    ("Lu & Carreira-Perpinan (2008)",        "Lu"),
    ("Our Constrained SC",                  "Ours"),
]

print(f"\n{'='*W}")
print(f"USPS subset  N={N_SAMPLES}, K={N_CLUSTERS}, k_nn={K_NN}, seed={RANDOM_STATE}")
print(f"{'='*W}")

# Header
hdr = f"{'Method':<40}"
for n_cl in NCL_LIST:
    hdr += f"  n_cl={n_cl:>2} ARI   NMI"
print(hdr)
print("-" * W)

for name, key in METHODS:
    row = f"{name:<40}"
    for n_cl in NCL_LIST:
        a, n = all_results[n_cl][key]
        row += f"  {a:>8.4f} {n:>6.4f}"
    print(row)

print("=" * W)
