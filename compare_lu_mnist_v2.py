"""
Comparison on MNIST (N=5000 subset) — corrected Lu implementation.

Lu & Carreira-Perpinan (2008) requires dense K^{-1}  (O(N^2) memory),
so a subset of N=5000 is used for all methods.

Key fixes vs v1:
  - M diagonal: M_ii = c_i/eps_c^2  (was c_i/(2*eps_c^2) due to double-counting)
  - Affinity K : heat diffusion kernel K = exp(-t * L_norm)
                 (low-rank + diagonal approximation via truncated eigsh)
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
    build_knn_graph, build_cannot_link_graph,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Load & subsample MNIST ────────────────────────────────────────────────────
print("Loading MNIST...")
try:
    import keras.datasets.mnist as km
    (X_full, y_full), _ = km.load_data()
except Exception:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X_full = mnist.data.astype(np.float32)
    y_full = mnist.target.astype(int)

X_full = np.reshape(X_full, (len(X_full), -1)).astype(np.float32)
y_full = np.array(y_full, dtype=int)

N_SAMPLES    = 5000
K_NN         = 3
N_CL         = 20
N_CLUSTERS   = 10
RANDOM_STATE = 42
EPS_CSC      = 0.7    # our method

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X = X_full[idx];  y = y_full[idx]
print(f"  Subset N={N_SAMPLES}, classes={np.unique(y)}")

# ── Shared graphs ──────────────────────────────────────────────────────────────
print("Building W_G (kNN)...")
W_G = build_knn_graph(X, k=K_NN, use_gaussian=False)
print(f"Building W_H (CL, n_cl={N_CL})...")
W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)

# ── Wang et al. (2014) ────────────────────────────────────────────────────────
def wang_algorithm2(A, Q, beta, K, n_extra=20, random_state=0, eps=1e-10):
    N = A.shape[0]
    vol = float(A.sum())
    deg = np.array(A.sum(axis=1)).ravel()
    isq = 1.0 / np.sqrt(np.maximum(deg, eps));  D_isq = diags(isq)
    I_sp = eye(N, format='csr')
    L_bar = I_sp - D_isq @ A @ D_isq
    Q_bar = D_isq @ Q @ D_isq
    n_q = min(K - 1 + n_extra, N - 2)
    qv, qvec = eigsh(Q_bar, k=n_q, which='LM')
    o = np.argsort(qv)[::-1];  qv = qv[o];  qvec = qvec[:, o]
    if beta >= qv[K - 2] * vol:
        raise ValueError("Infeasible")
    pm = qv > beta / vol
    V = qvec[:, pm];  md = qv[pm] - beta / vol
    if V.shape[1] < K - 1:
        raise ValueError(f"Only {V.shape[1]} feasible Ritz vectors")
    sv, svec = sla.eigh(V.T @ (L_bar @ V), np.diag(md))
    pi = np.where(sv > 1e-8)[0]
    if len(pi) < K - 1: pi = np.argsort(sv)[-(K - 1):]
    sel = pi[np.argsort(sv[pi])[:K - 1]]
    Vs = V @ svec[:, sel]
    Vs = Vs / np.maximum(np.linalg.norm(Vs, axis=0, keepdims=True), eps) * np.sqrt(vol)
    return KMeans(n_clusters=K, n_init=10, random_state=random_state).fit_predict(D_isq @ Vs)

# ── Lu & Carreira-Perpinan (2008) — Algorithm A ───────────────────────────────
def lu_csc(W_G, W_H, n_clusters, t=1.0, k_eig=150, eps_c=0.1, random_state=0, eps=1e-12):
    """
    Lu & Carreira-Perpinan (2008) Algorithm A.

    Affinity K : heat diffusion kernel K = exp(-t L_norm), approximated as
                 K ≈ V diag(exp(-t λ) − exp(-2t)) Vᵀ + exp(-2t) I
                 using the k_eig smallest eigenpairs of L_norm (Laplacian of W_G).
                 This gives a dense PD kernel capturing manifold structure.

    Constraint M (Eq. 3, cannot-link only):
                 M_ij  =  1/eps_c²   for (i,j) ∈ CL
                 M_ii  =  c_i/eps_c²     (c_i = #CL edges at node i)
                 In matrix form: M = (1/eps_c²)(D_H + W_H)
                 where D_H = diag(row sums of W_H).

    Modified affinity (Eq. 2):
                 K̄ = (K⁻¹ + M)⁻¹ = K (I + MK)⁻¹     [Woodbury]

    Final:  A = max(0, K̄),  spectral clustering on A.
    """
    N = W_G.shape[0]

    # ── Step 1: heat diffusion kernel K = exp(-t L_norm) ─────────────────
    print("    [Lu] Building heat diffusion kernel K...")
    deg  = np.array(W_G.sum(axis=1)).ravel()
    isqd = 1.0 / np.sqrt(np.maximum(deg, eps))
    L_sp = eye(N, format='csr') - diags(isqd) @ W_G @ diags(isqd)

    k_eig = min(k_eig, N - 2)
    lam, V = eigsh(L_sp, k=k_eig, which='SM', tol=1e-3)
    # K = V diag(exp(-tλ)-exp(-2t)) Vᵀ + exp(-2t)I  (PD, dense)
    scale  = np.exp(-t * lam) - np.exp(-2.0 * t)   # ≥ 0 since λ ≤ 2
    K = (V * scale[None, :]) @ V.T
    K += np.exp(-2.0 * t) * np.eye(N)

    # ── Step 2: constraint matrix M (Eq. 3) ──────────────────────────────
    print("    [Lu] Building constraint matrix M...")
    # M = (1/eps_c²)(D_H + W_H)
    w2    = 1.0 / (eps_c ** 2)
    deg_H = np.array(W_H.sum(axis=1)).ravel()
    # dense: W_H.toarray() has off-diagonal +1 for each CL pair
    # diagonal: deg_H[i] from all CL edges incident to i
    M  = w2 * W_H.toarray().astype(np.float64)
    M += w2 * np.diag(deg_H)

    # ── Step 3: K̄ = K(I + MK)⁻¹ via Woodbury ───────────────────────────
    print("    [Lu] Computing K̄  (dense N×N solve)...")
    MK   = M @ K
    IpMK = np.eye(N) + MK
    del MK

    # Solve (I+MK) Z = I  →  Z = (I+MK)⁻¹  →  K̄ = K Z
    Z     = np.linalg.solve(IpMK, np.eye(N))
    del IpMK
    K_bar = K @ Z
    del K, Z, M
    K_bar = (K_bar + K_bar.T) * 0.5   # enforce symmetry

    # ── Step 4: threshold  A = max(0, K̄) ────────────────────────────────
    A = np.maximum(0.0, K_bar);  del K_bar

    # ── Step 5: normalised-cut spectral clustering on A ──────────────────
    deg_a   = A.sum(axis=1)
    isqd_a  = 1.0 / np.sqrt(np.maximum(deg_a, eps))
    L_sym   = np.eye(N) - (isqd_a[:, None] * A * isqd_a[None, :])
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
# Normalize W_H to the same total weight as W_G before subtracting.
# Without this, W_H (n_cl=20, avg-degree ~40) is ~7x denser than W_G
# (k=3, avg-degree ~6), making Q dominated by -W_H and causing the
# algorithm to find no feasible eigenvectors.
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
# eps_c=0.1 → w2=100; with avg W_H degree ~40, M_ii≈4000 >> K^{-1}_{ii}≈1-7,
# so the constraint completely destroys the heat kernel.
# eps_c=1.0 → w2=1 gives a balanced modification.
lu_labels = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS,
                   t=1.0, k_eig=150, eps_c=1.0, random_state=RANDOM_STATE)
results['Lu'] = (adjusted_rand_score(y, lu_labels),
                 normalized_mutual_info_score(y, lu_labels))
print(f"   ARI={results['Lu'][0]:.4f}  NMI={results['Lu'][1]:.4f}")

print("\n[4/4] Our Constrained SC (eps=0.7)...")
ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS, exact=False,
                                  eps=EPS_CSC, random_seed=RANDOM_STATE)
results['Ours'] = (adjusted_rand_score(y, ol),
                   normalized_mutual_info_score(y, ol))
print(f"   ARI={results['Ours'][0]:.4f}  NMI={results['Ours'][1]:.4f}")

# ── Table ─────────────────────────────────────────────────────────────────────
print()
W = 68
print("=" * W)
print(f"MNIST subset  N={N_SAMPLES}, K={N_CLUSTERS}, k_nn={K_NN}, n_cl={N_CL}, seed={RANDOM_STATE}")
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
