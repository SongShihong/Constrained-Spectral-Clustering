"""
Full comparison: SC vs Wang vs Lu vs Ours-fixed vs Ours-adaptive.

For each (dataset, n_cl) cell:
  - SC               : unconstrained spectral clustering (uses only W_G)
  - Wang (2014)      : Algorithm 2 with β = 0
  - Lu (2008)        : t = 1, k_eig = 150, eps_c = 1.0
  - Ours fixed       : iterative_clustering with eps tuned per cell
                       (sweep over EPS_GRID, best ARI)
  - Ours adaptive    : iterative_clustering with eps* derived from data:
                          r       = vol(W_H) / vol(W_G)
                          eps*(r) = sigmoid( γ · (τ − r) )
                       γ = 3.0, τ = 2.75 (global, same for every cell)

Datasets:
  - USPS           (image, 256-d, k_nn=5, seed=63)
  - MNIST          (image, 784-d, k_nn=3, seed=42)
  - Fashion-MNIST  (image, 784-d, k_nn=5, seed=42)
  - Pendigits      (16-d, k_nn=5, seed=42)
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
from sklearn.datasets import fetch_openml
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils import (
    build_knn_graph, build_cannot_link_graph,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
    constrained_spectral_clustering,
)

# ── Configuration ─────────────────────────────────────────────────────────────
N_SAMPLES    = 5000
N_CLUSTERS   = 10
NCL_LIST     = [5, 10, 20]
EPS_GRID     = [0.05, 0.20, 0.50, 0.70, 0.85, 0.95]
LU_EPS_C     = 1.0

EPS_GAMMA = 3.0
EPS_TAU   = 2.75


# ── Adaptive eps + wrapper ────────────────────────────────────────────────────
def adaptive_eps(W_G, W_H, gamma=EPS_GAMMA, tau=EPS_TAU):
    vol_G = float(W_G.sum())
    vol_H = float(W_H.sum())
    r     = vol_H / max(vol_G, 1e-12)
    return 1.0 / (1.0 + np.exp(gamma * (r - tau)))


def iterative_clustering_adaptive(W_G, W_H, k, exact=False, random_seed=1,
                                  gamma=EPS_GAMMA, tau=EPS_TAU):
    eps = adaptive_eps(W_G, W_H, gamma=gamma, tau=tau)
    deg_g = np.array(W_G.sum(axis=1)).ravel()
    deg_h = np.array(W_H.sum(axis=1)).ravel()
    deg   = eps * deg_g + (1 - eps) * deg_h
    L_G   = normalized_laplacian_from_D_sp(W_G, deg)
    L_H   = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
    labels, _, _ = constrained_spectral_clustering(
        L_G_sym=L_G, L_H_sym_reg=L_H,
        n_clusters=k, random_state=random_seed, exact=exact,
    )
    return labels, eps


# ── Wang Algorithm 2 ──────────────────────────────────────────────────────────
def wang_algorithm2(A, Q, beta, K, n_extra=20, random_state=0, eps=1e-10):
    N = A.shape[0]
    vol = float(A.sum())
    deg = np.array(A.sum(axis=1)).ravel()
    isq = 1.0 / np.sqrt(np.maximum(deg, eps));  D_isq = diags(isq)
    L_bar = eye(N, format='csr') - D_isq @ A @ D_isq
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


# ── Lu & Carreira-Perpinan ────────────────────────────────────────────────────
def lu_csc(W_G, W_H, n_clusters, t=1.0, k_eig=150, eps_c=1.0,
           random_state=0, eps=1e-12):
    N = W_G.shape[0]
    deg  = np.array(W_G.sum(axis=1)).ravel()
    isqd = 1.0 / np.sqrt(np.maximum(deg, eps))
    L_sp = eye(N, format='csr') - diags(isqd) @ W_G @ diags(isqd)
    k_eig = min(k_eig, N - 2)
    lam, V = eigsh(L_sp, k=k_eig, which='SM', tol=1e-3)
    scale = np.exp(-t * lam) - np.exp(-2.0 * t)
    K = (V * scale[None, :]) @ V.T
    K += np.exp(-2.0 * t) * np.eye(N)
    w2    = 1.0 / (eps_c ** 2)
    deg_H = np.array(W_H.sum(axis=1)).ravel()
    M  = w2 * W_H.toarray().astype(np.float64)
    M += w2 * np.diag(deg_H)
    IpMK = np.eye(N) + M @ K
    Z = np.linalg.solve(IpMK, np.eye(N));  del IpMK
    K_bar = K @ Z;  del K, Z, M
    K_bar = (K_bar + K_bar.T) * 0.5
    A = np.maximum(0.0, K_bar);  del K_bar
    deg_a   = A.sum(axis=1)
    isqd_a  = 1.0 / np.sqrt(np.maximum(deg_a, eps))
    L_sym   = np.eye(N) - (isqd_a[:, None] * A * isqd_a[None, :])
    del A
    _, evecs = sla.eigh(L_sym, subset_by_index=[0, n_clusters - 1])
    return KMeans(n_clusters=n_clusters, n_init=10,
                  random_state=random_state).fit_predict(evecs)


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_usps():
    with h5py.File("./datasets/usps/usps.h5", 'r') as hf:
        X = np.array(hf['train']['data'][:]).reshape(-1, 256).astype(np.float32)
        y = np.array(hf['train']['target'][:], dtype=int)
    return X, y

def load_mnist():
    try:
        import keras.datasets.mnist as km
        (X, y), _ = km.load_data()
    except Exception:
        m = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X = m.data.astype(np.float32);  y = m.target.astype(int)
        return X, y
    return X.reshape(len(X), -1).astype(np.float32), y.astype(int)

def load_fashion_mnist():
    import keras.datasets.fashion_mnist as fm
    (X, y), _ = fm.load_data()
    return X.reshape(len(X), -1).astype(np.float32), y.astype(int)

def load_pendigits():
    d = fetch_openml('pendigits', version=1, as_frame=False, parser='liac-arff')
    return d.data.astype(np.float32), d.target.astype(int)

# (name, loader, k_nn, seed)
DATASETS = [
    ('USPS',          load_usps,          5, 63),
    ('MNIST',         load_mnist,         3, 42),
    ('Fashion-MNIST', load_fashion_mnist, 5, 42),
    ('Pendigits',     load_pendigits,     5, 42),
]


# ── Per-dataset run ───────────────────────────────────────────────────────────
def run_dataset(name, loader, k_nn, seed):
    print(f"\n{'#'*72}\n#  {name}  (k_nn={k_nn}, seed={seed})\n{'#'*72}")
    X_full, y_full = loader()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
    X = X_full[idx];  y = y_full[idx]

    print("Building W_G ...")
    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)
    vol_G = float(W_G.sum())
    print(f"  vol(W_G)={vol_G:.0f}  avg-deg={vol_G/N_SAMPLES:.1f}")

    print("Computing SC baseline ...")
    deg_g    = np.array(W_G.sum(axis=1)).ravel()
    L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_lab, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                        random_state=seed)
    sc_ari = adjusted_rand_score(y, sc_lab)
    sc_nmi = normalized_mutual_info_score(y, sc_lab)
    print(f"  SC  ARI={sc_ari:.4f}  NMI={sc_nmi:.4f}")

    out = {}
    for n_cl in NCL_LIST:
        print(f"\n--- {name}  n_cl={n_cl} ---")
        W_H = build_cannot_link_graph(y, n_cl=n_cl, random_state=seed)
        vol_H = float(W_H.sum())
        ratio = vol_H / vol_G

        # Wang
        try:
            scl = vol_G / vol_H
            Q   = (W_G - scl * W_H).astype(float)
            wl  = wang_algorithm2(W_G, Q, beta=0.0, K=N_CLUSTERS, random_state=seed)
            wang_a = adjusted_rand_score(y, wl)
            wang_n = normalized_mutual_info_score(y, wl)
        except Exception as e:
            wang_a = wang_n = float('nan')
            print(f"  Wang failed: {e}")
        print(f"  Wang        ARI={wang_a:.4f}  NMI={wang_n:.4f}")

        # Lu
        ll = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS, t=1.0, k_eig=150,
                    eps_c=LU_EPS_C, random_state=seed)
        lu_a = adjusted_rand_score(y, ll)
        lu_n = normalized_mutual_info_score(y, ll)
        print(f"  Lu          ARI={lu_a:.4f}  NMI={lu_n:.4f}")

        # Ours fixed (sweep EPS_GRID, pick best)
        best = (-1.0, -1.0, None)
        for ep in EPS_GRID:
            try:
                ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS,
                                                exact=False, eps=ep,
                                                random_seed=seed)
                a = adjusted_rand_score(y, ol)
                nn = normalized_mutual_info_score(y, ol)
            except Exception:
                a = nn = float('nan')
            if not np.isnan(a) and a > best[0]:
                best = (a, nn, ep)
        f_a, f_n, f_eps = best
        print(f"  Ours-fixed  ARI={f_a:.4f}  NMI={f_n:.4f}  (best eps={f_eps})")

        # Ours adaptive
        ol_a, used_eps = iterative_clustering_adaptive(
            W_G, W_H, k=N_CLUSTERS, exact=False, random_seed=seed,
        )
        a_a = adjusted_rand_score(y, ol_a)
        a_n = normalized_mutual_info_score(y, ol_a)
        print(f"  Ours-auto   ARI={a_a:.4f}  NMI={a_n:.4f}  (eps={used_eps:.3f})")

        out[n_cl] = {
            'r':         ratio,
            'eps_auto':  used_eps,
            'eps_fixed': f_eps,
            'SC':        (sc_ari, sc_nmi),
            'Wang':      (wang_a, wang_n),
            'Lu':        (lu_a, lu_n),
            'Fixed':     (f_a, f_n),
            'Auto':      (a_a, a_n),
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
all_results = {name: run_dataset(name, loader, k_nn, seed)
               for name, loader, k_nn, seed in DATASETS}


# ── Summary tables ────────────────────────────────────────────────────────────
def fmt(pair):
    a, n = pair
    return f"{a:>6.4f}/{n:<6.4f}"

W = 110
print(f"\n\n{'='*W}")
print("  FULL REPORT — ARI / NMI on N=5000 subsets, K=10")
print(f"  Adaptive eps:  eps* = sigmoid(γ·(τ − vol_H/vol_G)),  "
      f"γ={EPS_GAMMA}, τ={EPS_TAU}")
print(f"{'='*W}")

for name, res in all_results.items():
    print(f"\n[{name}]   SC baseline (n_cl-independent): "
          f"ARI={res[5]['SC'][0]:.4f}  NMI={res[5]['SC'][1]:.4f}")
    print(f"  {'n_cl':>4}  {'r':>5}  {'eps_a':>6}  {'eps_f':>6}  "
          f"{'Wang':>14}  {'Lu':>14}  {'Ours-fix':>14}  {'Ours-auto':>14}")
    for n_cl in NCL_LIST:
        r = res[n_cl]
        print(f"  {n_cl:>4}  {r['r']:>5.2f}  {r['eps_auto']:>6.3f}  "
              f"{r['eps_fixed']:>6.2f}  "
              f"{fmt(r['Wang']):>14}  {fmt(r['Lu']):>14}  "
              f"{fmt(r['Fixed']):>14}  {fmt(r['Auto']):>14}")
print(f"{'='*W}")

# Also print a per-method best-ARI summary
print("\n\nBest method per (dataset, n_cl) by ARI:")
print(f"{'Dataset':<16}{'n_cl':>5}   {'Best method':<14}{'ARI':>8}{'NMI':>8}")
for name, res in all_results.items():
    for n_cl in NCL_LIST:
        r = res[n_cl]
        cands = {'SC': r['SC'], 'Wang': r['Wang'], 'Lu': r['Lu'],
                 'Ours-fix': r['Fixed'], 'Ours-auto': r['Auto']}
        valid = {k: v for k, v in cands.items() if not np.isnan(v[0])}
        best_k = max(valid, key=lambda k: valid[k][0])
        a, n = valid[best_k]
        print(f"{name:<16}{n_cl:>5}   {best_k:<14}{a:>8.4f}{n:>8.4f}")
