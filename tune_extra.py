"""Parameter tuning for Fashion-MNIST and Pendigits (N=5000 subsets).

Tunes:
  - Ours: eps over a grid
  - Lu  : eps_c over a small grid
  - Wang: beta = 0 fixed (only cannot-link constraints)
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
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
)

N_SAMPLES    = 5000
N_CLUSTERS   = 10
RANDOM_STATE = 42
NCL_LIST     = [5, 10, 20]
EPS_GRID     = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95, 0.99]
LU_EPS_C_GRID = [0.5, 1.0, 2.0]


# ── Wang Algorithm 2 (compact, same as MNIST script) ──────────────────────────
def wang_alg2(A, Q, K, n_extra=20, random_state=0, eps=1e-10):
    N = A.shape[0]
    vol = float(A.sum())
    deg = np.array(A.sum(axis=1)).ravel()
    isq = 1.0 / np.sqrt(np.maximum(deg, eps)); D_isq = diags(isq)
    L_bar = eye(N, format='csr') - D_isq @ A @ D_isq
    Q_bar = D_isq @ Q @ D_isq
    n_q = min(K - 1 + n_extra, N - 2)
    qv, qvec = eigsh(Q_bar, k=n_q, which='LM')
    o = np.argsort(qv)[::-1]; qv = qv[o]; qvec = qvec[:, o]
    if 0.0 >= qv[K - 2] * vol:
        raise ValueError("infeasible")
    pm = qv > 0.0
    V = qvec[:, pm]; md = qv[pm]
    if V.shape[1] < K - 1:
        raise ValueError(f"only {V.shape[1]} feasible Ritz vectors")
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
    Z = np.linalg.solve(IpMK, np.eye(N))
    del IpMK
    K_bar = K @ Z; del K, Z, M
    K_bar = (K_bar + K_bar.T) * 0.5
    A = np.maximum(0.0, K_bar); del K_bar
    deg_a  = A.sum(axis=1)
    isqd_a = 1.0 / np.sqrt(np.maximum(deg_a, eps))
    L_sym  = np.eye(N) - (isqd_a[:, None] * A * isqd_a[None, :])
    del A
    _, evecs = sla.eigh(L_sym, subset_by_index=[0, n_clusters - 1])
    return KMeans(n_clusters=n_clusters, n_init=10,
                  random_state=random_state).fit_predict(evecs)


# ── Dataset loaders ───────────────────────────────────────────────────────────
def load_fmnist():
    import keras.datasets.fashion_mnist as fm
    (X, y), _ = fm.load_data()
    return X.reshape(len(X), -1).astype(np.float32), y.astype(int)

def load_pendigits():
    d = fetch_openml('pendigits', version=1, as_frame=False, parser='liac-arff')
    return d.data.astype(np.float32), d.target.astype(int)

DATASETS = {
    'Fashion-MNIST': (load_fmnist, 5),
    'Pendigits':     (load_pendigits, 5),
}

# ── Main tuning loop ──────────────────────────────────────────────────────────
all_best = {}

for ds_name, (loader, k_nn) in DATASETS.items():
    print(f"\n{'#'*72}\n# Dataset: {ds_name}  (k_nn={k_nn})\n{'#'*72}")
    X_full, y_full = loader()
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
    X = X_full[idx]; y = y_full[idx]
    print(f"  N={N_SAMPLES}  d={X.shape[1]}  classes={np.unique(y).tolist()}")

    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)
    print(f"  W_G avg-deg={W_G.sum()/N_SAMPLES:.1f}")

    # Unconstrained SC
    deg_g = np.array(W_G.sum(axis=1)).ravel()
    L_G = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_lab, _, _ = spectral_clustering(L_G, n_clusters=N_CLUSTERS,
                                        random_state=RANDOM_STATE)
    sc_ari = adjusted_rand_score(y, sc_lab)
    sc_nmi = normalized_mutual_info_score(y, sc_lab)
    print(f"  SC:  ARI={sc_ari:.4f}  NMI={sc_nmi:.4f}")

    ds_best = {'SC': (sc_ari, sc_nmi), 'NCL': {}}

    for N_CL in NCL_LIST:
        print(f"\n--- n_cl={N_CL} ---")
        W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
        print(f"  W_H avg-deg={W_H.sum()/N_SAMPLES:.1f}")

        # Wang (no tuning, beta=0)
        try:
            _scale = float(W_G.sum()) / float(W_H.sum())
            Q = (W_G - _scale * W_H).astype(float)
            wl = wang_alg2(W_G, Q, K=N_CLUSTERS, random_state=RANDOM_STATE)
            wang_ari = adjusted_rand_score(y, wl)
            wang_nmi = normalized_mutual_info_score(y, wl)
        except Exception as e:
            wang_ari = wang_nmi = float('nan')
            print(f"  Wang failed: {e}")
        print(f"  Wang: ARI={wang_ari:.4f}  NMI={wang_nmi:.4f}")

        # Lu sweep over eps_c
        best_lu = (-1, None, None, None)
        for ec in LU_EPS_C_GRID:
            try:
                ll = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS, t=1.0,
                            k_eig=150, eps_c=ec, random_state=RANDOM_STATE)
                a = adjusted_rand_score(y, ll)
                n = normalized_mutual_info_score(y, ll)
            except Exception as e:
                a = n = float('nan')
                print(f"  Lu eps_c={ec} failed: {e}")
                continue
            print(f"  Lu eps_c={ec}: ARI={a:.4f}  NMI={n:.4f}")
            if a > best_lu[0]:
                best_lu = (a, n, ec, ll)
        print(f"  -> Lu best eps_c={best_lu[2]} ARI={best_lu[0]:.4f}")

        # Ours: sweep eps
        best_ours = (-1, None, None)
        for e_ in EPS_GRID:
            try:
                ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS, exact=False,
                                                eps=e_, random_seed=RANDOM_STATE)
                a = adjusted_rand_score(y, ol)
                n = normalized_mutual_info_score(y, ol)
            except Exception as e:
                a = n = float('nan')
                print(f"  Ours eps={e_} failed: {e}")
                continue
            print(f"  Ours eps={e_:.2f}: ARI={a:.4f}  NMI={n:.4f}")
            if a > best_ours[0]:
                best_ours = (a, n, e_)
        print(f"  -> Ours best eps={best_ours[2]} ARI={best_ours[0]:.4f}")

        ds_best['NCL'][N_CL] = {
            'Wang': (wang_ari, wang_nmi),
            'Lu':   (best_lu[0], best_lu[1], best_lu[2]),
            'Ours': (best_ours[0], best_ours[1], best_ours[2]),
        }

    all_best[ds_name] = ds_best

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n\n{'='*72}\n  TUNED PARAMS\n{'='*72}")
for ds, info in all_best.items():
    print(f"\n[{ds}]  SC: ARI={info['SC'][0]:.4f}  NMI={info['SC'][1]:.4f}")
    for N_CL, r in info['NCL'].items():
        print(f"  n_cl={N_CL}:")
        print(f"    Wang  ARI={r['Wang'][0]:.4f}  NMI={r['Wang'][1]:.4f}")
        print(f"    Lu    ARI={r['Lu'][0]:.4f}  NMI={r['Lu'][1]:.4f}  (eps_c={r['Lu'][2]})")
        print(f"    Ours  ARI={r['Ours'][0]:.4f}  NMI={r['Ours'][1]:.4f}  (eps={r['Ours'][2]})")
