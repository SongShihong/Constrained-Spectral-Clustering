"""
compare_natural_datasets_kfn.py

Same datasets and methods as compare_natural_datasets.py, but the
cannot-link graph W_H is constructed from k-furthest neighbors in
feature space rather than from ground-truth labels.  This makes the
constraint generation fully unsupervised.

  W_G  — k-nearest-neighbor graph  (must-link proxy)
  W_H  — k-furthest-neighbor graph  (cannot-link proxy)

Methods compared:
  SC   — Unconstrained Spectral Clustering
  Wang — Wang et al. (2014) Algorithm 2
  Lu   — Lu & Carreira-Perpinan (2008) Algorithm A
  CSC  — Our Constrained Spectral Clustering (iterative_clustering)
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils_fixed import (
    build_knn_graph, build_kfn_graph,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
)

RANDOM_STATE = 42

# ── Baseline: Wang et al. (2014) Algorithm 2 ──────────────────────────────────

def wang_algorithm2(A, Q, beta, K, n_extra=20, random_state=0, eps=1e-10):
    N = A.shape[0]
    vol = float(A.sum())
    deg = np.array(A.sum(axis=1)).ravel()
    isq = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_isq = diags(isq)
    I_sp = eye(N, format='csr')
    L_bar = I_sp - D_isq @ A @ D_isq
    Q_bar = D_isq @ Q @ D_isq
    n_q = min(K - 1 + n_extra, N - 2)
    qv, qvec = eigsh(Q_bar, k=n_q, which='LM')
    o = np.argsort(qv)[::-1];  qv = qv[o];  qvec = qvec[:, o]
    if beta >= qv[K - 2] * vol:
        raise ValueError("Infeasible: beta too large relative to leading eigenvalue.")
    pm = qv > beta / vol
    V = qvec[:, pm];  md = qv[pm] - beta / vol
    if V.shape[1] < K - 1:
        raise ValueError(f"Only {V.shape[1]} feasible Ritz vectors (need {K-1}).")
    sv, svec = sla.eigh(V.T @ (L_bar @ V), np.diag(md))
    pi = np.where(sv > 1e-8)[0]
    if len(pi) < K - 1:
        pi = np.argsort(sv)[-(K - 1):]
    sel = pi[np.argsort(sv[pi])[:K - 1]]
    Vs = V @ svec[:, sel]
    Vs = Vs / np.maximum(np.linalg.norm(Vs, axis=0, keepdims=True), eps) * np.sqrt(vol)
    embed = D_isq @ Vs
    row_norms = np.linalg.norm(embed, axis=1, keepdims=True)
    embed = embed / np.maximum(row_norms, eps)
    return KMeans(n_clusters=K, n_init=10,
                  random_state=random_state).fit_predict(embed)


# ── Baseline: Lu & Carreira-Perpinan (2008) Algorithm A ───────────────────────

def lu_csc(W_G, W_H, n_clusters, t=1.0, k_eig=150, eps_c=1.0,
           random_state=0, eps=1e-12):
    N = W_G.shape[0]
    deg  = np.array(W_G.sum(axis=1)).ravel()
    isqd = 1.0 / np.sqrt(np.maximum(deg, eps))
    L_sp = eye(N, format='csr') - diags(isqd) @ W_G @ diags(isqd)
    k_eig = min(k_eig, N - 2)
    lam, V = eigsh(L_sp, k=k_eig, sigma=0, which='LM', tol=1e-3)
    scale  = np.exp(-t * lam) - np.exp(-2.0 * t)
    K_mat = (V * scale[None, :]) @ V.T
    K_mat += np.exp(-2.0 * t) * np.eye(N)
    w2    = 1.0 / (eps_c ** 2)
    deg_H = np.array(W_H.sum(axis=1)).ravel()
    M  = w2 * W_H.toarray().astype(np.float64)
    M += w2 * np.diag(deg_H)
    IpMK  = np.eye(N) + M @ K_mat
    Z     = np.linalg.solve(IpMK, np.eye(N));  del IpMK
    K_bar = K_mat @ Z;  del K_mat, Z, M
    K_bar = (K_bar + K_bar.T) * 0.5
    A = np.maximum(0.0, K_bar);  del K_bar
    deg_a  = A.sum(axis=1)
    isqd_a = 1.0 / np.sqrt(np.maximum(deg_a, eps))
    L_sym  = np.eye(N) - (isqd_a[:, None] * A * isqd_a[None, :]);  del A
    _, evecs = sla.eigh(L_sym, subset_by_index=[0, n_clusters - 1]);  del L_sym
    return KMeans(n_clusters=n_clusters, n_init=10,
                  random_state=random_state).fit_predict(evecs)


# ── Dataset loaders ────────────────────────────────────────────────────────────

def load_digits_data():
    from sklearn.datasets import load_digits as _ld
    d = _ld()
    return d.data.astype(np.float32), d.target.astype(int)


def load_olivetti_data():
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces(shuffle=True, random_state=RANDOM_STATE)
    return d.data.astype(np.float32), d.target.astype(int)


def load_wine_data():
    from sklearn.datasets import load_wine as _lw
    d = _lw()
    X = StandardScaler().fit_transform(d.data.astype(np.float32))
    return X, d.target.astype(int)


def load_newsgroups_data():
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    categories = [
        'sci.space', 'rec.sport.hockey',
        'talk.politics.guns', 'comp.graphics',
    ]
    raw = fetch_20newsgroups(
        subset='all', categories=categories,
        remove=('headers', 'footers', 'quotes'),
        random_state=RANDOM_STATE,
    )
    vec = TfidfVectorizer(max_features=5000, stop_words='english', sublinear_tf=True)
    X_tfidf = vec.fit_transform(raw.data)
    svd = TruncatedSVD(n_components=100, random_state=RANDOM_STATE)
    X = svd.fit_transform(X_tfidf).astype(np.float32)
    # L2-normalize so Euclidean distance == cosine distance (standard for text)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    y = np.array(raw.target, dtype=int)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(y), size=min(3900, len(y)), replace=False)
    return X[idx], y[idx]


# ── Experiment configuration ───────────────────────────────────────────────────
# (loader,               display_name,                   K,  k_nn, k_fn, eps_csc)
# k_fn: number of furthest neighbors used to build W_H
EXPERIMENTS = [
    (load_digits_data,    "Digits (sklearn)",             10,    5,   20,    0.7),
    (load_olivetti_data,  "Olivetti Faces",               40,    5,   10,    0.7),
    (load_wine_data,      "Wine (UCI)",                    3,    5,   10,    0.7),
    (load_newsgroups_data,"20 Newsgroups (4 topics)",      4,    3,   25,    0.5),
]


# ── Per-dataset runner ─────────────────────────────────────────────────────────

def run_experiment(loader, display_name, K, k_nn, k_fn, eps_csc):
    print(f"\n{'='*70}")
    print(f"  {display_name}")
    print(f"{'='*70}")

    print("  Loading data...")
    X, y = loader()
    N = len(y)
    print(f"  N={N}  K={K}  k_nn={k_nn}  k_fn={k_fn}  eps_csc={eps_csc}")

    print("  Building W_G (kNN similarity graph)...")
    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)

    print(f"  Building W_H (kFN cannot-link graph, k_fn={k_fn})...")
    W_H = build_kfn_graph(X, k=k_fn)

    results = {}

    # ── 1. SC ──────────────────────────────────────────────────────────────────
    print("\n  [1/4] Unconstrained Spectral Clustering...")
    deg_g = np.array(W_G.sum(axis=1)).ravel()
    L_G   = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_labels, _, _ = spectral_clustering(L_G, n_clusters=K, random_state=RANDOM_STATE)
    results['SC'] = (adjusted_rand_score(y, sc_labels),
                     normalized_mutual_info_score(y, sc_labels))
    print(f"       ARI={results['SC'][0]:.4f}  NMI={results['SC'][1]:.4f}")

    # ── 2. Wang ────────────────────────────────────────────────────────────────
    print("\n  [2/4] Wang et al. (2014) Algorithm 2...")
    _scale = float(W_G.sum()) / max(float(W_H.sum()), 1e-10) * 4.0
    Q_wang = (W_G - _scale * W_H).astype(float)
    try:
        wl = wang_algorithm2(W_G, Q_wang, beta=0.0, K=K, random_state=RANDOM_STATE)
        results['Wang'] = (adjusted_rand_score(y, wl),
                           normalized_mutual_info_score(y, wl))
    except Exception as e:
        results['Wang'] = (float('nan'), float('nan'))
        print(f"       FAILED: {e}")
    print(f"       ARI={results['Wang'][0]:.4f}  NMI={results['Wang'][1]:.4f}")

    # ── 3. Lu ──────────────────────────────────────────────────────────────────
    print("\n  [3/4] Lu & Carreira-Perpinan (2008)...")
    k_eig_lu = min(150, N - 2)
    try:
        lu_labels = lu_csc(W_G, W_H, n_clusters=K, t=1.0, k_eig=k_eig_lu,
                           eps_c=1.0, random_state=RANDOM_STATE)
        results['Lu'] = (adjusted_rand_score(y, lu_labels),
                         normalized_mutual_info_score(y, lu_labels))
    except Exception as e:
        results['Lu'] = (float('nan'), float('nan'))
        print(f"       FAILED: {e}")
    print(f"       ARI={results['Lu'][0]:.4f}  NMI={results['Lu'][1]:.4f}")

    # ── 4. CSC ─────────────────────────────────────────────────────────────────
    print("\n  [4/4] Our Constrained SC (CSC)...")
    ol, _, _ = iterative_clustering(W_G, W_H, k=K, exact=False,
                                     eps=eps_csc, random_seed=RANDOM_STATE)
    results['Ours'] = (adjusted_rand_score(y, ol),
                       normalized_mutual_info_score(y, ol))
    print(f"       ARI={results['Ours'][0]:.4f}  NMI={results['Ours'][1]:.4f}")

    return results, N


def print_table(display_name, N, K, k_nn, k_fn, results):
    W = 68
    print()
    print("=" * W)
    print(f"  {display_name}   N={N}, K={K}, k_nn={k_nn}, k_fn={k_fn}")
    print("=" * W)
    print(f"  {'Method':<43} {'ARI':>10} {'NMI':>10}")
    print("  " + "-" * (W - 2))
    for label, key in [
        ("Spectral Clustering (unconstrained)", "SC"),
        ("Wang et al. (2014) Algorithm 2",      "Wang"),
        ("Lu & Carreira-Perpinan (2008)",        "Lu"),
        ("Our Constrained SC (CSC)",             "Ours"),
    ]:
        a, n = results[key]
        print(f"  {label:<43} {a:>10.4f} {n:>10.4f}")
    print("=" * W)


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    all_results = []

    for loader, display_name, K, k_nn, k_fn, eps_csc in EXPERIMENTS:
        try:
            res, N = run_experiment(loader, display_name, K, k_nn, k_fn, eps_csc)
            all_results.append((display_name, N, K, k_nn, k_fn, res))
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] {display_name} raised: {exc}")
            traceback.print_exc()

    print("\n\n" + "#" * 70)
    print("  CONSOLIDATED RESULTS")
    print("#" * 70)
    for display_name, N, K, k_nn, k_fn, res in all_results:
        print_table(display_name, N, K, k_nn, k_fn, res)
