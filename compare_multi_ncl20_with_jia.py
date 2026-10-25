"""
Multi-dataset comparison at n_cl = 20 (fixed) — extends compare_multi_ncl20.py
with the dissimilarity-propagation-guided graph-Laplacian PCA method of

  Y. Jia, J. Hou, S. Kwong,
  "Constrained Clustering With Dissimilarity Propagation-Guided
   Graph-Laplacian PCA," IEEE TNNLS, 2021.

Same 16 datasets — 12 image/numeric + 2 text corpora x 2 embedding strategies.

  Image / numeric (subsample N <= 5000)
    1.  MNIST                 (28x28,  K=10)
    2.  USPS                  (16x16,  K=10) — local h5
    3.  Fashion-MNIST         (28x28,  K=10)
    4.  Pendigits             (16-d,   K=10)
    5.  Optdigits             (64-d,   K=10)
    6.  Letter-Recognition    (16-d,   K=26)
    7.  Digits-8x8 (sklearn)  (64-d,   K=10)
    8.  Olivetti-Faces        (4096-d, K=40)
    9.  COIL-20               (1024-d, K=20) — openml 'coil-20'
   10.  mfeat-pixel           (240-d,  K=10) — openml multi-feature digits
   11.  Semeion handwritten   (256-d,  K=10) — openml 'semeion'
   12.  Human Activity (HAR)  (561-d,  K=6)  — openml 'Human-Activity-Recognition'

  Text — same corpora under two embedding strategies, k_nn = 25
   13.  BBC-News + TF-IDF -> LSA-300            (no install)
   14.  BBC-News + Sentence-Transformers MiniLM (pip install sentence-transformers)
   15.  AG-News  + TF-IDF -> LSA-300            (no install)
   16.  AG-News  + Sentence-Transformers MiniLM (pip install sentence-transformers)

Methods compared at n_cl = 20:
  - Unconstrained Spectral Clustering          (SC)
  - Wang et al. (2014) Algorithm 2             (Wang, beta = 0)
  - Lu & Carreira-Perpinan (2008) Algorithm A  (Lu,   t = 1, k_eig = 150, eps_c = 1.0)
  - Jia, Hou & Kwong (TNNLS 2021)              (Jia,  alpha = 0.5, beta = 0.5, auto-scaled D̃)
  - Our Constrained SC                         (Ours, eps = best of small grid)

DP-GLPCA (Jia et al.) recap
---------------------------
Let W_G be the kNN similarity, S = D^{-1/2} W_G D^{-1/2} the symmetric-
normalized adjacency, and L = I - S the normalized Laplacian.
Let D0 be the cannot-link weight matrix (zero on must-link pairs / diagonal).
The dissimilarity is propagated along the data manifold by

    D_{t+1} = α S D_t S + (1 - α) D0   →   D̃   (fixed point)

and the embedding F (N × K) solves the GLPCA-style problem

    min  tr(F^T (L - β D̃) F)     s.t.  F^T F = I.

That is, F = K smallest eigenvectors of L - β D̃; row-normalize → k-means.
"""

import re
import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
from scipy.sparse import diags, eye

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import normalize
from scipy.sparse.linalg import eigsh

from utils import (
    build_knn_graph, build_cannot_link_graph,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Global configuration ─────────────────────────────────────────────────────
N_CL         = 20             # fixed
N_MAX        = 5000           # cap for image/numeric datasets
RANDOM_STATE = 42
EPS_GRID     = [0.05, 0.20, 0.50, 0.70, 0.85, 0.95]
LU_T         = 1.0
LU_K_EIG     = 150
LU_EPS_C     = 1.0
WANG_BETA    = 0.0
JIA_ALPHA    = 0.5
JIA_BETA     = 0.5            # interpreted in auto-scaled space (‖D̃‖_F = ‖L‖_F)
JIA_PROP_ITERS = 10
TOKEN_RE     = re.compile(r"[A-Za-z]+")

BOLD = "\033[1m";  RESET = "\033[0m"


# ── Wang et al. (2014) Algorithm 2 ────────────────────────────────────────────
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


# ── Lu & Carreira-Perpinan (2008) Algorithm A ─────────────────────────────────
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

    IpMK  = np.eye(N) + M @ K
    Z     = np.linalg.solve(IpMK, np.eye(N));  del IpMK
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


# ── Jia, Hou & Kwong (TNNLS 2021) — DP-GLPCA ─────────────────────────────────
def jia_propagate(W_G, W_H, alpha=0.5, n_prop_iters=15,
                  auto_scale=True, eps=1e-12):
    """
    Build the normalized Laplacian L of W_G and the propagated
    dissimilarity D̃ from W_H. With auto_scale=True, D̃ is rescaled so
    ‖D̃‖_F == ‖L‖_F — that way β has comparable meaning across datasets.
    """
    N = W_G.shape[0]
    deg  = np.array(W_G.sum(axis=1)).ravel()
    isq  = 1.0 / np.sqrt(np.maximum(deg, eps))
    D_isq = diags(isq)
    S_sp = (D_isq @ W_G @ D_isq).tocsr()             # symmetric
    L_sp = eye(N, format='csr') - S_sp

    D0 = W_H.toarray().astype(np.float64)
    np.fill_diagonal(D0, 0.0)

    # D ← α S D S + (1-α) D0  (S symmetric → (S(SD)^T)^T = S D S)
    D_t   = D0.copy()
    omA_D0 = (1.0 - alpha) * D0
    for _ in range(n_prop_iters):
        D_t = alpha * (S_sp @ (S_sp @ D_t).T).T + omA_D0
    D_tilde = (D_t + D_t.T) * 0.5
    np.fill_diagonal(D_tilde, 0.0)

    if auto_scale:
        fro_L = float(np.sqrt((L_sp.data ** 2).sum()))
        fro_D = float(np.linalg.norm(D_tilde))
        if fro_D > eps:
            D_tilde *= (fro_L / fro_D)
    return L_sp, D_tilde


def jia_solve(L_sp, D_tilde, beta, n_clusters,
              random_state=0, eps=1e-12):
    """min tr(Fᵀ(L - β D̃)F) s.t. FᵀF=I → smallest K eigvecs → row-norm → k-means."""
    M = L_sp.toarray() - beta * D_tilde
    M = (M + M.T) * 0.5
    _, F = sla.eigh(M, subset_by_index=[0, n_clusters - 1])
    norms = np.linalg.norm(F, axis=1, keepdims=True)
    F = F / np.maximum(norms, eps)
    return KMeans(n_clusters=n_clusters, n_init=10,
                  random_state=random_state).fit_predict(F)


def jia_dpglpca(W_G, W_H, n_clusters, alpha=0.5, beta=1.0,
                n_prop_iters=15, auto_scale=True,
                random_state=0, eps=1e-12):
    """One-shot wrapper: propagate + solve."""
    L_sp, D_tilde = jia_propagate(W_G, W_H, alpha, n_prop_iters,
                                  auto_scale=auto_scale, eps=eps)
    return jia_solve(L_sp, D_tilde, beta, n_clusters,
                     random_state=random_state, eps=eps)


# ── Image / numeric loaders ──────────────────────────────────────────────────
def _try_keras(name):
    if name == 'mnist':
        import keras.datasets.mnist as km
        (X, y), _ = km.load_data()
    else:
        import keras.datasets.fashion_mnist as fm
        (X, y), _ = fm.load_data()
    return X.reshape(len(X), -1).astype(np.float32), y.astype(int)

def load_mnist():
    try:
        return _try_keras('mnist')
    except Exception:
        from sklearn.datasets import fetch_openml
        d = fetch_openml('mnist_784', version=1, as_frame=False)
        return d.data.astype(np.float32), d.target.astype(int)

def load_fashion():
    try:
        return _try_keras('fashion')
    except Exception:
        from sklearn.datasets import fetch_openml
        d = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
        return d.data.astype(np.float32), d.target.astype(int)

def load_usps():
    import h5py
    path = '/home/songs/ConstrainedSpectralClustering/datasets/usps/usps.h5'
    with h5py.File(path, 'r') as hf:
        train = hf.get('train')
        X = train.get('data')[:]
        y = train.get('target')[:]
    return X.reshape(len(X), -1).astype(np.float32), np.asarray(y, dtype=int)

def load_pendigits():
    from sklearn.datasets import fetch_openml
    d = fetch_openml('pendigits', version=1, as_frame=False, parser='liac-arff')
    return d.data.astype(np.float32), d.target.astype(int)

def load_optdigits():
    from sklearn.datasets import fetch_openml
    d = fetch_openml('optdigits', version=1, as_frame=False, parser='liac-arff')
    return d.data.astype(np.float32), d.target.astype(int)

def load_letter():
    from sklearn.datasets import fetch_openml
    d = fetch_openml('letter', version=1, as_frame=False, parser='liac-arff')
    X = d.data.astype(np.float32)
    y = np.array([ord(c) - ord('A') for c in np.asarray(d.target)], dtype=int)
    return X, y

def load_digits8x8():
    from sklearn.datasets import load_digits
    d = load_digits()
    return d.data.astype(np.float32), d.target.astype(int)

def load_olivetti():
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces(random_state=RANDOM_STATE)
    return d.data.astype(np.float32), d.target.astype(int)

def _try_openml_first_match(candidates):
    """Try each (name, version) in order; return the first that loads."""
    from sklearn.datasets import fetch_openml
    last_err = None
    for cfg in candidates:
        try:
            d = fetch_openml(**cfg, as_frame=False, parser='auto')
            return d
        except Exception as e:
            last_err = e
    raise RuntimeError(f"All openml candidates failed: {last_err}")

def load_coil20():
    d = _try_openml_first_match([
        {'name': 'coil-20',  'version': 1},
        {'name': 'COIL-20',  'version': 1},
        {'data_id': 41960},
    ])
    X = np.asarray(d.data, dtype=np.float32)
    y_raw = np.asarray(d.target)
    try:
        y = y_raw.astype(int)
    except (ValueError, TypeError):
        _, y = np.unique(y_raw, return_inverse=True)
    return X, y

def load_mfeat_pixel():
    d = _try_openml_first_match([
        {'name': 'mfeat-pixel', 'version': 1},
        {'data_id': 20},
    ])
    X = np.asarray(d.data, dtype=np.float32)
    y_raw = np.asarray(d.target)
    try:
        y = y_raw.astype(int)
    except (ValueError, TypeError):
        _, y = np.unique(y_raw, return_inverse=True)
    return X, y

def load_semeion():
    d = _try_openml_first_match([
        {'name': 'semeion', 'version': 1},
        {'data_id': 1501},
    ])
    X = np.asarray(d.data, dtype=np.float32)
    y_raw = np.asarray(d.target)
    try:
        y = y_raw.astype(int)
    except (ValueError, TypeError):
        _, y = np.unique(y_raw, return_inverse=True)
    return X, y

def load_har():
    d = _try_openml_first_match([
        {'name': 'Human-Activity-Recognition-Using-Smartphones',
         'version': 1},
        {'name': 'har',       'version': 1},
        {'name': 'UCI_HAR',   'version': 1},
        {'data_id': 1478},
    ])
    X = np.asarray(d.data, dtype=np.float32)
    y_raw = np.asarray(d.target)
    try:
        y = y_raw.astype(int)
    except (ValueError, TypeError):
        _, y = np.unique(y_raw, return_inverse=True)
    return X, y


IMAGE_DATASETS = [
    # (display_name, loader, n_clusters, k_nn, n_max_override)
    ('MNIST',              load_mnist,       10,  5, N_MAX),
    ('USPS',               load_usps,        10,  5, N_MAX),
    ('Fashion-MNIST',      load_fashion,     10,  5, N_MAX),
    ('Pendigits',          load_pendigits,   10,  5, N_MAX),
    ('Optdigits',          load_optdigits,   10,  5, N_MAX),
    ('Letter-Recognition', load_letter,      26,  7, N_MAX),
    ('Digits-8x8',         load_digits8x8,   10,  5, None),   # full 1797
    ('Olivetti-Faces',     load_olivetti,    40,  3, None),   # full 400
    ('COIL-20',            load_coil20,      20,  5, N_MAX),
    ('mfeat-pixel',        load_mfeat_pixel, 10,  5, N_MAX),
    ('Semeion',            load_semeion,     10,  5, N_MAX),
    ('HAR',                load_har,          6,  7, N_MAX),
]


# ── Text embeddings ──────────────────────────────────────────────────────────
def emb_lsa300(docs, n_components=300):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    vec = TfidfVectorizer(stop_words='english', max_features=30000,
                          min_df=3, max_df=0.6, sublinear_tf=True)
    X = vec.fit_transform(docs)
    n_comp = min(n_components, min(X.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=RANDOM_STATE)
    H = svd.fit_transform(X)
    return normalize(H.astype(np.float32))

_ST_CACHE = {}
def emb_minilm(docs, model_name='all-MiniLM-L6-v2'):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers not installed; "
            "run `pip install sentence-transformers` to enable this embedding"
        ) from e
    if model_name not in _ST_CACHE:
        _ST_CACHE[model_name] = SentenceTransformer(model_name)
    model = _ST_CACHE[model_name]
    X = model.encode(docs, batch_size=64, show_progress_bar=False,
                     convert_to_numpy=True, normalize_embeddings=True)
    return X.astype(np.float32)


def _balance_subsample(docs, y, per_class):
    rng = np.random.default_rng(RANDOM_STATE)
    sel = []
    for c in np.unique(y):
        cls_idx = np.where(y == c)[0]
        chosen  = rng.choice(cls_idx, size=min(per_class, len(cls_idx)),
                             replace=False)
        sel.extend(chosen.tolist())
    sel = np.asarray(sel);  rng.shuffle(sel)
    return [docs[i] for i in sel], y[sel]


def load_bbc_news():
    """BBC news (5 categories: business, entertainment, politics, sport, tech)."""
    d = _try_openml_first_match([
        {'name': 'BBC-news',  'version': 1},
        {'name': 'bbc-news',  'version': 1},
        {'name': 'BBC',       'version': 1},
        {'data_id': 43759},
    ])
    docs = [str(t) for t in np.asarray(d.data).ravel()]
    y_raw = np.asarray(d.target)
    _, y = np.unique(y_raw, return_inverse=True)
    return docs, y

def load_ag_news():
    """AG News (4 classes: World, Sports, Business, Sci/Tech)."""
    d = _try_openml_first_match([
        {'name': 'ag_news',          'version': 1},
        {'name': 'AG_News',          'version': 1},
        {'name': 'ag-news-subset',   'version': 1},
        {'data_id': 45069},
    ])
    raw = np.asarray(d.data)
    if raw.ndim == 2 and raw.shape[1] >= 1:
        docs = [str(r[0]) for r in raw]
    else:
        docs = [str(t) for t in raw.ravel()]
    y_raw = np.asarray(d.target)
    _, y = np.unique(y_raw, return_inverse=True)
    return docs, y


# (display_name, corpus_id, doc_loader, embedding_fn, k_nn, per_class_cap)
TEXT_DATASETS = [
    ('BBC-News (TF-IDF→LSA-300)', 'bbc', load_bbc_news, emb_lsa300, 25, 1000),
    ('BBC-News (MiniLM-L6)',      'bbc', load_bbc_news, emb_minilm, 25, 1000),
    ('AG-News  (TF-IDF→LSA-300)', 'ag',  load_ag_news,  emb_lsa300, 25, 1250),
    ('AG-News  (MiniLM-L6)',      'ag',  load_ag_news,  emb_minilm, 25, 1250),
]


# ── Per-dataset evaluator ────────────────────────────────────────────────────
def evaluate(name, X, y, n_clusters, k_nn):
    """Run all five methods on (X, y); return dict of (ARI, NMI) per method."""
    N = X.shape[0]
    print(f"\n{'#'*72}\n#  {name}   N={N}  d={X.shape[1]}  K={n_clusters}  "
          f"k_nn={k_nn}\n{'#'*72}")

    print("  Building W_G (kNN)...")
    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)
    print(f"    W_G avg-deg={W_G.sum()/N:.1f}")

    print("  Building W_H (CL, n_cl=20)...")
    W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
    print(f"    W_H avg-deg={W_H.sum()/N:.1f}")

    res = {}

    print("  [SC]  ...")
    deg_g    = np.array(W_G.sum(axis=1)).ravel()
    L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_lab, _, _ = spectral_clustering(L_G_norm, n_clusters=n_clusters,
                                        random_state=RANDOM_STATE)
    res['SC'] = (adjusted_rand_score(y, sc_lab),
                 normalized_mutual_info_score(y, sc_lab))
    print(f"    ARI={res['SC'][0]:.4f}  NMI={res['SC'][1]:.4f}")

    print("  [Wang]...")
    try:
        scale = float(W_G.sum()) / float(W_H.sum())
        Q     = (W_G - scale * W_H).astype(float)
        wl = wang_algorithm2(W_G, Q, beta=WANG_BETA, K=n_clusters,
                              random_state=RANDOM_STATE)
        res['Wang'] = (adjusted_rand_score(y, wl),
                       normalized_mutual_info_score(y, wl))
    except Exception as e:
        res['Wang'] = (float('nan'), float('nan'))
        print(f"    FAILED: {e}")
    print(f"    ARI={res['Wang'][0]:.4f}  NMI={res['Wang'][1]:.4f}")

    print(f"  [Lu]   (eps_c={LU_EPS_C})...")
    try:
        ll = lu_csc(W_G, W_H, n_clusters=n_clusters, t=LU_T, k_eig=LU_K_EIG,
                    eps_c=LU_EPS_C, random_state=RANDOM_STATE)
        res['Lu'] = (adjusted_rand_score(y, ll),
                     normalized_mutual_info_score(y, ll))
    except Exception as e:
        res['Lu'] = (float('nan'), float('nan'))
        print(f"    FAILED: {e}")
    print(f"    ARI={res['Lu'][0]:.4f}  NMI={res['Lu'][1]:.4f}")

    print(f"  [Jia]  (alpha={JIA_ALPHA}, beta={JIA_BETA}, "
          f"iters={JIA_PROP_ITERS}, auto-scaled D̃)...")
    try:
        jl = jia_dpglpca(W_G, W_H, n_clusters=n_clusters,
                         alpha=JIA_ALPHA, beta=JIA_BETA,
                         n_prop_iters=JIA_PROP_ITERS,
                         auto_scale=True,
                         random_state=RANDOM_STATE)
        res['Jia'] = (adjusted_rand_score(y, jl),
                      normalized_mutual_info_score(y, jl))
    except Exception as e:
        res['Jia'] = (float('nan'), float('nan'))
        print(f"    FAILED: {e}")
    print(f"    ARI={res['Jia'][0]:.4f}  NMI={res['Jia'][1]:.4f}")

    print("  [Ours] sweeping eps...")
    best = (-np.inf, -np.inf, None)
    for ep in EPS_GRID:
        try:
            ol, _, _ = iterative_clustering(W_G, W_H, k=n_clusters, exact=False,
                                             eps=ep, random_seed=RANDOM_STATE)
            a = adjusted_rand_score(y, ol)
            n = normalized_mutual_info_score(y, ol)
        except Exception:
            a = n = float('nan')
        if not np.isnan(a) and a > best[0]:
            best = (a, n, ep)
    res['Ours']     = (best[0], best[1])
    res['Ours_eps'] = best[2]
    print(f"    ARI={best[0]:.4f}  NMI={best[1]:.4f}  (best eps={best[2]})")

    return res


def subsample(X, y, n_max):
    if n_max is None or len(X) <= n_max:
        return X, y
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(X), size=n_max, replace=False)
    return X[idx], y[idx]


def run_image_dataset(name, loader, n_clusters, k_nn, n_max):
    try:
        print(f"\n→ Loading {name}...")
        X, y = loader()
        print(f"   raw shape = {X.shape},  classes seen = {len(np.unique(y))}")
        X, y = subsample(X, y, n_max)
    except Exception as e:
        print(f"   SKIP {name}: load failed — {e}")
        return None
    return evaluate(name, X, y, n_clusters, k_nn)


_TEXT_CORPUS_CACHE = {}    # corpus_id -> (docs_balanced, y_balanced)
def _get_balanced_corpus(corpus_id, doc_loader, per_class):
    if corpus_id in _TEXT_CORPUS_CACHE:
        return _TEXT_CORPUS_CACHE[corpus_id]
    print(f"\n→ Loading raw corpus '{corpus_id}'...")
    docs, y = doc_loader()
    print(f"   raw N={len(docs)}, classes={len(np.unique(y))}, "
          f"counts={np.bincount(y).tolist()}")
    docs, y = _balance_subsample(docs, y, per_class)
    print(f"   balanced N={len(docs)}, counts={np.bincount(y).tolist()}")
    _TEXT_CORPUS_CACHE[corpus_id] = (docs, y)
    return docs, y


def run_text_dataset(name, corpus_id, doc_loader, emb_fn, k_nn, per_class):
    try:
        docs, y = _get_balanced_corpus(corpus_id, doc_loader, per_class)
        print(f"\n→ {name}: building embedding...")
        X = emb_fn(docs)
        print(f"   embedding shape = {X.shape}")
    except Exception as e:
        print(f"   SKIP {name}: {e}")
        return None
    return evaluate(name, X, y, len(np.unique(y)), k_nn)


# ── Main ─────────────────────────────────────────────────────────────────────
all_results = {}    # name -> dict of method -> (ari, nmi)

for name, loader, K, knn, nmax in IMAGE_DATASETS:
    t0 = time.time()
    r = run_image_dataset(name, loader, K, knn, nmax)
    if r is not None:
        all_results[name] = r
    print(f"   ({name} took {time.time()-t0:.1f}s)")

for name, corpus_id, doc_loader, emb_fn, knn, pc in TEXT_DATASETS:
    t0 = time.time()
    r = run_text_dataset(name, corpus_id, doc_loader, emb_fn, knn, pc)
    if r is not None:
        all_results[name] = r
    print(f"   ({name} took {time.time()-t0:.1f}s)")


# ── Summary table  (best method per dataset is BOLDED) ──────────────────────
W = 118
METHODS = [
    ("SC",                            "SC"),
    ("Wang (2014)",                   "Wang"),
    ("Lu & C-P  (2008)",              "Lu"),
    ("Jia et al. (2021)",             "Jia"),
    ("Ours",                          "Ours"),
]

print(f"\n\n{'='*W}")
print(f"  Summary  —  n_cl = {N_CL}  fixed,  seed = {RANDOM_STATE}    "
      f"({BOLD}bold{RESET} = best ARI per row)")
print(f"{'='*W}")

hdr = f"{'Dataset':<26}"
for label, _ in METHODS:
    hdr += f"  {label:>14} ARI / NMI"
print(hdr)
print("-" * W)

CELL_W = 24   # printable width of each "ARI / NMI" cell
for name, res in all_results.items():
    aris = [(key, res[key][0], res[key][1]) for _, key in METHODS]
    valid = [(k, a, n) for k, a, n in aris if not np.isnan(a)]
    best_key = max(valid, key=lambda t: t[1])[0] if valid else None

    row = f"{name:<26}"
    for _, key in METHODS:
        a, n = res[key]
        if np.isnan(a):
            cell_text = " " * 9 + "nan / nan"
        else:
            cell_text = f"  {a:>8.4f} / {n:>6.4f}"
        # pad to fixed visible width, then wrap with ANSI bold
        pad = max(0, CELL_W - len(cell_text))
        if key == best_key:
            row += " " * pad + BOLD + cell_text + RESET
        else:
            row += " " * pad + cell_text
    print(row)

print("=" * W)

# Per-dataset best parameter — Ours only (Jia uses fixed config)
print(f"\nJia — fixed (alpha={JIA_ALPHA}, beta={JIA_BETA}, "
      f"iters={JIA_PROP_ITERS}, auto-scaled D̃)")

print("\nOurs — best eps per dataset:")
for name, r in all_results.items():
    if 'Ours_eps' in r:
        print(f"  {name:<26}  eps = {r['Ours_eps']}")
