"""
Comparison on 20 Newsgroups text data with multiple embedding methods.

Subset: 10 selected categories, 500 docs each → N≈4980, K=10, k_nn=10, seed=42.

Three embedding methods are compared (each rebuilds the kNN graph, then runs
all four clustering methods):
  1. NMF-50              — Non-negative topic model on TF-IDF
  2. GloVe-50 (mean)     — averaged pretrained word vectors
                           ('glove-wiki-gigaword-50', gensim)
  3. GloVe-100 IDF-w.    — IDF-weighted average of pretrained word vectors
                           ('glove-wiki-gigaword-100', gensim)

All embeddings are L2-normalized so Euclidean kNN is equivalent to cosine kNN.

For each (embedding, n_cl) cell, four methods are evaluated:
  - Unconstrained Spectral Clustering
  - Wang et al. (2014) Algorithm 2
  - Lu & Carreira-Perpinan (2008)
  - Our Constrained SC

Hyperparameters:
  - Wang : beta = 0
  - Lu   : t = 1, k_eig = 150, eps_c = 1.0
  - Ours : eps swept on a small grid, best ARI reported per (embedding, n_cl)
"""

import sys, re
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh

from utils import (
    build_knn_graph, build_cannot_link_graph,
    spectral_clustering, iterative_clustering,
    normalized_laplacian_from_D_sp,
)

# ── Configuration ─────────────────────────────────────────────────────────────
N_PER_CLASS  = 500
K_NN         = 10        # text needs denser kNN than images
N_CLUSTERS   = 10
RANDOM_STATE = 42
NCL_LIST     = [5, 10, 20]

CATEGORIES = [
    'alt.atheism', 'comp.graphics', 'misc.forsale',
    'rec.autos', 'rec.sport.hockey', 'sci.crypt',
    'sci.med', 'sci.space', 'soc.religion.christian',
    'talk.politics.mideast',
]

EPS_GRID  = [0.05, 0.20, 0.50, 0.70, 0.85, 0.95]
LU_EPS_C  = 1.0

TOKEN_RE = re.compile(r"[A-Za-z]+")


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


# ── Embedding builders ────────────────────────────────────────────────────────
_GLOVE = {}
def _glove(name):
    if name not in _GLOVE:
        import gensim.downloader as api
        _GLOVE[name] = api.load(name)
    return _GLOVE[name]

def emb_nmf(docs, n_topics=50):
    """NMF on TF-IDF; rows are non-negative topic-weight vectors."""
    vec = TfidfVectorizer(stop_words='english', max_features=20000,
                          min_df=3, max_df=0.6, sublinear_tf=True)
    X_tfidf = vec.fit_transform(docs)
    nmf = NMF(n_components=n_topics, init='nndsvd',
              random_state=RANDOM_STATE, max_iter=200)
    H = nmf.fit_transform(X_tfidf)
    return normalize(H.astype(np.float32))

def emb_glove_mean(docs, model_name):
    kv = _glove(model_name);  dim = kv.vector_size
    out = np.zeros((len(docs), dim), dtype=np.float32)
    for i, doc in enumerate(docs):
        vecs = [kv[t] for t in TOKEN_RE.findall(doc.lower()) if t in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return normalize(out)

def emb_glove_idf(docs, model_name):
    """IDF-weighted mean of GloVe vectors."""
    vec = TfidfVectorizer(stop_words='english', max_features=20000,
                          min_df=3, max_df=0.6, sublinear_tf=True)
    vec.fit(docs)
    idf = dict(zip(vec.get_feature_names_out(), vec.idf_))
    kv = _glove(model_name);  dim = kv.vector_size
    out = np.zeros((len(docs), dim), dtype=np.float32)
    for i, doc in enumerate(docs):
        vecs, ws = [], []
        for t in TOKEN_RE.findall(doc.lower()):
            if t in kv and t in idf:
                vecs.append(kv[t]); ws.append(idf[t])
        if vecs:
            ws = np.asarray(ws); ws /= ws.sum()
            out[i] = np.dot(ws, np.asarray(vecs))
    return normalize(out)


EMBEDDINGS = [
    ('NMF-50',          lambda d: emb_nmf(d, n_topics=50)),
    ('GloVe-50 mean',   lambda d: emb_glove_mean(d, 'glove-wiki-gigaword-50')),
    ('GloVe-100 IDF-w', lambda d: emb_glove_idf(d, 'glove-wiki-gigaword-100')),
]


# ── Load + balance dataset ────────────────────────────────────────────────────
def load_20ng_balanced():
    print("Loading 20 Newsgroups...")
    d = fetch_20newsgroups(subset='train',
                           categories=CATEGORIES,
                           remove=('headers', 'footers', 'quotes'),
                           shuffle=False)
    docs   = d.data
    target = d.target
    rng = np.random.default_rng(RANDOM_STATE)
    sel_idx = []
    for c in range(len(CATEGORIES)):
        cls_idx = np.where(target == c)[0]
        chosen  = rng.choice(cls_idx, size=min(N_PER_CLASS, len(cls_idx)),
                              replace=False)
        sel_idx.extend(chosen.tolist())
    sel_idx = np.array(sel_idx);  rng.shuffle(sel_idx)
    docs_sel = [docs[i] for i in sel_idx]
    y_sel    = target[sel_idx]
    print(f"  N={len(docs_sel)}  classes={len(CATEGORIES)}  "
          f"per-class counts={np.bincount(y_sel).tolist()}")
    return docs_sel, y_sel


# ── Per-embedding run ─────────────────────────────────────────────────────────
def run_one_embedding(emb_name, X, y):
    N = X.shape[0]
    print(f"\n{'#'*72}\n#  Embedding: {emb_name}   shape={X.shape}\n{'#'*72}")

    print("Building W_G (kNN)...")
    W_G = build_knn_graph(X, k=K_NN, use_gaussian=False)
    print(f"  W_G avg-deg={W_G.sum()/N:.1f}")

    print("Computing unconstrained SC...")
    deg_g    = np.array(W_G.sum(axis=1)).ravel()
    L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
    sc_lab, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                        random_state=RANDOM_STATE)
    sc_ari = adjusted_rand_score(y, sc_lab)
    sc_nmi = normalized_mutual_info_score(y, sc_lab)
    print(f"  SC  ARI={sc_ari:.4f}  NMI={sc_nmi:.4f}")

    results = {}
    for N_CL in NCL_LIST:
        print(f"\n=== {emb_name}  n_cl={N_CL} ===")
        W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
        print(f"  W_H avg-deg={W_H.sum()/N:.1f}")

        res = {'SC': (sc_ari, sc_nmi)}

        try:
            _scale = float(W_G.sum()) / float(W_H.sum())
            Q_w = (W_G - _scale * W_H).astype(float)
            wl = wang_algorithm2(W_G, Q_w, beta=0.0, K=N_CLUSTERS,
                                  random_state=RANDOM_STATE)
            res['Wang'] = (adjusted_rand_score(y, wl),
                           normalized_mutual_info_score(y, wl))
        except Exception as e:
            res['Wang'] = (float('nan'), float('nan'))
            print(f"  Wang FAILED: {e}")
        print(f"  Wang  ARI={res['Wang'][0]:.4f}  NMI={res['Wang'][1]:.4f}")

        ll = lu_csc(W_G, W_H, n_clusters=N_CLUSTERS, t=1.0, k_eig=150,
                    eps_c=LU_EPS_C, random_state=RANDOM_STATE)
        res['Lu'] = (adjusted_rand_score(y, ll),
                     normalized_mutual_info_score(y, ll))
        print(f"  Lu    ARI={res['Lu'][0]:.4f}  NMI={res['Lu'][1]:.4f}  "
              f"(eps_c={LU_EPS_C})")

        best = (-1.0, -1.0, None)
        for ep in EPS_GRID:
            try:
                ol, _, _ = iterative_clustering(W_G, W_H, k=N_CLUSTERS,
                                                exact=False,
                                                eps=ep,
                                                random_seed=RANDOM_STATE)
                a = adjusted_rand_score(y, ol)
                n = normalized_mutual_info_score(y, ol)
            except Exception:
                a = n = float('nan')
            if not np.isnan(a) and a > best[0]:
                best = (a, n, ep)
        res['Ours'] = (best[0], best[1])
        res['Ours_eps'] = best[2]
        print(f"  Ours  ARI={best[0]:.4f}  NMI={best[1]:.4f}  (eps={best[2]})")

        results[N_CL] = res
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
docs, y = load_20ng_balanced()

all_results = {}
for emb_name, emb_fn in EMBEDDINGS:
    print(f"\n[Building embedding: {emb_name}]")
    X = emb_fn(docs)
    all_results[emb_name] = run_one_embedding(emb_name, X, y)


# ── Summary tables ────────────────────────────────────────────────────────────
W = 88
METHODS = [
    ("Spectral Clustering (unconstrained)", "SC"),
    ("Wang et al. (2014) Algorithm 2",      "Wang"),
    ("Lu & Carreira-Perpinan (2008)",        "Lu"),
    ("Our Constrained SC",                  "Ours"),
]

for emb_name, res in all_results.items():
    print(f"\n{'='*W}")
    print(f"20 Newsgroups  embedding={emb_name}  "
          f"N={len(y)}, K={N_CLUSTERS}, k_nn={K_NN}, seed={RANDOM_STATE}")
    print(f"{'='*W}")
    hdr = f"{'Method':<40}"
    for n_cl in NCL_LIST:
        hdr += f"  n_cl={n_cl:>2} ARI   NMI"
    print(hdr)
    print("-" * W)
    for name, key in METHODS:
        row = f"{name:<40}"
        for n_cl in NCL_LIST:
            a, n = res[n_cl][key]
            row += f"  {a:>8.4f} {n:>6.4f}"
        print(row)
    eps_str = "  ".join(f"n_cl={k}:{res[k]['Ours_eps']}" for k in NCL_LIST)
    print(f"{'(Ours best eps)':<40}  {eps_str}")
    print("=" * W)
