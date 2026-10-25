"""
Runtime benchmark — Ours vs Cucuringu et al. (2016).

For each dataset, both methods are run `REPEATS` times (after a single shared
graph construction) and we report mean +/- standard error of the wall-clock
time per call. Standard error = std(ddof=1) / sqrt(REPEATS).

Datasets:
  * the 16 image / numeric / text datasets from compare_multi_ncl20.py
  * SBM graphs at n in {1000, 2500, 5000, 7500, 10000} (K=10, p=0.2, q=0.13)

Tweak REPEATS (default 2) to control trial count.
"""

import sys
import time
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import networkx as nx

from utils import (
    SBM,
    build_knn_graph, build_cannot_link_graph,
    iterative_clustering,
)

# ---- Reuse the dataset definitions and Cucuringu impl ----------------------
from compare_multi_ncl20 import (
    IMAGE_DATASETS, TEXT_DATASETS,
    _get_balanced_corpus, subsample,
    N_CL, RANDOM_STATE,
)
from compare_multi_ncl20_with_cucuringu import (
    cucuringu_gsc, CUC_REG, CUC_K_EIG,
)

# ---- Benchmark configuration ------------------------------------------------
REPEATS    = 2                                  # trials per (dataset, method)
EPS_OURS   = 0.5                                # fixed eps for our method
SBM_NS     = [1000, 2500, 5000, 7500, 10000]    # SBM sizes
SBM_K      = 10
SBM_P      = 0.2
SBM_Q      = 0.13


# ---- Per-method timed wrappers (graph already built) -----------------------
def _time_cucuringu(W_G, W_H, n_clusters, seed):
    cucuringu_gsc(W_G, W_H, n_clusters=n_clusters,
                  k_eig=CUC_K_EIG, reg=CUC_REG, random_state=seed)

def _time_ours(W_G, W_H, n_clusters, seed):
    iterative_clustering(W_G, W_H, k=n_clusters, exact=False,
                         eps=EPS_OURS, random_seed=seed)


METHODS = [
    ("Cucuringu", "Cucuringu (2016)",            _time_cucuringu),
    ("Ours",      "Ours (eps=%.2f)" % EPS_OURS,  _time_ours),
]


# ---- Time a single method REPEATS times, return (times, ok_flag) ----------
def time_method(fn, W_G, W_H, n_clusters):
    times = []
    for r in range(REPEATS):
        seed = RANDOM_STATE + r
        try:
            t0 = time.perf_counter()
            fn(W_G, W_H, n_clusters, seed)
            times.append(time.perf_counter() - t0)
        except Exception as e:
            print(f"     trial {r} FAILED: {e}")
            times.append(np.nan)
    return np.asarray(times)


# ---- Per-dataset timing (both methods) ------------------------------------
def benchmark(name, W_G, W_H, n_clusters):
    print(f"\n{'='*72}\n  {name}    N={W_G.shape[0]}  K={n_clusters}\n{'='*72}")
    out = {}
    for key, label, fn in METHODS:
        print(f"  [{key}]  ({label}) ...")
        t = time_method(fn, W_G, W_H, n_clusters)
        out[key] = t
        valid = t[~np.isnan(t)]
        if len(valid) == 0:
            print(f"     all trials failed")
        else:
            mean = valid.mean()
            se   = valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
            print(f"     mean={mean:.3f}s   SE={se:.3f}s   trials={t.tolist()}")
    return out


# ---- Build (W_G, W_H) for image / numeric datasets ------------------------
def prep_image(loader, n_clusters, k_nn, n_max):
    print(f"   loading...")
    X, y = loader()
    X, y = subsample(X, y, n_max)
    print(f"   shape={X.shape}, K_true={len(np.unique(y))}")
    print(f"   building W_G (knn k={k_nn}) and W_H (CL n_cl={N_CL})...")
    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)
    W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
    return W_G, W_H


# ---- Build (W_G, W_H) for text datasets -----------------------------------
def prep_text(corpus_id, doc_loader, emb_fn, k_nn, per_class):
    docs, y = _get_balanced_corpus(corpus_id, doc_loader, per_class)
    X       = emb_fn(docs)
    print(f"   embedding shape={X.shape}, K_true={len(np.unique(y))}")
    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)
    W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
    return W_G, W_H, len(np.unique(y))


# ---- Build (W_G, W_H) for SBM ---------------------------------------------
def prep_sbm(n):
    block_size = n // SBM_K
    print(f"   generating SBM  block_size={block_size}, K={SBM_K}, "
          f"p={SBM_P}, q={SBM_Q}...")
    G, H = SBM(block_size=block_size, k=SBM_K, p=SBM_P, q=SBM_Q)
    W_G = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    W_H = nx.to_scipy_sparse_array(H, format='csr').astype(float)
    return W_G, W_H


# ---- Main loop ------------------------------------------------------------
all_times = {}    # name -> {method_key: np.array(REPEATS)}
order     = []    # preserve display order

print(f"\nRunning benchmark with REPEATS={REPEATS}\n")

# image / numeric
for name, loader, K, knn, nmax in IMAGE_DATASETS:
    print(f"\n--> {name}")
    try:
        W_G, W_H = prep_image(loader, K, knn, nmax)
    except Exception as e:
        print(f"   SKIP {name}: {e}")
        continue
    all_times[name] = benchmark(name, W_G, W_H, K)
    order.append((name, K, W_G.shape[0]))

# text
for name, corpus_id, doc_loader, emb_fn, knn, pc in TEXT_DATASETS:
    print(f"\n--> {name}")
    try:
        W_G, W_H, K = prep_text(corpus_id, doc_loader, emb_fn, knn, pc)
    except Exception as e:
        print(f"   SKIP {name}: {e}")
        continue
    all_times[name] = benchmark(name, W_G, W_H, K)
    order.append((name, K, W_G.shape[0]))

# SBM
for n in SBM_NS:
    name = f"SBM (n={n})"
    print(f"\n--> {name}")
    try:
        W_G, W_H = prep_sbm(n)
    except Exception as e:
        print(f"   SKIP {name}: {e}")
        continue
    all_times[name] = benchmark(name, W_G, W_H, SBM_K)
    order.append((name, SBM_K, W_G.shape[0]))


# ---- Summary table --------------------------------------------------------
def fmt_cell(t):
    valid = t[~np.isnan(t)]
    if len(valid) == 0:
        return "      nan"
    mean = valid.mean()
    se   = valid.std(ddof=1) / np.sqrt(len(valid)) if len(valid) > 1 else 0.0
    return f"{mean:7.3f} +/- {se:6.3f}"

W = 26 + 8 + 5 + 2 * 22
print("\n\n" + "=" * W)
print(f"  Runtime summary  (seconds, mean +/- SE over REPEATS={REPEATS})")
print("=" * W)
hdr = f"{'Dataset':<26}{'N':>8}{'K':>5}"
for key, _, _ in METHODS:
    hdr += f"  {key:>18}"
print(hdr)
print("-" * W)

for name, K, N in order:
    row = f"{name:<26}{N:>8}{K:>5}"
    res = all_times[name]
    for key, _, _ in METHODS:
        row += f"  {fmt_cell(res[key]):>18}"
    print(row)

print("=" * W)
print(f"\nNotes:")
print(f"  * Times exclude graph (W_G, W_H) construction.")
print(f"  * Cucuringu may report nan when the generalized eigsolver fails")
print(f"    (falls back to dense; OOM on large N).")
print(f"  * Ours uses fixed eps={EPS_OURS} (no grid sweep, single call per trial).")
