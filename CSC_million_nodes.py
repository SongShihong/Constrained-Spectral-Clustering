"""
CSC efficiency test on a million-node graph.

Instead of SBM with fixed probabilities (which creates O(n^2) edges),
each node is assigned a fixed number of must-link and cannot-link neighbors,
keeping the graph sparse (O(n) edges total).

Graph structure:
  - n = 1_000_000 nodes, 10 clusters of 100_000 nodes each
  - Must-link graph G : each node connects to k_ml random same-cluster nodes
  - Cannot-link graph H: each node connects to k_cl random different-cluster nodes
"""

import numpy as np
import time
import os
from scipy.sparse import csr_matrix
from utils import (
    normalized_laplacian_from_D_sp,
    iterative_clustering,
)

# ── configuration ────────────────────────────────────────────────────────────
N_NODES   = 1_000_000
N_CLUSTERS = 10
K_ML      = 10   # must-link  neighbors per node (same cluster)
K_CL      = 10   # cannot-link neighbors per node (different cluster)
RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────────────────────────

timings = {}
rng = np.random.default_rng(RANDOM_SEED)

print(f"Configuration: n={N_NODES:,}, k={N_CLUSTERS}, k_ml={K_ML}, k_cl={K_CL}")
print("=" * 60)


# ── 1. assign true labels ─────────────────────────────────────────────────────
t0 = time.perf_counter()
block_size = N_NODES // N_CLUSTERS
true_labels = np.repeat(np.arange(N_CLUSTERS), block_size)
# node indices grouped by cluster
cluster_idx = [np.where(true_labels == c)[0] for c in range(N_CLUSTERS)]
timings["label_assignment"] = time.perf_counter() - t0
print(f"[{timings['label_assignment']:.2f}s] Labels assigned")


# ── 2. build must-link graph G (constant-degree sparse) ─────────────────────
t0 = time.perf_counter()

rows_G, cols_G = [], []
for c in range(N_CLUSTERS):
    idx = cluster_idx[c]
    n_c = len(idx)
    # for each node pick k_ml distinct neighbors (within same cluster, excluding self)
    k = min(K_ML, n_c - 1)
    # draw all neighbors at once: shape (n_c, n_c-1), then trim to k columns
    all_others = np.tile(np.arange(n_c), (n_c, 1))
    mask = ~np.eye(n_c, dtype=bool)
    # use rng to shuffle each row independently
    shuffled = rng.permuted(all_others * mask + (1 - mask.astype(int)) * n_c, axis=1)
    # take first k valid (non-self) columns
    chosen_local = shuffled[:, :k]   # local indices within cluster
    src_global = np.repeat(idx, k)
    dst_global = idx[chosen_local.ravel()]
    rows_G.append(src_global)
    cols_G.append(dst_global)

rows_G = np.concatenate(rows_G)
cols_G = np.concatenate(cols_G)
vals_G = np.ones(len(rows_G), dtype=np.float32)
W_G = csr_matrix((vals_G, (rows_G, cols_G)), shape=(N_NODES, N_NODES))
W_G = (W_G + W_G.T)   # symmetrize
W_G.data = np.ones(len(W_G.data), dtype=np.float32)  # binarize weights

timings["build_G"] = time.perf_counter() - t0
print(f"[{timings['build_G']:.2f}s] Must-link graph G built  "
      f"({W_G.nnz:,} edges, {W_G.nnz / N_NODES:.1f} avg degree)")


# ── 3. build cannot-link graph H (constant-degree sparse) ───────────────────
t0 = time.perf_counter()

rows_H, cols_H = [], []
for c in range(N_CLUSTERS):
    idx = cluster_idx[c]
    n_c = len(idx)
    # pool all nodes NOT in this cluster
    other_idx = np.concatenate([cluster_idx[o] for o in range(N_CLUSTERS) if o != c])
    k = min(K_CL, len(other_idx))
    # sample k cannot-link neighbors for every node in this cluster
    chosen = rng.integers(0, len(other_idx), size=(n_c, k))
    src_global = np.repeat(idx, k)
    dst_global = other_idx[chosen.ravel()]
    rows_H.append(src_global)
    cols_H.append(dst_global)

rows_H = np.concatenate(rows_H)
cols_H = np.concatenate(cols_H)
vals_H = np.ones(len(rows_H), dtype=np.float32)
W_H = csr_matrix((vals_H, (rows_H, cols_H)), shape=(N_NODES, N_NODES))
W_H = (W_H + W_H.T)
W_H.data = np.ones(len(W_H.data), dtype=np.float32)

timings["build_H"] = time.perf_counter() - t0
print(f"[{timings['build_H']:.2f}s] Cannot-link graph H built "
      f"({W_H.nnz:,} edges, {W_H.nnz / N_NODES:.1f} avg degree)")


# ── 4. build Laplacians ───────────────────────────────────────────────────────
t0 = time.perf_counter()
eps = 0.9
deg_g = np.array(W_G.sum(axis=1)).ravel()
deg_h = np.array(W_H.sum(axis=1)).ravel()
deg   = eps * deg_g + (1 - eps) * deg_h

L_G = normalized_laplacian_from_D_sp(W_G, deg)
L_H = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
timings["build_laplacians"] = time.perf_counter() - t0
print(f"[{timings['build_laplacians']:.2f}s] Laplacians built")


# ── 5. constrained spectral clustering (LOBPCG) ───────────────────────────────
t0 = time.perf_counter()
from sklearn.cluster import KMeans
from scipy.sparse.linalg import lobpcg

np.random.seed(RANDOM_SEED)
X0 = np.random.randn(N_NODES, N_CLUSTERS).astype(np.float64)
print("Running LOBPCG eigensolver …")
vals, vecs = lobpcg(L_G, X0, B=L_H, largest=False, tol=1e-5, maxiter=300)
timings["eigensolver"] = time.perf_counter() - t0
print(f"[{timings['eigensolver']:.2f}s] LOBPCG converged, eigenvalues: {np.sort(vals)}")


# ── 6. k-means on embedding ───────────────────────────────────────────────────
t0 = time.perf_counter()
print("Running KMeans …")
kmeans = KMeans(n_clusters=N_CLUSTERS, n_init=5, random_state=RANDOM_SEED)
pred_labels = kmeans.fit_predict(vecs)
timings["kmeans"] = time.perf_counter() - t0
print(f"[{timings['kmeans']:.2f}s] KMeans done")


# ── 7. evaluate ───────────────────────────────────────────────────────────────
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
ari = adjusted_rand_score(true_labels, pred_labels)
nmi = normalized_mutual_info_score(true_labels, pred_labels)
print(f"\nARI = {ari:.4f}   NMI = {nmi:.4f}")


# ── 8. summary ────────────────────────────────────────────────────────────────
total = sum(timings.values())
timings["total"] = total

print("\n" + "=" * 60)
print("TIMING SUMMARY")
print("=" * 60)
for k, v in timings.items():
    print(f"  {k:<25s} {v:8.2f} s")
print("=" * 60)
print(f"  {'TOTAL':<25s} {total:8.2f} s")


# ── 9. save report ────────────────────────────────────────────────────────────
report_path = os.path.join(os.path.dirname(__file__), "csc_million_nodes_report.txt")
with open(report_path, "w") as f:
    f.write("CSC Million-Node Efficiency Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"Date        : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Nodes       : {N_NODES:,}\n")
    f.write(f"Clusters    : {N_CLUSTERS}\n")
    f.write(f"k_ml        : {K_ML}  (must-link neighbors/node)\n")
    f.write(f"k_cl        : {K_CL}  (cannot-link neighbors/node)\n")
    f.write(f"G edges     : {W_G.nnz:,}  (avg degree {W_G.nnz/N_NODES:.1f})\n")
    f.write(f"H edges     : {W_H.nnz:,}  (avg degree {W_H.nnz/N_NODES:.1f})\n")
    f.write(f"epsilon     : {eps}\n")
    f.write("-" * 60 + "\n")
    f.write("TIMING BREAKDOWN\n")
    for k, v in timings.items():
        f.write(f"  {k:<25s} {v:8.2f} s\n")
    f.write("-" * 60 + "\n")
    f.write("CLUSTERING QUALITY\n")
    f.write(f"  ARI = {ari:.4f}\n")
    f.write(f"  NMI = {nmi:.4f}\n")
    f.write("=" * 60 + "\n")

print(f"\nReport saved to: {report_path}")
