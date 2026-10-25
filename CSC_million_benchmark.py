"""
CSC Efficiency Benchmark on Large Graphs (up to 1 million nodes)

Instead of probability-based SBM (which creates O(n^2 * p) edges),
each node has exactly k_ml must-link neighbors (same cluster)
and exactly k_cl cannot-link neighbors (different cluster).

Graph construction is fully vectorized using NumPy; no NetworkX.
"""

import numpy as np
from scipy.sparse import csr_matrix
import time
import traceback
import warnings

warnings.filterwarnings("ignore", message=".*KMeans is known to have a memory leak.*")

from utils import normalized_laplacian_from_D_sp, constrained_spectral_clustering


# ---------------------------------------------------------------------------
# Sparse graph builders
# ---------------------------------------------------------------------------

def build_must_link_sparse(n_nodes, n_clusters, k_ml=5, seed=42):
    """
    For each node, sample exactly k_ml neighbors from the SAME cluster.

    Vectorized trick (no Python loops over nodes):
      - Sample rand_idx in [0, n_c - 1) for each node
      - For node i: if rand_idx[i] >= i, increment by 1  → skips self
    """
    rng = np.random.default_rng(seed)

    base = n_nodes // n_clusters
    cluster_sizes = [base] * n_clusters
    cluster_sizes[-1] += n_nodes - base * n_clusters  # last cluster gets remainder

    all_rows, all_cols = [], []
    offset = 0

    for n_c in cluster_sizes:
        k = min(k_ml, n_c - 1)
        node_ids = np.arange(n_c, dtype=np.int32)

        # shape (n_c, k): random indices in [0, n_c-1)
        rand_idx = rng.integers(0, n_c - 1, size=(n_c, k), dtype=np.int32)
        # shift to skip self
        rand_idx[rand_idx >= node_ids[:, np.newaxis]] += 1

        all_rows.append(np.repeat(node_ids, k) + offset)
        all_cols.append(rand_idx.ravel() + offset)
        offset += n_c

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.ones(len(rows), dtype=np.float32)

    W = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    W = (W + W.T)
    W.data[:] = 1.0   # binarize (symmetrization may double some)
    return W


def build_cannot_link_sparse(n_nodes, n_clusters, k_cl=5, seed=99):
    """
    For each node, sample exactly k_cl neighbors from DIFFERENT clusters.

    Vectorized trick (no Python loops over nodes):
      - For cluster at [offset, offset+n_c): "other" nodes are
        [0, offset) ∪ [offset+n_c, n_nodes), total n_other = n_nodes - n_c
      - Sample rand_idx in [0, n_other) per node
      - Map: if rand_idx >= offset  →  add n_c  (skip own cluster range)
    """
    rng = np.random.default_rng(seed)

    base = n_nodes // n_clusters
    cluster_sizes = [base] * n_clusters
    cluster_sizes[-1] += n_nodes - base * n_clusters

    all_rows, all_cols = [], []
    offset = 0

    for n_c in cluster_sizes:
        n_other = n_nodes - n_c
        k = min(k_cl, n_other)
        node_ids = np.arange(n_c, dtype=np.int32)

        # shape (n_c, k): random indices in [0, n_other)
        rand_idx = rng.integers(0, n_other, size=(n_c, k), dtype=np.int32)
        # remap: skip the cluster's own range
        rand_idx[rand_idx >= offset] += n_c

        all_rows.append(np.repeat(node_ids, k) + offset)
        all_cols.append(rand_idx.ravel())      # already global indices
        offset += n_c

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    vals = np.ones(len(rows), dtype=np.float32)

    W = csr_matrix((vals, (rows, cols)), shape=(n_nodes, n_nodes))
    W = (W + W.T)
    W.data[:] = 1.0
    return W


# ---------------------------------------------------------------------------
# CSC core (delegates to utils.py)
# ---------------------------------------------------------------------------

def csc_clustering(W_G, W_H, k, eps=0.9, random_seed=0):
    """Constrained Spectral Clustering using utils.py functions."""
    deg_g = np.array(W_G.sum(axis=1)).ravel()
    deg_h = np.array(W_H.sum(axis=1)).ravel()
    deg   = eps * deg_g + (1 - eps) * deg_h

    t0 = time.perf_counter()
    L_G = normalized_laplacian_from_D_sp(W_G, deg)
    L_H = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
    t_lap = time.perf_counter() - t0

    t0 = time.perf_counter()
    labels, _, _ = constrained_spectral_clustering(
        L_G, L_H, n_clusters=k, random_state=random_seed, exact=False
    )
    t_csc = time.perf_counter() - t0

    return labels, dict(laplacian=t_lap, eigensolver=t_csc)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(n_nodes, n_clusters=10, k_ml=5, k_cl=5, seed=42):
    print(f"\n{'='*65}")
    print(f"  n = {n_nodes:>10,}   clusters={n_clusters}   k_ml={k_ml}   k_cl={k_cl}")
    print(f"{'='*65}")
    result = dict(n_nodes=n_nodes)

    # --- Build must-link ---
    t0 = time.perf_counter()
    W_G = build_must_link_sparse(n_nodes, n_clusters, k_ml=k_ml, seed=seed)
    result['t_build_ml'] = time.perf_counter() - t0
    result['nnz_G'] = W_G.nnz
    print(f"  Must-link   : {result['nnz_G']:>12,} nnz  | {result['t_build_ml']:6.2f}s")

    # --- Build cannot-link ---
    t0 = time.perf_counter()
    W_H = build_cannot_link_sparse(n_nodes, n_clusters, k_cl=k_cl, seed=seed+1)
    result['t_build_cl'] = time.perf_counter() - t0
    result['nnz_H'] = W_H.nnz
    print(f"  Cannot-link : {result['nnz_H']:>12,} nnz  | {result['t_build_cl']:6.2f}s")

    # --- CSC ---
    t0 = time.perf_counter()
    labels, sub_times = csc_clustering(W_G, W_H, k=n_clusters, random_seed=seed)
    result['t_csc_total'] = time.perf_counter() - t0
    result.update({f't_{k}': v for k, v in sub_times.items()})

    print(f"  Laplacians  :              {'':12}  | {sub_times['laplacian']:6.2f}s")
    print(f"  LOBPCG+KM   :              {'':12}  | {sub_times['eigensolver']:6.2f}s")
    print(f"  CSC total   :              {'':12}  | {result['t_csc_total']:6.2f}s")

    n_uniq = len(np.unique(labels))
    print(f"  Unique clusters found: {n_uniq}")

    result['error'] = None
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    N_CLUSTERS = 10
    K_ML = 5   # must-link neighbors per node
    K_CL = 5   # cannot-link neighbors per node

    sizes = [10_000, 100_000, 500_000, 1_000_000]

    results = []
    for n in sizes:
        try:
            r = run_benchmark(n, n_clusters=N_CLUSTERS, k_ml=K_ML, k_cl=K_CL)
        except Exception as e:
            print(f"\nFAILED for n={n:,}: {e}")
            traceback.print_exc()
            r = dict(n_nodes=n, error=str(e))
        results.append(r)

    # -----------------------------------------------------------------------
    # Build report
    # -----------------------------------------------------------------------
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        "=" * 80,
        "  CSC Efficiency Benchmark Report",
        f"  Generated: {ts}",
        "=" * 80,
        "",
        "Setup",
        "-----",
        f"  Number of clusters : {N_CLUSTERS}",
        f"  k_ml               : {K_ML}  (must-link neighbors per node, same cluster)",
        f"  k_cl               : {K_CL}  (cannot-link neighbors per node, diff cluster)",
        "  Graph construction : vectorized NumPy (no NetworkX, no SBM probabilities)",
        "  Eigensolver        : LOBPCG (k eigenvectors, tol=1e-5, maxiter=300)",
        "  Clustering         : KMeans (n_init=5)",
        "",
        "Timing Results",
        "--------------",
        "",
        f"{'N Nodes':>12} | {'Build ML':>9} | {'Build CL':>9} | {'Laplacian':>10} | "
        f"{'LOBPCG+KM':>10} | {'CSC Total':>10} | {'NNZ (ML)':>12} | {'NNZ (CL)':>12}",
        "-" * 105,
    ]

    for r in results:
        n = r['n_nodes']
        if r.get('error'):
            lines.append(f"{n:>12,} | ERROR: {r['error']}")
        else:
            lines.append(
                f"{n:>12,} | {r['t_build_ml']:>9.2f} | {r['t_build_cl']:>9.2f} | "
                f"{r.get('t_laplacian', float('nan')):>10.2f} | "
                f"{r.get('t_eigensolver', float('nan')):>10.2f} | "
                f"{r.get('t_csc_total', float('nan')):>10.2f} | "
                f"{r.get('nnz_G', 0):>12,} | {r.get('nnz_H', 0):>12,}"
            )

    lines += [
        "",
        "Notes",
        "-----",
        "  - Build ML/CL: time to construct the sparse must-/cannot-link matrices.",
        "  - Laplacian:   time to compute both normalized Laplacians L_G and L_H.",
        "  - LOBPCG+KM:   eigensolver + KMeans (inside constrained_spectral_clustering).",
        "  - CSC Total:   Laplacian + LOBPCG+KM combined.",
        "  - NNZ after symmetrization (some edges may be doubled).",
        "",
    ]

    report = "\n".join(lines)
    print("\n\n" + report)

    report_path = "CSC_benchmark_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
