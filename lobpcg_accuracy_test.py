"""
LOBPCG accuracy experiment.

Verifies that LOBPCG converges to the true k smallest eigenvectors of a
graph Laplacian, and measures how results change across iteration budgets.

Experiment design
-----------------
  Graphs  : (A) SBM with k=5 blocks, block_size=200, N=1000   (sparse, moderate)
            (B) SBM with k=10 blocks, block_size=200, N=2000  (sparse, harder)
  Ground truth : scipy.linalg.eigh on the dense Laplacian
  LOBPCG budgets tested : 50, 100, 200, 500, 1000

Metrics
-------
  For each budget we report:
    1. Eigenvalue error   : max |λ_i^lobpcg - λ_i^true| / max(|λ_i^true|, 1e-12)
                           for i = 1 … k
    2. Subspace error     : ||P_true - P_lobpcg||_F  where P = V V^T (projector)
                           (invariant to sign and rotation within degenerate subspaces)
    3. K-means ARI        : k-means on LOBPCG eigenvectors vs. ground-truth labels
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import lobpcg
from scipy.linalg import eigh
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
import networkx as nx

from utils import SBM, normalized_laplacian_from_D_sp

RANDOM_STATE = 42
K_CLUSTERS   = [5, 10]          # k for the two SBM experiments
BLOCK_SIZE   = 200
P, Q         = 0.2, 0.12        # harder: p/q close → small eigengap
BUDGETS      = [50, 100, 200, 500, 1000]


# ── helpers ──────────────────────────────────────────────────────────────────

def sbm_to_sparse_laplacian(block_size, k, p, q, seed=42):
    """Build normalised graph Laplacian from SBM must-link graph."""
    G, _ = SBM(block_size=block_size, k=k, p=p, q=q)
    # NetworkX → sparse adjacency
    N     = G.number_of_nodes()
    edges = list(G.edges())
    rows  = [u for u, v in edges] + [v for u, v in edges]
    cols  = [v for u, v in edges] + [u for u, v in edges]
    vals  = np.ones(len(rows), dtype=float)
    W     = csr_matrix((vals, (rows, cols)), shape=(N, N))
    deg   = np.array(W.sum(axis=1)).ravel()
    L     = normalized_laplacian_from_D_sp(W, deg)
    labels = np.array([G.nodes[i]['block'] for i in range(N)])
    return L, labels, deg


def subspace_error(V_true, V_approx):
    """Frobenius norm of the difference between orthogonal projectors."""
    P_true   = V_true   @ V_true.T
    P_approx = V_approx @ V_approx.T
    return np.linalg.norm(P_true - P_approx, 'fro')


def run_lobpcg(L, k, maxiter, rng):
    """Run LOBPCG for k smallest eigenpairs; returns (vals, vecs, elapsed)."""
    N  = L.shape[0]
    X0 = rng.standard_normal((N, k))
    X0, _ = np.linalg.qr(X0)   # orthonormalise starting vectors
    t0 = time.time()
    vals, vecs = lobpcg(L, X0, largest=False, maxiter=maxiter, tol=0)
    elapsed = time.time() - t0
    order   = np.argsort(vals)
    return vals[order], vecs[:, order], elapsed


def run_eigh_truth(L_dense, k):
    """Dense ground-truth: k smallest eigenpairs via eigh."""
    t0 = time.time()
    vals, vecs = eigh(L_dense, subset_by_index=[0, k - 1])
    elapsed = time.time() - t0
    return vals, vecs, elapsed


def kmeans_ari(vecs, true_labels, k, n_init=10):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=RANDOM_STATE)
    pred = km.fit_predict(vecs)
    return adjusted_rand_score(true_labels, pred)


# ── main experiment ───────────────────────────────────────────────────────────

for k in K_CLUSTERS:
    N = k * BLOCK_SIZE
    print("=" * 70)
    print(f"Experiment: SBM  k={k}  block_size={BLOCK_SIZE}  N={N}  p={P}  q={Q}")
    print("=" * 70)

    # Build Laplacian
    print("  Building graph and Laplacian ...", end=" ", flush=True)
    L_sp, true_labels, deg = sbm_to_sparse_laplacian(BLOCK_SIZE, k, P, Q)
    L_dense = L_sp.toarray()
    print("done")

    # Ground truth via dense eigh
    print("  Computing ground truth (dense eigh) ...", end=" ", flush=True)
    vals_true, vecs_true, t_eigh = run_eigh_truth(L_dense, k)
    ari_true = kmeans_ari(vecs_true, true_labels, k)
    print(f"done  ({t_eigh:.2f}s)")
    print(f"  True eigenvalues λ_1 … λ_k: "
          f"{' '.join(f'{v:.4f}' for v in vals_true)}")
    print(f"  ARI (ground truth eigenvectors): {ari_true:.4f}")
    print()

    # LOBPCG across budgets
    rng = np.random.default_rng(RANDOM_STATE)

    print(f"  {'Budget':>8}  {'Time(s)':>8}  {'MaxEigErr':>12}  "
          f"{'SubspaceErr':>13}  {'ARI':>8}  {'ΔEig_1-k':>20}")
    print("  " + "-" * 77)

    prev_vecs = None
    prev_budget = None

    for budget in BUDGETS:
        vals_l, vecs_l, t_l = run_lobpcg(L_sp, k, budget, rng)

        # eigenvalue relative error
        eig_err = np.max(
            np.abs(vals_l - vals_true) / np.maximum(np.abs(vals_true), 1e-12)
        )

        # subspace error (rotation-invariant)
        sub_err = subspace_error(vecs_true, vecs_l)

        # ARI of k-means on LOBPCG vectors
        ari_l = kmeans_ari(vecs_l, true_labels, k)

        # per-eigenvalue absolute errors (for display)
        eig_diffs = " ".join(f"{abs(vals_l[i]-vals_true[i]):.2e}"
                             for i in range(min(k, 5)))
        if k > 5:
            eig_diffs += " …"

        print(f"  {budget:>8}  {t_l:>8.3f}  {eig_err:>12.2e}  "
              f"{sub_err:>13.4f}  {ari_l:>8.4f}  {eig_diffs}")

        prev_vecs   = vecs_l
        prev_budget = budget

    # Consecutive-budget comparison
    print()
    print(f"  Subspace distances between consecutive LOBPCG runs")
    print(f"  {'Budgets':>16}  {'SubspaceErr':>13}")
    print("  " + "-" * 32)
    vecs_by_budget = {}
    rng2 = np.random.default_rng(RANDOM_STATE)
    for budget in BUDGETS:
        v, vx, _ = run_lobpcg(L_sp, k, budget, rng2)
        vecs_by_budget[budget] = vx
    for i in range(len(BUDGETS) - 1):
        b1, b2 = BUDGETS[i], BUDGETS[i + 1]
        d = subspace_error(vecs_by_budget[b1], vecs_by_budget[b2])
        print(f"  {b1:>6} → {b2:<6}      {d:>13.4f}")

    print()


# ── final interpretation ──────────────────────────────────────────────────────
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print("""
Metrics explained
-----------------
  MaxEigErr    : max relative eigenvalue error vs. dense ground truth
                 ≈ 0 means LOBPCG found the correct eigenvalues
  SubspaceErr  : ||P_true - P_lobpcg||_F  (projector Frobenius norm)
                 0 = perfect subspace match; 2√k = random subspace
  ARI          : k-means clustering quality on the LOBPCG eigenvectors
                 vs. true SBM block labels

How to read the results
-----------------------
  - If SubspaceErr barely changes from budget 100 → 200 → 500, then
    100 iterations is already sufficient for this problem size.
  - If ARI stays high (≥0.95) even at budget=50, eigenvectors are
    good enough for clustering even before full convergence.
  - Large SubspaceErr + low ARI at small budgets signals under-convergence.
""")
