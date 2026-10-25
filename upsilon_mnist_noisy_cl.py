"""
Upsilon analysis on MNIST with noisy cannot-link constraints.

Graphs
------
  G (must-link)   : 3-NN graph on raw pixel features  (binary weights,
                    same as CSC_MNIST.py)
  H (cannot-link) : 20-NN cannot-link graph from TRUE labels, but with
                    2 WRONG edges per vertex (same-class, i.e. noisy)
                    and 18 correct edges (other-class)

Target quantity
---------------
  k   = 10  (MNIST has 10 digit classes)
  φ   = cut_G(y) / cut_H(y)   under the true-label partition
        Note: wrong H-edges (same-class) do NOT cross boundaries,
        so cut_H(y) < |E(H)|.
  Υ   = λ_{k+1} / φ

  λ_{k+1} computed approximately via eigsh (sparse, Lanczos).
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict
from scipy.sparse import csr_matrix, triu, eye
from scipy.sparse.linalg import eigsh

import keras.datasets.mnist

from utils import (
    build_knn_graph,
    normalized_laplacian_from_D_sp,
)

# ── Parameters (matching CSC_MNIST.py) ────────────────────────────────────────
N_SAMPLES    = 10000   # increase to 60000 for full MNIST
K_NN         = 3       # must-link: k-NN graph
N_CL         = 20      # total cannot-link edges per vertex
N_WRONG      = 2       # wrong cannot-link edges per vertex (same-class)
N_CORRECT    = N_CL - N_WRONG   # = 18 correct (other-class) edges per vertex
N_CLUSTERS   = 10
EPS          = 0.9
REG          = 1e-8
RANDOM_STATE = 42
EIGSH_TOL    = 1e-5
EIGSH_MAXITER = 3000


# ── Helper: noisy cannot-link graph ───────────────────────────────────────────

def build_noisy_cannot_link_graph(y, n_cl=20, n_wrong=2, random_state=0):
    """Build a cannot-link graph where each vertex has:
      - n_wrong  edges to nodes in the SAME class   (wrong/noisy constraints)
      - n_cl - n_wrong  edges to nodes in OTHER classes  (correct constraints)

    Returns the symmetrised CSR adjacency matrix and two index arrays
    (wrong_pairs, correct_pairs) listing the directed edges before symmetrisation.
    """
    n_samples = len(y)
    rng = np.random.default_rng(random_state)

    label_to_idx = defaultdict(list)
    for i, label in enumerate(y):
        label_to_idx[label].append(i)
    labels = np.unique(y)

    rows_wrong,   cols_wrong   = [], []
    rows_correct, cols_correct = [], []

    for i in range(n_samples):
        my_label = y[i]

        # ── wrong edges: same class ──────────────────────────────────────────
        same_idx = [j for j in label_to_idx[my_label] if j != i]
        n_w = min(n_wrong, len(same_idx))
        if n_w > 0:
            chosen = rng.choice(same_idx, size=n_w, replace=False)
            rows_wrong.extend([i] * n_w)
            cols_wrong.extend(chosen.tolist())

        # ── correct edges: other classes ─────────────────────────────────────
        n_c = n_cl - n_wrong
        other_idx = np.concatenate(
            [np.array(label_to_idx[l]) for l in labels if l != my_label]
        )
        n_c = min(n_c, len(other_idx))
        chosen = rng.choice(other_idx, size=n_c, replace=False)
        rows_correct.extend([i] * n_c)
        cols_correct.extend(chosen.tolist())

    all_rows = rows_wrong   + rows_correct
    all_cols = cols_wrong   + cols_correct
    vals     = np.ones(len(all_rows), dtype=float)

    W_H = csr_matrix((vals, (all_rows, all_cols)), shape=(n_samples, n_samples))
    W_H = W_H + W_H.T

    return W_H, (rows_wrong, cols_wrong), (rows_correct, cols_correct)


# ── Helper: sparse cut ────────────────────────────────────────────────────────

def sparse_cut(W, labels):
    """Cut weight of W under label partition (upper triangle, no double-count)."""
    W_tri      = triu(W, k=1).tocsr()
    rows, cols = W_tri.nonzero()
    data       = np.array(W_tri[rows, cols]).ravel()
    cross      = labels[rows] != labels[cols]
    return float(data[cross].sum())


# ── 1. Load MNIST ──────────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1 — Load MNIST")
t0 = time.time()

(X_full, y_full), _ = keras.datasets.mnist.load_data()
X_full = X_full.reshape(len(X_full), -1).astype(np.float32)
y_full = y_full.astype(int)

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X, y = X_full[idx], y_full[idx]

print(f"  N={N_SAMPLES}  classes={np.unique(y).tolist()}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 2. Build must-link graph G ────────────────────────────────────────────────
print(f"\nStep 2 — Build must-link graph G  (k={K_NN} NN, binary)")
t0 = time.time()

W_G   = build_knn_graph(X, k=K_NN, use_gaussian=False)
deg_g = np.array(W_G.sum(axis=1)).ravel()

print(f"  |V|={W_G.shape[0]}  |E|={W_G.nnz // 2}  avg_deg={deg_g.mean():.2f}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 3. Build noisy cannot-link graph H ────────────────────────────────────────
print(f"\nStep 3 — Build noisy cannot-link graph H")
print(f"  n_cl={N_CL} total  |  {N_CORRECT} correct (other-class)  "
      f"+  {N_WRONG} wrong (same-class) per vertex")
t0 = time.time()

W_H, (rw, cw), (rc, cc) = build_noisy_cannot_link_graph(
    y, n_cl=N_CL, n_wrong=N_WRONG, random_state=RANDOM_STATE
)
deg_h = np.array(W_H.sum(axis=1)).ravel()

# Count wrong vs correct edges in the symmetrised matrix
wrong_edges   = sparse_cut(
    csr_matrix((np.ones(len(rw)), (rw, cw)), shape=W_H.shape) +
    csr_matrix((np.ones(len(rw)), (cw, rw)), shape=W_H.shape),
    y
)  # wrong edges that DO cross class (should be ~0)
total_H_edges = W_H.nnz // 2

# Empirical audit: fraction of H edges that are wrong (same-class)
W_H_tri        = triu(W_H, k=1).tocsr()
tri_rows, tri_cols = W_H_tri.nonzero()
n_same_class   = int((y[tri_rows] == y[tri_cols]).sum())
n_cross_class  = total_H_edges - n_same_class

print(f"  |V|={W_H.shape[0]}  |E(H)|={total_H_edges}")
print(f"  avg_deg={deg_h.mean():.2f}")
print(f"  Same-class (wrong) edges : {n_same_class}  "
      f"({100*n_same_class/total_H_edges:.2f}%)")
print(f"  Cross-class (correct) edges: {n_cross_class}  "
      f"({100*n_cross_class/total_H_edges:.2f}%)")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 4. Compute double-normalised Laplacians ────────────────────────────────────
print(f"\nStep 4 — Compute sparse Laplacians  (eps={EPS})")
t0 = time.time()

deg     = EPS * deg_g + (1.0 - EPS) * deg_h
L_G     = normalized_laplacian_from_D_sp(W_G, deg)
L_H     = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
N       = W_G.shape[0]
L_H_reg = L_H + REG * eye(N, format='csr')

print(f"  deg_G mean={deg_g.mean():.2f}  deg_H mean={deg_h.mean():.2f}  "
      f"combined D mean={deg.mean():.2f}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 5. Approximate λ_1 … λ_{k+1} via eigsh ───────────────────────────────────
print(f"\nStep 5 — Compute {N_CLUSTERS+1} smallest eigenvalues via eigsh")
print("  Solving:  L_G v = λ L_H_reg v")
t0 = time.time()

np.random.seed(RANDOM_STATE)
vals, vecs = eigsh(
    L_G,
    k=N_CLUSTERS + 1,
    M=L_H_reg,
    which='SM',
    tol=EIGSH_TOL,
    maxiter=EIGSH_MAXITER,
)
vals = np.sort(np.real(vals))

print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 6. Compute φ = cut_G(y) / cut_H(y) ────────────────────────────────────────
print("\nStep 6 — Compute φ = cut_G(y*) / cut_H(y*)")

cut_G = sparse_cut(W_G, y)
cut_H = sparse_cut(W_H, y)   # only cross-class H edges (wrong edges excluded)

print(f"  cut_G(y*)             = {cut_G:.0f}")
print(f"  cut_H(y*)             = {cut_H:.0f}  "
      f"(cross-class H edges only;  wrong edges = {n_same_class} not counted)")
print(f"  total H edges         = {total_H_edges}")
print(f"  cut_H / total_H_edges = {cut_H / total_H_edges:.5f}")

phi = cut_G / max(cut_H, 1.0)
print(f"  φ = {phi:.6f}")


# ── 7. Report results ──────────────────────────────────────────────────────────
lk1     = vals[N_CLUSTERS]
lk      = vals[N_CLUSTERS - 1]
gap     = lk1 - lk
upsilon = lk1 / max(phi, 1e-12)

print()
print("=" * 70)
print("RESULTS — Upsilon analysis on MNIST (noisy cannot-link)")
print("=" * 70)
print(f"  Dataset   : MNIST  N={N_SAMPLES}  k={N_CLUSTERS} classes")
print(f"  G         : {K_NN}-NN binary graph  (must-link)")
print(f"  H         : {N_CL}-NN cannot-link  "
      f"({N_CORRECT} correct + {N_WRONG} wrong per vertex)")
print(f"  eps       : {EPS}")
print()

print("  Eigenvalues λ_1 … λ_{k+1}:")
for i, v in enumerate(vals):
    marker = " ← λ_k" if i == N_CLUSTERS - 1 else \
             (" ← λ_{k+1}  ***" if i == N_CLUSTERS else "")
    print(f"    λ_{i+1:>2} = {v:.6f}{marker}")

print()
print(f"  φ   = cut_G(y*) / cut_H(y*) = {cut_G:.0f} / {cut_H:.0f} = {phi:.6f}")
print(f"  λ_k     = {lk:.6f}")
print(f"  λ_{{k+1}} = {lk1:.6f}")
print(f"  gap     = λ_{{k+1}} - λ_k = {gap:.6f}")
print()
print(f"  ┌─────────────────────────────────────┐")
print(f"  │  Υ = λ_{{k+1}} / φ = {upsilon:>10.4f}        │")
print(f"  └─────────────────────────────────────┘")


# ── 8. Compare with clean cannot-link (reference) ─────────────────────────────
print()
print("=" * 70)
print("EFFECT OF WRONG EDGES (reference: upsilon_mnist.py with n_cl=20, 0 wrong)")
print("=" * 70)
print(f"  Wrong edges per vertex  : {N_WRONG} / {N_CL}  "
      f"({100*N_WRONG/N_CL:.0f}% noise rate)")
print(f"  Same-class H edges      : {n_same_class} / {total_H_edges}  "
      f"({100*n_same_class/total_H_edges:.2f}%)")
print()
print(f"  φ with noise = {phi:.6f}")
print(f"  (With 0 wrong edges, cut_H = all {total_H_edges} edges,")
phi_clean = cut_G / total_H_edges
print(f"   φ_clean would be ≈ {phi_clean:.6f}  — noise inflates φ by "
      f"{100*(phi/phi_clean - 1):.1f}%)")
upsilon_clean = lk1 / max(phi_clean, 1e-12)
print(f"  Υ_clean ≈ {upsilon_clean:.4f}  vs  Υ_noisy = {upsilon:.4f}  "
      f"(Υ drops by {100*(1 - upsilon/upsilon_clean):.1f}% due to noise)")
