"""
Upsilon analysis on MNIST.

Graphs
------
  G (must-link)   : 3-NN graph on raw pixel features  (binary weights,
                    same construction as CSC_MNIST.py)
  H (cannot-link) : 10-NN cannot-link graph built from TRUE labels
                    (each node connected to 10 random nodes in other classes)

Target quantity
---------------
  k   = 10  (MNIST has 10 digit classes)
  φ   = cut_G(y) / cut_H(y)   under the true-label partition
  Υ   = λ_{k+1} / φ

  λ_{k+1} is the 11th smallest eigenvalue of the generalised problem
       L_G v = λ L_H v
  computed approximately via scipy.sparse.linalg.eigsh (Lanczos, sparse).

  Dense eigh is infeasible for N ≥ 10 000.  eigsh converges to the
  smallest eigenvalues without forming any N×N dense matrix.

Parameters
----------
  N_SAMPLES : number of MNIST digits to use  (set lower for quick tests,
              60 000 for the full dataset as in CSC_MNIST.py)
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import time
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse import triu, eye
from scipy.sparse.linalg import eigsh

import keras.datasets.mnist

from utils import (
    build_knn_graph,
    build_cannot_link_graph,
    normalized_laplacian_from_D_sp,
)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SAMPLES    = 10000      # increase to 60000 for full MNIST (slower)
K_NN         = 3          # must-link: k-NN graph
N_CL         = 20         # cannot-link: neighbours per node from other classes
N_CLUSTERS   = 10         # number of MNIST classes
EPS          = 0.9        # degree mixing  D = eps*deg_G + (1-eps)*deg_H
REG          = 1e-8       # ridge added to L_H to ensure positive definiteness
RANDOM_STATE = 42
EIGSH_TOL    = 1e-5       # convergence tolerance for eigsh
EIGSH_MAXITER = 3000      # max Lanczos iterations


# ── 1. Load MNIST ──────────────────────────────────────────────────────────────
print("=" * 70)
print("Step 1 — Load MNIST")
t0 = time.time()

(X_full, y_full), _ = keras.datasets.mnist.load_data()
X_full = X_full.reshape(len(X_full), -1).astype(np.float32)
y_full = y_full.astype(int)

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X   = X_full[idx]
y   = y_full[idx]

print(f"  N={N_SAMPLES}  classes={np.unique(y).tolist()}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 2. Build must-link graph G (3-NN, binary) ──────────────────────────────────
print("\nStep 2 — Build must-link graph G  (k={} NN, binary)".format(K_NN))
t0 = time.time()

W_G    = build_knn_graph(X, k=K_NN, use_gaussian=False)
deg_g  = np.array(W_G.sum(axis=1)).ravel()

print(f"  |V|={W_G.shape[0]}  |E|={W_G.nnz // 2}  "
      f"avg_deg={deg_g.mean():.2f}  max_deg={deg_g.max():.0f}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 3. Build cannot-link graph H (10-NN, true labels) ─────────────────────────
print(f"\nStep 3 — Build cannot-link graph H  (n_cl={N_CL}, true labels)")
t0 = time.time()

W_H   = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
deg_h = np.array(W_H.sum(axis=1)).ravel()

print(f"  |V|={W_H.shape[0]}  |E|={W_H.nnz // 2}  "
      f"avg_deg={deg_h.mean():.2f}")
print(f"  All H-edges cross class boundaries by construction.")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 4. Compute double-normalised Laplacians ───────────────────────────────────
print("\nStep 4 — Compute sparse Laplacians  (eps={})".format(EPS))
t0 = time.time()

deg   = EPS * deg_g + (1.0 - EPS) * deg_h          # combined degree
L_G   = normalized_laplacian_from_D_sp(W_G, deg)
L_H   = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
N     = W_G.shape[0]
L_H_reg = L_H + REG * eye(N, format='csr')          # ensure positive definite

print(f"  deg_G mean={deg_g.mean():.2f}   deg_H mean={deg_h.mean():.2f}   "
      f"combined D mean={deg.mean():.2f}")
print(f"  L_G nnz={L_G.nnz}   L_H nnz={L_H.nnz}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 5. Approximate λ_1 … λ_{k+1} via eigsh (Lanczos) ─────────────────────────
print(f"\nStep 5 — Compute {N_CLUSTERS+1} smallest eigenvalues via eigsh  "
      f"(tol={EIGSH_TOL}, maxiter={EIGSH_MAXITER})")
print("  Solving generalised eigenproblem:  L_G v = λ L_H_reg v")
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

def sparse_cut(W, labels):
    """Cut weight of W under label partition.  Uses upper triangle to avoid
    double-counting the symmetrised matrix."""
    W_tri  = triu(W, k=1).tocsr()
    rows, cols = W_tri.nonzero()
    data   = np.array(W_tri[rows, cols]).ravel()
    cross  = labels[rows] != labels[cols]
    return float(data[cross].sum())

cut_G = sparse_cut(W_G, y)
cut_H = sparse_cut(W_H, y)   # should equal W_H.sum()/2 (all H edges are cross-class)

# Sanity check: all H edges should cross class boundaries
h_total = W_H.sum() / 2
print(f"  cut_G(y) = {cut_G:.0f}")
print(f"  cut_H(y) = {cut_H:.0f}   (total H edges = {h_total:.0f}  "
      f"→ fraction cross-class = {cut_H/h_total:.5f})")

phi = cut_G / max(cut_H, 1.0)
print(f"  φ        = {phi:.6f}")


# ── 7. Report results ──────────────────────────────────────────────────────────
print()
print("=" * 70)
print("RESULTS — Upsilon analysis on MNIST")
print("=" * 70)
print(f"  Dataset   : MNIST  N={N_SAMPLES}  k={N_CLUSTERS} classes")
print(f"  G         : {K_NN}-NN binary graph  (must-link)")
print(f"  H         : {N_CL}-NN cannot-link graph from true labels")
print(f"  eps       : {EPS}")
print()

lk1 = vals[N_CLUSTERS]      # λ_{k+1}  (0-indexed: vals[10] = 11th eigenvalue)
lk  = vals[N_CLUSTERS - 1]  # λ_k
gap = lk1 - lk
upsilon = lk1 / max(phi, 1e-12)

print("  Eigenvalues λ_1 … λ_{k+1}:")
for i, v in enumerate(vals):
    marker = " ← λ_k" if i == N_CLUSTERS - 1 else (" ← λ_{k+1}  ***" if i == N_CLUSTERS else "")
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
print()

# ── 8. Context: φ interpretation ──────────────────────────────────────────────
print("=" * 70)
print("INTERPRETATION")
print("=" * 70)
print(f"  φ = {phi:.4f}:  must-link cut / cannot-link cut under true labels.")
print(f"  Since H is built from true labels, cut_H = all H edges = {cut_H:.0f}.")
print(f"  φ < 1 means G has fewer cross-cluster edges than H has by construction.")
print()
print(f"  Υ = {upsilon:.4f}:")
if upsilon > 1:
    print("  Υ > 1: λ_{k+1} is larger than φ — spectral gap is 'boosted' relative")
    print("  to the cut ratio. Constrained spectral clustering is expected to work well.")
else:
    print("  Υ ≤ 1: λ_{k+1} is comparable to or smaller than φ — the spectral gap")
    print("  may not reliably separate clusters from the cannot-link penalty.")
