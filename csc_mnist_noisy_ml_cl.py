"""
MNIST: SC vs CSC with noisy must-link and noisy cannot-link graphs.

Graph construction (class-based, using true labels)
----------------------------------------------------
  G (must-link)   : 4 edges per vertex
                      3 correct  — random nodes from the SAME class
                      1 wrong    — random node  from a DIFFERENT class

  H (cannot-link) : 20 edges per vertex
                      19 correct — random nodes from DIFFERENT classes
                       1 wrong   — random node  from the SAME class

Methods compared
----------------
  SC  : standard spectral clustering on G only
  CSC : constrained spectral clustering on G + H  (iterative_clustering)
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
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

import keras.datasets.mnist

from utils import (
    normalized_laplacian_from_D_sp,
    spectral_clustering,
    iterative_clustering,
)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SAMPLES      = 10000
N_ML           = 4      # must-link edges per vertex
N_ML_WRONG     = 1      # wrong must-link per vertex (diff-class)
N_ML_CORRECT   = N_ML - N_ML_WRONG     # 3
N_CL           = 20     # cannot-link edges per vertex
N_CL_WRONG     = 1      # wrong cannot-link per vertex (same-class)
N_CL_CORRECT   = N_CL - N_CL_WRONG     # 19
N_CLUSTERS     = 10
EPS            = 0.9
REG            = 1e-8
RANDOM_STATE   = 42
EIGSH_TOL      = 1e-5
EIGSH_MAXITER  = 3000


# ── Graph builders ─────────────────────────────────────────────────────────────

def build_noisy_must_link_graph(y, n_ml=4, n_wrong=1, random_state=0):
    """Each vertex gets:
      n_ml - n_wrong  edges to SAME-class nodes      (correct must-link)
      n_wrong         edges to DIFFERENT-class nodes  (wrong must-link)
    """
    n_samples = len(y)
    rng = np.random.default_rng(random_state)

    label_to_idx = defaultdict(list)
    for i, label in enumerate(y):
        label_to_idx[label].append(i)
    labels = np.unique(y)

    rows, cols = [], []

    for i in range(n_samples):
        my_label = y[i]

        # correct: same class
        same_idx = [j for j in label_to_idx[my_label] if j != i]
        n_c = min(n_ml - n_wrong, len(same_idx))
        if n_c > 0:
            chosen = rng.choice(same_idx, size=n_c, replace=False)
            rows.extend([i] * n_c)
            cols.extend(chosen.tolist())

        # wrong: different class
        other_idx = np.concatenate(
            [np.array(label_to_idx[l]) for l in labels if l != my_label]
        )
        n_w = min(n_wrong, len(other_idx))
        if n_w > 0:
            chosen = rng.choice(other_idx, size=n_w, replace=False)
            rows.extend([i] * n_w)
            cols.extend(chosen.tolist())

    vals = np.ones(len(rows), dtype=float)
    W = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    return W + W.T


def build_noisy_cannot_link_graph(y, n_cl=20, n_wrong=1, random_state=0):
    """Each vertex gets:
      n_cl - n_wrong  edges to DIFFERENT-class nodes  (correct cannot-link)
      n_wrong         edges to SAME-class nodes        (wrong cannot-link)
    """
    n_samples = len(y)
    rng = np.random.default_rng(random_state)

    label_to_idx = defaultdict(list)
    for i, label in enumerate(y):
        label_to_idx[label].append(i)
    labels = np.unique(y)

    rows, cols = [], []

    for i in range(n_samples):
        my_label = y[i]

        # correct: different class
        other_idx = np.concatenate(
            [np.array(label_to_idx[l]) for l in labels if l != my_label]
        )
        n_c = min(n_cl - n_wrong, len(other_idx))
        chosen = rng.choice(other_idx, size=n_c, replace=False)
        rows.extend([i] * n_c)
        cols.extend(chosen.tolist())

        # wrong: same class
        same_idx = [j for j in label_to_idx[my_label] if j != i]
        n_w = min(n_wrong, len(same_idx))
        if n_w > 0:
            chosen = rng.choice(same_idx, size=n_w, replace=False)
            rows.extend([i] * n_w)
            cols.extend(chosen.tolist())

    vals = np.ones(len(rows), dtype=float)
    W = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    return W + W.T


def sparse_cut(W, labels):
    W_tri  = triu(W, k=1).tocsr()
    r, c   = W_tri.nonzero()
    data   = np.array(W_tri[r, c]).ravel()
    return float(data[labels[r] != labels[c]].sum())


def edge_audit(W, labels, same_is_wrong=True):
    """(total_edges, wrong_edges, wrong_pct)
    same_is_wrong=True  → wrong = same-class  (cannot-link)
    same_is_wrong=False → wrong = diff-class  (must-link)
    """
    W_tri  = triu(W, k=1).tocsr()
    r, c   = W_tri.nonzero()
    same   = (labels[r] == labels[c])
    total  = len(r)
    wrong  = int(same.sum()) if same_is_wrong else int((~same).sum())
    return total, wrong, 100 * wrong / max(total, 1)


# ── 1. Load MNIST ──────────────────────────────────────────────────────────────
print("=" * 65)
print("Loading MNIST")
t0 = time.time()

(X_full, y_full), _ = keras.datasets.mnist.load_data()
X_full = X_full.reshape(len(X_full), -1).astype(np.float32)
y_full = y_full.astype(int)

rng  = np.random.default_rng(RANDOM_STATE)
idx  = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X, y = X_full[idx], y_full[idx]
print(f"  N={N_SAMPLES}  elapsed: {time.time()-t0:.1f}s")


# ── 2. Build noisy must-link graph G ─────────────────────────────────────────
print(f"\nBuilding noisy must-link graph G")
print(f"  {N_ML} edges/vertex: {N_ML_CORRECT} correct (same-class) "
      f"+ {N_ML_WRONG} wrong (diff-class)")
t0 = time.time()

W_G   = build_noisy_must_link_graph(y, n_ml=N_ML, n_wrong=N_ML_WRONG,
                                     random_state=RANDOM_STATE)
deg_g = np.array(W_G.sum(axis=1)).ravel()
total_G, wrong_G, wrong_G_pct = edge_audit(W_G, y, same_is_wrong=False)

print(f"  |E(G)|={total_G}  avg_deg={deg_g.mean():.2f}")
print(f"  Wrong (diff-class) edges  : {wrong_G}  ({wrong_G_pct:.1f}%)")
print(f"  Correct (same-class) edges: {total_G - wrong_G}  ({100-wrong_G_pct:.1f}%)")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 3. Build noisy cannot-link graph H ────────────────────────────────────────
print(f"\nBuilding noisy cannot-link graph H")
print(f"  {N_CL} edges/vertex: {N_CL_CORRECT} correct (diff-class) "
      f"+ {N_CL_WRONG} wrong (same-class)")
t0 = time.time()

W_H   = build_noisy_cannot_link_graph(y, n_cl=N_CL, n_wrong=N_CL_WRONG,
                                       random_state=RANDOM_STATE)
deg_h = np.array(W_H.sum(axis=1)).ravel()
total_H, wrong_H, wrong_H_pct = edge_audit(W_H, y, same_is_wrong=True)

print(f"  |E(H)|={total_H}  avg_deg={deg_h.mean():.2f}")
print(f"  Wrong (same-class) edges  : {wrong_H}  ({wrong_H_pct:.1f}%)")
print(f"  Correct (diff-class) edges: {total_H - wrong_H}  ({100-wrong_H_pct:.1f}%)")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 4. Compute φ and Υ ────────────────────────────────────────────────────────
print(f"\nComputing φ and Υ  (eigsh, eps={EPS})")
t0 = time.time()

deg     = EPS * deg_g + (1.0 - EPS) * deg_h
L_G     = normalized_laplacian_from_D_sp(W_G, deg)
L_H     = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
N       = W_G.shape[0]
L_H_reg = L_H + REG * eye(N, format='csr')

np.random.seed(RANDOM_STATE)
vals, _ = eigsh(L_G, k=N_CLUSTERS + 1, M=L_H_reg,
                which='SM', tol=EIGSH_TOL, maxiter=EIGSH_MAXITER)
vals    = np.sort(np.real(vals))
lk1     = vals[N_CLUSTERS]

cut_G   = sparse_cut(W_G, y)
cut_H   = sparse_cut(W_H, y)   # only cross-class H edges
phi     = cut_G / max(cut_H, 1.0)
upsilon = lk1 / max(phi, 1e-12)

print(f"  cut_G(y*)={cut_G:.0f}  cut_H(y*)={cut_H:.0f}")
print(f"  φ={phi:.5f}  λ_{{k+1}}={lk1:.5f}  Υ={upsilon:.4f}")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 5. Spectral Clustering on G only ─────────────────────────────────────────
print("\nRunning Spectral Clustering (SC) ...")
t0 = time.time()

L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
sc_labels, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                       random_state=RANDOM_STATE)
sc_ari = adjusted_rand_score(y, sc_labels)
sc_nmi = normalized_mutual_info_score(y, sc_labels)
print(f"  ARI={sc_ari:.4f}  NMI={sc_nmi:.4f}  elapsed: {time.time()-t0:.1f}s")


# ── 6. Constrained Spectral Clustering on G + H ───────────────────────────────
print("\nRunning Constrained Spectral Clustering (CSC) ...")
t0 = time.time()

csc_labels, _, _ = iterative_clustering(
    W_G, W_H, k=N_CLUSTERS, exact=False, eps=EPS, random_seed=RANDOM_STATE
)
csc_ari = adjusted_rand_score(y, csc_labels)
csc_nmi = normalized_mutual_info_score(y, csc_labels)
print(f"  ARI={csc_ari:.4f}  NMI={csc_nmi:.4f}  elapsed: {time.time()-t0:.1f}s")


# ── 7. Summary ────────────────────────────────────────────────────────────────
delta_ari = csc_ari - sc_ari
delta_nmi = csc_nmi - sc_nmi

print()
print("=" * 65)
print(f"SUMMARY — MNIST  N={N_SAMPLES:,}  k={N_CLUSTERS}")
print("=" * 65)
print(f"  G : {N_ML} must-link/vertex  "
      f"({N_ML_CORRECT} correct + {N_ML_WRONG} wrong = {100*N_ML_WRONG/N_ML:.0f}% noise)")
print(f"  H : {N_CL} cannot-link/vertex  "
      f"({N_CL_CORRECT} correct + {N_CL_WRONG} wrong = {100*N_CL_WRONG/N_CL:.0f}% noise)")
print()
print(f"  {'Method':<30}  {'ARI':>8}  {'NMI':>8}")
print(f"  {'-'*30}  {'-'*8}  {'-'*8}")
print(f"  {'SC  (G only)':30}  {sc_ari:8.4f}  {sc_nmi:8.4f}")
print(f"  {'CSC (G + noisy H)':30}  {csc_ari:8.4f}  {csc_nmi:8.4f}")
sign_a = "+" if delta_ari >= 0 else ""
sign_n = "+" if delta_nmi >= 0 else ""
print(f"  {'Δ (CSC − SC)':30}  {sign_a}{delta_ari:7.4f}  {sign_n}{delta_nmi:7.4f}")
print()
print(f"  Eigenvalues λ_1 … λ_{{k+1}}:")
for i, v in enumerate(vals):
    marker = " ← λ_k" if i == N_CLUSTERS - 1 else \
             (" ← λ_{k+1}" if i == N_CLUSTERS else "")
    print(f"    λ_{i+1:>2} = {v:.6f}{marker}")
print()
print(f"  φ        = cut_G(y*)/cut_H(y*) = {cut_G:.0f}/{cut_H:.0f} = {phi:.5f}")
print(f"  λ_{{k+1}}  = {lk1:.5f}")
print(f"  gap      = {vals[N_CLUSTERS]-vals[N_CLUSTERS-1]:.6f}")
print(f"  Υ        = {upsilon:.4f}")
