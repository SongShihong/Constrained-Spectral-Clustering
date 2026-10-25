"""
MNIST: Spectral Clustering vs Constrained Spectral Clustering
with noisy cannot-link constraints (cl=20, 2 wrong per vertex).

Methods compared
----------------
  SC  : standard spectral clustering on G only
  CSC : constrained spectral clustering on G + H  (iterative_clustering)

Cannot-link graph H
-------------------
  20 edges per vertex total:
    18 correct  — random nodes from OTHER classes
     2 wrong    — random nodes from the SAME class  (noisy constraints)

Other settings match CSC_MNIST.py:
  G : 3-NN binary graph, eps=0.9
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
    build_knn_graph,
    normalized_laplacian_from_D_sp,
    spectral_clustering,
    iterative_clustering,
)

# ── Parameters ─────────────────────────────────────────────────────────────────
N_SAMPLES    = 10000
K_NN         = 3
N_CL         = 20
N_WRONG      = 1
N_CORRECT    = N_CL - N_WRONG    # 18
N_CLUSTERS   = 10
EPS          = 0.9
REG          = 1e-8
RANDOM_STATE = 42
EIGSH_TOL    = 1e-5
EIGSH_MAXITER = 3000


# ── Noisy cannot-link graph builder ───────────────────────────────────────────

def build_noisy_cannot_link_graph(y, n_cl=20, n_wrong=2, random_state=0):
    """Each vertex gets:
      n_wrong         edges to same-class nodes   (wrong constraints)
      n_cl - n_wrong  edges to other-class nodes  (correct constraints)
    Returns symmetrised CSR adjacency matrix.
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

        # wrong edges: same class
        same_idx = [j for j in label_to_idx[my_label] if j != i]
        n_w = min(n_wrong, len(same_idx))
        if n_w > 0:
            chosen = rng.choice(same_idx, size=n_w, replace=False)
            rows.extend([i] * n_w)
            cols.extend(chosen.tolist())

        # correct edges: other classes
        other_idx = np.concatenate(
            [np.array(label_to_idx[l]) for l in labels if l != my_label]
        )
        n_c = min(n_cl - n_wrong, len(other_idx))
        chosen = rng.choice(other_idx, size=n_c, replace=False)
        rows.extend([i] * n_c)
        cols.extend(chosen.tolist())

    vals = np.ones(len(rows), dtype=float)
    W_H  = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    return W_H + W_H.T


def sparse_cut(W, labels):
    W_tri      = triu(W, k=1).tocsr()
    r, c       = W_tri.nonzero()
    data       = np.array(W_tri[r, c]).ravel()
    return float(data[labels[r] != labels[c]].sum())


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


# ── 2. Build G ────────────────────────────────────────────────────────────────
print("\nBuilding must-link graph G  (k={} NN)".format(K_NN))
t0 = time.time()
W_G   = build_knn_graph(X, k=K_NN, use_gaussian=False)
deg_g = np.array(W_G.sum(axis=1)).ravel()
print(f"  |E|={W_G.nnz//2}  avg_deg={deg_g.mean():.2f}  elapsed: {time.time()-t0:.1f}s")


# ── 3. Build noisy H ──────────────────────────────────────────────────────────
print(f"\nBuilding noisy cannot-link graph H  "
      f"(n_cl={N_CL}: {N_CORRECT} correct + {N_WRONG} wrong per vertex)")
t0 = time.time()
W_H   = build_noisy_cannot_link_graph(y, n_cl=N_CL, n_wrong=N_WRONG,
                                       random_state=RANDOM_STATE)
deg_h = np.array(W_H.sum(axis=1)).ravel()

# edge audit
W_tri = triu(W_H, k=1).tocsr()
tr, tc = W_tri.nonzero()
n_wrong_edges  = int((y[tr] == y[tc]).sum())
n_correct_edges = W_H.nnz // 2 - n_wrong_edges
print(f"  |E(H)|={W_H.nnz//2}  avg_deg={deg_h.mean():.2f}")
print(f"  Wrong (same-class) edges : {n_wrong_edges}  "
      f"({100*n_wrong_edges/(W_H.nnz//2):.1f}%)")
print(f"  Correct edges            : {n_correct_edges}  "
      f"({100*n_correct_edges/(W_H.nnz//2):.1f}%)")
print(f"  elapsed: {time.time()-t0:.1f}s")


# ── 4. Compute φ and Υ ────────────────────────────────────────────────────────
print("\nComputing φ and Υ  (approximate eigsh)")
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

cut_G = sparse_cut(W_G, y)
cut_H = sparse_cut(W_H, y)   # only cross-class H edges
phi   = cut_G / max(cut_H, 1.0)
upsilon = lk1 / max(phi, 1e-12)
print(f"  φ={phi:.5f}  λ_{{k+1}}={lk1:.5f}  Υ={upsilon:.4f}  elapsed: {time.time()-t0:.1f}s")


# ── 5. Spectral Clustering (SC) on G only ────────────────────────────────────
print("\nRunning Spectral Clustering (SC) ...")
t0 = time.time()

L_G_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
sc_labels, _, _ = spectral_clustering(L_G_norm, n_clusters=N_CLUSTERS,
                                       random_state=RANDOM_STATE)
sc_ari = adjusted_rand_score(y, sc_labels)
sc_nmi = normalized_mutual_info_score(y, sc_labels)
print(f"  ARI={sc_ari:.4f}  NMI={sc_nmi:.4f}  elapsed: {time.time()-t0:.1f}s")


# ── 6. Constrained Spectral Clustering (CSC) on G + H ────────────────────────
print("\nRunning Constrained Spectral Clustering (CSC) ...")
t0 = time.time()

csc_labels, csc_vals, _ = iterative_clustering(
    W_G, W_H, k=N_CLUSTERS, exact=False, eps=EPS, random_seed=RANDOM_STATE
)
csc_ari = adjusted_rand_score(y, csc_labels)
csc_nmi = normalized_mutual_info_score(y, csc_labels)
print(f"  ARI={csc_ari:.4f}  NMI={csc_nmi:.4f}  elapsed: {time.time()-t0:.1f}s")


# ── 7. Summary ────────────────────────────────────────────────────────────────
print()
print("=" * 65)
print("SUMMARY — MNIST  N={:,}  G: {}-NN  H: n_cl={} ({} wrong/vertex)".format(
    N_SAMPLES, K_NN, N_CL, N_WRONG))
print("=" * 65)
print(f"  {'Method':<35}  {'ARI':>8}  {'NMI':>8}")
print(f"  {'-'*35}  {'-'*8}  {'-'*8}")
print(f"  {'SC  (G only)':35}  {sc_ari:8.4f}  {sc_nmi:8.4f}")
print(f"  {'CSC (G + noisy H)':35}  {csc_ari:8.4f}  {csc_nmi:8.4f}")
delta_ari = csc_ari - sc_ari
delta_nmi = csc_nmi - sc_nmi
sign_ari  = "+" if delta_ari >= 0 else ""
sign_nmi  = "+" if delta_nmi >= 0 else ""
print(f"  {'Δ (CSC − SC)':35}  {sign_ari}{delta_ari:7.4f}  {sign_nmi}{delta_nmi:7.4f}")
print()
print(f"  φ        = {phi:.5f}   (cut_G={cut_G:.0f} / cut_H={cut_H:.0f})")
print(f"  λ_{{k+1}}  = {lk1:.5f}")
print(f"  Υ        = {upsilon:.4f}")
print()
print(f"  Noise: {N_WRONG}/{N_CL} = {100*N_WRONG/N_CL:.0f}% wrong constraints per vertex")
print(f"         {n_wrong_edges}/{W_H.nnz//2} = {100*n_wrong_edges/(W_H.nnz//2):.1f}% wrong edges in H")
