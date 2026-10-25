"""
Lambda, Phi, and Upsilon analysis on SBM.

For each (p, q) parameter setting and k in {2, ..., 10}:
  - Generate SBM with block_size=100
  - Compute generalised eigenvalues λ_1 … λ_{k+1} of  L_G v = λ L_H v
  - Compute φ = cut_G(y*) / cut_H(y*)  under the ground-truth partition y*
    (must-link cut divided by cannot-link cut at the optimal clustering)
  - Compute Υ (upsilon) = λ_{k+1} / φ

L_G, L_H are the double-normalised Laplacians from iterative_clustering:
    D   = eps * deg_G + (1-eps) * deg_H
    L_G = I - D^{-1/2} W_G D^{-1/2}
    L_H = I - D_H^{-1/2} W_H D_H^{-1/2},  with D_H = max(deg_H, D)

Eigenvalues are computed with scipy.linalg.eigh (dense, exact).
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from utils import SBM, normalized_laplacian_from_D_sp

# ── Fixed settings ─────────────────────────────────────────────────────────────
BLOCK_SIZE = 100
EPS        = 0.9
SEED       = 42
K_RANGE    = range(2, 11)

# ── Parameter sets (p, q) to sweep ────────────────────────────────────────────
PARAM_SETS = [
    (0.20, 0.05),   # high SNR
    (0.20, 0.10),   # medium SNR
    (0.20, 0.15),   # low SNR
    (0.20, 0.18),   # very low SNR
]


# ── Helper functions ───────────────────────────────────────────────────────────

def compute_eigenvalues(W_G, W_H, n_keep, eps=0.9, reg=1e-8):
    """Return the n_keep smallest generalised eigenvalues of L_G v = λ L_H v."""
    deg_g = np.array(W_G.sum(1)).ravel()
    deg_h = np.array(W_H.sum(1)).ravel()
    deg   = eps * deg_g + (1.0 - eps) * deg_h

    L_G = normalized_laplacian_from_D_sp(W_G, deg).toarray()
    L_H = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg)).toarray()
    L_H += reg * np.eye(W_G.shape[0])

    vals = sla.eigh(L_G, L_H, eigvals_only=True)
    return np.sort(np.real(vals))[:n_keep]


def compute_phi(G, H, true_labels):
    """φ = |cut edges in G| / |cut edges in H| under the ground-truth partition."""
    g_cut = sum(1 for u, v in G.edges() if true_labels[u] != true_labels[v])
    h_cut = sum(1 for u, v in H.edges() if true_labels[u] != true_labels[v])
    return g_cut / max(h_cut, 1)


# ── Collect all results in one pass ───────────────────────────────────────────
# results[(p,q)][k] = {'eigs': array, 'phi': float, 'upsilon': float}
results = {}

for p, q in PARAM_SETS:
    results[(p, q)] = {}
    for k in K_RANGE:
        np.random.seed(SEED)
        G, H = SBM(block_size=BLOCK_SIZE, k=k, p=p, q=q)
        W_G  = nx.to_scipy_sparse_array(G, format='csr').astype(float)
        W_H  = nx.to_scipy_sparse_array(H, format='csr').astype(float)
        true_labels = np.array([G.nodes[i]['block'] for i in G.nodes()])

        eigs     = compute_eigenvalues(W_G, W_H, n_keep=k + 1, eps=EPS)
        phi      = compute_phi(G, H, true_labels)
        upsilon  = eigs[k] / max(phi, 1e-12)   # Υ = λ_{k+1} / φ

        results[(p, q)][k] = {'eigs': eigs, 'phi': phi, 'upsilon': upsilon}


# ── Per-parameter detailed tables ─────────────────────────────────────────────
for p, q in PARAM_SETS:
    snr = (p - q) / (p + q)
    W   = 120
    print()
    print("=" * W)
    print(f"SBM:  p={p:.2f}  q={q:.2f}  SNR=(p-q)/(p+q)={snr:.3f}   "
          f"block_size={BLOCK_SIZE}  eps={EPS}")
    print("=" * W)
    print(f"  Υ (upsilon) = λ_{{k+1}} / φ")
    print("-" * W)
    print(f"{'k':>3}  {'N':>5}  {'φ':>9}  {'Υ':>9}  "
          f"eigenvalues  λ_1 … λ_{{k+1}}")
    print("-" * W)

    for k in K_RANGE:
        r       = results[(p, q)][k]
        eigs    = r['eigs']
        phi     = r['phi']
        upsilon = r['upsilon']
        eig_str = "  ".join(f"{v:7.4f}" for v in eigs)
        print(f"k={k:2d}  N={k*BLOCK_SIZE:4d}  "
              f"φ={phi:8.5f}  Υ={upsilon:8.4f}  [{eig_str}]")
    print()


# ── Summary tables ─────────────────────────────────────────────────────────────
COL_W  = 16
SEP    = f"  p={PARAM_SETS[0][0]:.2f}"   # just for width calc
HDR    = f"{'k':>4}" + "".join(
    f"  p={p:.2f},q={q:.2f}".rjust(COL_W) for p, q in PARAM_SETS
)
DIV    = "-" * (4 + len(PARAM_SETS) * (COL_W + 2))

# ── φ summary ─────────────────────────────────────────────────────────────────
print()
print("=" * len(HDR))
print("SUMMARY TABLE 1 — φ  (must-link cut / cannot-link cut at ground-truth partition)")
print("=" * len(HDR))
print(HDR)
print(DIV)
for k in K_RANGE:
    row = f"{k:>4}"
    for p, q in PARAM_SETS:
        row += f"{results[(p,q)][k]['phi']:>{COL_W}.6f}"
    print(row)

# ── λ_{k+1} summary ───────────────────────────────────────────────────────────
print()
print("=" * len(HDR))
print("SUMMARY TABLE 2 — λ_{k+1}  (first out-of-cluster eigenvalue)")
print("=" * len(HDR))
print(HDR)
print(DIV)
for k in K_RANGE:
    row = f"{k:>4}"
    for p, q in PARAM_SETS:
        row += f"{results[(p,q)][k]['eigs'][k]:>{COL_W}.6f}"
    print(row)

# ── Υ summary ─────────────────────────────────────────────────────────────────
print()
print("=" * len(HDR))
print("SUMMARY TABLE 3 — Υ = λ_{k+1} / φ  (upsilon)")
print("=" * len(HDR))
print(HDR)
print(DIV)
for k in K_RANGE:
    row = f"{k:>4}"
    for p, q in PARAM_SETS:
        row += f"{results[(p,q)][k]['upsilon']:>{COL_W}.6f}"
    print(row)

# ── eigenvalue gap λ_{k+1} - λ_k summary ─────────────────────────────────────
print()
print("=" * len(HDR))
print("SUMMARY TABLE 4 — gap = λ_{k+1} - λ_k")
print("=" * len(HDR))
print(HDR)
print(DIV)
for k in K_RANGE:
    row = f"{k:>4}"
    for p, q in PARAM_SETS:
        eigs = results[(p, q)][k]['eigs']
        row += f"{eigs[k] - eigs[k-1]:>{COL_W}.6f}"
    print(row)

print()
