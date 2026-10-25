"""
Upsilon trend analysis on SBM.

Fixed:  p=0.5, q=0.01, block_size=100
Vary:   k = 2 … 20

For each k:
  - Generate SBM (N = k * 100 nodes)
  - Compute λ_1 … λ_{k+1}  (generalised eigenproblem L_G v = λ L_H v)
  - Compute φ = cut_G(y*) / cut_H(y*)  under the ground-truth partition y*
  - Compute Υ = λ_{k+1} / φ
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from utils import SBM, normalized_laplacian_from_D_sp

# ── Settings ───────────────────────────────────────────────────────────────────
P          = 0.5
Q          = 0.01
BLOCK_SIZE = 100
EPS        = 0.9
SEED       = 42
K_RANGE    = range(2, 21)


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_eigenvalues(W_G, W_H, n_keep, eps=EPS, reg=1e-8):
    deg_g = np.array(W_G.sum(1)).ravel()
    deg_h = np.array(W_H.sum(1)).ravel()
    deg   = eps * deg_g + (1.0 - eps) * deg_h

    L_G = normalized_laplacian_from_D_sp(W_G, deg).toarray()
    L_H = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg)).toarray()
    L_H += reg * np.eye(W_G.shape[0])

    vals = sla.eigh(L_G, L_H, eigvals_only=True)
    return np.sort(np.real(vals))[:n_keep]


def compute_phi(G, H, true_labels):
    g_cut = sum(1 for u, v in G.edges() if true_labels[u] != true_labels[v])
    h_cut = sum(1 for u, v in H.edges() if true_labels[u] != true_labels[v])
    return g_cut / max(h_cut, 1)


# ── Main sweep ─────────────────────────────────────────────────────────────────
print("=" * 110)
print(f"SBM:  p={P}  q={Q}  block_size={BLOCK_SIZE}  eps={EPS}  seed={SEED}")
print(f"Υ (upsilon) = λ_{{k+1}} / φ")
print("=" * 110)
print(f"{'k':>4}  {'N':>6}  {'φ':>10}  {'λ_k':>10}  {'λ_{k+1}':>10}  "
      f"{'gap':>10}  {'Υ':>10}  eigenvalues λ_1 … λ_{{k+1}}")
print("-" * 110)

upsilons = []
phis     = []
lk1s     = []

for k in K_RANGE:
    np.random.seed(SEED)
    G, H = SBM(block_size=BLOCK_SIZE, k=k, p=P, q=Q)
    W_G  = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    W_H  = nx.to_scipy_sparse_array(H, format='csr').astype(float)
    true_labels = np.array([G.nodes[i]['block'] for i in G.nodes()])

    eigs    = compute_eigenvalues(W_G, W_H, n_keep=k + 1)
    phi     = compute_phi(G, H, true_labels)
    lk      = eigs[k - 1]          # λ_k  (last in-cluster eigenvalue)
    lk1     = eigs[k]               # λ_{k+1} (first out-of-cluster eigenvalue)
    gap     = lk1 - lk
    upsilon = lk1 / max(phi, 1e-12)

    upsilons.append(upsilon)
    phis.append(phi)
    lk1s.append(lk1)

    eig_str = "  ".join(f"{v:7.4f}" for v in eigs)
    print(f"k={k:>2}  N={k*BLOCK_SIZE:>5}  "
          f"φ={phi:9.5f}  λ_k={lk:9.5f}  λ_{{k+1}}={lk1:9.5f}  "
          f"gap={gap:9.5f}  Υ={upsilon:9.4f}  [{eig_str}]")

# ── Upsilon trend summary ──────────────────────────────────────────────────────
print()
print("=" * 60)
print("UPSILON TREND SUMMARY")
print(f"{'k':>4}  {'Υ = λ_{k+1}/φ':>14}  {'direction':>12}")
print("-" * 60)
for i, k in enumerate(K_RANGE):
    if i == 0:
        direction = "—"
    else:
        delta = upsilons[i] - upsilons[i - 1]
        direction = f"↑ +{delta:.4f}" if delta > 0 else f"↓ {delta:.4f}"
    print(f"k={k:>2}  Υ={upsilons[i]:>12.6f}  {direction:>12}")

print()
print(f"  min Υ = {min(upsilons):.6f}  at k={list(K_RANGE)[np.argmin(upsilons)]}")
print(f"  max Υ = {max(upsilons):.6f}  at k={list(K_RANGE)[np.argmax(upsilons)]}")
print(f"  φ is stable: mean={np.mean(phis):.6f}  std={np.std(phis):.6f}")
print("=" * 60)
