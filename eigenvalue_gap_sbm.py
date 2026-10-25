"""
Eigenvalue gap analysis on SBM — precise computation via dense eigh.

Settings: block_size=100, p=0.2, q=0.12, eps=0.9, k=2..10
Outputs:  λ_1 … λ_{k+1} for each k (generalized eigenproblem L_G v = λ L_H v)
Solver:   scipy.linalg.eigh (dense, exact) — feasible since N = k*100 ≤ 1000
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from utils import SBM, normalized_laplacian_from_D_sp

BLOCK_SIZE = 100
P          = 0.2
Q          = 0.12
EPS        = 0.9
SEED       = 42

print(f"SBM: block_size={BLOCK_SIZE}, p={P}, q={Q}, eps={EPS}")
print(f"{'k':>4}  {'λ_1...λ_{k+1}'}")
print("-" * 100)

for k in range(2, 11):
    np.random.seed(SEED)
    G, H = SBM(block_size=BLOCK_SIZE, k=k, p=P, q=Q)
    W_G  = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    W_H  = nx.to_scipy_sparse_array(H, format='csr').astype(float)

    deg_g = np.array(W_G.sum(1)).ravel()
    deg_h = np.array(W_H.sum(1)).ravel()
    deg   = EPS * deg_g + (1 - EPS) * deg_h

    L_G = normalized_laplacian_from_D_sp(W_G, deg).toarray()
    L_H = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg)).toarray()

    # Regularise L_H to ensure positive-definiteness (L_H is only PSD in general)
    N = W_G.shape[0]
    L_H += 1e-8 * np.eye(N)

    # Full dense generalised eigenproblem — exact, no truncation
    vals = sla.eigh(L_G, L_H, eigvals_only=True)
    vals = np.sort(np.real(vals))

    # Print λ_1 … λ_{k+1}
    shown = vals[:k + 1]
    eig_str = "  ".join(f"λ{i+1}={v:.6f}" for i, v in enumerate(shown))
    gap   = shown[k] - shown[k - 1]
    ratio = shown[k] / max(shown[k - 1], 1e-12)
    print(f"k={k:2d}  {eig_str}  | gap={gap:.6f}  ratio={ratio:.4f}")
