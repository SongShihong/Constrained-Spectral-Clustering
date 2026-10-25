"""
Eigenvalue Gap Analysis for the Generalised Laplacian
======================================================
Verifies the spectral gap  λ_k vs λ_{k+1}  in the generalised eigenproblem

    L_G_sym v = λ L_H_sym v

used by iterative_clustering() (same normalisation).

Datasets
--------
  SBM   : k = 2 … 10  (block_size=500, p=0.2, q=0.155, eps=0.9)
  MNIST : K = 10,  n_cl ∈ {5, 10, 20}  (k_nn=3, eps=0.7)
  USPS  : K = 10,  n_cl ∈ {5, 10, 20}  (k_nn=5, eps=0.9/0.9/0.5)

Figure layout (3 rows × 2 cols)
---------------------------------
  Row 0 : SBM    — [eigenvalue spectrum, k=2..10]  |  [gap vs k]
  Row 1 : MNIST  — [eigenvalue spectrum, n_cl×3]   |  [gap vs n_cl]
  Row 2 : USPS   — [eigenvalue spectrum, n_cl×3]   |  [gap vs n_cl]
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

from scipy.sparse.linalg import eigsh
import networkx as nx
import h5py

from utils import (
    SBM, build_knn_graph, build_cannot_link_graph,
    normalized_laplacian_from_D_sp,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def gen_eigenvalues(W_G, W_H, k, eps, n_eigs, seed=42):
    """
    Compute the n_eigs smallest generalised eigenvalues of
        L_G_sym v = λ L_H_sym v
    using exactly the same path as iterative_clustering():
      lobpcg(L_G, X0, B=L_H, largest=False, tol=1e-6, maxiter=200)
    n_eigs must be <= n_clusters argument passed; we request n_eigs eigenvalues.
    """
    from scipy.sparse.linalg import lobpcg

    deg_g = np.array(W_G.sum(1)).ravel()
    deg_h = np.array(W_H.sum(1)).ravel()
    deg   = eps * deg_g + (1 - eps) * deg_h
    L_G   = normalized_laplacian_from_D_sp(W_G, deg)
    L_H   = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))

    N = W_G.shape[0]
    n_eigs = min(n_eigs, N - 2)

    np.random.seed(seed)
    X0 = np.random.randn(N, n_eigs)
    vals, _ = lobpcg(L_G, X0, B=L_H, largest=False, tol=1e-6, maxiter=200)
    return np.sort(np.real(vals))


def gap_and_ratio(vals, k):
    """Absolute gap and ratio at position k (1-indexed, so vals[k-1] and vals[k])."""
    lk   = vals[k - 1]   # λ_k
    lk1  = vals[k]       # λ_{k+1}
    return lk1 - lk, lk1 / max(lk, 1e-12)


# ── Colour / style helpers ────────────────────────────────────────────────────
COLORS_NCI = {'5': '#2196F3', '10': '#FF9800', '20': '#4CAF50'}
SEED = 42

# ═════════════════════════════════════════════════════════════════════════════
# Load data (once)
# ═════════════════════════════════════════════════════════════════════════════
print("Loading MNIST …")
try:
    import keras.datasets.mnist as km
    (X_mnist, y_mnist), _ = km.load_data()
except Exception:
    from sklearn.datasets import fetch_openml
    _m = fetch_openml("mnist_784", version=1, as_frame=False)
    X_mnist, y_mnist = _m.data.astype(np.float32), _m.target.astype(int)
X_mnist = np.reshape(X_mnist, (len(X_mnist), -1)).astype(np.float32)
y_mnist = np.array(y_mnist, dtype=int)

print("Loading USPS …")
with h5py.File('./datasets/usps/usps.h5', 'r') as hf:
    X_usps = hf['train']['data'][:]
    y_usps = hf['train']['target'][:]
X_usps = np.reshape(X_usps, (len(X_usps), -1)).astype(np.float32)
y_usps = np.array(y_usps, dtype=int)

print("Subsampling MNIST to N=10 000 …")
rng_sub = np.random.default_rng(SEED)
idx_m   = rng_sub.choice(len(X_mnist), size=10000, replace=False)
X_mnist = X_mnist[idx_m];  y_mnist = y_mnist[idx_m]

print("Building MNIST kNN graph (k=3, N=10 000) …")
W_G_mnist = build_knn_graph(X_mnist, k=3, use_gaussian=False)
print("Building USPS  kNN graph (k=5, N=7291) …")
W_G_usps  = build_knn_graph(X_usps,  k=5, use_gaussian=False)

# ═════════════════════════════════════════════════════════════════════════════
# Compute eigenvalues
# ═════════════════════════════════════════════════════════════════════════════

# ── SBM ──────────────────────────────────────────────────────────────────────
print("\nSBM eigenvalues …")
sbm_records = {}   # k -> {'vals', 'gap', 'ratio'}

for k in range(2, 11):
    G, H = SBM(block_size=500, k=k, p=0.2, q=0.155)
    W_G  = nx.to_scipy_sparse_array(G)
    W_H  = nx.to_scipy_sparse_array(H)
    vals = gen_eigenvalues(W_G, W_H, k=k, eps=0.9, n_eigs=k+3, seed=SEED)
    g, r = gap_and_ratio(vals, k)
    sbm_records[k] = {'vals': vals, 'gap': g, 'ratio': r}
    print(f"  k={k:2d}  λ_k={vals[k-1]:.6f}  λ_{{k+1}}={vals[k]:.6f}"
          f"  gap={g:.6f}  ratio={r:.4f}")

# ── MNIST ─────────────────────────────────────────────────────────────────────
print("\nMNIST eigenvalues …")
K_MNIST = 10
mnist_records = {}

for n_cl in [5, 10, 20]:
    W_H  = build_cannot_link_graph(y_mnist, n_cl=n_cl, random_state=SEED)
    vals = gen_eigenvalues(W_G_mnist, W_H, k=K_MNIST, eps=0.7, n_eigs=K_MNIST+3, seed=SEED)
    g, r = gap_and_ratio(vals, K_MNIST)
    mnist_records[n_cl] = {'vals': vals, 'gap': g, 'ratio': r}
    print(f"  n_cl={n_cl:2d}  λ_k={vals[K_MNIST-1]:.6f}  λ_{{k+1}}={vals[K_MNIST]:.6f}"
          f"  gap={g:.6f}  ratio={r:.4f}")

# ── USPS ──────────────────────────────────────────────────────────────────────
print("\nUSPS eigenvalues …")
K_USPS = 10
usps_records = {}

for n_cl in [5, 10, 20]:
    eps_usps = 0.5 if n_cl == 20 else 0.9
    W_H  = build_cannot_link_graph(y_usps, n_cl=n_cl, random_state=SEED)
    vals = gen_eigenvalues(W_G_usps, W_H, k=K_USPS, eps=eps_usps, n_eigs=K_USPS+3, seed=SEED)
    g, r = gap_and_ratio(vals, K_USPS)
    usps_records[n_cl] = {'vals': vals, 'gap': g, 'ratio': r}
    print(f"  n_cl={n_cl:2d}  λ_k={vals[K_USPS-1]:.6f}  λ_{{k+1}}={vals[K_USPS]:.6f}"
          f"  gap={g:.6f}  ratio={r:.4f}")

# ═════════════════════════════════════════════════════════════════════════════
# Plot
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(13, 13))
gs  = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.38)

# ── helper: annotate bar with value ──────────────────────────────────────────
def annotate_bars(ax, rects, fmt='{:.4f}'):
    for r in rects:
        h = r.get_height()
        ax.text(r.get_x() + r.get_width()/2, h * 1.02, fmt.format(h),
                ha='center', va='bottom', fontsize=7.5)

# ─────────────────────────────────────────────────────────────────────────────
# Row 0: SBM
# ─────────────────────────────────────────────────────────────────────────────
ax0L = fig.add_subplot(gs[0, 0])
ax0R = fig.add_subplot(gs[0, 1])

# Left: eigenvalue spectra λ_1…λ_{k+1} for all k=2..10
cmap = plt.cm.Blues
for i, k in enumerate(range(2, 11)):
    rec  = sbm_records[k]
    vals = rec['vals'][:k + 1]   # show λ_1 … λ_{k+1}
    col  = cmap(0.3 + 0.07 * i)
    ax0L.plot(range(1, k + 2), vals, 'o-', color=col, lw=1.4, ms=4,
              label=f'k={k}')
    # mark the gap with a vertical segment at index k+1
    ax0L.vlines(k + 1, vals[k - 1], vals[k], color=col, lw=2.5, alpha=0.6)

ax0L.set_xlabel('Eigenvalue index')
ax0L.set_ylabel('Eigenvalue value')
ax0L.set_title('SBM — eigenvalue spectra λ₁…λ_{k+1}  (k = 2…10)')
ax0L.legend(fontsize=7, loc='upper left', ncol=2)
ax0L.set_xticks(range(1, 13))

# Right: gap vs k
ks   = list(sbm_records.keys())
gaps = [sbm_records[k]['gap']   for k in ks]
rats = [sbm_records[k]['ratio'] for k in ks]
bars = ax0R.bar(ks, gaps, color=plt.cm.Blues(0.6), edgecolor='white', width=0.6, zorder=2)
annotate_bars(ax0R, bars)
ax0R.set_xlabel('Number of clusters k')
ax0R.set_ylabel('Eigenvalue gap  λ_{k+1} − λ_k', labelpad=4)
ax0R.set_title('SBM — eigenvalue gap vs k')
ax0R.set_xticks(ks)
ax0R.grid(axis='y', ls='--', alpha=0.4)

ax0R_twin = ax0R.twinx()
ax0R_twin.plot(ks, rats, 's--', color='tomato', lw=1.4, ms=5, label='ratio λ_{k+1}/λ_k')
ax0R_twin.set_ylabel('Ratio  λ_{k+1} / λ_k', color='tomato')
ax0R_twin.tick_params(axis='y', labelcolor='tomato')
ax0R_twin.legend(loc='upper right', fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Row 1: MNIST
# ─────────────────────────────────────────────────────────────────────────────
ax1L = fig.add_subplot(gs[1, 0])
ax1R = fig.add_subplot(gs[1, 1])

for n_cl in [5, 10, 20]:
    rec  = mnist_records[n_cl]
    vals = rec['vals']
    n    = len(vals)
    c    = COLORS_NCI[str(n_cl)]
    ax1L.plot(range(1, n+1), vals, 'o-', color=c, lw=1.6, ms=5,
              label=f'n_cl={n_cl}  (gap={rec["gap"]:.4f})')

ax1L.axvline(K_MNIST + 0.5, color='gray', ls='--', lw=1.2, alpha=0.7, label=f'k={K_MNIST}')
ax1L.set_xlabel('Eigenvalue index')
ax1L.set_ylabel('Eigenvalue value')
ax1L.set_title(f'MNIST — eigenvalue spectra (K={K_MNIST})')
ax1L.legend(fontsize=8)
ax1L.set_xticks(range(1, K_MNIST+4))

n_cl_vals = [5, 10, 20]
gaps_mnist = [mnist_records[n]['gap']   for n in n_cl_vals]
rats_mnist = [mnist_records[n]['ratio'] for n in n_cl_vals]
bars = ax1R.bar(n_cl_vals,
                gaps_mnist,
                color=[COLORS_NCI[str(n)] for n in n_cl_vals],
                edgecolor='white', width=3, zorder=2)
annotate_bars(ax1R, bars)
ax1R.set_xlabel('n_cl (cannot-link edges per node)')
ax1R.set_ylabel('Eigenvalue gap  λ_{k+1} − λ_k')
ax1R.set_title(f'MNIST — eigenvalue gap vs n_cl')
ax1R.set_xticks(n_cl_vals)
ax1R.grid(axis='y', ls='--', alpha=0.4)

ax1R_twin = ax1R.twinx()
ax1R_twin.plot(n_cl_vals, rats_mnist, 's--', color='tomato', lw=1.4, ms=5,
               label='ratio λ_{k+1}/λ_k')
ax1R_twin.set_ylabel('Ratio  λ_{k+1} / λ_k', color='tomato')
ax1R_twin.tick_params(axis='y', labelcolor='tomato')
ax1R_twin.legend(loc='upper left', fontsize=8)

# ─────────────────────────────────────────────────────────────────────────────
# Row 2: USPS
# ─────────────────────────────────────────────────────────────────────────────
ax2L = fig.add_subplot(gs[2, 0])
ax2R = fig.add_subplot(gs[2, 1])

for n_cl in [5, 10, 20]:
    rec  = usps_records[n_cl]
    vals = rec['vals']
    n    = len(vals)
    c    = COLORS_NCI[str(n_cl)]
    ax2L.plot(range(1, n+1), vals, 'o-', color=c, lw=1.6, ms=5,
              label=f'n_cl={n_cl}  (gap={rec["gap"]:.4f})')

ax2L.axvline(K_USPS + 0.5, color='gray', ls='--', lw=1.2, alpha=0.7, label=f'k={K_USPS}')
ax2L.set_xlabel('Eigenvalue index')
ax2L.set_ylabel('Eigenvalue value')
ax2L.set_title(f'USPS — eigenvalue spectra (K={K_USPS})')
ax2L.legend(fontsize=8)
ax2L.set_xticks(range(1, K_USPS+4))

gaps_usps = [usps_records[n]['gap']   for n in n_cl_vals]
rats_usps = [usps_records[n]['ratio'] for n in n_cl_vals]
bars = ax2R.bar(n_cl_vals,
                gaps_usps,
                color=[COLORS_NCI[str(n)] for n in n_cl_vals],
                edgecolor='white', width=3, zorder=2)
annotate_bars(ax2R, bars)
ax2R.set_xlabel('n_cl (cannot-link edges per node)')
ax2R.set_ylabel('Eigenvalue gap  λ_{k+1} − λ_k')
ax2R.set_title(f'USPS — eigenvalue gap vs n_cl')
ax2R.set_xticks(n_cl_vals)
ax2R.grid(axis='y', ls='--', alpha=0.4)

ax2R_twin = ax2R.twinx()
ax2R_twin.plot(n_cl_vals, rats_usps, 's--', color='tomato', lw=1.4, ms=5,
               label='ratio λ_{k+1}/λ_k')
ax2R_twin.set_ylabel('Ratio  λ_{k+1} / λ_k', color='tomato')
ax2R_twin.tick_params(axis='y', labelcolor='tomato')
ax2R_twin.legend(loc='upper left', fontsize=8)

fig.suptitle('Generalised Laplacian Eigenvalue Gap  (λ_k vs λ_{k+1})',
             fontsize=14, fontweight='bold', y=0.995)

out = '/home/songs/ConstrainedSpectralClustering/eigenvalue_gap.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nFigure saved → {out}")

# ═════════════════════════════════════════════════════════════════════════════
# Numerical summary table
# ═════════════════════════════════════════════════════════════════════════════
W = 72
print("\n" + "="*W)
print("NUMERICAL SUMMARY")
print("="*W)

print("\n── SBM (block_size=500, p=0.2, q=0.155, eps=0.9) ──────────────────────")
for k in range(2, 11):
    v = sbm_records[k]['vals']
    eig_str = "  ".join(f"λ{i+1}={v[i]:.4f}" for i in range(k + 1))
    print(f"k={k:2d}  {eig_str}  | gap={sbm_records[k]['gap']:.6f}  ratio={sbm_records[k]['ratio']:.4f}")

print("\n── MNIST (K=10, k_nn=3, eps=0.7) ──────────────────────────────────────")
for n_cl in [5, 10, 20]:
    v = mnist_records[n_cl]['vals']
    eig_str = "  ".join(f"λ{i+1}={v[i]:.4f}" for i in range(K_MNIST + 1))
    print(f"n_cl={n_cl:2d}  {eig_str}  | gap={mnist_records[n_cl]['gap']:.6f}  ratio={mnist_records[n_cl]['ratio']:.4f}")

print("\n── USPS (K=10, k_nn=5, eps=0.9/0.9/0.5 for n_cl=5/10/20) ─────────────")
for n_cl in [5, 10, 20]:
    v = usps_records[n_cl]['vals']
    eig_str = "  ".join(f"λ{i+1}={v[i]:.4f}" for i in range(K_USPS + 1))
    print(f"n_cl={n_cl:2d}  {eig_str}  | gap={usps_records[n_cl]['gap']:.6f}  ratio={usps_records[n_cl]['ratio']:.4f}")
print("="*W)
