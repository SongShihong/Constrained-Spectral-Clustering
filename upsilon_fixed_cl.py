"""
Upsilon trend with fixed 5-NN cannot-link graph.

Construction
------------
  G (must-link)   : SBM with fixed p=0.5, q=0.01  (same as before)
  H (cannot-link) : for each node, connect to 5 randomly chosen nodes
                    from OTHER ground-truth clusters  (fixed degree = 5)

Why this changes the analysis
------------------------------
  With SBM-based H:
      cut_H(y*) ≈ C(k,2)·n²·p   →   φ = cut_G/cut_H ≈ q/p  (constant, k cancels)

  With 5-NN cannot-link H:
      Every edge in H is a between-cluster edge by construction, so
      cut_H(y*) = |E(H)| = n·k·5/2   (ALL edges cross boundaries; grows with k)
      cut_G(y*) ≈ C(k,2)·n²·q = k(k-1)/2 · n²·q   (grows as k²)

      φ = cut_G / cut_H
        ≈ [k(k-1)·n²·q/2] / [n·k·5/2]
        = (k-1)·n·q / 5
        ~ k·n·q / 5          ← grows linearly with k (k no longer cancels!)

  Therefore  Υ = λ_{k+1} / φ  now depends on how λ_{k+1} grows vs. k.

Fixed:  p=0.5, q=0.01, block_size=100, n_cl=5, k=2..20
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import scipy.linalg as sla
import networkx as nx
import warnings
warnings.filterwarnings("ignore")

from utils import SBM, normalized_laplacian_from_D_sp, build_cannot_link_graph

# ── Settings ───────────────────────────────────────────────────────────────────
P          = 0.5
Q          = 0.01
BLOCK_SIZE = 100
N_CL       = 5        # cannot-link neighbours per node
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


def compute_phi(G, H_nx, true_labels):
    """φ = cut_G(y*) / cut_H(y*)."""
    g_cut = sum(1 for u, v in G.edges()    if true_labels[u] != true_labels[v])
    h_cut = sum(1 for u, v in H_nx.edges() if true_labels[u] != true_labels[v])
    return g_cut / max(h_cut, 1)


def phi_theory(k, n, q, n_cl):
    """Expected φ under the new construction.

    cut_G(y*) ≈ k(k-1)/2 · n² · q
    cut_H(y*) = |E(H)| = n·k·n_cl / 2    (all H edges cross boundaries)
    φ ≈ [k(k-1)·n²·q] / [n·k·n_cl] = (k-1)·n·q / n_cl
    """
    return (k - 1) * n * q / n_cl


# ── Main sweep ─────────────────────────────────────────────────────────────────
print("=" * 120)
print(f"G: SBM p={P} q={Q} block_size={BLOCK_SIZE}   "
      f"H: {N_CL}-NN cannot-link (ground-truth clusters)   eps={EPS}")
print()
print("Theory:  cut_H(y*) = |E(H)| = n·k·5/2  (all edges cross clusters by construction)")
print("         φ ≈ (k-1)·n·q / 5   →   grows linearly with k")
print("         Υ = λ_{k+1} / φ   —  trend to be determined empirically")
print("=" * 120)
print(f"{'k':>4}  {'N':>5}  {'deg_H':>7}  {'φ_emp':>10}  {'φ_theory':>10}  "
      f"{'λ_k':>9}  {'λ_{k+1}':>9}  {'gap':>9}  {'Υ':>10}")
print("-" * 120)

upsilons   = []
phis_emp   = []
phis_th    = []
lk1s       = []

for k in K_RANGE:
    n = BLOCK_SIZE

    # Generate must-link graph via SBM
    np.random.seed(SEED)
    G, _ = SBM(block_size=n, k=k, p=P, q=Q)
    W_G  = nx.to_scipy_sparse_array(G, format='csr').astype(float)
    true_labels = np.array([G.nodes[i]['block'] for i in G.nodes()])

    # Build cannot-link graph: 5 neighbours from OTHER ground-truth clusters
    W_H = build_cannot_link_graph(true_labels, n_cl=N_CL, random_state=SEED)
    H_nx = nx.from_scipy_sparse_array(W_H)

    # Degrees
    deg_h_mean = float(W_H.sum()) / W_H.shape[0]

    # Eigenvalues
    eigs    = compute_eigenvalues(W_G, W_H, n_keep=k + 1)
    lk      = eigs[k - 1]
    lk1     = eigs[k]
    gap     = lk1 - lk

    # φ: empirical and theoretical
    phi_emp = compute_phi(G, H_nx, true_labels)
    phi_th  = phi_theory(k, n, Q, N_CL)

    upsilon = lk1 / max(phi_emp, 1e-12)

    upsilons.append(upsilon)
    phis_emp.append(phi_emp)
    phis_th.append(phi_th)
    lk1s.append(lk1)

    print(f"k={k:>2}  N={k*n:>5}  "
          f"deg_H={deg_h_mean:6.2f}  "
          f"φ={phi_emp:9.4f}  "
          f"φ_th={phi_th:9.4f}  "
          f"λ_k={lk:8.5f}  "
          f"λ_{{k+1}}={lk1:8.5f}  "
          f"gap={gap:8.5f}  "
          f"Υ={upsilon:9.4f}")


# ── Upsilon trend ──────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("UPSILON TREND  (5-NN cannot-link H)")
print("=" * 70)
print(f"{'k':>4}  {'Υ':>10}  {'φ_emp':>10}  {'direction':>12}")
print("-" * 70)
for i, k in enumerate(K_RANGE):
    if i == 0:
        direction = "—"
    else:
        delta     = upsilons[i] - upsilons[i - 1]
        direction = f"↑ +{delta:.4f}" if delta > 0 else f"↓  {delta:.4f}"
    print(f"k={k:>2}  Υ={upsilons[i]:>9.4f}  φ={phis_emp[i]:>9.4f}  {direction:>12}")

print()
print(f"  min Υ = {min(upsilons):.4f}  at k={list(K_RANGE)[int(np.argmin(upsilons))]}")
print(f"  max Υ = {max(upsilons):.4f}  at k={list(K_RANGE)[int(np.argmax(upsilons))]}")


# ── Power-law fit ──────────────────────────────────────────────────────────────
ks        = np.array(list(K_RANGE), dtype=float)
ups       = np.array(upsilons)
phi_e     = np.array(phis_emp)
phi_t     = np.array(phis_th)

alpha_up, log_c_up = np.polyfit(np.log(ks), np.log(ups),   1)
alpha_ph, log_c_ph = np.polyfit(np.log(ks), np.log(phi_e), 1)
alpha_l1, log_c_l1 = np.polyfit(np.log(ks), np.log(np.array(lk1s)), 1)

print()
print("=" * 70)
print("POWER-LAW FITS   f ≈ C · k^alpha  (fit over k=2..20)")
print("=" * 70)
print(f"  φ (empirical)  :  C={np.exp(log_c_ph):.4f}  alpha={alpha_ph:.3f}  "
      f"(theory predicts alpha ≈ 1)")
print(f"  λ_{{k+1}}        :  C={np.exp(log_c_l1):.4f}  alpha={alpha_l1:.3f}")
print(f"  Υ = λ/φ        :  C={np.exp(log_c_up):.4f}  alpha={alpha_up:.3f}  "
      f"(= alpha_lambda - alpha_phi)")

print()
print("=" * 70)
print("COMPARISON: SBM-H  vs.  5-NN-H")
print("=" * 70)
print(f"  {'quantity':<25}  {'SBM-H':>12}  {'5-NN-H':>12}")
print(f"  {'-'*25}  {'-'*12}  {'-'*12}")
print(f"  {'φ growth (alpha)':25}  {'≈ 0  (const)':>12}  {alpha_ph:>11.3f}")
print(f"  {'Υ growth (alpha)':25}  {'≈ 0  (bnd)':>12}  {alpha_up:>11.3f}")
print(f"  {'Υ bounded?':25}  {'YES (by p/q)':>12}  {'NO' if alpha_up > 0.1 else 'YES':>12}")
print()
print("  Interpretation:")
if alpha_up > 0.5:
    print(f"  Υ grows as k^{alpha_up:.2f} with 5-NN cannot-link — φ grows slower than λ_{{k+1}}.")
elif alpha_up < -0.1:
    print(f"  Υ DECREASES as k^{alpha_up:.2f} — φ grows faster than λ_{{k+1}}.")
else:
    print(f"  Υ is approximately flat (alpha={alpha_up:.3f}) — φ and λ_{{k+1}} grow at similar rates.")
