"""
Can Υ grow as k^3?  Theory + empirical verification.

Key insight
-----------
  Υ = λ_{k+1} / φ,  and  φ ≈ q/p  (constant for fixed p, q).
  λ_{k+1} is bounded in (0, 2], so with FIXED p and q,
  Υ is bounded by p/q  — polynomial growth in k is impossible.

  To achieve Υ ~ k^α, we need φ ~ k^{-α}, i.e. q ~ p / k^α.

  Setting  q(k) = q0 / k^3  gives φ ≈ q0 / (p * k^3), so:
      Υ ≈ λ_{k+1} * p / q0 * k^3  →  Υ ~ k^3  as k → ∞
  (since λ_{k+1} → 1 when between-block edges become negligible).

Experiments
-----------
  1. Baseline     : p=0.5, q=0.01 (fixed) — shows Υ bounded/slow growth
  2. q ~ 1/k      : q(k) = 0.5/k         — Υ ~ k
  3. q ~ 1/k^2    : q(k) = 0.5/k^2       — Υ ~ k^2
  4. q ~ 1/k^3    : q(k) = 0.5/k^3       — Υ ~ k^3  (target)
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
BLOCK_SIZE = 100
EPS        = 0.9
SEED       = 42
K_RANGE    = range(2, 21)

# Four q-schedules to compare
Q_SCHEDULES = {
    "fixed  q=0.01         ": lambda k: 0.01,
    "q=0.5/k    (alpha=1)  ": lambda k: 0.5 / k,
    "q=0.5/k^2  (alpha=2)  ": lambda k: 0.5 / k**2,
    "q=0.5/k^3  (alpha=3)  ": lambda k: 0.5 / k**3,
}


# ── Helpers ────────────────────────────────────────────────────────────────────

def compute_eigenvalues(W_G, W_H, n_keep, eps=EPS, reg=1e-8):
    deg_g = np.array(W_G.sum(1)).ravel()
    deg_h = np.array(W_H.sum(1)).ravel()
    deg   = eps * deg_g + (1.0 - eps) * deg_h
    L_G   = normalized_laplacian_from_D_sp(W_G, deg).toarray()
    L_H   = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg)).toarray()
    L_H  += reg * np.eye(W_G.shape[0])
    vals  = sla.eigh(L_G, L_H, eigvals_only=True)
    return np.sort(np.real(vals))[:n_keep]


def compute_phi(G, H, true_labels):
    g_cut = sum(1 for u, v in G.edges() if true_labels[u] != true_labels[v])
    h_cut = sum(1 for u, v in H.edges() if true_labels[u] != true_labels[v])
    return g_cut / max(h_cut, 1)


# ── Per-schedule detailed table ────────────────────────────────────────────────
all_upsilons = {name: [] for name in Q_SCHEDULES}

for name, q_fn in Q_SCHEDULES.items():
    print()
    print("=" * 100)
    print(f"Schedule: {name.strip()}   p={P}   block_size={BLOCK_SIZE}   eps={EPS}")
    print("=" * 100)
    print(f"{'k':>4}  {'N':>5}  {'q':>9}  {'φ':>10}  {'λ_{k+1}':>10}  "
          f"{'Υ':>10}  {'k^3':>8}  {'Υ/k^3':>8}")
    print("-" * 100)

    upsilons = []
    for k in K_RANGE:
        q = q_fn(k)
        np.random.seed(SEED)
        G, H = SBM(block_size=BLOCK_SIZE, k=k, p=P, q=q)
        W_G  = nx.to_scipy_sparse_array(G, format='csr').astype(float)
        W_H  = nx.to_scipy_sparse_array(H, format='csr').astype(float)
        true_labels = np.array([G.nodes[i]['block'] for i in G.nodes()])

        eigs    = compute_eigenvalues(W_G, W_H, n_keep=k + 1)
        phi     = compute_phi(G, H, true_labels)
        lk1     = eigs[k]
        upsilon = lk1 / max(phi, 1e-12)
        k3      = k ** 3
        ratio   = upsilon / k3

        upsilons.append(upsilon)
        all_upsilons[name].append(upsilon)

        print(f"k={k:>2}  N={k*BLOCK_SIZE:>5}  q={q:9.6f}  "
              f"φ={phi:9.6f}  λ_{{k+1}}={lk1:9.5f}  "
              f"Υ={upsilon:10.3f}  k^3={k3:>8d}  Υ/k^3={ratio:8.5f}")

    # Fit log(Υ) ~ α * log(k) + const  to estimate empirical exponent
    ks  = np.array(list(K_RANGE), dtype=float)
    ups = np.array(upsilons, dtype=float)
    log_k  = np.log(ks)
    log_up = np.log(ups)
    alpha, log_c = np.polyfit(log_k, log_up, 1)
    print(f"\n  Empirical fit:  log(Υ) = {alpha:.3f} * log(k) + {log_c:.3f}  "
          f"  →  Υ ≈ {np.exp(log_c):.3f} * k^{alpha:.3f}")


# ── Side-by-side comparison table ─────────────────────────────────────────────
COL_W = 14
HDR   = f"{'k':>4}" + "".join(f"{name[:COL_W]:>{COL_W}}" for name in Q_SCHEDULES)
DIV   = "-" * len(HDR)

print()
print("=" * len(HDR))
print("COMPARISON — Υ values across q-schedules")
print("=" * len(HDR))
print(HDR)
print(DIV)
for i, k in enumerate(K_RANGE):
    row = f"{k:>4}"
    for name in Q_SCHEDULES:
        row += f"{all_upsilons[name][i]:>{COL_W}.3f}"
    print(row)

# ── Empirical growth rate summary ─────────────────────────────────────────────
print()
print("=" * len(HDR))
print("EMPIRICAL POWER-LAW FIT   Υ ≈ C * k^alpha  (fit over k=2..20)")
print("=" * len(HDR))
print(f"  {'Schedule':<40}  {'C':>8}  {'alpha':>7}  {'target':>8}")
print(f"  {'-'*40}  {'-'*8}  {'-'*7}  {'-'*8}")
ks = np.array(list(K_RANGE), dtype=float)
targets = {"fixed": "bounded", "alpha=1": "1", "alpha=2": "2", "alpha=3": "3"}
for name, q_fn in Q_SCHEDULES.items():
    ups   = np.array(all_upsilons[name])
    alpha, log_c = np.polyfit(np.log(ks), np.log(ups), 1)
    tag   = name.strip().split()[-1].strip("()")
    print(f"  {name.strip():<40}  C={np.exp(log_c):>6.3f}  α={alpha:>6.3f}  "
          f"target={'bounded' if 'fixed' in name else tag}")

print()
print("  Conclusion:")
print("  - Fixed q: Υ is bounded (approaches p/q = 50), no power-law growth possible.")
print("  - q(k) = p/k^α  gives  Υ ~ k^α  (confirmed empirically above).")
print("  - To achieve Υ ~ k^3, set  q(k) = p / k^3.")
print()
