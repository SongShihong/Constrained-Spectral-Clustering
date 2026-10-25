"""
Diagnostic for Jia et al. DP-GLPCA on MNIST (5000-sample, n_cl=20).

Sweeps (alpha, n_iter, beta) with Frobenius auto-scaling of D̃ so that
beta has comparable magnitude across configurations. Builds W_G and W_H
once, and reuses the propagated D̃ across all beta values.

Reports:
  - SC baseline (no constraints)
  - the full sweep grid
  - the best Jia config and its margin over SC
"""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import (
    build_knn_graph, build_cannot_link_graph,
    spectral_clustering, normalized_laplacian_from_D_sp,
)
from compare_multi_ncl20_with_jia import (
    load_mnist, subsample, jia_propagate, jia_solve,
    N_CL, N_MAX, RANDOM_STATE,
)

# ── Load + graphs (once) ─────────────────────────────────────────────────────
print("→ Loading MNIST...")
X, y = load_mnist()
X, y = subsample(X, y, N_MAX)
N, K = X.shape[0], 10
print(f"  N={N},  K={K}")

print("→ Building W_G (k=5)...")
W_G = build_knn_graph(X, k=5, use_gaussian=False)
print(f"  W_G avg-deg = {W_G.sum()/N:.2f}")

print(f"→ Building W_H (n_cl={N_CL})...")
W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
print(f"  W_H avg-deg = {W_H.sum()/N:.2f}")

# ── SC baseline ──────────────────────────────────────────────────────────────
print("\n→ SC baseline (no constraints)...")
deg_g = np.array(W_G.sum(axis=1)).ravel()
L_norm = normalized_laplacian_from_D_sp(W_G, deg_g)
sc_lab, _, _ = spectral_clustering(L_norm, n_clusters=K, random_state=RANDOM_STATE)
SC_ARI = adjusted_rand_score(y, sc_lab)
SC_NMI = normalized_mutual_info_score(y, sc_lab)
print(f"  SC: ARI={SC_ARI:.4f}  NMI={SC_NMI:.4f}")

# ── Jia sweep ────────────────────────────────────────────────────────────────
ALPHAS  = [0.1, 0.3, 0.5, 0.7]
N_ITERS = [3, 5, 10, 20]
BETAS   = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0]

print(f"\n→ Jia sweep — auto-scaled D̃ (‖D̃‖_F=‖L‖_F), 4×4×7 = "
      f"{len(ALPHAS)*len(N_ITERS)*len(BETAS)} configs")
print(f"\n  {'alpha':>5}  {'n_it':>4}  {'beta':>5}  {'ARI':>7}  {'NMI':>7}  {'Δ vs SC':>8}")
print("  " + "-" * 50)

best = (-np.inf, None, None)  # (ari, (a,n,b), nmi)
t_start = time.time()

for alpha in ALPHAS:
    for n_it in N_ITERS:
        t0 = time.time()
        L_sp, D_tilde = jia_propagate(W_G, W_H, alpha=alpha, n_prop_iters=n_it,
                                       auto_scale=True)
        prep = time.time() - t0
        for beta in BETAS:
            try:
                lab = jia_solve(L_sp, D_tilde, beta=beta, n_clusters=K,
                                random_state=RANDOM_STATE)
                ari = adjusted_rand_score(y, lab)
                nmi = normalized_mutual_info_score(y, lab)
            except Exception as e:
                ari = nmi = float('nan')
            tag = ""
            if not np.isnan(ari) and ari > best[0]:
                best = (ari, (alpha, n_it, beta), nmi)
                tag = "  <-- best"
            delta = ari - SC_ARI if not np.isnan(ari) else float('nan')
            print(f"  {alpha:>5.2f}  {n_it:>4d}  {beta:>5.2f}  "
                  f"{ari:>7.4f}  {nmi:>7.4f}  {delta:>+8.4f}{tag}")
        print(f"    (alpha={alpha}, n_it={n_it}: "
              f"prep {prep:.1f}s, total {time.time()-t0:.1f}s)")

print(f"\n  total sweep time = {time.time()-t_start:.1f}s")
print(f"\n  BEST Jia: ARI={best[0]:.4f}  NMI={best[2]:.4f}  "
      f"@ (alpha, n_it, beta) = {best[1]}")
print(f"  SC baseline: ARI={SC_ARI:.4f}  NMI={SC_NMI:.4f}")
print(f"  Δ ARI = {best[0]-SC_ARI:+.4f}")
