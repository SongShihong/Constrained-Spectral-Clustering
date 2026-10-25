"""Sweep eps values for our Constrained SC on USPS subset (N=5000)."""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import h5py
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from utils import (
    build_knn_graph, build_cannot_link_graph, iterative_clustering,
)

with h5py.File("./datasets/usps/usps.h5", 'r') as hf:
    train = hf.get('train')
    X_full = train.get('data')[:]
    y_full = train.get('target')[:]
X_full = np.reshape(X_full, (len(X_full), -1)).astype(np.float32)
y_full = np.array(y_full, dtype=int)

N_SAMPLES    = 5000
K_NN         = 5
N_CLUSTERS   = 10
RANDOM_STATE = 63

# Per-n_cl finer grids based on coarse sweep
GRIDS = {
     5: [0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.82, 0.85, 0.88, 0.90, 0.92, 0.95],
    10: [0.65, 0.70, 0.75, 0.80, 0.82, 0.85, 0.87, 0.88, 0.89, 0.90, 0.91, 0.92, 0.93, 0.94],
    20: [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30],
}

rng = np.random.default_rng(RANDOM_STATE)
idx = rng.choice(len(X_full), size=N_SAMPLES, replace=False)
X = X_full[idx]; y = y_full[idx]

W_G = build_knn_graph(X, k=K_NN, use_gaussian=False)
print(f"USPS subset N={N_SAMPLES}, K={N_CLUSTERS}, k_nn={K_NN}, seed={RANDOM_STATE}")
print(f"W_G avg-deg={W_G.sum()/N_SAMPLES:.1f}\n")

results = {}
for N_CL, grid in GRIDS.items():
    print(f"\n=== n_cl = {N_CL} ===")
    W_H = build_cannot_link_graph(y, n_cl=N_CL, random_state=RANDOM_STATE)
    print(f"W_H avg-deg={W_H.sum()/N_SAMPLES:.1f}")
    print(f"{'eps':>6}  {'ARI':>8}  {'NMI':>8}")
    results[N_CL] = {}
    for eps in grid:
        try:
            labels, _, _ = iterative_clustering(
                W_G, W_H, k=N_CLUSTERS, exact=False,
                eps=eps, random_seed=RANDOM_STATE,
            )
            ari = adjusted_rand_score(y, labels)
            nmi = normalized_mutual_info_score(y, labels)
        except Exception as e:
            ari, nmi = float('nan'), float('nan')
            print(f"  eps={eps}: FAILED {e}")
            continue
        results[N_CL][eps] = (ari, nmi)
        print(f"  {eps:>4.2f}  {ari:>8.4f}  {nmi:>8.4f}")

print("\n\n=== Best eps per n_cl (fine grid) ===")
for N_CL in GRIDS:
    best_eps = max(results[N_CL], key=lambda e: results[N_CL][e][0])
    a, n = results[N_CL][best_eps]
    print(f"  n_cl={N_CL}: best eps={best_eps}  ARI={a:.4f}  NMI={n:.4f}")
