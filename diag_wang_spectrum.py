"""Diagnose Wang's drop at n_cl=20: count positive eigenvalues of Q_bar."""
import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')
import numpy as np, h5py, warnings
warnings.filterwarnings("ignore")
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh
from sklearn.datasets import fetch_openml
from utils import build_knn_graph, build_cannot_link_graph

def load_usps():
    with h5py.File("./datasets/usps/usps.h5", 'r') as hf:
        X = np.array(hf['train']['data'][:]).reshape(-1, 256).astype(np.float32)
        y = np.array(hf['train']['target'][:], dtype=int)
    return X, y

def load_pendigits():
    d = fetch_openml('pendigits', version=1, as_frame=False, parser='liac-arff')
    return d.data.astype(np.float32), d.target.astype(int)

DATASETS = [('USPS', load_usps, 5, 63), ('Pendigits', load_pendigits, 5, 42)]
N_SAMPLES, K = 5000, 10

for name, loader, k_nn, seed in DATASETS:
    print(f"\n=== {name} (k_nn={k_nn}) ===")
    X, y = loader()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=N_SAMPLES, replace=False)
    X = X[idx]; y = y[idx]
    W_G = build_knn_graph(X, k=k_nn, use_gaussian=False)
    vol_G = float(W_G.sum())
    deg = np.array(W_G.sum(axis=1)).ravel()
    isq = 1.0 / np.sqrt(np.maximum(deg, 1e-10))
    D_isq = diags(isq)

    print(f"  vol_G={vol_G:.0f}")
    print(f"  {'n_cl':>5}  {'vol_H':>8}  {'α=vol_G/vol_H':>14}  "
          f"{'top29 eigs of Q_bar':<60}  {'#pos':>5}  "
          f"{'top9 spread':>12}")
    for n_cl in [5, 10, 20, 40]:
        W_H   = build_cannot_link_graph(y, n_cl=n_cl, random_state=seed)
        vol_H = float(W_H.sum())
        alpha = vol_G / vol_H
        Q     = (W_G - alpha * W_H).astype(float)
        Q_bar = D_isq @ Q @ D_isq

        n_q = min(K - 1 + 20, N_SAMPLES - 2)
        qv, _ = eigsh(Q_bar, k=n_q, which='LM')
        qv_sorted = np.sort(qv)[::-1]   # descending
        n_pos = int((qv > 0).sum())
        top9_min = qv_sorted[:K - 1].min()
        top9_max = qv_sorted[:K - 1].max()
        eigs_short = ", ".join(f"{x:+.3f}" for x in qv_sorted[:6])
        print(f"  {n_cl:>5}  {vol_H:>8.0f}  {alpha:>14.4f}  "
              f"{eigs_short:<60}  {n_pos:>5d}  "
              f"[{top9_min:+.3f},{top9_max:+.3f}]")
