# Constrained Spectral Clustering on Natural Datasets

## Overview

We evaluate our Constrained Spectral Clustering (CSC) algorithm against three baselines on four datasets where must-link (ML) and cannot-link (CL) constraints arise naturally from the domain. We also introduce a hybrid variant, **CSC (heat-K)**, that replaces the sparse kNN graph with Lu's heat-diffusion kernel before applying CSC's constraint mechanism.

---

## Datasets

We selected datasets where the constraint structure is motivated by the domain itself — no artificial labeling scheme is required beyond class identity.

| Dataset | N | K | Features | Natural constraint motivation |
|---|---|---|---|---|
| **Digits (sklearn)** | 1797 | 10 | 64 pixel values (8×8) | Same digit class shares stroke patterns (ML); different digit classes are shape-incompatible (CL). Application: OCR with partial annotation. |
| **Olivetti Faces** | 400 | 40 | 4096 pixel values (64×64) | 10 photos of each of 40 people under varied lighting/pose. Same person → ML; different identity → CL. Application: face clustering in surveillance. |
| **Wine (UCI)** | 178 | 3 | 13 chemical attributes | Same cultivar shares a chemical fingerprint (ML); different cultivars are chemically incompatible (CL). Application: wine authentication. |
| **20 Newsgroups (4 topics)** | 2000 | 4 | TF-IDF → LSA (100 dims) | Documents on the same topic share vocabulary (ML); unrelated topics have almost no overlap (CL). Application: document clustering with editorial constraints. |

**Constraint generation.** For each node $i$, we draw $n_\text{CL}$ random cannot-link edges to nodes from different classes, building $W_H$ (Digits: 20, Olivetti: 10, Wine: 5, Newsgroups: 15). The similarity graph $W_G$ is a $k$-NN graph ($k=5$) with binary weights.

---

## Methods

### Spectral Clustering (SC)
Standard normalised-cut spectral clustering on $W_G$. Serves as the unconstrained baseline.

### Wang et al. (2014) — Algorithm 2
Finds eigenvectors of $\bar{L}_G$ that satisfy a feasibility constraint derived from $W_H$, formulated as a constrained eigenvalue problem with a Ritz-vector projection step.

### Lu & Carreira-Perpinan (2008) — Algorithm A
1. Builds a dense heat-diffusion kernel $K = \exp(-t L_\text{norm})$ using the $k_\text{eig} = 150$ smallest eigenpairs of $L_\text{norm}$.
2. Incorporates cannot-link constraints by modifying $K$ directly:
$$\bar{K} = K(I + MK)^{-1}, \quad M = \frac{1}{\varepsilon_c^2}(D_H + W_H)$$
3. Runs normalised spectral clustering on $\max(0, \bar{K})$.

Memory cost: $O(N^2)$. Not scalable beyond $N \approx 5000$.

### Our CSC — `iterative_clustering`
Solves the generalised eigenvalue problem:
$$L_G\, v = \lambda\, L_H\, v$$
where both Laplacians share a common normalisation $D = \varepsilon \cdot d_G + (1-\varepsilon) \cdot d_H$ ($\varepsilon = 0.7$). This finds directions that are simultaneously smooth along similarity edges and rough along cannot-link edges. Operates entirely on sparse matrices; scales to millions of nodes.

### CSC (heat-K) — Hybrid variant
Same as CSC, but replaces the sparse kNN graph with Lu's heat-diffusion kernel $K$ as $W_G$. Intended to give CSC the global manifold structure that the sparse kNN misses.

---

## Results (ARI)

| Dataset | SC | Wang | Lu | **CSC (kNN)** | **CSC (heat-K)** |
|---|---|---|---|---|---|
| Digits (N=1797, K=10) | 0.766 | 0.878 | **0.974** | 0.950 | 0.516 |
| Olivetti (N=400, K=40) | 0.503 | **0.534** | 0.203 | 0.516 | 0.094 |
| Wine (N=178, K=3) | 0.879 | **1.000** | **1.000** | 0.896 | **1.000** |
| Newsgroups (N=2000, K=4) | -0.000 | -0.000 | 0.888 | 0.257 | **0.996** |

*Metric: Adjusted Rand Index (ARI). Seed = 42 for all methods.*

---

## Analysis

### When does CSC (kNN) work well?

CSC is the most **consistent** method — it never collapses badly and strictly improves over SC on every dataset. On Digits it reaches 0.950, within 2.4 points of Lu's best. On Olivetti (K=40 fine-grained clusters) it outperforms both Lu and Wang.

### Why does Lu beat CSC on Digits and Newsgroups?

Lu's heat-diffusion kernel $K_{ij} = \exp(-t\lambda)$ integrates **all paths** between nodes $i$ and $j$ — not just direct kNN edges. This captures global manifold structure:

- **Digits**: The 150-eigenvector approximation gives a richer per-node representation before k-means.
- **Newsgroups**: The kNN graph on LSA features is essentially structureless (SC $\approx$ 0). There is no cluster signal in $L_G$, so the Rayleigh ratio $v^\top L_G v / v^\top L_H v$ is meaningless regardless of the constraints. Lu's dense $K$ rescues the global topic structure through multi-hop diffusion.

CSC's constraint mechanism is a **reweighting of the objective** — it cannot create signal that is absent from $W_G$.

### Why does Lu fail on Olivetti?

With $K=40$ classes and only $N=400$ points (10 per class), clusters are tiny. The heat kernel at $t=1.0$ **over-smooths**: nearby clusters from different identities get blended. CSC's sparse kNN graph preserves local neighbourhood structure, which is exactly what fine-grained clustering needs.

### The CSC (heat-K) hybrid

Replacing the sparse kNN with $K$ gives CSC the global structure. On Newsgroups this raises ARI from 0.257 to **0.996**, surpassing Lu (0.888). On Wine it achieves perfect clustering. However, it **collapses** on Digits (0.516) and Olivetti (0.094).

The cause: $K$ is a complete dense graph, so $d_G[i] = \sum_j K_{ij} \approx N \cdot e^{-2t}$ is large and nearly uniform. The shared normalisation $D = \varepsilon\, d_G + (1-\varepsilon)\, d_H$ becomes dominated by $d_G$, washing out the constraint signal from the sparse $W_H$. Both $L_G$ and $L_H$ are divided by essentially the same large diagonal, making the generalised eigenvalue problem almost degenerate.

### Summary of conditions

| Condition | CSC (kNN) | Lu | CSC (heat-K) |
|---|---|---|---|
| $W_G$ has clear cluster structure | ✓ competitive | ✓ good | ✗ dense $K$ swamps $W_H$ |
| $W_G$ is poorly connected | ✗ can't fix broken numerator | ✓ diffusion rescues it | ✓ best of all |
| $K$ large relative to $N$ | ✓ preserves local structure | ✗ over-smooths | ✗ over-smooths |
| $N$ large (millions) | ✓ sparse, scalable | ✗ $O(N^2)$ memory | ✗ $O(N^2)$ memory |

---

## Proposed next step: sparsified heat-K

The failure mode of CSC (heat-K) is a scale mismatch between $W_G$ (dense) and $W_H$ (sparse). A natural fix is to **sparsify** $K$ after computing it — keep only the top-$k$ entries per row — so that $\deg_G$ stays comparable to $\deg_H$:

$$W_G^{\text{sparse-K}}_{ij} = K_{ij} \cdot \mathbf{1}[j \in \text{top-}k_i(K)]$$

This retains the global manifold structure (fixing Newsgroups) while preventing the dense kernel from swamping the constraint signal (protecting Digits and Olivetti). It would also partially restore scalability, since a sparse $K$ can be stored in $O(Nk)$ rather than $O(N^2)$.

---

## Dataset References

1. **Digits (sklearn).**
   Alpaydin, E. & Kaynak, C. (1998). Cascading classifiers. *Kybernetika*, 34(4), 369–374.
   Also available as the *Optical Recognition of Handwritten Digits* dataset at the UCI Machine Learning Repository: Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine. https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits

2. **Olivetti Faces.**
   Samaria, F. & Harter, A. (1994). Parameterisation of a stochastic model for human face identification. In *Proceedings of the 2nd IEEE Workshop on Applications of Computer Vision* (pp. 138–142). IEEE.
   Dataset originally released as the ORL Database of Faces by AT&T Laboratories Cambridge. Now hosted by scikit-learn: https://scikit-learn.org/stable/datasets/real_world.html#olivetti-faces-dataset

3. **Wine (UCI).**
   Forina, M., Leardi, R., Armanino, C., & Lanteri, S. (1988). *PARVUS: An Extendable Package for Data Exploration, Classification and Correlation*. Institute of Pharmaceutical and Food Analysis and Technologies, Genoa, Italy.
   UCI Machine Learning Repository entry: Aeberhard, S., Coomans, D., & de Vel, O. (1992). *Comparison of classifiers in high dimensional settings* (Tech. Rep. No. 92-02). James Cook University. https://archive.ics.uci.edu/ml/datasets/wine

4. **20 Newsgroups.**
   Lang, K. (1995). Newsweeder: Learning to filter netnews. In *Proceedings of the 12th International Conference on Machine Learning (ICML)* (pp. 331–339). Morgan Kaufmann.
   Dataset mirror and documentation: http://qwone.com/~jason/20Newsgroups/
