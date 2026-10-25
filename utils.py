
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from collections import defaultdict

from scipy.sparse import csr_matrix, diags, eye
from scipy.sparse.linalg import eigsh, lobpcg
import networkx as nx


def SBM(block_size, k, p, q):
    """ Generate a Stochastic Block Model graph.
    
    Args:
        block_size: Number of nodes in each block.
        k: Number of blocks.
        p: Probability of edges within blocks.
        q: Probability of edges between blocks.
        
    Returns:
        G: A NetworkX graph generated from the SBM.
        G: A NetworkX graph generated from the SBM with inverted probabilities.
    """

    ## --- Graph configuration ---
    block_size = block_size     # Block sizes: number of nodes in each community
    num_blocks = k      # Number of communities
    sizes = [block_size] * num_blocks  # The size vector of blocks

    ## --- Affinity matrix ---
    Affinity_G = np.full((num_blocks, num_blocks), q)  # Start with inter-community probabilities
    np.fill_diagonal(Affinity_G, p)  # Set intra-community probabilities

    Affinity_H = np.full((num_blocks, num_blocks), p)  # Start with intra-community probabilities
    np.fill_diagonal(Affinity_H, q)  # Set inter-community probabilities

    ## --- Generate SBM graphs ---
    G = nx.stochastic_block_model(sizes, Affinity_G, seed=42)   # The graph for must-link constraints
    H = nx.stochastic_block_model(sizes, Affinity_H, seed=42)   # The graph for cannot-link constraints
    
    return G, H

def load_mnist_subset(n_samples=5000, random_state=0):
    """
    Load MNIST from openml, take a subset, standardize.
    """
    mnist = fetch_openml("mnist_784", version=1, as_frame=False)
    X = mnist.data.astype(np.float32)
    y = mnist.target.astype(int)

    rng = np.random.default_rng(random_state)
    idx = rng.choice(X.shape[0], size=n_samples, replace=False)
    X = X[idx]
    y = y[idx]

    X = StandardScaler().fit_transform(X)
    return X, y


def build_knn_graph(X, k=10, use_gaussian=True, sigma=None):
    """
    Build a symmetric k-NN graph with either binary or Gaussian weights.
    Returns a CSR adjacency matrix W_G (n x n).
    """
    n_samples = X.shape[0]

    nn = NearestNeighbors(n_neighbors=k+1, metric="euclidean")
    nn.fit(X)
    dists, indices = nn.kneighbors(X)

    # neighbors for each point are indices[:, 1:] (skip self)
    rows = np.repeat(np.arange(n_samples), k)
    cols = indices[:, 1:].ravel()

    if use_gaussian:
        # distances to neighbors (flatten)
        dists_flat = dists[:, 1:].ravel()
        if sigma is None:
            # crude: use median distance as sigma
            sigma = np.median(dists_flat)
        vals = np.exp(-(dists_flat ** 2) / (2.0 * sigma ** 2))
    else:
        vals = np.ones_like(rows, dtype=float)

    W = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    # symmetrize by averaging
    W = (W + W.T)
    return W


def build_cannot_link_graph(y, n_cl=5, random_state=0):
    """
    For each node i, connect it to n_cl nodes from different classes.
    Returns a symmetric CSR adjacency matrix W_H (n x n) with weight 1.
    """
    n_samples = len(y)
    rng = np.random.default_rng(random_state)

    # map from label -> list of indices
    label_to_idx = defaultdict(list)
    for i, label in enumerate(y):
        label_to_idx[label].append(i)

    rows = []
    cols = []

    labels = np.unique(y)

    for i in range(n_samples):
        my_label = y[i]
        # indices not in my_label
        other_idx = np.concatenate(
            [np.array(label_to_idx[l]) for l in labels if l != my_label]
        )
        if other_idx.size < n_cl:
            chosen = other_idx  # fallback
        else:
            chosen = rng.choice(other_idx, size=n_cl, replace=False)

        rows.extend([i] * len(chosen))
        cols.extend(chosen.tolist())

    vals = np.ones(len(rows), dtype=float)
    W_H = csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
    W_H = (W_H + W_H.T)
    return W_H


def normalized_laplacian_sp(W, eps_deg=1e-12):
    """
    Compute symmetric normalized Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.
    """
    # n = W.shape[0]
    deg = np.array(W.sum(axis=1)).ravel()
    return normalized_laplacian_from_D_sp(W, deg, eps_deg=eps_deg)

def normalized_laplacian_from_D_sp(W, D, eps_deg=1e-12):
    # n = W.shape[0]
    # deg = np.array(W.sum(axis=1)).ravel()
    # D = diags(deg)
    # Lap = D - W
    # inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(J, eps_deg))
    # J_inv_sqrt = diags(inv_sqrt_deg)
    # L_sym = J_inv_sqrt @ Lap @ J_inv_sqrt
    # return L_sym
    n = W.shape[0]
    inv_sqrt_deg = 1.0 / np.sqrt(np.maximum(D, eps_deg))
    D_inv_sqrt = diags(inv_sqrt_deg)
    I = eye(n, format="csr")
    L_sym = I - D_inv_sqrt @ W @ D_inv_sqrt
    return L_sym

def double_normalized_laplacian_sp(W_G, W_H, eps_deg=1e-12, eps=0.5):
    """
    Compute double normalized Laplacian:
    L_G^sym = I - D_G^{-1/2} W_G D_G^{-1/2}
    L_H^sym = I - D_H^{-1/2} W_H D_H^{-1/2}
    """
    n = W_G.shape[0]

    deg_G = np.array(W_G.sum(axis=1)).ravel()
    deg_H = np.array(W_H.sum(axis=1)).ravel()
    # D = np.maximum(deg_H, deg_G)
    # eps = 0.5
    # D = np.mean([deg_H, deg_G], axis=0)
    D = deg_G * eps + deg_H * (1 - eps)
    L_G_sym = normalized_laplacian_from_D_sp(W_G, D, eps_deg=eps_deg)
    L_H_sym = normalized_laplacian_from_D_sp(W_H, D, eps_deg=eps_deg)

    return L_G_sym, L_H_sym


def constrained_spectral_clustering(L_G_sym, L_H_sym_reg, n_clusters=10, random_state=0, exact = False):
    """
    Solve L_G^sym v = lambda L_H^sym v (regularized)
    Cluster with k-means on eigenvectors
    """

    np.random.seed(0)
    n_samples = L_G_sym.shape[0]
    if exact is True:
        vals, vecs = eigsh(L_G_sym, k=n_clusters, M=L_H_sym_reg, which="SM")

    else:
        X0 = np.random.randn(n_samples, n_clusters)
        vals, vecs = lobpcg(L_G_sym, X0, B=L_H_sym_reg, largest=False, tol=1e-6, maxiter=200)

    # Use eigenvectors as embedding for k-means
    X_embed = vecs

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10,
        random_state=random_state
    )
    labels_pred = kmeans.fit_predict(X_embed)

    return labels_pred, vals, vecs

def spectral_clustering(
    L_G_sym, 
    n_clusters=10,
    random_state=0
):
    vals, vecs = eigsh(
        L_G_sym,
        k=n_clusters,
        which="SM"
    )
    X_embed = vecs

    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=10, 
        random_state=random_state
    )
    labels_pred = kmeans.fit_predict(X_embed)

    return labels_pred, vals, vecs

def iterative_clustering(W_G, W_H, k, eps = 0.9, exact = False, random_seed = 1):
    deg_g = np.array(W_G.sum(axis=1)).ravel()
    deg_h = np.array(W_H.sum(axis=1)).ravel()
    deg = eps * deg_g + (1-eps) * deg_h
    normalized_Lap_G = normalized_laplacian_from_D_sp(W_G, deg)
    normalized_Lap_H = normalized_laplacian_from_D_sp(W_H, np.maximum(deg_h, deg))
    return constrained_spectral_clustering(
        L_G_sym=normalized_Lap_G,
        L_H_sym_reg=normalized_Lap_H,
        n_clusters = k,
        random_state=random_seed,
        exact=exact
    )

def cut(G, labels):
    """ Compute the unweighted cut size of graph G given the clustering labels.
    Args:
        G: A NetworkX graph.
        labels: Cluster labels for each node.
    Returns:
        cut_size: Unweighted cut size.
    """
    # Find cut edges
    G_cut_edges = [(u, v) for u, v in G.edges() if labels[u] != labels[v]]
    # print("Cut edges:", G_cut_edges)
    # Unweighted cut size
    cut_size = len(G_cut_edges)
    return cut_size


def ratio_cut(G, H, labels):
    """ Compute the ratio cut value phi(G, H) for the clustering defined by labels.
    Args:
        G: A NetworkX graph (must-link graph).
        H: A NetworkX graph (cannot-link graph).
        labels: Cluster labels for each node.
    Returns:
        ratio_cut: The ratio cut value phi(G, H).
    """
    return cut(G, labels) / cut(H, labels)

def conductance_all_clusters(G, labels):
    # group nodes by cluster label
    clusters = defaultdict(list)
    for node, lab in zip(G.nodes(), labels):
        clusters[lab].append(node)

    results = {}

    for lab, nodes_S in clusters.items():
        S = set(nodes_S)
        T = set(G.nodes()) - S

        cut_size = nx.cut_size(G, S, T)
        vol_S = sum(dict(G.degree(S)).values())
        vol_T = sum(dict(G.degree(T)).values())

        phi = cut_size / min(vol_S, vol_T)
        results[lab] = phi

    return results

def ratiocut_all_clusters(G, H, labels):
    clusters = defaultdict(list)
    for node, lab in zip(G.nodes(), labels):
        clusters[lab].append(node)

    results = {}

    for lab, nodes_S in clusters.items():
        S = set(nodes_S)
        T = set(G.nodes()) - S
        G_cut_size = nx.cut_size(G, S, T)
        H_cut_size = nx.cut_size(H, S, T)
        phi = G_cut_size / max(1e-3, H_cut_size)
        results[lab] = phi
    return results