"""
Wrong-edge analysis on SBM.

For each vertex, what fraction of must-link edges connect to a DIFFERENT block
(wrong must-link), and what fraction of cannot-link edges connect to the SAME
block (wrong cannot-link)?

G (must-link)   : SBM intra-prob = p, inter-prob = q
H (cannot-link) : SBM intra-prob = q, inter-prob = p  (inverted)

Theory (law of large numbers, n -> inf)
-------
  E[deg_G_within]  = (n-1)*p  ≈  n*p
  E[deg_G_between] = (k-1)*n*q
  wrong_G %  =  (k-1)*n*q / [(n-1)*p + (k-1)*n*q]

  E[deg_H_within]  = (n-1)*q  ≈  n*q
  E[deg_H_between] = (k-1)*n*p
  wrong_H %  =  (n-1)*q / [(n-1)*q + (k-1)*n*p]
"""

import sys
sys.path.insert(0, '/home/songs/ConstrainedSpectralClustering')

import numpy as np
import networkx as nx
from utils import SBM

P          = 0.2
Q          = 0.15
K          = 10
BLOCK_SIZE = 1000
SEED       = 42


# ── Theoretical predictions ────────────────────────────────────────────────────
n = BLOCK_SIZE

deg_G_within  = (n - 1) * P
deg_G_between = (K - 1) * n * Q
deg_G_total   = deg_G_within + deg_G_between
wrong_G_pct   = 100 * deg_G_between / deg_G_total

deg_H_within  = (n - 1) * Q
deg_H_between = (K - 1) * n * P
deg_H_total   = deg_H_within + deg_H_between
wrong_H_pct   = 100 * deg_H_within / deg_H_total

print("=" * 65)
print(f"SBM:  p={P}  q={Q}  k={K}  block_size={BLOCK_SIZE}")
print("=" * 65)
print()
print("THEORETICAL  (expected edges per vertex)")
print("-" * 65)
print(f"  Must-link graph G  (intra=p={P}, inter=q={Q}):")
print(f"    within-block edges (correct) : {deg_G_within:8.1f}")
print(f"    between-block edges (WRONG)  : {deg_G_between:8.1f}")
print(f"    total                        : {deg_G_total:8.1f}")
print(f"    → wrong must-link  = {deg_G_between:.1f} / {deg_G_total:.1f} = {wrong_G_pct:.2f}%")
print()
print(f"  Cannot-link graph H  (intra=q={Q}, inter=p={P}):")
print(f"    within-block edges (WRONG)   : {deg_H_within:8.1f}")
print(f"    between-block edges (correct): {deg_H_between:8.1f}")
print(f"    total                        : {deg_H_total:8.1f}")
print(f"    → wrong cannot-link = {deg_H_within:.1f} / {deg_H_total:.1f} = {wrong_H_pct:.2f}%")


# ── Empirical verification ─────────────────────────────────────────────────────
print()
print("EMPIRICAL  (single SBM sample, seed={})".format(SEED))
print("-" * 65)

np.random.seed(SEED)
G, H = SBM(block_size=BLOCK_SIZE, k=K, p=P, q=Q)
true_labels = np.array([G.nodes[i]['block'] for i in G.nodes()])

# Count per-vertex wrong edges in G
total_G_edges   = 0
wrong_G_edges   = 0
total_H_edges   = 0
wrong_H_edges   = 0

for u, v in G.edges():
    total_G_edges += 1
    if true_labels[u] != true_labels[v]:
        wrong_G_edges += 1

for u, v in H.edges():
    total_H_edges += 1
    if true_labels[u] == true_labels[v]:
        wrong_H_edges += 1

emp_wrong_G_pct = 100 * wrong_G_edges / total_G_edges
emp_wrong_H_pct = 100 * wrong_H_edges / total_H_edges

# Per-vertex averages
N_nodes = K * BLOCK_SIZE
print(f"  Must-link graph G:")
print(f"    total edges             : {total_G_edges:,}")
print(f"    wrong (between-block)   : {wrong_G_edges:,}  ({emp_wrong_G_pct:.2f}%)")
print(f"    avg wrong edges/vertex  : {2*wrong_G_edges/N_nodes:.1f}  "
      f"(theory: {deg_G_between:.1f})")
print()
print(f"  Cannot-link graph H:")
print(f"    total edges             : {total_H_edges:,}")
print(f"    wrong (within-block)    : {wrong_H_edges:,}  ({emp_wrong_H_pct:.2f}%)")
print(f"    avg wrong edges/vertex  : {2*wrong_H_edges/N_nodes:.1f}  "
      f"(theory: {deg_H_within:.1f})")

print()
print("=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"  {'':30}  {'Theory':>10}  {'Empirical':>10}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}")
print(f"  {'Wrong must-link edges (%)':30}  {wrong_G_pct:>9.2f}%  {emp_wrong_G_pct:>9.2f}%")
print(f"  {'Wrong cannot-link edges (%)':30}  {wrong_H_pct:>9.2f}%  {emp_wrong_H_pct:>9.2f}%")
print()
print(f"  Interpretation:")
print(f"  With p={P} and q={Q} close together and k={K} large blocks,")
print(f"  {wrong_G_pct:.1f}% of must-link edges are WRONG (cross-block).")
print(f"  Only {wrong_H_pct:.1f}% of cannot-link edges are wrong (within-block).")
print(f"  The must-link graph is severely contaminated at this (p,q,k) setting.")
