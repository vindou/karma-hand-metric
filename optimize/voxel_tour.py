"""A smooth visiting order for the reachable voxels of a KaRMA result.

Phase 1 of the metric reaches every voxel from its parent by one rolling
primitive, so the stored joint configurations are continuous along the edges
of that search tree and NOT, in general, between voxels that are merely
adjacent in space (the hand is redundant and the two may hold the sphere with
different postures). An Euler tour of the tree from the seed therefore moves
the hand one voxel at a time with continuous joints, retracing its path when it
backtracks.

`tree_and_tour(result)` returns:
  nodes : dict voxel -> dict(q, centre, bins, coverage, parent)   (Phase-1 configuration per voxel)
  tour  : list of voxels (len ~ 2 * n_voxels - 1), consecutive entries are tree-adjacent
  first_visit : list of bools, True where the tour reaches a voxel for the first time
"""
from __future__ import annotations

import numpy as np


def tree_and_tour(res):
    # parent voxel per voxel, and the parent's Phase-1 orientation bin
    parent_of: dict = {}
    phase1_bin: dict = {}
    by_voxel: dict = {}
    for (v, b), node in res.state_nodes.items():
        by_voxel.setdefault(v, {})[b] = node
        if node.parent is not None:
            pv, pb = node.parent
            parent_of[v] = pv
            phase1_bin[pv] = pb
    root = next((v for v in by_voxel if v not in parent_of), (0, 0, 0))
    cov = res.rotational_coverage_per_voxel
    nodes = {}
    # choose each voxel's Phase-1 configuration: the bin referenced by its children,
    # else (leaves) the stored node closest in joint space to the parent's configuration
    order = sorted(by_voxel, key=lambda v: 0 if v == root else 1)
    for v in order:
        bins = by_voxel[v]
        if v in phase1_bin and phase1_bin[v] in bins:
            node = bins[phase1_bin[v]]
        elif v == root:
            node = next(iter(bins.values()))
        else:
            pq = nodes[parent_of[v]]["q"] if parent_of.get(v) in nodes else None
            node = (min(bins.values(), key=lambda n: float(np.linalg.norm(np.asarray(n.q) - pq))) if pq is not None
                    else next(iter(bins.values())))
        nodes[v] = dict(q=np.asarray(node.q, float), centre=np.asarray(node.centre_world, float),
                        bins=set(res.voxel_ori_bins.get(v, ())), coverage=float(cov.get(v, 0.0)),
                        parent=parent_of.get(v))
    # second pass for leaves whose parent came later in `order`
    for v in order:
        if v != root and v in parent_of and parent_of[v] in nodes and (v not in phase1_bin or phase1_bin[v] not in by_voxel[v]):
            pq = nodes[parent_of[v]]["q"]
            node = min(by_voxel[v].values(), key=lambda n: float(np.linalg.norm(np.asarray(n.q) - pq)))
            nodes[v]["q"] = np.asarray(node.q, float)

    children: dict = {v: [] for v in nodes}
    for v, p in parent_of.items():
        if p in children:
            children[p].append(v)
    # visit children nearest-first in joint space for a calmer motion
    for p in children:
        children[p].sort(key=lambda c: float(np.linalg.norm(nodes[c]["q"] - nodes[p]["q"])))

    tour, first = [], []
    seen = set()
    stack = [(root, iter(children[root]))]
    tour.append(root); first.append(True); seen.add(root)
    while stack:
        v, it = stack[-1]
        nxt = next(it, None)
        if nxt is None:
            stack.pop()
            if stack:
                tour.append(stack[-1][0]); first.append(False)
            continue
        tour.append(nxt); first.append(nxt not in seen); seen.add(nxt)
        stack.append((nxt, iter(children[nxt])))
    return nodes, tour, first


def tour_order(res):
    """Voxels in first-visit order along the tour (one entry per voxel)."""
    nodes, tour, first = tree_and_tour(res)
    return [v for v, f in zip(tour, first) if f], nodes
