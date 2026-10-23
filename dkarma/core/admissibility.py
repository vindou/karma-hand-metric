"""Admissibility of a design (SPEC.md §8.1 D18, owner's decision §16 item 10, 2026-10-13).

A design is admissible only if no two modelled fingers can occupy the same space. The
configuration-independent part is the base separation: any two finger bases at least
``2 r_l + eps_col`` apart (24.1 mm for 11.8 mm capsules and the 0.5 mm clearance). The reach
bound (250 mm per finger, the sum of a finger's link lengths including the tip) is the global
hand envelope. Configuration-dependent clearance between all fingers is the self-collision gate
of every evaluator (``dkarma.core.contacts.self_clearance``; ``karma_jax`` with
``build_flat_model(..., self_collision_all=True)``).

Smooth penalties for the optimizer live in :mod:`dkarma.optimization.objectives`; this module is
the host-side hard check every optimizer applies to a candidate before it is evaluated or
returned (CLAUDE.md rule 11 b).
"""
from __future__ import annotations

import numpy as np

from .morphology import HandParams, DEFAULT_CAPSULE_RADIUS_M
from .topology import Topology

EPS_COL_M = 0.0005
MIN_BASE_SEPARATION_M = 2.0 * DEFAULT_CAPSULE_RADIUS_M + EPS_COL_M      # 0.0241 m
REACH_MAX_M = 0.25


def base_separations(params: HandParams, topo: Topology) -> dict:
    """Pairwise distances (m) between finger bases, keyed by finger-name pairs."""
    b = np.asarray(params.base_xyz, float)
    names = topo.fingers
    return {(names[i], names[j]): float(np.linalg.norm(b[i] - b[j])) for i in range(len(names)) for j in range(i + 1, len(names))}


def finger_reaches(params: HandParams, topo: Topology) -> dict:
    """Sum of link lengths (tip included) per finger (m)."""
    ll = np.asarray(params.link_len, float)
    return {name: float(ll[f, : topo.n_joints[f]].sum()) for f, name in enumerate(topo.fingers)}


def admissibility(params: HandParams, topo: Topology, min_sep_m: float = MIN_BASE_SEPARATION_M, reach_max_m: float = REACH_MAX_M) -> dict:
    """Hard D18 check of a design: ``dict(admissible, min_base_separation_m, max_reach_m,
    base_separations, reaches, violations)``."""
    seps = base_separations(params, topo)
    reaches = finger_reaches(params, topo)
    min_sep = min(seps.values()) if seps else float("inf")
    max_reach = max(reaches.values())
    viol = []
    if min_sep < min_sep_m:
        viol.append(f"base separation {min_sep * 1e3:.1f} mm < {min_sep_m * 1e3:.1f} mm")
    if max_reach > reach_max_m:
        viol.append(f"reach {max_reach * 1e3:.1f} mm > {reach_max_m * 1e3:.0f} mm")
    return dict(admissible=not viol, min_base_separation_m=min_sep, max_reach_m=max_reach,
                base_separations={f"{a}-{b}": v for (a, b), v in seps.items()}, reaches=reaches, violations=viol,
                min_sep_required_m=min_sep_m, reach_max_m=reach_max_m)


__all__ = ["EPS_COL_M", "MIN_BASE_SEPARATION_M", "REACH_MAX_M", "base_separations", "finger_reaches", "admissibility"]
