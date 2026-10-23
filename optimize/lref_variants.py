"""Alternative definitions of KaRMA's normalising length L_ref (arm F).

The shipped definition (karma.lref) is

    L_ref = mxPr + mF

with mxPr the max pairwise distance between ALL finger knuckles and mF the
MEDIAN wire length over ALL finger chains, plus a rule that advances a
finger's knuckle distally when its proximal joint has a small range or its
wire is an outlier. optimize/exploits.py shows both parts can be gamed by
chains that never touch the object.

Variants here:
  pinch      L_ref from the two pinch fingers only: thumb-index knuckle
             distance + mean of their two wires. No knuckle advancement.
  noadvance  the shipped definition with the knuckle-advancement rule removed
             (isolates that rule's effect).
  shipped    the original, for reference.

`patch(variant)` monkeypatches karma.config.compute_lref so that load_config()
uses the variant. L_ref is baked into the frozen MetricConfig that the worker
processes receive, so patching the parent process is sufficient.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from karma import lref as L
from karma import config as karma_config


def _pinch_chain(robot: dict, joints_db, c2p, p2c, finger: str):
    base = robot["base_link"]
    links_cfg = robot.get("links", {}) or {}
    first_active = robot["fingers"][finger]["active_joints"][0]
    seed_link = robot["seed"][f"{finger}_link"]
    # leaf below the seed link: the longest chain (mirrors karma.lref's choice)
    leaves = L._all_leaves(seed_link, p2c) or [seed_link]
    best = None
    for leaf in leaves:
        chain = L._chain_links(base, leaf, c2p)
        wire = sum(float(np.linalg.norm(joints_db[jn]["xyz"])) for jn, _ in chain)
        if best is None or wire > best[0]:
            best = (wire, leaf, chain)
    _, leaf, chain = best
    cfg = links_cfg.get(leaf)
    tip_off = None
    if cfg and isinstance(cfg, dict) and "tip_length_m" in cfg:
        tip_off = np.array([float(cfg["tip_length_m"]), 0.0, 0.0])
    elif seed_link != leaf:
        pass
    else:
        tip_off = None
    rev_idx = [jn for jn, _ in chain].index(first_active)
    kpos, wire, _, _, _ = L._analyze_chain(chain, joints_db, tip_off, start_rev_idx=rev_idx)
    return kpos, wire


def lref_pinch_breakdown(robot_yaml_path) -> dict:
    robot_yaml_path = Path(robot_yaml_path)
    robot = yaml.safe_load(robot_yaml_path.read_text())
    joints_db, c2p, p2c = L._parse_urdf(robot_yaml_path.parent / robot["urdf_path"])
    kt, wt = _pinch_chain(robot, joints_db, c2p, p2c, "thumb")
    ki, wi = _pinch_chain(robot, joints_db, c2p, p2c, "index")
    mxPr = float(np.linalg.norm(kt - ki))
    mF = 0.5 * (wt + wi)
    return {"L_ref_m": mxPr + mF, "mxPr_m": mxPr, "mF_m": mF, "n_fingers": 2, "finger_wires_m": [wt, wi]}


def lref_noadvance_breakdown(robot_yaml_path) -> dict:
    orig = L._advance_knuckles
    L._advance_knuckles = lambda fingers, joints_db: None
    try:
        return L.compute_lref_breakdown(robot_yaml_path)
    finally:
        L._advance_knuckles = orig


VARIANTS = {
    "shipped": L.compute_lref_breakdown,
    "pinch": lref_pinch_breakdown,
    "noadvance": lref_noadvance_breakdown,
}


def patch(variant: str) -> None:
    fn = VARIANTS[variant]
    karma_config.compute_lref = lambda p: fn(p)["L_ref_m"]


if __name__ == "__main__":
    import sys
    repo = Path(__file__).resolve().parent.parent
    robots = sorted(p for p in (repo / "robots").glob("robot_*.yaml") if "template" not in p.name)
    print(f"{'hand':<22s} {'shipped':>8s} {'noadvance':>10s} {'pinch':>8s}   pinch mxPr / mF")
    for p in robots:
        s = L.compute_lref_breakdown(p)["L_ref_m"] * 1e3
        n = lref_noadvance_breakdown(p)["L_ref_m"] * 1e3
        b = lref_pinch_breakdown(p)
        print(f"{p.stem[6:]:<22s} {s:8.1f} {n:10.1f} {b['L_ref_m']*1e3:8.1f}   {b['mxPr_m']*1e3:6.1f} / {b['mF_m']*1e3:6.1f}")
