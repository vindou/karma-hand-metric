"""Render the best design of an optimization run (or any design JSON) to a PNG,
without waiting for the official metric.

    python optimize/render_best.py --run optimize/runs/T                 # -> optimize/runs/T/best_design/best.png
    python optimize/render_best.py --design some_hand.json --out x.png
    python optimize/render_best.py --run optimize/runs/T --no-cloud      # instant: hand only, no BFS

The reachable-voxel cloud comes from the fast proxy's best seed (translation
BFS only, so voxels are coloured by distance from the seed, not by orientation
coverage). Use evaluate_full.py + render_hand.py for the official picture.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from karma.config import load_config                                               # noqa: E402
from karma.robot import load_robot, active_position_indices, active_velocity_indices, capsule_endpoints_world, update_kinematics  # noqa: E402
from karma.seed_selection import _polish_seeds, _compute_trial_frame, _run_trial_bfs   # noqa: E402
from karma.orientation import compute_motion_primitives_frame                      # noqa: E402
from optimize.hand_family import HandSpec, HandFamily, FamilyConfig                 # noqa: E402
from optimize.fast_eval import generate_seeds, FastEvalConfig, METRIC_CFG            # noqa: E402
from optimize.evaluate_full import load_design                                      # noqa: E402
from optimize.render_hand import draw_hand, set_equal                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="run directory containing best.json")
    ap.add_argument("--design", help="HandSpec/best/genotype json (alternative to --run)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--budget", type=int, default=2500, help="BFS voxel cap for the cloud")
    ap.add_argument("--no-cloud", action="store_true")
    a = ap.parse_args()
    logging.disable(logging.WARNING)

    if a.run:
        best = json.loads((Path(a.run) / "best.json").read_text())
        meta = json.loads((Path(a.run) / "meta.json").read_text())
        fam = HandFamily(FamilyConfig(**meta["family"]))
        hand = fam.decode(np.array(best["x"]))
        label = f"{Path(a.run).name}: best of generation {best['gen']} (proxy obj {best['objective']:.4f})"
        out = a.out or str(Path(a.run) / "best_design" / "best.png")
    else:
        hand = load_design(a.design)
        label = Path(a.design).stem
        out = a.out or str(Path(a.design).with_suffix(".png"))
    print(hand.describe())

    tmp = Path(tempfile.mkdtemp(prefix="karma_render_"))
    _, yaml_path = hand.write(tmp)
    cfg = load_config(METRIC_CFG, yaml_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]
    q0 = np.zeros(ctx.model.nq)

    seed = None; disc = {}
    t0 = time.perf_counter()
    seeds, _ = generate_seeds(cfg, ctx, FastEvalConfig.loop())
    if seeds:
        seeds = _polish_seeds(cfg, ctx, seeds, q_ids, v_ids, q_lower, q_upper)
        ranked = []
        for s in seeds:
            R = _compute_trial_frame(cfg, ctx, s, v_ids)
            prims = compute_motion_primitives_frame(R, cfg.voxel_size_m)
            st = _run_trial_bfs(cfg, ctx, s, R, 15, q_ids, v_ids, q_lower, q_upper, prims)
            ranked.append((st.n_voxels, st.fail_moves / max(st.attempts, 1), s, R, prims))
        ranked.sort(key=lambda t: (-t[0], t[1]))
        _, _, seed, R, prims = ranked[0]
        if not a.no_cloud:
            st, disc = _run_trial_bfs(cfg, ctx, seed, R, a.budget, q_ids, v_ids, q_lower, q_upper, prims, return_configs=True)
    print(f"seeds: {len(seeds)}, cloud voxels: {len(disc)}  ({time.perf_counter() - t0:.0f}s)")

    V = np.array([n.p for n in disc.values()]) * 1e3 if disc else np.zeros((0, 3))
    far = max(disc.values(), key=lambda n: np.linalg.norm(n.p - seed.sphere_centre_world)) if disc else None
    panels = [
        ("Neutral pose (q = 0)", q0, None, False, (28, -55)),
        ("Seed pinch + reachable sphere-centre voxels (translation BFS)", seed.q if seed else q0,
         seed.sphere_centre_world if seed else None, True, (28, -55)),
        ("Hand at the farthest reached voxel", far.q if far else (seed.q if seed else q0),
         far.p if far else (seed.sphere_centre_world if seed else None), True, (28, -55)),
        ("Seed pinch, view along -z", seed.q if seed else q0, seed.sphere_centre_world if seed else None, False, (90, -90)),
    ]
    fig = plt.figure(figsize=(16, 9)); fig.patch.set_facecolor("white")
    axes, all_pts = [], []
    for i, (title, q, c, cloud, (el, az)) in enumerate(panels, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d"); axes.append(ax)
        draw_hand(ax, ctx, cfg, q.copy(), c, alpha=0.85 if cloud else 0.95)
        pts = [np.zeros(3)]
        for name in ctx.link_capsules:
            aa, bb = capsule_endpoints_world(ctx, name); pts += [np.asarray(aa) * 1e3, np.asarray(bb) * 1e3]
        if cloud and V.size:
            d = np.linalg.norm(V - seed.sphere_centre_world * 1e3, axis=1)
            sc = ax.scatter(V[:, 0], V[:, 1], V[:, 2], c=d, cmap="viridis", s=14, marker="s", alpha=0.45, depthshade=False, linewidths=0)
            pts += list(V)
        all_pts += pts
        ax.view_init(elev=el, azim=az); ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]"); ax.tick_params(labelsize=7)
    for ax in axes:
        set_equal(ax, all_pts)
    if V.size:
        cb = fig.colorbar(sc, ax=axes, shrink=0.5, pad=0.02); cb.set_label("distance from seed pinch [mm]")
    T = len(disc) * cfg.voxel_size_m ** 3 / cfg.l_ref_m ** 3 if disc else float("nan")
    fig.suptitle(f"{label}\n{len(disc)} voxels from the best seed (single-seed proxy T = {T:.4f}), "
                 f"L_ref = {cfg.l_ref_m*1e3:.0f} mm, voxel = {cfg.voxel_size_m*1e3:.1f} mm, "
                 f"thumb = orange, index = blue, grey bar = palm (knuckle to knuckle)", fontsize=10)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
