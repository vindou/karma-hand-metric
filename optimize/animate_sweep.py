"""Animate a scored hand sweeping through every reached voxel, smoothly.

    python optimize/animate_sweep.py --run-dir optimize/runs/A1/final --out optimize/runs/A1/final/sweep.mp4

The hand follows an Euler tour of the metric's own search tree (optimize/voxel_tour.py):
one rolling step per frame while exploring, retracing faster when backtracking, with
joint-space interpolation between frames. The reachable set is painted in as it is visited.
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
import matplotlib.colors
from mpl_toolkits.mplot3d.art3d import Line3DCollection
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from karma.config import load_config                                                   # noqa: E402
from karma.robot import load_robot, update_kinematics, capsule_endpoints_world           # noqa: E402
from karma.storage import load_result                                                   # noqa: E402
from optimize.voxel_tour import tree_and_tour                                            # noqa: E402
from optimize.viz import hand_pose, voxel_quads, seed_frame, THUMB_C, INDEX_C, THUMB_DIM, INDEX_DIM, PALM_C, CMAP  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402
import matplotlib.cm as mcm  # noqa: E402



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--sub", type=int, default=2, help="interpolated frames per forward tree edge")
    ap.add_argument("--back-stride", type=int, default=3, help="tree edges per frame while retracing")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after this many frames (0 = all)")
    ap.add_argument("--elev", type=float, default=24); ap.add_argument("--azim", type=float, default=-58)
    ap.add_argument("--spin", type=float, default=0.0, help="degrees of camera azimuth per frame")
    ap.add_argument("--dpi", type=int, default=110)
    a = ap.parse_args()
    rd = Path(a.run_dir)
    res, _ = load_result(rd / "current.pkl")
    cfg = load_config(REPO / "karma_config.yaml", next(rd.glob("robot_*.yaml")))
    ctx = load_robot(cfg.urdf_path, cfg)
    nodes, tour, first = tree_and_tour(res)
    n_vox = len(nodes)
    pair_of = {}
    for (v, b), node in res.state_nodes.items():
        pair_of.setdefault(v, tuple(node.contact_pair))

    # frame plan: (q, centre, voxel_visited_or_None)
    frames = []
    i = 0
    while i < len(tour) - 1:
        v0, v1 = tour[i], tour[i + 1]
        if first[i + 1]:
            q0, q1, c0, c1 = nodes[v0]["q"], nodes[v1]["q"], nodes[v0]["centre"], nodes[v1]["centre"]
            for k in range(1, a.sub + 1):
                t = k / a.sub
                frames.append(((1 - t) * q0 + t * q1, (1 - t) * c0 + t * c1, v1 if k == a.sub else None))
            i += 1
        else:  # retracing: jump back-stride edges per frame while they are all retraces
            j = i + 1
            while j < len(tour) - 1 and not first[j + 1] and j - i < a.back_stride:
                j += 1
            frames.append((nodes[tour[j]]["q"], nodes[tour[j]]["centre"], None))
            i = j
    frames.insert(0, (nodes[tour[0]]["q"], nodes[tour[0]]["centre"], tour[0]))
    if a.max_frames:
        frames = frames[: a.max_frames]
    print(f"{n_vox} voxels, tour length {len(tour)}, {len(frames)} frames = {len(frames)/a.fps:.0f} s at {a.fps} fps")

    r_mm = cfg.link_radius_m * 1e3
    edge = cfg.voxel_size_m * 1e3
    R_grid = seed_frame(res)
    vox_keys = list(nodes)
    P_all = np.array([nodes[v]["centre"] for v in vox_keys]) * 1e3
    idx_of = {v: k for k, v in enumerate(vox_keys)}
    cov = np.array([nodes[v]["coverage"] for v in vox_keys])
    cmap = matplotlib.colormaps[CMAP]; vmax = max(0.25, float(cov.max()))
    quads, owner = voxel_quads(P_all, edge, R_grid, shrink=0.9)              # (6N,4,3)
    quad_cols = cmap(cov[owner] / vmax); quad_cols[:, 3] = 0.55
    lim_pts = np.vstack([P_all, np.zeros((1, 3))])
    c = 0.5 * (lim_pts.max(0) + lim_pts.min(0)); r = 0.5 * float((lim_pts.max(0) - lim_pts.min(0)).max()) * 1.15 + 30

    fig = plt.figure(figsize=(12, 9), dpi=a.dpi); fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r); ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    sens = res.seed_sensitivity or {}
    fig.suptitle(f"{rd.parent.name}: KaRMA-T {res.translational_score:.3f}  R {res.global_rotational_score:.3f}  S {sens.get('karma_s', float('nan')):.3f}   "
                 f"{n_vox} voxels, L_ref {res.l_ref_m*1e3:.0f} mm   (voxels coloured by orientation coverage, {CMAP})", fontsize=12)
    px_per_mm = (fig.get_figheight() * a.dpi * 0.7) / (2 * r)
    pt_per_mm = px_per_mm * 72 / a.dpi
    ghost = ax.scatter(P_all[:, 0], P_all[:, 1], P_all[:, 2], s=(edge * pt_per_mm * 0.4) ** 2, c="#cccccc", marker=".", alpha=0.12, depthshade=False, linewidths=0)
    visited_mask = np.zeros(n_vox, bool)
    cubes = Poly3DCollection(np.zeros((0, 4, 3)), facecolors=[], edgecolors="none", zorder=1)
    ax.add_collection3d(cubes)
    sph = ax.scatter([], [], [], s=(cfg.sphere_radius_m * 1e3 * 2 * pt_per_mm) ** 2, c="#222222", depthshade=False, zorder=6)
    link_lines = {}
    for name in ctx.link_capsules:
        is_thumb = name in set(cfg.thumb_contact_links)
        link_lines[name] = ax.plot([], [], [], lw=max(r_mm * 2 * pt_per_mm, 2), solid_capstyle="round",
                                   color=THUMB_C if is_thumb else INDEX_C, alpha=0.95, zorder=5)[0]
    palm_line = ax.plot([], [], [], lw=5, color=PALM_C, solid_capstyle="round", zorder=4)[0]
    axis_lines = []  # one shaft per joint, with a dot at the arrow tip
    alen = 4.6 * r_mm
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=10)
    cbar = fig.colorbar(mcm.ScalarMappable(norm=matplotlib.colors.Normalize(0, vmax), cmap=cmap), ax=ax, shrink=0.55, pad=0.02)
    cbar.set_label("per-voxel orientation coverage (fraction of 228 bins)")

    writer = FFMpegWriter(fps=a.fps, bitrate=5000, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    out = a.out or (rd / "sweep.mp4")
    t0 = time.perf_counter()
    with writer.saving(fig, str(out), a.dpi):
        for fi, (q, centre, vis) in enumerate(frames):
            pair = pair_of.get(vis) if vis is not None else None
            g = hand_pose(ctx, cfg, q, None, None)
            thumb_set = set(cfg.thumb_contact_links)
            for name, p0, p1, is_thumb, _ in g["capsules"]:
                p0, p1 = p0 * 1e3, p1 * 1e3
                ln = link_lines[name]
                ln.set_data_3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]])
                dim = pair is not None and name not in pair
                ln.set_color((THUMB_DIM if dim else THUMB_C) if is_thumb else (INDEX_DIM if dim else INDEX_C))
            K = g["knuckles"] * 1e3
            palm_line.set_data_3d(K[:, 0], K[:, 1], K[:, 2])
            # joint axes
            if not axis_lines:
                for j in g["joints"]:
                    col = "#7a3b00" if j["is_thumb"] else "#0b2f73"
                    axis_lines.append((ax.plot([], [], [], lw=2.2, color=col, zorder=7)[0], ax.plot([], [], [], marker="^", ms=5, color=col, zorder=7)[0]))
            for (shaft, tip), j in zip(axis_lines, g["joints"]):
                o = j["origin"] * 1e3; axv = j["axis"]
                p0, p1 = o - 0.5 * alen * axv, o + 0.5 * alen * axv
                shaft.set_data_3d([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]]); tip.set_data_3d([p1[0]], [p1[1]], [p1[2]])
            cc = np.asarray(centre) * 1e3
            sph._offsets3d = ([cc[0]], [cc[1]], [cc[2]])
            if vis is not None:
                visited_mask[idx_of[vis]] = True
            m6 = np.repeat(visited_mask, 6)
            cubes.set_verts(quads[m6]); cubes.set_facecolor(quad_cols[m6])
            txt.set_text(f"frame {fi+1}/{len(frames)}   visited {int(visited_mask.sum())}/{n_vox} voxels")
            if a.spin:
                ax.view_init(elev=a.elev, azim=a.azim + a.spin * fi)
            elif fi == 0:
                ax.view_init(elev=a.elev, azim=a.azim)
            writer.grab_frame()
            if fi % 500 == 0:
                print(f"  frame {fi}/{len(frames)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"wrote {out} ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
