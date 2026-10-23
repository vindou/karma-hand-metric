"""Render a scored design: finger capsules at the seed pinch, the test sphere,
and the reachable sphere-centre voxel cloud coloured by orientation coverage.

    python optimize/render_hand.py --run-dir optimize/runs/T/final --out optimize/runs/T/final/hand.png

<run-dir> is the output of evaluate_full.py (holds current.pkl + robot_<name>.yaml).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from karma.config import load_config                                  # noqa: E402
from karma.robot import load_robot, update_kinematics, capsule_endpoints_world  # noqa: E402
from karma.storage import load_result                                 # noqa: E402
from karma.viser_geometry import capsule_mesh, sphere_mesh            # noqa: E402
from optimize.viz import hand_pose, voxel_quads, seed_frame, THUMB_C, INDEX_C, THUMB_DIM, INDEX_DIM, PALM_C, CMAP  # noqa: E402
import matplotlib.colors                                              # noqa: E402
import matplotlib.cm                                                  # noqa: E402


def _rot_z_to(d: np.ndarray) -> np.ndarray:
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); c = float(np.dot(z, d))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


def _add_mesh(ax, verts, faces, color, alpha=0.95):
    tri = verts[faces]
    pc = Poly3DCollection(tri, facecolor=color, edgecolor="none", alpha=alpha)
    ax.add_collection3d(pc)


def draw_hand(ax, ctx, cfg, q, centre, scale=1e3, alpha=0.95, pair=None, axes=True):
    """Capsule hand (contact links saturated, others dimmed), joint axis arrows, palm bar, sphere, contact normals."""
    g = hand_pose(ctx, cfg, q, centre, pair)
    r = cfg.link_radius_m * scale
    for name, a, b, is_thumb, is_contact in g["capsules"]:
        a, b = np.asarray(a) * scale, np.asarray(b) * scale
        seg = b - a; L = np.linalg.norm(seg)
        verts, faces = capsule_mesh(r, max(L / 2, 1e-6), n_rings=8, n_segments=12)
        R = _rot_z_to(seg) if L > 1e-9 else np.eye(3)
        verts = (verts @ R.T) + 0.5 * (a + b)
        dim = pair is not None and not is_contact
        _add_mesh(ax, verts, faces, (THUMB_DIM if dim else THUMB_C) if is_thumb else (INDEX_DIM if dim else INDEX_C), alpha)
    if axes:
        alen = 4.6 * r
        for j in g["joints"]:
            o = np.asarray(j["origin"]) * scale; ax_ = np.asarray(j["axis"])
            col = "#7a3b00" if j["is_thumb"] else "#0b2f73"
            p0 = o - 0.5 * alen * ax_
            ax.quiver(p0[0], p0[1], p0[2], ax_[0], ax_[1], ax_[2], length=alen, arrow_length_ratio=0.35, color=col, lw=1.8, zorder=10)
    if centre is not None:
        sv, sf = sphere_mesh(cfg.sphere_radius_m * scale, 10, 14)
        _add_mesh(ax, np.asarray(sv) + np.asarray(centre) * scale, np.asarray(sf), "#3a3a3a", 0.95)
        for ct in g["contacts"]:
            pt = ct["point"] * scale; n = ct["normal"]; nl = 1.5 * cfg.sphere_radius_m * scale
            ax.plot([pt[0], pt[0] + nl * n[0]], [pt[1], pt[1] + nl * n[1]], [pt[2], pt[2] + nl * n[2]], color="#00c2a8", lw=2.2, zorder=11)
            ax.scatter([pt[0]], [pt[1]], [pt[2]], color="#00c2a8", s=18, zorder=11)
    K = np.asarray(g["knuckles"]) * scale
    if len(K) >= 2:
        ax.plot(K[:, 0], K[:, 1], K[:, 2], color=PALM_C, lw=6, solid_capstyle="round")


def draw_voxels(ax, centres_mm, values, edge_mm, R_grid, vmin, vmax, alpha=0.5, shrink=0.9):
    """True voxel cubes on the seed-frame grid, coloured on the magma scale."""
    quads, owner = voxel_quads(np.asarray(centres_mm), edge_mm, R_grid, shrink=shrink)
    cmap = matplotlib.colormaps[CMAP]
    cols = cmap((np.asarray(values)[owner] - vmin) / max(vmax - vmin, 1e-9)); cols[:, 3] = alpha
    pc = Poly3DCollection(quads, facecolors=cols, edgecolors="none")
    ax.add_collection3d(pc)
    return pc


def set_equal(ax, pts):
    pts = np.asarray(pts)
    c = 0.5 * (pts.max(0) + pts.min(0)); r = 0.5 * float((pts.max(0) - pts.min(0)).max()) * 1.05 + 1e-3
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_box_aspect((1, 1, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--title", default=None)
    a = ap.parse_args()
    rd = Path(a.run_dir)
    res, meta = load_result(rd / "current.pkl")
    yaml_path = next(rd.glob("robot_*.yaml"))
    cfg = load_config(REPO / "karma_config.yaml", yaml_path)
    ctx = load_robot(cfg.urdf_path, cfg)

    # one representative node per voxel
    vox_centre, vox_q, vox_cov, vox_pair = {}, {}, {}, {}
    for (v, _b), node in res.state_nodes.items():
        if v not in vox_centre:
            vox_centre[v] = node.centre_world; vox_q[v] = node.q; vox_pair[v] = tuple(node.contact_pair)
            vox_cov[v] = res.rotational_coverage_per_voxel.get(v, 0.0)
    R_grid = seed_frame(res)
    seed_pair = tuple(res.seed_contact_pair)
    V = np.array(list(vox_centre.values())) * 1e3 if vox_centre else np.zeros((0, 3))
    C = np.array([vox_cov[v] for v in vox_centre]) if vox_centre else np.zeros(0)
    seed_q, seed_c = res.seed_q, res.seed_centre

    # the farthest reached voxel from the seed (to show the hand "elsewhere" in its workspace)
    far_v = max(vox_centre, key=lambda v: np.linalg.norm(vox_centre[v] - seed_c)) if vox_centre else None

    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("white")
    views = [
        ("Seed pinch + reachable sphere-centre voxels", seed_q, seed_c, seed_pair, True, (28, -55)),
        ("Hand at the farthest reached voxel", vox_q[far_v] if far_v else seed_q, vox_centre[far_v] if far_v else seed_c,
         vox_pair[far_v] if far_v else seed_pair, True, (28, -55)),
        ("Seed pinch, view along -z (palm normal)", seed_q, seed_c, seed_pair, False, (90, -90)),
        ("Seed pinch, view along -y", seed_q, seed_c, seed_pair, False, (0, -90)),
    ]
    vmax = max(0.25, float(C.max()) if C.size else 0.25)
    norm = Normalize(vmin=0.0, vmax=vmax)
    all_pts = []
    axes = []
    sc = None
    for i, (title, q, c, pair, show_cloud, (elev, azim)) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, i, projection="3d", computed_zorder=False); axes.append(ax)
        pts = [np.zeros(3)]
        if show_cloud and V.size:   # cloud first, hand on top (no depth sorting), so the pose stays readable inside it
            draw_voxels(ax, V, C, cfg.voxel_size_m * 1e3, R_grid, 0.0, vmax, alpha=0.3)
            pts += list(V)
        draw_hand(ax, ctx, cfg, q, c, alpha=1.0, pair=pair)
        for name in ctx.link_capsules:
            aa, bb = capsule_endpoints_world(ctx, name); pts += [np.asarray(aa) * 1e3, np.asarray(bb) * 1e3]
        all_pts += pts
        set_equal(ax, pts)
        ax.view_init(elev=elev, azim=azim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
        ax.tick_params(labelsize=7)
    # same frame for all panels
    for ax in axes:
        set_equal(ax, all_pts)
    if V.size:
        cb = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=matplotlib.colormaps[CMAP]), ax=axes, shrink=0.5, pad=0.02)
        cb.set_label("per-voxel orientation coverage (fraction of 228 bins)")
    sens = res.seed_sensitivity or {}
    title = a.title or rd.name
    fig.suptitle(
        f"{title}:  KaRMA-T = {res.translational_score:.4f}   KaRMA-R = {res.global_rotational_score:.3f}   "
        f"KaRMA-S = {sens.get('karma_s', float('nan')):.3f}   |   {res.n_voxels_reached} voxels, "
        f"L_ref = {res.l_ref_m * 1e3:.0f} mm, voxel = {cfg.voxel_size_m * 1e3:.1f} mm   "
        f"(thumb orange, index blue, dimmed = not in contact; arrows = joint axes; teal = contact normals)", fontsize=10)
    out = a.out or (rd / "hand.png")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
