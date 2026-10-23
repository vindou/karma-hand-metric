"""Animate a hand pivoting through its operating range, one joint at a time, then all together.

    python optimize/animate_joints.py --run-dir optimize/runs/A1/final --out optimize/runs/A1/final/joints.mp4
    python optimize/animate_joints.py --design optimize/runs/A1/best.json --base neutral     # no metric result needed

Segment per active joint: hold the base pose (seed pinch by default, or neutral), sweep that
joint lower -> upper -> lower with an eased profile while its axis arrow is highlighted and
the fingertip path is traced; a final segment moves every joint on its own sinusoid.
"""
from __future__ import annotations
import argparse, json, sys, tempfile, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from karma.config import load_config                                                     # noqa: E402
from karma.robot import load_robot, active_position_indices                               # noqa: E402
from karma.viser_geometry import capsule_mesh, sphere_mesh                                # noqa: E402
from optimize.viz import hand_pose, rot_z_to, THUMB_C, INDEX_C, THUMB_DIM, INDEX_DIM, PALM_C  # noqa: E402

AXIS_LEN_R = 4.6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="scored run dir (uses its robot yaml and seed pose)")
    ap.add_argument("--design", help="design json instead of a run dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--base", choices=["seed", "neutral"], default="seed", help="pose the other joints hold")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames-per-joint", type=int, default=72)
    ap.add_argument("--wiggle-frames", type=int, default=240)
    ap.add_argument("--spin", type=float, default=0.15, help="camera azimuth degrees per frame")
    ap.add_argument("--dpi", type=int, default=100)
    a = ap.parse_args()

    if a.run_dir:
        rd = Path(a.run_dir); yml = next(rd.glob("robot_*.yaml")); out = a.out or (rd / "joints.mp4")
        from karma.storage import load_result
        res, _ = load_result(rd / "current.pkl"); seed_q, seed_c, seed_pair = np.asarray(res.seed_q), np.asarray(res.seed_centre), tuple(res.seed_contact_pair)
        label = rd.parent.name
    else:
        from optimize.evaluate_full import load_design
        hand = load_design(a.design); tmp = Path(tempfile.mkdtemp()); _, yml = hand.write(tmp)
        out = a.out or Path(a.design).with_name("joints.mp4"); res = None; seed_q = seed_c = seed_pair = None; label = Path(a.design).stem
    cfg = load_config(REPO / "karma_config.yaml", yml); ctx = load_robot(cfg.urdf_path, cfg)
    model = ctx.model
    q_ids = active_position_indices(model, cfg.finger_joint_names)
    lo = np.array(model.lowerPositionLimit); hi = np.array(model.upperPositionLimit)
    base = np.zeros(model.nq) if (a.base == "neutral" or seed_q is None) else seed_q.copy()
    base = np.clip(base, lo, hi)
    names = list(model.names)
    active = [(j, names[j]) for j in range(1, model.njoints) if names[j] in cfg.finger_joint_names]
    active.sort(key=lambda t: cfg.finger_joint_names.index(t[1]))   # thumb chain first, then index, proximal to distal

    # ── frame plan: list of (q, highlight_joint_name, segment_label) ──
    plan = []
    n = a.frames_per_joint
    ease = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)   # noqa: E731
    for j, jn in active:
        iq = model.joints[j].idx_q
        for k in range(n):
            t = k / (n - 1)
            # base -> lo -> hi -> base, eased
            if t < 0.25:
                s_ = ease(t / 0.25); val = base[iq] + s_ * (lo[iq] - base[iq])
            elif t < 0.75:
                s_ = ease((t - 0.25) / 0.5); val = lo[iq] + s_ * (hi[iq] - lo[iq])
            else:
                s_ = ease((t - 0.75) / 0.25); val = hi[iq] + s_ * (base[iq] - hi[iq])
            q = base.copy(); q[iq] = val
            plan.append((q, jn, f"{jn}: {np.degrees(val):+.0f}°  (limits {np.degrees(lo[iq]):+.0f}° … {np.degrees(hi[iq]):+.0f}°)"))
    rng = np.random.RandomState(0)
    freqs = 1.0 + 0.6 * rng.rand(len(active)); phases = 2 * np.pi * rng.rand(len(active))
    for k in range(a.wiggle_frames):
        t = k / a.fps
        q = base.copy()
        env = ease(min(1.0, k / (0.15 * a.wiggle_frames))) * ease(min(1.0, (a.wiggle_frames - 1 - k) / (0.15 * a.wiggle_frames)))
        for (j, jn), f, ph in zip(active, freqs, phases):
            iq = model.joints[j].idx_q
            mid, amp = 0.5 * (lo[iq] + hi[iq]), 0.5 * (hi[iq] - lo[iq])
            target = mid + amp * np.sin(2 * np.pi * f * t / 4.0 + ph)
            q[iq] = base[iq] + env * (target - base[iq])
        plan.append((q, None, "all joints: independent sinusoids spanning each range"))
    print(f"{len(active)} active joints, {len(plan)} frames = {len(plan)/a.fps:.0f} s at {a.fps} fps")

    # ── bounds from a coarse pass over the plan ──
    pts = [np.zeros(3)]
    for q, _, _ in plan[:: max(1, len(plan) // 150)]:
        g = hand_pose(ctx, cfg, q)
        for _, p0, p1, _, _ in g["capsules"]:
            pts += [p0 * 1e3, p1 * 1e3]
    pts = np.array(pts); c = 0.5 * (pts.max(0) + pts.min(0)); r = 0.5 * float((pts.max(0) - pts.min(0)).max()) * 1.1 + 20

    fig = plt.figure(figsize=(12, 9), dpi=a.dpi); fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r); ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x [mm]"); ax.set_ylabel("y [mm]"); ax.set_zlabel("z [mm]")
    fig.suptitle(f"{label}: operating range, one joint at a time ({a.base} base pose)", fontsize=12)
    txt = ax.text2D(0.02, 0.95, "", transform=ax.transAxes, fontsize=11)
    r_mm = cfg.link_radius_m * 1e3
    thumb_links = set(cfg.thumb_contact_links)
    tip_links = (cfg.seed_thumb_link, cfg.seed_index_link)
    trail = {ln: [] for ln in tip_links}
    trail_lines = {ln: ax.plot([], [], [], lw=1.6, color="#ff2d55", alpha=0.8, zorder=12)[0] for ln in tip_links}
    writer = FFMpegWriter(fps=a.fps, bitrate=5000, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2"])
    t0 = time.perf_counter(); last_hl = None
    with writer.saving(fig, str(out), a.dpi):
        for fi, (q, hl, lab) in enumerate(plan):
            for coll in list(ax.collections):
                coll.remove()
            for ln in list(ax.lines):
                if ln not in trail_lines.values():
                    ln.remove()
            if hl != last_hl:
                for ln in tip_links:
                    trail[ln] = []
                last_hl = hl
            g = hand_pose(ctx, cfg, q, seed_c if (a.base == "seed" and seed_c is not None and hl is None and fi == 0) else None, None)
            for name, p0, p1, is_thumb, _ in g["capsules"]:
                p0, p1 = p0 * 1e3, p1 * 1e3; seg = p1 - p0; L = np.linalg.norm(seg)
                verts, faces = capsule_mesh(r_mm, max(L / 2, 1e-6), n_rings=8, n_segments=12)
                verts = verts @ (rot_z_to(seg) if L > 1e-9 else np.eye(3)).T + 0.5 * (p0 + p1)
                ax.add_collection3d(Poly3DCollection(verts[faces], facecolor=THUMB_C if is_thumb else INDEX_C, edgecolor="none", alpha=0.95))
                if name in tip_links:
                    trail[name].append(p1)
            for ln in tip_links:
                if len(trail[ln]) > 1:
                    T = np.array(trail[ln]); trail_lines[ln].set_data_3d(T[:, 0], T[:, 1], T[:, 2])
                else:
                    trail_lines[ln].set_data_3d([], [], [])
            alen = AXIS_LEN_R * r_mm
            for j in g["joints"]:
                o = j["origin"] * 1e3; axv = j["axis"]; is_hl = (j["name"] == hl)
                col = "#ff2d55" if is_hl else ("#7a3b00" if j["is_thumb"] else "#0b2f73")
                L_ = alen * (1.4 if is_hl else 1.0); p0 = o - 0.5 * L_ * axv
                ax.quiver(p0[0], p0[1], p0[2], axv[0], axv[1], axv[2], length=L_, arrow_length_ratio=0.3, color=col, lw=3.0 if is_hl else 1.8, zorder=10)
            K = g["knuckles"] * 1e3
            if len(K) >= 2:
                ax.plot(K[:, 0], K[:, 1], K[:, 2], color=PALM_C, lw=6, solid_capstyle="round", zorder=4)
            txt.set_text(lab)
            ax.view_init(elev=24, azim=-58 + a.spin * fi)
            writer.grab_frame()
            if fi % 200 == 0:
                print(f"  frame {fi}/{len(plan)} ({time.perf_counter()-t0:.0f}s)", flush=True)
    print(f"wrote {out} ({time.perf_counter()-t0:.0f}s)")


if __name__ == "__main__":
    main()
