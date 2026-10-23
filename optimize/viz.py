"""Shared drawing geometry for the explorer notebook, the sweep video and the static renders.

Everything here is backend-agnostic numpy: capsule segments, joint axis arrows, contact
points/normals, and true voxel cubes aligned with the seed frame (the grid the metric
actually counts on). Colour conventions: thumb orange, index blue, non-contact links
desaturated, voxel coverage on the magma colourscale.
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from karma.robot import update_kinematics, capsule_endpoints_world
from karma.contacts import contact_kinematics

THUMB_C, INDEX_C, THUMB_DIM, INDEX_DIM, PALM_C = "#e07b2a", "#2f6fd1", "#f1c7a3", "#b7cdf2", "#6f6f6f"
CMAP = "magma"


def hand_pose(ctx, cfg, q, centre=None, contact_pair=None):
    """Geometry of the hand at configuration q (metres).

    Returns dict with
      capsules: list of (name, a, b, is_thumb, is_contact)
      joints:   list of dict(name, origin, axis (unit, world), is_thumb, q, lo, hi, active)
      knuckles: (K,3) root joint origins
      contacts: list of dict(link, point, normal, gap) for the contact pair (if centre given)
    """
    q = np.array(q, dtype=float)
    update_kinematics(ctx, q)
    model, data = ctx.model, ctx.data
    thumb_links = set(cfg.thumb_contact_links)
    active = set(cfg.finger_joint_names)
    contact_pair = tuple(contact_pair) if contact_pair else ()
    caps = []
    for name in ctx.link_capsules:
        a, b = capsule_endpoints_world(ctx, name)
        caps.append((name, np.asarray(a).copy(), np.asarray(b).copy(), name in thumb_links, name in contact_pair))
    joints = []
    names = list(model.names)
    thumb_joint_names = set(cfg.thumb_joint_names)
    for j in range(1, model.njoints):
        jm = model.joints[j]
        if jm.nv != 1:
            continue
        Jw = pin.getJointJacobian(model, data, j, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[3:, jm.idx_v]
        nrm = np.linalg.norm(Jw)
        axis = Jw / nrm if nrm > 1e-9 else np.array([0.0, 0.0, 1.0])
        joints.append(dict(name=names[j], origin=np.asarray(data.oMi[j].translation).copy(), axis=axis,
                           is_thumb=names[j] in thumb_joint_names, q=float(q[jm.idx_q]),
                           lo=float(model.lowerPositionLimit[jm.idx_q]), hi=float(model.upperPositionLimit[jm.idx_q]),
                           active=names[j] in active))
    roots = [j for j in range(1, model.njoints) if model.parents[j] == 0]
    knuckles = np.array([np.asarray(data.oMi[j].translation) for j in roots])
    contacts = []
    if centre is not None and contact_pair:
        for link in contact_pair:
            if link in ctx.link_capsules:
                ck = contact_kinematics(ctx, q, np.asarray(centre, float), link, cfg.sphere_radius_m, cfg.link_radius_m)
                contacts.append(dict(link=link, point=np.asarray(ck.sphere_point_world).copy(),
                                     normal=np.asarray(ck.normal_world).copy(), gap=float(ck.g_m)))
    return dict(capsules=caps, joints=joints, knuckles=knuckles, contacts=contacts)


def rot_z_to(d):
    d = d / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(z, d); s = np.linalg.norm(v); c = float(np.dot(z, d))
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)


# ── voxel cubes ───────────────────────────────────────────────────────────────

_CUBE_V = np.array([[x, y, z] for x in (-0.5, 0.5) for y in (-0.5, 0.5) for z in (-0.5, 0.5)], float)  # 8 corners
_CUBE_F = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                    [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]], int)       # 12 triangles
_CUBE_QUADS = np.array([[0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6], [0, 2, 6, 4], [1, 5, 7, 3]], int)


def voxel_cubes(centres, edge, R=None, shrink=0.92):
    """Triangle mesh of one cube per centre: verts (8N,3), faces (12N,3), vertex_owner (8N,).
    R (3x3, columns = grid axes) orients the cubes with the seed frame."""
    C = np.asarray(centres, float).reshape(-1, 3)
    R = np.eye(3) if R is None else np.asarray(R, float)
    corners = (_CUBE_V * edge * shrink) @ R.T                         # (8,3)
    verts = (C[:, None, :] + corners[None, :, :]).reshape(-1, 3)
    faces = (_CUBE_F[None, :, :] + 8 * np.arange(len(C))[:, None, None]).reshape(-1, 3)
    owner = np.repeat(np.arange(len(C)), 8)
    return verts, faces, owner


def voxel_quads(centres, edge, R=None, shrink=0.92):
    """Quad faces per cube for matplotlib Poly3DCollection: (6N,4,3) and owner (6N,)."""
    C = np.asarray(centres, float).reshape(-1, 3)
    R = np.eye(3) if R is None else np.asarray(R, float)
    corners = (_CUBE_V * edge * shrink) @ R.T
    V = C[:, None, :] + corners[None, :, :]                             # (N,8,3)
    quads = V[:, _CUBE_QUADS, :].reshape(-1, 4, 3)
    return quads, np.repeat(np.arange(len(C)), 6)


def seed_frame(res):
    R = np.asarray(getattr(res, "seed_frame", np.eye(3)), float)
    return R if R.shape == (3, 3) else np.eye(3)


# ── joint sweep plan (shared by the Plotly animation and the mp4) ───────────

def joint_sweep_plan(ctx, cfg, base_q, frames_per_joint=16, wiggle_frames=48, fps=20):
    """List of (q, highlighted_joint_name_or_None, label). One eased lower->upper->base sweep
    per active joint (thumb chain then index), then every joint on its own sinusoid."""
    model = ctx.model
    lo = np.array(model.lowerPositionLimit); hi = np.array(model.upperPositionLimit)
    base = np.clip(np.array(base_q, float), lo, hi)
    names = list(model.names)
    active = [(j, names[j]) for j in range(1, model.njoints) if names[j] in cfg.finger_joint_names]
    active.sort(key=lambda t: cfg.finger_joint_names.index(t[1]))
    ease = lambda t: 0.5 - 0.5 * np.cos(np.pi * t)  # noqa: E731
    plan = []
    n = max(frames_per_joint, 4)
    for j, jn in active:
        iq = model.joints[j].idx_q
        for k in range(n):
            t = k / (n - 1)
            if t < 0.25:
                val = base[iq] + ease(t / 0.25) * (lo[iq] - base[iq])
            elif t < 0.75:
                val = lo[iq] + ease((t - 0.25) / 0.5) * (hi[iq] - lo[iq])
            else:
                val = hi[iq] + ease((t - 0.75) / 0.25) * (base[iq] - hi[iq])
            q = base.copy(); q[iq] = val
            plan.append((q, jn, f"{jn} {np.degrees(val):+.0f}° [{np.degrees(lo[iq]):+.0f}°, {np.degrees(hi[iq]):+.0f}°]"))
    rng = np.random.RandomState(0)
    freqs = 1.0 + 0.6 * rng.rand(len(active)); phases = 2 * np.pi * rng.rand(len(active))
    for k in range(wiggle_frames):
        t = k / fps
        env = ease(min(1.0, k / max(1, 0.15 * wiggle_frames))) * ease(min(1.0, (wiggle_frames - 1 - k) / max(1, 0.15 * wiggle_frames)))
        q = base.copy()
        for (j, jn), f, ph in zip(active, freqs, phases):
            iq = model.joints[j].idx_q
            mid, amp = 0.5 * (lo[iq] + hi[iq]), 0.5 * (hi[iq] - lo[iq])
            q[iq] = base[iq] + env * (mid + amp * np.sin(2 * np.pi * f * t / 4.0 + ph) - base[iq])
        plan.append((q, None, "all joints, independent sinusoids"))
    return plan
