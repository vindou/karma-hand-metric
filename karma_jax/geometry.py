"""Capsule geometry: endpoints, sphere-capsule gaps, contact kinematics,
collision and grasp checks, for one state (vmap for batches).

One :func:`capsule_pass` computes, for every capsule at once, the closest
point on its axis to the sphere centre, the signed gap and the normal. That
single tensor then serves contact-pair selection, the object-collision check
and the two active contacts, which the reference computes with three separate
code paths (``contact_kinematics``, ``contact_gap_only``, ``check_collisions``).
Every function reproduces the reference's formulas and thresholds; the
data-dependent branches become ``jnp.where``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .model import DeviceModel
from .kinematics import FKResult, frame_jacobian, skew
from .linalg import mm, mv, vm

_T_EPS = 1e-12      # interior/endpoint threshold on the segment parameter (reference)
_SSQ_EPS = 1e-18    # degenerate (zero-length) segment threshold (reference)
_DIST_EPS = 1e-12   # sphere centre on the axis (reference)


class CapsulePass(NamedTuple):
    a: object        # [C,3] axis start (link frame origin)
    b: object        # [C,3] axis end
    t: object        # [C]   closest-point parameter in [0,1]
    axis_pt: object  # [C,3] closest point on the axis
    normal: object   # [C,3] unit vector axis_pt -> sphere centre
    dist: object     # [C]   |centre - axis_pt|
    gap: object      # [C]   dist - r_s - r_l (signed)


class ContactKin(NamedTuple):
    """Mirror of ``karma.contacts.ContactKinematics`` for one capsule."""
    cap: object          # scalar capsule index
    axis_point: object   # [3]
    sphere_point: object # [3]
    normal: object       # [3]
    g: object            # scalar signed gap
    Jv: object           # [3,J] velocity of the closest axis point (full q)
    Jw: object           # [3,J] link angular velocity
    origin: object       # [3]   link frame origin
    Jv_origin: object    # [3,J]


def capsule_endpoints(model: DeviceModel, fk: FKResult):
    """World axis endpoints ``(a, b)`` of every capsule, each ``[C,3]``."""
    a = fk.p_frame[model.cap_frame]
    distal = model.cap_distal
    has_distal = distal >= 0
    b_distal = fk.p_frame[jnp.where(has_distal, distal, 0)]
    R_link = fk.R_frame[model.cap_frame]
    b_tip = a + jnp.einsum("cab,cb->ca", R_link, model.cap_tip)
    b = jnp.where(has_distal[:, None], b_distal, b_tip)
    return a, b


def capsule_gap(model: DeviceModel, fk: FKResult, c, centre, r_s, r_l):
    """Signed gap of ONE capsule ``c`` (reference ``contact_gap_only``), cheaper than a full pass."""
    f = model.cap_frame[c]
    a = fk.p_frame[f]
    distal = model.cap_distal[c]
    has_distal = distal >= 0
    b = jnp.where(has_distal, fk.p_frame[jnp.where(has_distal, distal, 0)],
                  a + fk.R_frame[f] @ model.cap_tip[c])
    seg = b - a
    ssq = jnp.dot(seg, seg)
    degenerate = ssq < _SSQ_EPS
    t = jnp.where(degenerate, 0.0, jnp.clip(jnp.dot(centre - a, seg) / jnp.where(degenerate, 1.0, ssq), 0.0, 1.0))
    axis_pt = a + t * seg
    return jnp.linalg.norm(centre - axis_pt) - r_s - r_l


def _fallback_normal(seg):
    """Deterministic normal when the centre lies on the axis (reference logic)."""
    ex = jnp.array([1.0, 0.0, 0.0], seg.dtype)
    ey = jnp.array([0.0, 1.0, 0.0], seg.dtype)
    ez = jnp.array([0.0, 0.0, 1.0], seg.dtype)
    n1 = jnp.cross(seg, ex)
    n = jnp.where(jnp.linalg.norm(n1, axis=-1, keepdims=True) < 1e-12, jnp.cross(seg, ey), n1)
    nrm = jnp.linalg.norm(n, axis=-1, keepdims=True)
    return jnp.where(nrm > 1e-12, n / jnp.where(nrm > 1e-12, nrm, 1.0), ez)


def capsule_pass(model: DeviceModel, fk: FKResult, centre, r_s, r_l) -> CapsulePass:
    """Closest point, gap and normal of every capsule w.r.t. a sphere at ``centre``."""
    a, b = capsule_endpoints(model, fk)
    seg = b - a
    ssq = jnp.sum(seg * seg, axis=-1)
    degenerate = ssq < _SSQ_EPS
    safe_ssq = jnp.where(degenerate, 1.0, ssq)
    t_raw = jnp.sum((centre[None, :] - a) * seg, axis=-1) / safe_ssq
    t = jnp.where(degenerate, 0.0, jnp.clip(t_raw, 0.0, 1.0))
    axis_pt = a + t[:, None] * seg
    diff = centre[None, :] - axis_pt
    dist = jnp.linalg.norm(diff, axis=-1)
    on_axis = dist < _DIST_EPS
    normal = jnp.where(on_axis[:, None], _fallback_normal(seg),
                       diff / jnp.where(on_axis, 1.0, dist)[:, None])
    gap = dist - r_s - r_l
    return CapsulePass(a, b, t, axis_pt, normal, dist, gap)


def contact_kinematics(model: DeviceModel, fk: FKResult, cp: CapsulePass, c, centre, r_s) -> ContactKin:
    """Contact kinematics of capsule ``c`` (scalar, may be traced), as the reference."""
    a = cp.a[c]
    b = cp.b[c]
    t = cp.t[c]
    seg = b - a
    ssq = jnp.sum(seg * seg)
    normal = cp.normal[c]
    gap = cp.gap[c]
    axis_pt = cp.axis_pt[c]
    sphere_pt = centre - r_s * normal

    link_f = model.cap_frame[c]
    Jv_a, Jw_a = frame_jacobian(model, fk, link_f)
    distal = model.cap_distal[c]
    has_distal = distal >= 0
    Jv_b_distal, _ = frame_jacobian(model, fk, jnp.where(has_distal, distal, 0))
    R_a = fk.R_frame[link_f]
    lever_ab = R_a @ model.cap_tip[c]
    Jv_b_tip = Jv_a - mm(skew(lever_ab), Jw_a)
    Jv_b = jnp.where(has_distal, Jv_b_distal, Jv_b_tip)

    # Jacobian of the closest point on the segment (reference formula).
    J_s = Jv_b - Jv_a
    w = centre - a
    u = jnp.where(ssq < _SSQ_EPS, 1.0, ssq)
    ws = jnp.dot(w, seg)
    dt = (-vm(seg, Jv_a) + vm(w, J_s)) / u - (ws / (u * u)) * (2.0 * vm(seg, J_s))
    Jv_interior = Jv_a + t * J_s + jnp.outer(seg, dt)
    at_start = t <= 0.0 + _T_EPS
    at_end = t >= 1.0 - _T_EPS
    Jv_axis = jnp.where(at_start, Jv_a, jnp.where(at_end, Jv_b, Jv_interior))

    return ContactKin(c, axis_pt, sphere_pt, normal, gap, Jv_axis, Jw_a, a, Jv_a)


# ── Pair selection ───────────────────────────────────────────────────────────

def _best_link(gap, caps, current, hysteresis):
    """Reference ``select_contact_pair._best``: argmin |gap| over ``caps`` (padded
    with -1), keeping ``current`` unless the best is better by ``hysteresis``."""
    valid = caps >= 0
    ag = jnp.abs(gap[jnp.where(valid, caps, 0)])
    ag = jnp.where(valid, ag, jnp.inf)
    k = jnp.argmin(ag)              # first minimum, as the reference's strict '<' scan
    best = caps[k]
    best_gap = ag[k]
    cur_gap = jnp.abs(gap[current])
    keep = (current != best) & (hysteresis > 0) & (cur_gap - best_gap < hysteresis)
    return jnp.where(keep, current, best)


def select_contact_pair(model: DeviceModel, cp: CapsulePass, current_pair, hysteresis):
    """``(thumb_cap, index_cap)`` as ``karma.contacts.select_contact_pair``."""
    t = _best_link(cp.gap, model.thumb_caps, current_pair[0], hysteresis)
    i = _best_link(cp.gap, model.index_caps, current_pair[1], hysteresis)
    return jnp.stack([t, i])


# ── Validations ──────────────────────────────────────────────────────────────

def check_collisions(model: DeviceModel, cp: CapsulePass, pair, eps_col):
    """Sphere vs every capsule except the active pair. Returns ``(ok, min_clearance)``.

    ``ok`` is identical to the reference. ``min_clearance`` is the true minimum
    over all capsules, whereas the reference's AABB prefilter skips far capsules
    and can report a larger value; the two agree whenever the value is below
    ``eps_col``, which is the only regime in which the reference consumes it.
    """
    C = cp.gap.shape[0]
    idx = jnp.arange(C)
    mask = model.cap_valid & (idx != pair[0]) & (idx != pair[1])
    clearance = jnp.where(mask, cp.gap, jnp.inf)
    min_clearance = jnp.min(clearance)
    return min_clearance >= eps_col, min_clearance


def segment_segment_dist(a0, b0, a1, b1):
    """Closest distance between segments [a0,b0], [a1,b1]; broadcasts over leading axes.

    Branch-for-branch port of ``karma.collisions._segment_segment_dist``.
    """
    EPS = 1e-12
    d0 = b0 - a0
    d1 = b1 - a1
    r = a0 - a1
    a = jnp.sum(d0 * d0, axis=-1)
    e = jnp.sum(d1 * d1, axis=-1)
    f = jnp.sum(d1 * r, axis=-1)
    c = jnp.sum(d0 * r, axis=-1)
    b = jnp.sum(d0 * d1, axis=-1)
    a_tiny = a <= EPS
    e_tiny = e <= EPS
    sa = jnp.where(a_tiny, 1.0, a)
    se = jnp.where(e_tiny, 1.0, e)

    # general case
    denom = a * e - b * b
    s = jnp.where(jnp.abs(denom) > EPS, jnp.clip((b * f - c * e) / jnp.where(jnp.abs(denom) > EPS, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / se
    s = jnp.where(t < 0.0, jnp.clip(-c / sa, 0.0, 1.0),
                  jnp.where(t > 1.0, jnp.clip((b - c) / sa, 0.0, 1.0), s))
    t = jnp.clip(t, 0.0, 1.0)

    # degenerate cases
    s = jnp.where(a_tiny, 0.0, s)
    t = jnp.where(a_tiny, jnp.clip(f / se, 0.0, 1.0), t)
    s = jnp.where(e_tiny & ~a_tiny, jnp.clip(-c / sa, 0.0, 1.0), s)
    t = jnp.where(e_tiny, 0.0, t)
    s = jnp.where(a_tiny & e_tiny, 0.0, s)

    closest = (a0 + s[..., None] * d0) - (a1 + t[..., None] * d1)
    return jnp.linalg.norm(closest, axis=-1)


def check_self_collisions(model: DeviceModel, cp: CapsulePass, r_l, eps_col):
    """Reference self-collision gate: thumb capsules vs index capsules. With ``model.cap_group``
    set (``build_flat_model(..., self_collision_all=True)``, Phase 9.2 / SPEC §8.1 D18) the gate
    is :func:`check_self_collisions_all` instead. Returns ``(ok, min_clearance)``."""
    if getattr(model, "cap_group", None) is not None:
        return check_self_collisions_all(model, cp, r_l, eps_col)
    return check_self_collisions_pair(model, cp, r_l, eps_col)


def check_self_collisions_all(model: DeviceModel, cp: CapsulePass, r_l, eps_col):
    """Every pair of valid capsules on different kinematic chains (``model.cap_group``):
    clearance ``dist - 2 r_l``. Returns ``(ok, min_clearance)``."""
    C = cp.a.shape[0]
    g = model.cap_group
    idx = jnp.arange(C)
    mask = (idx[:, None] < idx[None, :]) & model.cap_valid[:, None] & model.cap_valid[None, :] \
        & (g[:, None] != g[None, :]) & (g[:, None] >= 0) & (g[None, :] >= 0)
    d = segment_segment_dist(cp.a[:, None, :], cp.b[:, None, :], cp.a[None, :, :], cp.b[None, :, :])   # [C,C]
    clearance = d - 2.0 * r_l
    min_clearance = jnp.min(jnp.where(mask, clearance, jnp.inf))
    return min_clearance >= eps_col, min_clearance


def check_self_collisions_pair(model: DeviceModel, cp: CapsulePass, r_l, eps_col):
    """Thumb capsules vs index capsules (the reference's ``check_self_collisions``)."""
    tc = model.thumb_caps
    ic = model.index_caps
    tv = tc >= 0
    iv = ic >= 0
    ti = jnp.where(tv, tc, 0)
    ii = jnp.where(iv, ic, 0)
    a0 = cp.a[ti][:, None, :]
    b0 = cp.b[ti][:, None, :]
    a1 = cp.a[ii][None, :, :]
    b1 = cp.b[ii][None, :, :]
    d = segment_segment_dist(a0, b0, a1, b1)           # [L,L]
    clearance = d - 2.0 * r_l
    valid = tv[:, None] & iv[None, :]
    min_clearance = jnp.min(jnp.where(valid, clearance, jnp.inf))
    return min_clearance >= eps_col, min_clearance


def check_antipodal_friction(n0, n1, mu):
    """Reference ``grasp.check_antipodal_friction``. Returns ``(feasible, cos_val)``."""
    norm0 = jnp.linalg.norm(n0)
    norm1 = jnp.linalg.norm(n1)
    degenerate = (norm0 < 1e-12) | (norm1 < 1e-12)
    u0 = n0 / jnp.where(degenerate, 1.0, norm0)
    u1 = n1 / jnp.where(degenerate, 1.0, norm1)
    cos_val = jnp.dot(-u0, u1)
    threshold = jnp.cos(2.0 * jnp.arctan(mu))
    feasible = (~degenerate) & (cos_val >= threshold)
    return feasible, jnp.where(degenerate, 0.0, cos_val)
