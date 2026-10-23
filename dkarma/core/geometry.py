"""Capsule/sphere geometry primitives for a :class:`~dkarma.core.morphology.HandModel`,
with per-capsule radii and smooth aggregates (SPEC.md §3.3, §4.3).

Exact quantities reuse ``karma_jax.geometry`` (closest points, gaps, normals,
contact kinematics, segment distances). This module adds the per-capsule radius,
masked hard/soft minima and maxima, the softplus hinge used by the smooth
residuals, and the capsule-capsule clearance table that the self-collision gate
and its smooth version are built from (:mod:`dkarma.core.contacts`).

Sharpness convention: ``softmin(x, beta) = -(1/beta) * logsumexp(-beta * x)``
with ``beta`` in ``1 / units(x)`` (``1/m`` for clearances), so ``softmin <= min``
and the gap closes like ``log(n) / beta``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from karma_jax.geometry import (CapsulePass, ContactKin, capsule_pass as _capsule_pass,
                                contact_kinematics as _contact_kinematics, segment_segment_dist,
                                check_antipodal_friction)
from karma_jax.kinematics import FKResult

from .morphology import HandModel


# ── scalar helpers ───────────────────────────────────────────────────────────

def safe_norm(v, eps: float = 1e-12, axis=-1):
    """``(norm, is_small)``: the norm with a finite gradient at zero (reported as ``is_small``)."""
    ssq = jnp.sum(v * v, axis=axis)
    small = ssq < eps * eps
    return jnp.sqrt(jnp.where(small, eps * eps, ssq)), small


def hard_min(x, mask=None):
    if mask is not None:
        x = jnp.where(mask, x, jnp.inf)
    return jnp.min(x)


def hard_min_clearance(clearance, mask):
    """Minimum over a masked table (any shape)."""
    return hard_min(clearance, mask)


def softmin(x, beta, mask=None, axis=None):
    """``-(1/beta) logsumexp(-beta x)`` over the (masked) entries; -> min as beta -> inf."""
    if mask is not None:
        x = jnp.where(mask, x, jnp.inf)
    return -jax.scipy.special.logsumexp(-beta * x, axis=axis) / beta


def softmax(x, beta, mask=None, axis=None):
    if mask is not None:
        x = jnp.where(mask, x, -jnp.inf)
    return jax.scipy.special.logsumexp(beta * x, axis=axis) / beta


def hinge(x, eps):
    """Smooth one-sided penalty ``eps * softplus(x / eps)`` (≈ max(x, 0) for |x| >> eps)."""
    return jax.nn.softplus(x / eps) * eps


# ── capsules vs the sphere ───────────────────────────────────────────────────

def capsule_pass(hm: HandModel, fkr: FKResult, centre, r_s) -> CapsulePass:
    """Closest point, normal and signed gap ``dist - r_s - cap_radius[c]`` of every capsule."""
    return _capsule_pass(hm.dm, fkr, centre, r_s, hm.cap_radius)


def gap_of(cp: CapsulePass, c):
    return cp.gap[c]


def contact_kinematics(hm: HandModel, fkr: FKResult, cp: CapsulePass, c, centre, r_s) -> ContactKin:
    """Contact kinematics of capsule ``c`` (normal, closest-point Jacobian, sphere point)."""
    return _contact_kinematics(hm.dm, fkr, cp, c, centre, r_s)


# ── capsules vs capsules ─────────────────────────────────────────────────────

def capsule_pair_clearance(hm: HandModel, cp: CapsulePass):
    """``[C, C]`` clearance ``dist - r_i - r_j`` between every pair of capsules of *different*
    fingers, and the mask of pairs to consider (upper triangle, valid, different fingers)."""
    C = cp.a.shape[0]
    i = jnp.arange(C)
    mask = (i[:, None] < i[None, :]) & (hm.cap_finger[:, None] != hm.cap_finger[None, :])
    mask = mask & hm.dm.cap_valid[:, None] & hm.dm.cap_valid[None, :]
    # Masked pairs (a capsule with itself or its neighbour) have zero distance, where the norm's
    # gradient is NaN and a later ``where`` cannot remove it; move those second segments far away
    # so every entry is well defined. Valid pairs are untouched.
    shift = jnp.where(mask, 0.0, 10.0)[..., None]
    a1 = cp.a[None, :, :] + shift
    b1 = cp.b[None, :, :] + shift
    d = segment_segment_dist(jnp.broadcast_to(cp.a[:, None, :], (C, C, 3)),
                             jnp.broadcast_to(cp.b[:, None, :], (C, C, 3)), a1, b1)
    clr = d - hm.cap_radius[:, None] - hm.cap_radius[None, :]
    return clr, mask


def finger_pair_clearance(hm: HandModel, cp: CapsulePass, f_a, f_b):
    """``[L, L]`` clearances between the capsules of fingers ``f_a`` and ``f_b`` and their mask."""
    ca, cb = hm.finger_caps[f_a], hm.finger_caps[f_b]
    va, vb = ca >= 0, cb >= 0
    ia, ib = jnp.where(va, ca, 0), jnp.where(vb, cb, 0)
    d = segment_segment_dist(cp.a[ia][:, None, :], cp.b[ia][:, None, :], cp.a[ib][None, :, :], cp.b[ib][None, :, :])
    return d - hm.cap_radius[ia][:, None] - hm.cap_radius[ib][None, :], va[:, None] & vb[None, :]


def antipodal(n0, n1, mu):
    """Reference two-contact friction-cone check: ``(feasible, cos)``."""
    return check_antipodal_friction(n0, n1, mu)


__all__ = ["CapsulePass", "ContactKin", "safe_norm", "hard_min", "hard_min_clearance", "softmin", "softmax",
           "hinge", "capsule_pass", "gap_of", "contact_kinematics", "capsule_pair_clearance",
           "finger_pair_clearance", "antipodal", "segment_segment_dist"]


def exp_so3_safe(omega):
    """``karma_jax.steps.exp_so3`` (rotation vector -> matrix, identity below the reference's 1e-10
    angle guard) with a **finite derivative at omega = 0**. The reference takes the norm of ``omega``
    first, whose gradient at exactly zero is NaN; the guard ``where(small, I, R)`` removes the value but
    not that NaN (Phase 9.1, docs/phase9_report.md §9.1: a failed rotation QP zeroes the joint step, the
    achieved rotation is exactly zero, and the NaN reached the morphology gradient through the contact
    Jacobians). Here the guard is decided on ``stop_gradient(omega)`` and the norm is taken of
    ``where(small, 1, omega)``, so the forward value is **bit-identical** to the reference everywhere
    (same operations on the same inputs where ``|omega| >= 1e-10``, the identity below) and the
    derivative equals the reference's where ``|omega| >= 1e-10`` and is zero below."""
    from karma_jax.kinematics import rot_axis_angle
    small = jnp.linalg.norm(jax.lax.stop_gradient(omega)) < 1e-10
    w = jnp.where(small, jnp.ones_like(omega), omega)
    angle = jnp.linalg.norm(w)
    axis = w / jnp.where(small, 1.0, angle)
    R = rot_axis_angle(axis, angle)
    return jnp.where(small, jnp.eye(3, dtype=omega.dtype), R)
