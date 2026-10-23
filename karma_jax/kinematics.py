"""Batched forward kinematics and analytic frame Jacobians.

All functions here take a :class:`~karma_jax.model.DeviceModel` and operate on
ONE configuration; batch them with ``jax.vmap`` (see :func:`fk_batched`). They
are pure ``jax.numpy`` and work under ``jit`` in fp32 or fp64.

Conventions match Pinocchio (see :mod:`karma_jax.model`):

* ``q`` is the full configuration ``[J_pad]`` (not just the active joints).
  Mimic couplings are applied inside :func:`forward_kinematics` via the affine
  map stored in the model, which is what ``karma.robot.update_kinematics`` does
  in place.
* Frame Jacobians are ``LOCAL_WORLD_ALIGNED``: the linear part is the world
  velocity of the frame origin, the angular part the world angular velocity.
  For a revolute joint ``j`` with world axis ``w_j`` and origin ``o_j`` moving
  frame ``f`` at ``p_f``, column ``j`` is ``[w_j x (p_f - o_j); w_j]``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .model import DeviceModel


class FKResult(NamedTuple):
    R_joint: object   # [J,3,3] world rotation of each joint frame (includes q rotation)
    o_joint: object   # [J,3]   world origin of each joint
    w_axis: object    # [J,3]   world revolute axis of each joint
    R_frame: object   # [F,3,3] world rotation of each frame
    p_frame: object   # [F,3]   world position of each frame


def rot_axis_angle(axis, theta):
    """Rodrigues: rotation of ``theta`` about unit ``axis``. Shapes broadcast."""
    x, y, z = axis[..., 0], axis[..., 1], axis[..., 2]
    c = jnp.cos(theta)
    s = jnp.sin(theta)
    C = 1.0 - c
    R = jnp.stack([
        jnp.stack([c + x * x * C, x * y * C - z * s, x * z * C + y * s], axis=-1),
        jnp.stack([y * x * C + z * s, c + y * y * C, y * z * C - x * s], axis=-1),
        jnp.stack([z * x * C - y * s, z * y * C + x * s, c + z * z * C], axis=-1),
    ], axis=-2)
    return R


def skew(v):
    """[...,3] -> [...,3,3] skew-symmetric matrix."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = jnp.zeros_like(x)
    return jnp.stack([
        jnp.stack([o, -z, y], axis=-1),
        jnp.stack([z, o, -x], axis=-1),
        jnp.stack([-y, x, o], axis=-1),
    ], axis=-2)


def apply_mimic(model: DeviceModel, q):
    """``q[target] = mult * q[source] + offset`` for every mimic joint.

    Written as one gather over the per-joint tables, ``q_c[j] = jm_mult[j] *
    q[jm_source[j]] + jm_offset[j]`` with ``(source j, mult 1, offset 0)`` for
    joints that are not mimics (``1.0 * x + 0.0 == x`` exactly). Two reasons
    not to use a scatter or a dense ``A_mimic @ q`` here:

    * ``A_mimic @ q`` runs in TF32 on GPUs and perturbs ``q`` by ~1e-3.
    * ``q.at[mimic_target].set(...)`` with the model closed over as a constant
      is miscompiled by XLA (jax 0.7.1) whenever the mimic table has as many
      (padded, duplicate) entries as ``q``: the scatter is rewritten into a
      gather and every joint receives ``q[0]``. That was the "J_pad ==
      n_joints breaks FK" defect on the optimizer's 8-joint hands (K_pad == 8).
    """
    return model.jm_mult * jnp.take(q, model.jm_source) + model.jm_offset


def forward_kinematics(model: DeviceModel, q) -> FKResult:
    """FK for one configuration ``q`` [J_pad]. Mimic couplings are applied here."""
    qc = apply_mimic(model, q)
    J = model.parent.shape[0]
    dtype = model.t_joint.dtype
    eye = jnp.eye(3, dtype=dtype)

    def body(j, carry):
        R_all, o_all = carry
        par = model.parent[j]
        has_par = par >= 0
        par_i = jnp.where(has_par, par, 0)
        R_p = jnp.where(has_par, R_all[par_i], eye)
        o_p = jnp.where(has_par, o_all[par_i], jnp.zeros(3, dtype))
        o_j = o_p + R_p @ model.t_joint[j]
        R_pj = R_p @ model.R_joint[j]
        R_j = R_pj @ rot_axis_angle(model.axis[j], qc[j])
        return R_all.at[j].set(R_j), o_all.at[j].set(o_j)

    R0 = jnp.broadcast_to(eye, (J, 3, 3))
    o0 = jnp.zeros((J, 3), dtype)
    R_all, o_all = lax.fori_loop(0, J, body, (R0, o0))

    # World axis: rotation about the axis commutes with it, so R_j @ a == R_pj @ a.
    w_axis = jnp.einsum("jab,jb->ja", R_all, model.axis)

    # Frames hang off joints (or the universe).
    fp = model.frame_parent
    has_fp = fp >= 0
    fp_i = jnp.where(has_fp, fp, 0)
    R_fp = jnp.where(has_fp[:, None, None], R_all[fp_i], eye)
    o_fp = jnp.where(has_fp[:, None], o_all[fp_i], 0.0)
    R_frame = jnp.einsum("fab,fbc->fac", R_fp, model.R_frame)
    p_frame = o_fp + jnp.einsum("fab,fb->fa", R_fp, model.t_frame)

    return FKResult(R_all, o_all, w_axis, R_frame, p_frame)


def frame_jacobian(model: DeviceModel, fk: FKResult, f):
    """``LOCAL_WORLD_ALIGNED`` Jacobian of frame ``f`` (scalar index, may be traced).

    Returns ``(Jv, Jw)`` each ``[3, J_pad]`` over the FULL configuration (no
    mimic folding; see :func:`fold_active`).
    """
    p_f = fk.p_frame[f]                                   # [3]
    mask = model.anc[f].astype(fk.w_axis.dtype)           # [J]
    lever = p_f[None, :] - fk.o_joint                     # [J,3]
    lin = jnp.cross(fk.w_axis, lever) * mask[:, None]     # [J,3]
    ang = fk.w_axis * mask[:, None]                       # [J,3]
    return lin.T, ang.T


def fold_active(model: DeviceModel, J):
    """Fold mimic columns into their sources and select the active joints.

    ``J`` is ``[..., J_pad]`` over the full configuration. Returns
    ``[..., M_pad]``; padding lanes are zero. Equals
    ``karma.robot.fold_mimic_jacobian(J, v_ids, ctx)`` including its order of
    additions: ``J_eff[:, s] = (J[:, s] + m_1 J[:, t_1]) + m_2 J[:, t_2] ...``
    over the mimic list in order, so hands whose mimics share a source (Inspire,
    SVH) fold bit-identically to the reference. Gather-only (see apply_mimic);
    the static loop over the small mimic table fuses into a few elementwise ops.
    """
    dtype = J.dtype
    J_act = jnp.take(J, model.q_ids, axis=-1) * model.active_valid.astype(dtype)   # [...,M]
    for k in range(model.mimic_target.shape[0]):
        valid_k = model.mimic_valid[k]
        contrib = jnp.take(J, model.mimic_target[k], axis=-1) * (model.mimic_mult[k] * valid_k)  # [...]
        sel = (model.q_ids == model.mimic_source[k]) & valid_k & model.active_valid          # [M]
        J_act = J_act + jnp.where(sel, contrib[..., None], jnp.zeros((), dtype))
    return J_act


def active_limits(model: DeviceModel):
    """Lower/upper position limits of the active joints, ``[M_pad]`` each."""
    lo = jnp.take(model.q_lower, model.q_ids)
    hi = jnp.take(model.q_upper, model.q_ids)
    return lo, hi


def set_active(model: DeviceModel, q, q_active):
    """Write active joint values ``q_active`` [M_pad] into full ``q`` [J_pad].

    Gather form, ``q[j] = q_active[active_lane[j]]`` where ``is_active[j]``:
    padded lanes of ``q_active`` are never read. The former scatter
    ``q.at[q_ids].set(...)`` sent every padded lane (``q_ids == 0``) to joint 0
    together with the real lane-0 update, and XLA does not define which of the
    duplicate writes wins; that was the "M_pad > n_active corrupts the
    projection" defect.
    """
    return jnp.where(model.is_active, jnp.take(q_active, model.active_lane), q)


def get_active(model: DeviceModel, q):
    return jnp.take(q, model.q_ids) * model.active_valid.astype(q.dtype)


fk_batched = jax.vmap(forward_kinematics, in_axes=(None, 0))
