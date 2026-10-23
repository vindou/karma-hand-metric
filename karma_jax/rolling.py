"""Rolling-contact translation and rotation steps (ports of ``karma/rolling_qp.py``).

Both functions take the two :class:`~karma_jax.geometry.ContactKin` objects
(thumb, index) of ONE state and return array-valued results; ``vmap`` for a
batch. The QP build follows the reference line by line, including the
non-dimensionalization by the sphere radius, the ``lambda_theta = 1e5`` prior,
the ``0.2 rad`` rotation-variable bound and the slip/spin decomposition of the
residual. The QP itself is solved by :mod:`karma_jax.qp` instead of OSQP.

Padding: the active-joint axis is padded to ``M_pad``. Padding columns of the
rolling map are zero; to keep the QP well posed they get unit curvature, zero
cost and a ``[-1, 1]`` box, so their solution is exactly zero and they decouple
from the real variables in the Newton system. They are also masked out of the
solver's scalars (``mu``, step ratios, tolerance scales) through
``row_mask``/``var_mask``, so a padded problem reproduces the unpadded solution
(``karma_jax/tests/test_padding.py``).
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp

from .geometry import ContactKin
from .kinematics import DeviceModel, fold_active, skew
from .qp import QPSettings, solve_box_qp
from .linalg import mm, mv, vm, mtm

LAMBDA_THETA = 10.0 / (0.010 ** 2)   # == 1e5, as the reference
THETA_BOUND = 0.2                    # rad, per-step bound on the sphere-rotation variable


class RollingStepResult(NamedTuple):
    success: object            # bool: QP solved and slip within threshold
    qp_ok: object              # bool: QP solved
    slip_ok: object            # bool: tangential slip within threshold
    dq_active: object          # [M_pad]
    dtheta: object             # [3]
    roll_residual: object      # max |M x - b| (non-dimensional)
    tangential_slip_mm: object # mm
    spin_slip: object          # |dtheta - ideal_theta|


class RotationStepResult(NamedTuple):
    success: object            # bool: QP solved
    dq_active: object          # [M_pad]
    dtheta_achieved: object    # [3]
    roll_residual: object
    efficiency: object         # in [0,1]
    tangential_slip_mm: object
    spin_achieved: object


class StepParams(NamedTuple):
    """Scalar parameters shared by both steps (from MetricConfig)."""
    sphere_radius: object
    link_radius: object
    eps_g: object
    max_joint_step: object     # rad
    qp_regularization: object
    slip_threshold: object     # metres; use jnp.inf for "allow slip"
    friction_coeff: object
    eps_col: object


def compute_rolling_rotation(n_thumb, delta_p, r_s):
    """Geometric pure-rolling rotation for a translation (reference formula)."""
    pinch = n_thumb / jnp.linalg.norm(n_thumb)
    dp_mag = jnp.linalg.norm(delta_p)
    dp_perp = delta_p - jnp.dot(delta_p, pinch) * pinch
    perp_mag = jnp.linalg.norm(dp_perp)
    ok1 = (dp_mag >= 1e-12) & (perp_mag >= 1e-12)
    perp_dir = dp_perp / jnp.where(ok1, perp_mag, 1.0)
    rot_axis = jnp.cross(pinch, perp_dir)
    ra_norm = jnp.linalg.norm(rot_axis)
    ok = ok1 & (ra_norm >= 1e-12)
    rot_axis = rot_axis / jnp.where(ok, ra_norm, 1.0)
    theta = perp_mag / r_s
    return jnp.where(ok, theta * rot_axis, 0.0)


def _surface_jacobian(model: DeviceModel, ck: ContactKin, r_l):
    """``Jv_surf = Jv_origin - skew(r) @ Jw`` on the active joints, ``[3, M_pad]``."""
    p_capsule = ck.axis_point + r_l * ck.normal
    r = p_capsule - ck.origin
    Jv_o = fold_active(model, ck.Jv_origin)
    Jw = fold_active(model, ck.Jw)
    return Jv_o - mm(skew(r), Jw)


def _dq_bounds(model: DeviceModel, q_active, q_lo, q_hi, max_step):
    dq_lo = jnp.maximum(-max_step, q_lo - q_active)
    dq_hi = jnp.minimum(max_step, q_hi - q_active)
    valid = model.active_valid
    return jnp.where(valid, dq_lo, -1.0), jnp.where(valid, dq_hi, 1.0)


def _tangential_slip(residual, normals):
    """Max over contacts of the residual component perpendicular to the normal."""
    res = residual.reshape(2, 3)
    normal_comp = jnp.sum(res * normals, axis=-1, keepdims=True) * normals
    tang = res - normal_comp
    return jnp.max(jnp.linalg.norm(tang, axis=-1))


class RollingQP(NamedTuple):
    """The assembled (non-dimensional) rolling QP, exposed for oracle tests."""
    P: object
    c: object
    A: object
    l: object
    u: object
    M_roll: object
    b_roll: object
    ideal_theta: object
    normals: object


def build_rolling_qp(
    model: DeviceModel, ck_t: ContactKin, ck_i: ContactKin, delta_p,
    q_active, q_lo, q_hi, params: StepParams,
) -> RollingQP:
    r_s = params.sphere_radius
    M_pad = q_active.shape[0]
    n = M_pad + 3
    dtype = delta_p.dtype

    normals = jnp.stack([ck_t.normal, ck_i.normal])                 # [2,3]
    ideal_theta = compute_rolling_rotation(ck_t.normal, delta_p, r_s)

    rows = []
    gap_rows = []
    for ck in (ck_t, ck_i):
        Jv_surf = _surface_jacobian(model, ck, params.link_radius)   # [3,M]
        rows.append(jnp.concatenate([Jv_surf, -r_s * skew(ck.normal)], axis=1))
        gap_rows.append(jnp.concatenate([vm(ck.normal, fold_active(model, ck.Jv)), jnp.zeros((3,), dtype)]))
    M_roll = jnp.concatenate(rows, axis=0) / r_s                     # [6,n]
    b_roll = jnp.concatenate([delta_p, delta_p]) / r_s               # [6]
    A_gap = jnp.stack(gap_rows) / r_s                                # [2,n]
    nT_dp = mv(normals, delta_p)                                     # [2]
    g = jnp.stack([ck_t.g, ck_i.g])
    l_gap = (g + nT_dp - params.eps_g) / r_s
    u_gap = (g + nT_dp + params.eps_g) / r_s

    dq_lo, dq_hi = _dq_bounds(model, q_active, q_lo, q_hi, params.max_joint_step)
    x_lo = jnp.concatenate([dq_lo, jnp.full((3,), -THETA_BOUND, dtype)])
    x_hi = jnp.concatenate([dq_hi, jnp.full((3,), THETA_BOUND, dtype)])

    P = mtm(M_roll)
    diag = jnp.concatenate([jnp.where(model.active_valid, params.qp_regularization / r_s ** 2, 1.0),
                            jnp.full((3,), LAMBDA_THETA, dtype)])
    P = P + jnp.diag(diag)
    c = -vm(b_roll, M_roll)
    c = c.at[M_pad:].add(-LAMBDA_THETA * ideal_theta)

    A = jnp.concatenate([A_gap, jnp.eye(n, dtype=dtype)], axis=0)
    l = jnp.concatenate([l_gap, x_lo])
    u = jnp.concatenate([u_gap, x_hi])
    return RollingQP(P, c, A, l, u, M_roll, b_roll, ideal_theta, normals)


def solve_rolling_step(
    model: DeviceModel, ck_t: ContactKin, ck_i: ContactKin, delta_p,
    q_active, q_lo, q_hi, params: StepParams, qp_settings: QPSettings,
) -> RollingStepResult:
    """Translation sub-step: port of ``karma.rolling_qp.solve_rolling_step``."""
    r_s = params.sphere_radius
    M_pad = q_active.shape[0]
    valid = model.active_valid.astype(delta_p.dtype)
    qp = build_rolling_qp(model, ck_t, ck_i, delta_p, q_active, q_lo, q_hi, params)
    M_roll, b_roll, ideal_theta, normals = qp.M_roll, qp.b_roll, qp.ideal_theta, qp.normals

    var_mask = jnp.concatenate([model.active_valid, jnp.ones((3,), bool)])        # [M_pad+3]
    row_mask = jnp.concatenate([jnp.ones((2,), bool), var_mask])                   # [A_gap; I]
    res = solve_box_qp(qp.P, qp.c, qp.A, qp.l, qp.u, qp_settings, row_mask=row_mask, var_mask=var_mask)
    x = res.x
    dq = x[:M_pad] * valid
    dtheta = x[M_pad:]

    residual = mv(M_roll, x) - b_roll
    roll_res = jnp.max(jnp.abs(residual))
    max_tang = _tangential_slip(residual, normals)
    slip_mm = max_tang * r_s * 1000.0
    spin_slip = jnp.linalg.norm(dtheta - ideal_theta)
    slip_ok = max_tang * r_s <= params.slip_threshold
    qp_ok = res.success
    success = qp_ok & slip_ok
    zero_dq = jnp.zeros_like(dq)
    return RollingStepResult(
        success=success, qp_ok=qp_ok, slip_ok=slip_ok,
        dq_active=jnp.where(qp_ok, dq, zero_dq),
        dtheta=jnp.where(qp_ok, dtheta, 0.0),
        roll_residual=jnp.where(qp_ok, roll_res, jnp.inf),
        tangential_slip_mm=jnp.where(qp_ok, slip_mm, 0.0),
        spin_slip=jnp.where(qp_ok, spin_slip, 0.0),
    )


class RotationQP(NamedTuple):
    P: object
    c: object
    A: object
    l: object
    u: object
    M_roll: object
    b_roll: object
    normals: object


def build_rotation_qp(
    model: DeviceModel, ck_t: ContactKin, ck_i: ContactKin, dtheta_target,
    q_active, q_lo, q_hi, params: StepParams,
) -> RotationQP:
    r_s = params.sphere_radius
    M_pad = q_active.shape[0]
    dtype = dtheta_target.dtype
    normals = jnp.stack([ck_t.normal, ck_i.normal])

    rows, b_rows, gap_rows = [], [], []
    for ck in (ck_t, ck_i):
        rows.append(_surface_jacobian(model, ck, params.link_radius))
        b_rows.append(r_s * mv(skew(ck.normal), dtheta_target))
        gap_rows.append(vm(ck.normal, fold_active(model, ck.Jv)))
    M_roll = jnp.concatenate(rows, axis=0) / r_s                     # [6,M]
    b_roll = jnp.concatenate(b_rows) / r_s
    A_gap = jnp.stack(gap_rows) / r_s                                # [2,M]
    g = jnp.stack([ck_t.g, ck_i.g])
    l_gap = (g - params.eps_g) / r_s
    u_gap = (g + params.eps_g) / r_s

    dq_lo, dq_hi = _dq_bounds(model, q_active, q_lo, q_hi, params.max_joint_step)

    P = mtm(M_roll) + jnp.diag(jnp.where(model.active_valid, params.qp_regularization / r_s ** 2, 1.0))
    c = -vm(b_roll, M_roll)
    A = jnp.concatenate([A_gap, jnp.eye(M_pad, dtype=dtype)], axis=0)
    l = jnp.concatenate([l_gap, dq_lo])
    u = jnp.concatenate([u_gap, dq_hi])
    return RotationQP(P, c, A, l, u, M_roll, b_roll, normals)


def solve_rotation_step(
    model: DeviceModel, ck_t: ContactKin, ck_i: ContactKin, dtheta_target,
    q_active, q_lo, q_hi, params: StepParams, qp_settings: QPSettings,
) -> RotationStepResult:
    """Pure rotation step: port of ``karma.rolling_qp.solve_rotation_step``."""
    r_s = params.sphere_radius
    dtype = dtheta_target.dtype
    valid = model.active_valid.astype(dtype)
    qp = build_rotation_qp(model, ck_t, ck_i, dtheta_target, q_active, q_lo, q_hi, params)
    M_roll, b_roll, normals = qp.M_roll, qp.b_roll, qp.normals

    var_mask = model.active_valid                                                  # [M_pad]
    row_mask = jnp.concatenate([jnp.ones((2,), bool), var_mask])                   # [A_gap; I]
    res = solve_box_qp(qp.P, qp.c, qp.A, qp.l, qp.u, qp_settings, row_mask=row_mask, var_mask=var_mask)
    dq = res.x * valid

    achieved = mv(M_roll, dq)
    residual = achieved - b_roll
    roll_res = jnp.max(jnp.abs(residual))

    A_theta = jnp.concatenate([skew(ck_t.normal), skew(ck_i.normal)], axis=0)   # [6,3]
    AtA = mtm(A_theta) + 1e-4 * jnp.eye(3, dtype=dtype)
    dtheta_achieved = jnp.linalg.solve(AtA, vm(achieved, A_theta))

    target_norm = jnp.linalg.norm(dtheta_target)
    has_target = target_norm > 1e-10
    alignment = jnp.dot(dtheta_target, dtheta_achieved) / jnp.where(has_target, target_norm ** 2, 1.0)
    efficiency = jnp.where(has_target, jnp.clip(alignment, 0.0, 1.0), 0.0)

    max_tang = _tangential_slip(residual, normals)
    slip_mm = max_tang * r_s * 1000.0
    spin = jnp.sum(jnp.abs(mv(normals, dtheta_achieved)))

    ok = res.success
    return RotationStepResult(
        success=ok,
        dq_active=jnp.where(ok, dq, 0.0),
        dtheta_achieved=jnp.where(ok, dtheta_achieved, 0.0),
        roll_residual=jnp.where(ok, roll_res, jnp.inf),
        efficiency=jnp.where(ok, efficiency, 0.0),
        tangential_slip_mm=jnp.where(ok, slip_mm, jnp.inf),
        spin_achieved=jnp.where(ok, spin, 0.0),
    )
