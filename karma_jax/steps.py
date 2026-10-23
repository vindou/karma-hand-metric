"""One translation primitive step and one rotation step, with all validations.

Ports of ``karma.seed_selection._attempt_primitive_step`` (Phase 1) and the loop
body of ``karma.metric._explore_rotations_at_voxel`` (Phase 2) for ONE lane;
``vmap`` for a batch. The reference's early returns become an ``alive`` mask
and an integer fail code, so every lane does fixed-shape work and the
``fail_reasons`` tallies survive.

Fail codes (translation): see :data:`STEP_FAIL_NAMES`; (rotation): see
:data:`ROT_FAIL_NAMES`. ``0`` is success in both.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .model import DeviceModel
from .kinematics import (forward_kinematics, active_limits, get_active, set_active,
                         apply_mimic, rot_axis_angle)
from .geometry import (capsule_pass, contact_kinematics, select_contact_pair,
                       check_collisions, check_self_collisions, check_antipodal_friction)
from .projection import project_onto_contacts
from .rolling import StepParams, solve_rolling_step, solve_rotation_step
from .qp import QPSettings

STEP_FAIL_NAMES = ("ok", "qp_fail", "projection", "gap_post_proj", "collision",
                   "self_collision", "grasp", "joint_limits")
ROT_FAIL_NAMES = ("ok", "rot_qp_fail", "rot_low_eff", "rot_slip", "rot_projection", "rot_gap",
                  "rot_collision", "rot_self_collision", "rot_grasp", "rot_joint_limits")

ROTATION_EFFICIENCY_MIN = 0.3   # reference guard
JOINT_LIMIT_SLACK = 1e-6        # reference tolerance on the final joint-limit check


class Constraints(NamedTuple):
    """Static ablation switches (``karma.ablation``). Joint limits are ablated by
    widening the model's limits, not here."""
    collisions: bool = True
    antipodal: bool = True


class GraspState(NamedTuple):
    q: object       # [J_pad] full configuration
    p: object       # [3] sphere centre (world)
    R: object       # [3,3] sphere body orientation
    pair: object    # [2] int32 capsule indices (thumb, index)


class StepOutcome(NamedTuple):
    ok: object          # bool
    code: object        # int32 fail code (0 = ok)
    state: object       # GraspState after the step (undefined where not ok)
    n_t: object         # [3] thumb contact normal at the final state
    n_i: object         # [3] index contact normal at the final state


class RotationOutcome(NamedTuple):
    ok: object
    code: object
    state: object
    axis_body: object   # [3] pinch axis in the sphere body frame (for binning)
    efficiency: object
    slip_mm: object


def exp_so3(omega):
    """Rotation vector -> matrix, identity below the reference's 1e-10 angle guard."""
    angle = jnp.linalg.norm(omega)
    small = angle < 1e-10
    axis = omega / jnp.where(small, 1.0, angle)
    R = rot_axis_angle(axis, angle)
    return jnp.where(small, jnp.eye(3, dtype=omega.dtype), R)


def reorthogonalize(R):
    """Project onto SO(3) via SVD with det(+1), as the reference."""
    U, _, Vt = jnp.linalg.svd(R)
    Rc = U @ Vt
    neg = jnp.linalg.det(Rc) < 0
    U = jnp.where(neg, U.at[:, -1].multiply(-1.0), U)
    return U @ Vt


def pinch_axis_body(n_thumb, R):
    """Reference ``contact_line_axis_body``: unit(R^T @ unit(-n_thumb))."""
    axis_w = -n_thumb
    axis_w = axis_w / jnp.maximum(jnp.linalg.norm(axis_w), 1e-12)
    a = R.T @ axis_w
    return a / jnp.maximum(jnp.linalg.norm(a), 1e-12)


def _validate(model, pr, params: StepParams, cons: Constraints, codes):
    """Shared post-projection validation. ``pr`` is a ProjectionResult at the
    final state; ``codes`` maps names to this phase's fail codes. Returns a code
    (0 = pass) following the reference order."""
    q_lo, q_hi = active_limits(model)
    gap_bad = jnp.maximum(jnp.abs(pr.ck_t.g), jnp.abs(pr.ck_i.g)) > params.eps_g
    pair = jnp.stack([pr.ck_t.cap, pr.ck_i.cap])
    if cons.collisions:
        col_ok, _ = check_collisions(model, pr.cp, pair, params.eps_col)
        self_ok, _ = check_self_collisions(model, pr.cp, params.link_radius, params.eps_col)
    else:
        col_ok = self_ok = jnp.asarray(True)
    if cons.antipodal:
        grasp_ok, _ = check_antipodal_friction(pr.ck_t.normal, pr.ck_i.normal, params.friction_coeff)
    else:
        grasp_ok = jnp.asarray(True)
    qa = get_active(model, pr.q)
    valid = model.active_valid
    jl_bad = jnp.any(valid & ((qa < q_lo - JOINT_LIMIT_SLACK) | (qa > q_hi + JOINT_LIMIT_SLACK)))
    code = jnp.where(gap_bad, codes["gap"],
           jnp.where(~col_ok, codes["collision"],
           jnp.where(~self_ok, codes["self_collision"],
           jnp.where(~grasp_ok, codes["grasp"],
           jnp.where(jl_bad, codes["joint_limits"], 0)))))
    return code


_STEP_CODES = dict(gap=3, collision=4, self_collision=5, grasp=6, joint_limits=7)
_ROT_CODES = dict(gap=5, collision=6, self_collision=7, grasp=8, joint_limits=9)


def attempt_primitive_step(
    model: DeviceModel, state: GraspState, delta_p_full, n_sub: int,
    params: StepParams, qp_settings: QPSettings, cons: Constraints = Constraints(),
) -> StepOutcome:
    """Roll the sphere by ``delta_p_full`` in ``n_sub`` sub-steps, project, validate."""
    r_s, r_l = params.sphere_radius, params.link_radius
    q_lo, q_hi = active_limits(model)
    sub_dp = delta_p_full / n_sub
    hyst = 0.5 * params.eps_g

    def sub_step(_, carry):
        st, alive, code, _pr = carry
        fk = forward_kinematics(model, st.q)
        cp = capsule_pass(model, fk, st.p, r_s, r_l)
        ck_t = contact_kinematics(model, fk, cp, st.pair[0], st.p, r_s)
        ck_i = contact_kinematics(model, fk, cp, st.pair[1], st.p, r_s)
        qa = get_active(model, st.q)
        roll = solve_rolling_step(model, ck_t, ck_i, sub_dp, qa, q_lo, q_hi, params, qp_settings)

        q1 = set_active(model, st.q, qa + roll.dq_active)
        p1 = st.p + sub_dp
        R1 = exp_so3(roll.dtheta) @ st.R
        fk1 = forward_kinematics(model, q1)
        cp1 = capsule_pass(model, fk1, p1, r_s, r_l)
        pair1 = select_contact_pair(model, cp1, st.pair, hyst)
        pr = project_onto_contacts(model, q1, p1, pair1, r_s, r_l, params.eps_g)
        R1 = exp_so3(pr.dtheta) @ R1

        new_code = jnp.where(~roll.success, 1, jnp.where(~pr.ok, 2, 0))
        step_ok = alive & (new_code == 0)
        st_new = GraspState(pr.q, p1, R1, pair1)
        st = jax.tree_util.tree_map(lambda a, b: jnp.where(step_ok, a, b), st_new, st)
        code = jnp.where(alive & ~step_ok, new_code, code)
        pr_keep = jax.tree_util.tree_map(lambda a, b: jnp.where(step_ok, a, b), pr, _pr)
        return st, step_ok, code, pr_keep

    # A dummy projection result to seed the carry (overwritten on the first live sub-step).
    pr0 = project_onto_contacts(model, state.q, state.p, state.pair, r_s, r_l, params.eps_g, max_iters=0)
    st, alive, code, pr = lax.fori_loop(
        0, n_sub, sub_step, (state, jnp.asarray(True), jnp.asarray(0, jnp.int32), pr0))

    vcode = _validate(model, pr, params, cons, _STEP_CODES)
    code = jnp.where(alive, vcode, code)
    ok = alive & (vcode == 0)
    return StepOutcome(ok, code, st, pr.ck_t.normal, pr.ck_i.normal)


def rotation_step(
    model: DeviceModel, state: GraspState, dtheta_dir,
    params: StepParams, qp_settings: QPSettings, cons: Constraints = Constraints(),
) -> RotationOutcome:
    """One Phase-2 tilt step (body of the reference's per-direction loop)."""
    r_s, r_l = params.sphere_radius, params.link_radius
    q_lo, q_hi = active_limits(model)
    hyst = 0.5 * params.eps_g

    fk = forward_kinematics(model, state.q)
    cp = capsule_pass(model, fk, state.p, r_s, r_l)
    ck_t = contact_kinematics(model, fk, cp, state.pair[0], state.p, r_s)
    ck_i = contact_kinematics(model, fk, cp, state.pair[1], state.p, r_s)
    qa = get_active(model, state.q)
    rot = solve_rotation_step(model, ck_t, ck_i, dtheta_dir, qa, q_lo, q_hi, params, qp_settings)

    slip_bad = rot.tangential_slip_mm > params.slip_threshold * 1000.0
    pre_code = jnp.where(~rot.success, 1,
               jnp.where(rot.efficiency < ROTATION_EFFICIENCY_MIN, 2,
               jnp.where(slip_bad, 3, 0)))

    q1 = set_active(model, state.q, qa + rot.dq_active)
    R1 = exp_so3(rot.dtheta_achieved) @ state.R
    fk1 = forward_kinematics(model, q1)
    cp1 = capsule_pass(model, fk1, state.p, r_s, r_l)
    pair1 = select_contact_pair(model, cp1, state.pair, hyst)
    pr = project_onto_contacts(model, q1, state.p, pair1, r_s, r_l, params.eps_g)
    R1 = exp_so3(pr.dtheta) @ R1

    vcode = _validate(model, pr, params, cons, _ROT_CODES)
    code = jnp.where(pre_code != 0, pre_code, jnp.where(~pr.ok, 4, vcode))
    ok = code == 0
    new_state = GraspState(pr.q, state.p, R1, pair1)
    axis_body = pinch_axis_body(pr.ck_t.normal, R1)
    return RotationOutcome(ok, code, new_state, axis_body, rot.efficiency, rot.tangential_slip_mm)


def make_batched_step(model: DeviceModel, n_sub: int, params: StepParams,
                      qp_settings: QPSettings, cons: Constraints = Constraints()):
    """``jit``-compiled batched translation step: (GraspState[B], delta_p[B,3]) -> StepOutcome[B]."""
    def one(state, dp):
        return attempt_primitive_step(model, state, dp, n_sub, params, qp_settings, cons)
    return jax.jit(jax.vmap(one))


def make_batched_rotation_step(model: DeviceModel, params: StepParams,
                               qp_settings: QPSettings, cons: Constraints = Constraints()):
    """``jit``-compiled batched rotation step: (GraspState[B], dtheta[B,3]) -> RotationOutcome[B]."""
    def one(state, dth):
        return rotation_step(model, state, dth, params, qp_settings, cons)
    return jax.jit(jax.vmap(one))
