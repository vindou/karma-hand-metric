"""Exact validation of rollouts with the reference step kernels (CLAUDE.md rule 2).

The smooth rollout earns coverage through surrogate weights; whether a
trajectory is *actually* feasible is decided here by ``karma_jax.steps``, the
oracle-tested port of the reference's ``_attempt_primitive_step`` (translation,
``n_sub`` sub-steps then the full gate set) and of the Phase-2 rotation step.
These kernels re-select the contact pair by nearest link with hysteresis, so a
smooth rollout (fixed assignment) and the exact one can diverge where the
reference would switch links; the outcome records the final pair so callers
can tell those cases apart.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from karma_jax.qp import QPSettings, default_settings
from karma_jax.rolling import StepParams
from karma_jax.steps import (GraspState as RefGraspState, attempt_primitive_step, rotation_step, Constraints,
                             STEP_FAIL_NAMES, ROT_FAIL_NAMES)

from ..core.morphology import HandModel
from ..core.contacts import ContactAssignment
from .rollout import GraspState


class ExactOutcome(NamedTuple):
    ok: object        # [K] bool per primitive (translation) or per step (rotation)
    code: object      # [K] int32 fail code (0 = ok); names in STEP_FAIL_NAMES / ROT_FAIL_NAMES
    q: object         # [K+1, J] states after each primitive (initial first)
    p: object         # [K+1, 3]
    R: object         # [K+1, 3, 3]
    pair: object      # [K+1, 2] contact pair the reference used
    axis_body: object # [K, 3] rotation steps only (pinch axis in the body frame), zeros otherwise


def _ref_state(state: GraspState, asg: ContactAssignment) -> RefGraspState:
    return RefGraspState(state.q, state.p, state.R, asg.caps[:2].astype(jnp.int32))


def validate_translation(hm: HandModel, state0: GraspState, dps, asg: ContactAssignment, sp: StepParams,
                         n_sub: int, qp_settings: QPSettings | None = None,
                         cons: Constraints = Constraints()) -> ExactOutcome:
    """``dps`` [H,3] sub-step displacements with ``H = K * n_sub``; each block of ``n_sub``
    is one reference primitive step (QP + projection per sub-step, gates at the end).
    Lanes stop at the first failed primitive (later entries repeat the last state)."""
    if qp_settings is None:
        qp_settings = default_settings(state0.q.dtype)
    H = dps.shape[0]
    K = H // n_sub
    blocks = dps[: K * n_sub].reshape(K, n_sub, 3).sum(axis=1)       # delta_p_full per primitive

    def body(carry, dp_full):
        st, alive = carry
        out = attempt_primitive_step(hm.dm, st, dp_full, n_sub, sp, qp_settings, cons)
        ok = alive & out.ok
        st_new = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), out.state, st)
        code = jnp.where(alive, out.code, -1)
        return (st_new, ok), (ok, code, st_new, jnp.zeros((3,), st.q.dtype))

    st0 = _ref_state(state0, asg)
    (_, _), (ok, code, states, axis) = jax.lax.scan(body, (st0, jnp.asarray(True)), blocks)
    q = jnp.concatenate([st0.q[None], states.q])
    p = jnp.concatenate([st0.p[None], states.p])
    R = jnp.concatenate([st0.R[None], states.R])
    pair = jnp.concatenate([st0.pair[None], states.pair])
    return ExactOutcome(ok, code, q, p, R, pair, axis)


def validate_rotation(hm: HandModel, state0: GraspState, dthetas, asg: ContactAssignment, sp: StepParams,
                      qp_settings: QPSettings | None = None, cons: Constraints = Constraints()) -> ExactOutcome:
    """``dthetas`` [H,3] per-step rotation commands (the reference uses 0.15 rad tilts)."""
    if qp_settings is None:
        qp_settings = default_settings(state0.q.dtype)

    def body(carry, dth):
        st, alive = carry
        out = rotation_step(hm.dm, st, dth, sp, qp_settings, cons)
        ok = alive & out.ok
        st_new = jax.tree_util.tree_map(lambda a, b: jnp.where(ok, a, b), out.state, st)
        code = jnp.where(alive, out.code, -1)
        return (st_new, ok), (ok, code, st_new, out.axis_body)

    st0 = _ref_state(state0, asg)
    (_, _), (ok, code, states, axis) = jax.lax.scan(body, (st0, jnp.asarray(True)), dthetas)
    q = jnp.concatenate([st0.q[None], states.q])
    p = jnp.concatenate([st0.p[None], states.p])
    R = jnp.concatenate([st0.R[None], states.R])
    pair = jnp.concatenate([st0.pair[None], states.pair])
    return ExactOutcome(ok, code, q, p, R, pair, axis)


def fail_names(kind: str) -> tuple[str, ...]:
    return STEP_FAIL_NAMES if kind == "translation" else ROT_FAIL_NAMES


__all__ = ["ExactOutcome", "validate_translation", "validate_rotation", "fail_names"]
