"""Gauss-Newton projection onto the two-contact manifold (port of ``karma/projection.py``).

:func:`project_onto_contacts` closes both contact gaps for ONE state with a
fixed budget of damped Gauss-Newton iterations and the reference's
three-candidate backtracking, accumulating the sphere rotation that keeps the
contacts rolling. ``vmap`` for a batch.

Data-dependent control flow of the reference and its masked equivalent here:

* ``break`` once ``max|g| <= eps_g``: a ``converged`` flag freezes the state
  for the remaining iterations.
* backtracking "first alpha in {1, 1/2, 1/4} that reduces |g|, else 1/8":
  the three candidates are evaluated in one vmapped FK and the first
  improving one is selected with ``jnp.where``.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .model import DeviceModel
from .kinematics import (FKResult, forward_kinematics, fold_active, active_limits,
                         get_active, set_active, apply_mimic, skew)
from .geometry import CapsulePass, ContactKin, capsule_pass, capsule_gap, contact_kinematics
from .linalg import mm, mv, vm, mtm

GN_DAMPING = 1e-2      # * r_s^2, reference
DTHETA_DAMPING = 1e-4  # * r_s^2, reference
MAX_ITERS = 10
_ALPHAS = (1.0, 0.5, 0.25)
_ALPHA_FALLBACK = 0.125


class ProjectionResult(NamedTuple):
    q: object              # [J_pad] projected configuration (mimic-consistent)
    dtheta: object         # [3] accumulated rolling rotation
    ok: object             # bool: both gaps within eps_g at the end
    fk: object             # FKResult at q
    cp: object             # CapsulePass at (q, centre)
    ck_t: object           # ContactKin of pair[0]
    ck_i: object           # ContactKin of pair[1]


def compute_rolling_dtheta(model: DeviceModel, ck0: ContactKin, ck1: ContactKin, dq_active, r_s, r_l):
    """Sphere rotation that keeps both contacts rolling under joint motion ``dq`` (dp = 0)."""
    rows_A, rows_b = [], []
    for ck in (ck0, ck1):
        n = ck.normal
        Jw = fold_active(model, ck.Jw)
        p_capsule = ck.axis_point + r_l * n
        r = p_capsule - ck.origin
        Jv_surf = fold_active(model, ck.Jv_origin) - mm(skew(r), Jw)
        rows_A.append(r_s * skew(n))
        rows_b.append(mv(Jv_surf, dq_active))
    A = jnp.concatenate(rows_A, axis=0)      # [6,3]
    b = jnp.concatenate(rows_b)              # [6]
    AtA = mtm(A) + (DTHETA_DAMPING * r_s ** 2) * jnp.eye(3, dtype=A.dtype)
    return jnp.linalg.solve(AtA, vm(b, A))


def _contacts_at(model, q, centre, pair, r_s, r_l):
    fk = forward_kinematics(model, q)
    cp = capsule_pass(model, fk, centre, r_s, r_l)
    ck0 = contact_kinematics(model, fk, cp, pair[0], centre, r_s)
    ck1 = contact_kinematics(model, fk, cp, pair[1], centre, r_s)
    return fk, cp, ck0, ck1


def project_onto_contacts(
    model: DeviceModel, q, centre, pair, r_s, r_l, eps_g, max_iters: int = MAX_ITERS,
) -> ProjectionResult:
    """Port of ``karma.projection._project_onto_contacts`` for one state."""
    dtype = q.dtype
    q_lo, q_hi = active_limits(model)
    valid = model.active_valid.astype(dtype)
    M = q_lo.shape[0]
    alphas = jnp.asarray(_ALPHAS, dtype)

    def clip_active(qa):
        return jnp.clip(qa, q_lo, q_hi) * valid

    q = set_active(model, q, clip_active(get_active(model, q)))
    total_dtheta = jnp.zeros((3,), dtype)

    fk_gaps = jax.vmap(forward_kinematics, in_axes=(None, 0))

    def body(_, state):
        q, total_dtheta, converged = state
        fk, cp, ck0, ck1 = _contacts_at(model, q, centre, pair, r_s, r_l)
        g = jnp.stack([ck0.g, ck1.g])
        conv_now = jnp.max(jnp.abs(g)) <= eps_g

        # Gauss-Newton step on the two gaps: dg/dq = -n^T dp_axis/dq
        J = jnp.stack([-vm(ck0.normal, fold_active(model, ck0.Jv)),
                       -vm(ck1.normal, fold_active(model, ck1.Jv))])            # [2,M]
        JtJ = mtm(J) + (GN_DAMPING * r_s ** 2) * jnp.eye(M, dtype=dtype)
        dq = -jnp.linalg.solve(JtJ, vm(g, J)) * valid

        # Backtracking: evaluate the three candidates at once.
        qa = get_active(model, q)
        qa_try = clip_active(qa[None, :] + alphas[:, None] * dq[None, :])      # [3,M]
        q_try = jax.vmap(lambda a: set_active(model, q, a))(qa_try)           # [3,J]
        fk_try = fk_gaps(model, q_try)
        g0t = jax.vmap(lambda f: capsule_gap(model, f, pair[0], centre, r_s, r_l))(fk_try)
        g1t = jax.vmap(lambda f: capsule_gap(model, f, pair[1], centre, r_s, r_l))(fk_try)
        norms = jnp.sqrt(g0t ** 2 + g1t ** 2)
        improve = norms < jnp.linalg.norm(g)
        first = jnp.argmax(improve)                                            # first True (0 if none)
        alpha = jnp.where(jnp.any(improve), alphas[first], _ALPHA_FALLBACK)

        actual_dq = alpha * dq
        q_new = set_active(model, q, clip_active(qa + actual_dq))
        step_dtheta = compute_rolling_dtheta(model, ck0, ck1, actual_dq, r_s, r_l)

        upd = ~(converged | conv_now)
        q = jnp.where(upd, q_new, q)
        total_dtheta = jnp.where(upd, total_dtheta + step_dtheta, total_dtheta)
        return q, total_dtheta, converged | conv_now

    q, total_dtheta, _ = lax.fori_loop(0, max_iters, body, (q, total_dtheta, jnp.asarray(False)))

    fk, cp, ck0, ck1 = _contacts_at(model, q, centre, pair, r_s, r_l)
    ok = jnp.maximum(jnp.abs(ck0.g), jnp.abs(ck1.g)) <= eps_g
    return ProjectionResult(apply_mimic(model, q), total_dtheta, ok, fk, cp, ck0, ck1)
