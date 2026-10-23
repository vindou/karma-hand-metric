"""Gauss-Newton projection onto the ``Cn``-contact manifold (fixed assignment), the
masked-lane generalization of ``karma_jax.projection.project_onto_contacts``.

For two valid lanes the iterates, backtracking choices and accumulated rolling
rotation equal the reference port to roundoff (``dkarma/tests/test_projection.py``);
with ``backtrack=False`` the step is always the full damped Gauss-Newton step,
which removes the discrete alpha selection from the smooth path. Everything is a
fixed-trip ``fori_loop`` with ``where`` masks, so it is reverse-differentiable.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from karma_jax.kinematics import forward_kinematics, fold_active, skew
from karma_jax.linalg import mm, mv, vm, mtm
from karma_jax.projection import GN_DAMPING, DTHETA_DAMPING, MAX_ITERS, _ALPHAS, _ALPHA_FALLBACK
from karma_jax.rolling import StepParams

from ..core.morphology import HandModel
from ..core.contacts import ContactAssignment, ContactSet, contact_pass
from ..core import geometry as geo


class ProjectionResult(NamedTuple):
    q: object          # [J]
    dtheta: object     # [3] accumulated rolling rotation
    ok: object         # bool: all valid gaps within eps_g
    fk: object         # FKResult at q
    cp: object         # CapsulePass at (q, centre)
    cs: object         # ContactSet of the assignment at (q, centre)
    gap_max: object    # max |gap| over valid lanes at the end


def rolling_dtheta(cs: ContactSet, valid, dq, r_s):
    """Sphere rotation that keeps the valid contacts rolling under ``dq`` with the centre fixed."""
    vf = valid.astype(dq.dtype)
    A = (jax.vmap(skew)(cs.normal) * r_s * vf[:, None, None]).reshape(-1, 3)   # [3Cn,3]
    b = (jnp.einsum("caj,j->ca", cs.J_surf, dq) * vf[:, None]).reshape(-1)
    AtA = mtm(A) + (DTHETA_DAMPING * r_s ** 2) * jnp.eye(3, dtype=dq.dtype)
    return jnp.linalg.solve(AtA, vm(b, A))


def project(hm: HandModel, q, centre, asg: ContactAssignment, sp: StepParams, *,
            max_iters: int = MAX_ITERS, backtrack: bool = True) -> ProjectionResult:
    dtype = q.dtype
    r_s = sp.sphere_radius
    q_lo, q_hi = hm.dm.q_lower, hm.dm.q_upper
    J = q.shape[0]
    valid = asg.valid
    vf = valid.astype(dtype)
    alphas = jnp.asarray(_ALPHAS, dtype)

    def contacts(qq):
        fkr = forward_kinematics(hm.dm, qq)
        cp = geo.capsule_pass(hm, fkr, centre, r_s)
        cs = contact_pass(hm, fkr, centre, r_s, asg)
        return fkr, cp, cs

    def gap_norm(qq):
        fkr = forward_kinematics(hm.dm, qq)
        cp = geo.capsule_pass(hm, fkr, centre, r_s)
        g = cp.gap[jnp.where(valid, asg.caps, 0)] * vf
        return jnp.linalg.norm(g)

    q = jnp.clip(q, q_lo, q_hi)
    total = jnp.zeros((3,), dtype)

    def body(_, state):
        q, total, converged = state
        fkr, cp, cs = contacts(q)
        g = cs.gap * vf
        conv_now = jnp.max(jnp.abs(g)) <= sp.eps_g
        Jg = -jnp.einsum("ca,caj->cj", cs.normal, jax.vmap(lambda Jv: fold_active(hm.dm, Jv))(cs.Jv)) * vf[:, None]
        JtJ = mtm(Jg) + (GN_DAMPING * r_s ** 2) * jnp.eye(J, dtype=dtype)
        dq = -jnp.linalg.solve(JtJ, vm(g, Jg))
        if backtrack:
            q_try = jnp.clip(q[None, :] + alphas[:, None] * dq[None, :], q_lo, q_hi)
            norms = jax.vmap(gap_norm)(q_try)
            improve = norms < jnp.linalg.norm(g)
            first = jnp.argmax(improve)
            alpha = jnp.where(jnp.any(improve), alphas[first], _ALPHA_FALLBACK)
        else:
            alpha = jnp.asarray(1.0, dtype)
        actual = alpha * dq
        q_new = jnp.clip(q + actual, q_lo, q_hi)
        step_dtheta = rolling_dtheta(cs, valid, actual, r_s)
        upd = ~(converged | conv_now)
        return (jnp.where(upd, q_new, q), jnp.where(upd, total + step_dtheta, total), converged | conv_now)

    q, total, _ = lax.fori_loop(0, max_iters, body, (q, total, jnp.asarray(False)))
    fkr, cp, cs = contacts(q)
    gap_max = jnp.max(jnp.abs(cs.gap * vf))
    return ProjectionResult(q, total, gap_max <= sp.eps_g, fkr, cp, cs, gap_max)


__all__ = ["ProjectionResult", "project", "rolling_dtheta"]
