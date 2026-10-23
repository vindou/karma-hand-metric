"""The rolling-contact step problem for ``Cn`` masked contacts, with two solvers:
a soft closed-form solve and the reference's hard QP with KKT-based derivatives
(SPEC.md §4.2).

Problem (non-dimensionalized by the sphere radius ``r_s`` exactly as
``karma/rolling_qp.py`` and ``karma_jax/rolling.py``): for contact lane ``c`` with
normal ``n_c``, surface Jacobian ``J_surf,c`` (velocity of the finger material point
under the contact) and closest-point Jacobian ``Jv_c``,

    rolling rows   (J_surf,c dq - r_s [n_c]x dtheta) / r_s = dp / r_s          (3 per lane)
    gap band       (g_c + n_c.dp - eps_g)/r_s <= n_c.Jv_c dq / r_s <= (g_c + n_c.dp + eps_g)/r_s
    boxes          dq in [max(-dq_max, lo - q), min(dq_max, hi - q)],  dtheta in [-0.2, 0.2]
    objective      |rows|^2 + reg/r_s^2 |dq|^2 + lambda_theta |dtheta - theta_c|^2

``theta_c`` is the pure-rolling rotation of the commanded translation plus the
commanded extra rotation. With ``free_theta=False`` the rotation is not a variable
but fixed to the command (the reference's *rotation* QP: ``J_surf dq = r_s [n]x
dtheta_cmd``). Masked lanes contribute zero rows and trivially feasible bounds. For
two lanes (thumb, index) the assembled matrices equal
``karma_jax.rolling.build_rolling_qp`` / ``build_rotation_qp``
(``dkarma/tests/test_rolling_qp.py``).

Solvers
-------
* :func:`solve_soft`: minimizes the objective without the gap band and the boxes
  (one Cholesky solve), then clips ``dq`` to the joint box. Differentiable
  everywhere; infeasibilities show up in the step residuals, not in the solve.
* :func:`solve_hard`: the reference's interior-point QP (``karma_jax.qp``) for the
  value, and for the derivative the solution of the equality-constrained QP on
  the identified active set, written as one fixed-shape KKT linear system
  (inactive rows carry the equation ``lambda_i = 0``). The returned ``x`` has the
  QP's value and the KKT solution's gradient (``x_qp + (x_kkt - stop_grad(x_kkt))``);
  ``kkt_gap = |x_kkt - x_qp|`` reports how well the active set was identified.
  Away from active-set changes this is the exact derivative of the QP solution
  map; at a change the solution is only directionally differentiable and the
  gradient belongs to one side (SPEC §4.2: validate near such points).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_factor, cho_solve

from karma_jax.kinematics import fold_active, skew
from karma_jax.linalg import mm, mv, vm, mtm
from karma_jax.qp import QPSettings, solve_box_qp
from karma_jax.rolling import StepParams, compute_rolling_rotation, LAMBDA_THETA, THETA_BOUND

from ..core.morphology import HandModel
from ..core.contacts import ContactSet, ContactAssignment


class RollingProblem(NamedTuple):
    P: object          # [n,n]
    c: object          # [n]
    A: object          # [Cn+n, n]  gap rows then identity
    l: object          # [Cn+n]
    u: object          # [Cn+n]
    M_roll: object     # [3Cn, n]
    b_roll: object     # [3Cn]
    theta_c: object    # [3] rotation prior centre (free_theta) or the fixed rotation
    normals: object    # [Cn,3]
    valid: object      # bool [Cn]
    row_mask: object   # bool [Cn+n]
    n_dq: int          # static
    free_theta: bool   # static


class StepSolution(NamedTuple):
    dq: object             # [J]
    dtheta: object         # [3] (== theta_c when not free)
    residual: object       # [3Cn] rolling residual M x - b (non-dimensional)
    slip_m: object         # max over lanes of the tangential residual, in metres
    spin_err: object       # |dtheta - theta_c| (free) or 0
    clip_m: object         # how much dq was clipped to the joint box (rad, max)
    qp_ok: object          # bool (hard) / True (soft)
    kkt_gap: object        # |x_kkt - x_qp| (hard) / 0


def assemble(hm: HandModel, cs: ContactSet, asg: ContactAssignment, dp, dtheta_cmd, q,
             sp: StepParams, *, free_theta: bool = True, lambda_theta=LAMBDA_THETA,
             theta_bound=THETA_BOUND) -> RollingProblem:
    """Build the step problem at state ``(q, p)`` with contacts ``cs`` for the command
    ``(dp, dtheta_cmd)`` (object displacement and extra rotation, world frame)."""
    r_s = sp.sphere_radius
    dtype = jnp.asarray(dp).dtype
    J = hm.dm.q_ids.shape[0]
    Cn = cs.gap.shape[0]
    valid = asg.valid
    vf = valid.astype(dtype)

    # pure-rolling rotation of the translation, from the first valid lane's normal (thumb)
    n0 = cs.normal[0]
    theta_roll = compute_rolling_rotation(n0, dp, r_s)
    theta_c = theta_roll + dtheta_cmd

    J_surf = cs.J_surf * vf[:, None, None]                               # [Cn,3,J]
    Jv_f = jax.vmap(lambda Jv: fold_active(hm.dm, Jv))(cs.Jv) * vf[:, None, None]
    normals = cs.normal * vf[:, None]
    nskew = jax.vmap(skew)(normals)                                      # [Cn,3,3]

    if free_theta:
        rows = jnp.concatenate([J_surf, -r_s * nskew], axis=2)           # [Cn,3,J+3]
        b = jnp.broadcast_to(dp, (Cn, 3)) * vf[:, None]
        n = J + 3
    else:
        rows = J_surf                                                    # [Cn,3,J]
        b = (dp[None, :] + r_s * jnp.einsum("cab,b->ca", nskew, dtheta_cmd)) * vf[:, None]
        n = J
    M_roll = rows.reshape(3 * Cn, n) / r_s
    b_roll = b.reshape(3 * Cn) / r_s

    gap_rows = jnp.einsum("ca,caj->cj", normals, Jv_f)                   # [Cn,J]
    if free_theta:
        gap_rows = jnp.concatenate([gap_rows, jnp.zeros((Cn, 3), dtype)], axis=1)
    A_gap = gap_rows / r_s
    n_dp = mv(normals, dp)
    g = cs.gap * vf
    l_gap = jnp.where(valid, (g + n_dp - sp.eps_g) / r_s, -1.0)
    u_gap = jnp.where(valid, (g + n_dp + sp.eps_g) / r_s, 1.0)

    q_lo, q_hi = hm.dm.q_lower, hm.dm.q_upper
    dq_lo = jnp.maximum(-sp.max_joint_step, q_lo - q)
    dq_hi = jnp.minimum(sp.max_joint_step, q_hi - q)
    if free_theta:
        x_lo = jnp.concatenate([dq_lo, jnp.full((3,), -theta_bound, dtype)])
        x_hi = jnp.concatenate([dq_hi, jnp.full((3,), theta_bound, dtype)])
        diag = jnp.concatenate([jnp.full((J,), sp.qp_regularization / r_s ** 2, dtype),
                                jnp.full((3,), lambda_theta, dtype)])
    else:
        x_lo, x_hi = dq_lo, dq_hi
        diag = jnp.full((J,), sp.qp_regularization / r_s ** 2, dtype)

    P = mtm(M_roll) + jnp.diag(diag)
    c = -vm(b_roll, M_roll)
    if free_theta:
        c = c.at[J:].add(-lambda_theta * theta_c)
    A = jnp.concatenate([A_gap, jnp.eye(n, dtype=dtype)], axis=0)
    l = jnp.concatenate([l_gap, x_lo])
    u = jnp.concatenate([u_gap, x_hi])
    row_mask = jnp.concatenate([valid, jnp.ones((n,), bool)])
    return RollingProblem(P, c, A, l, u, M_roll, b_roll, theta_c, normals, valid, row_mask, J, free_theta)


def _finish(prob: RollingProblem, x, qp_ok, kkt_gap, sp: StepParams) -> StepSolution:
    J = prob.n_dq
    dq_raw = x[:J]
    dtheta = x[J:] if prob.free_theta else prob.theta_c
    # the joint box of the problem (rows J.. of A are the identity): clip for safety
    lo = prob.l[-x.shape[0]:][:J]
    hi = prob.u[-x.shape[0]:][:J]
    dq = jnp.clip(dq_raw, lo, hi)
    clip_m = jnp.max(jnp.abs(dq - dq_raw))
    x_used = x.at[:J].set(dq)
    residual = mv(prob.M_roll, x_used) - prob.b_roll
    Cn = prob.normals.shape[0]
    res = residual.reshape(Cn, 3)
    tang = res - jnp.sum(res * prob.normals, axis=-1, keepdims=True) * prob.normals
    slip = jnp.max(jnp.where(prob.valid, jnp.linalg.norm(tang, axis=-1), 0.0)) * sp.sphere_radius
    spin = jnp.linalg.norm(dtheta - prob.theta_c) if prob.free_theta else jnp.zeros((), x.dtype)
    return StepSolution(dq, dtheta, residual, slip, spin, clip_m, qp_ok, kkt_gap)


def solve_soft(prob: RollingProblem, sp: StepParams) -> StepSolution:
    """Unconstrained minimizer (one Cholesky solve), then ``dq`` clipped to the joint box."""
    L = cho_factor(prob.P, lower=True)
    x = -cho_solve(L, prob.c)
    return _finish(prob, x, jnp.asarray(True), jnp.zeros((), x.dtype), sp)


def _independent_rows(Gm, m, order, rel_tol):
    """Sequential Gram-Schmidt over the active rows of ``Gm`` (rows of ``G`` pre-masked by ``m``)
    in the processing order ``order``: a row is kept when its residual after projection on the
    rows kept so far exceeds ``rel_tol`` times its own norm. Returns the reduced mask (float)."""
    m2, n = Gm.shape
    dtype = Gm.dtype

    def body(carry, i):
        Q, k = carry
        g = Gm[i]
        r = g - jnp.dot(Q.T, jnp.dot(Q, g))
        r = r - jnp.dot(Q.T, jnp.dot(Q, r))                       # re-orthogonalize once
        nr = jnp.linalg.norm(r)
        ng = jnp.linalg.norm(g)
        keep = (m[i] > 0.5) & (ng > 0) & (nr > rel_tol * ng)
        Q = Q.at[k].set(jnp.where(keep, r / jnp.where(keep, nr, 1.0), 0.0))
        return (Q, k + keep.astype(jnp.int32)), keep

    (_, _), keeps = jax.lax.scan(body, (jnp.zeros((m2, n), dtype), jnp.zeros((), jnp.int32)), order)
    return jnp.zeros((m2,), dtype).at[order].set(keeps.astype(dtype))


def kkt_row_order(n_gap: int, n: int):
    """Processing order of the ``2 (n_gap + n)`` constraint rows for :func:`_independent_rows`:
    box rows first (exact identity rows, kept), then the gap rows (which can depend on them).
    Rows of ``G = [A; -A]`` with ``A = [gap; I]``: upper gap, upper box, lower gap, lower box."""
    m1 = n_gap + n
    upper_gap = list(range(0, n_gap)); upper_box = list(range(n_gap, m1))
    lower_gap = list(range(m1, m1 + n_gap)); lower_box = list(range(m1 + n_gap, 2 * m1))
    return jnp.asarray(upper_box + lower_box + upper_gap + lower_gap, jnp.int32)


KKT_METHODS = ("exact", "reg", "pinv", "reduced")


def kkt_resolve(P, c, A, l, u, x_qp, act_tol, reg=1e-10, method: str = "exact", rcond: float = 1e-10,
                dep_tol: float = 1e-6, n_gap: int | None = None):
    """Equality-constrained QP on the active set of ``x_qp`` as one KKT solve.

    Constraints ``G x <= h`` with ``G = [A; -A]``, ``h = [u; -l]``; row ``i`` is active
    when its slack is within ``act_tol * (1 + |h_i|)``. Inactive rows get ``lambda_i = 0``.
    Returns ``(x_kkt, active_mask)``; the active set is a stop-gradient constant.

    ``method`` (Phase 9.1, SPEC §11.2 item 9.1; the comparison is in
    ``docs/phase9_report.md`` §9.1):

    * ``"exact"``: the Phase 2 derivative, the KKT system with ``-reg`` (1e-10) on the active
      dual block solved by LU; singular to working precision when the active rows are dependent.
    * ``"reg"``: the same system with a larger ``reg`` (the derivative of the penalty-smoothed
      problem ``lambda = (G_a x - h_a) / reg``).
    * ``"pinv"``: the minimum-norm least-squares solution ``pinv(K) rhs`` with relative cutoff
      ``rcond`` (JAX's constant-rank pseudoinverse derivative).
    * ``"reduced"``: the active-set derivative with linearly dependent active rows removed
      (sequential Gram-Schmidt, relative tolerance ``dep_tol``, box rows kept in preference to
      gap rows; ``n_gap`` = number of gap rows of ``A``, default ``A.shape[0] - P.shape[0]``);
      the KKT matrix of the kept rows is nonsingular because ``P`` is positive definite.
    """
    G = jnp.concatenate([A, -A], axis=0)
    h = jnp.concatenate([u, -l])
    slack = h - mv(G, jax.lax.stop_gradient(x_qp))
    m = jax.lax.stop_gradient(slack <= act_tol * (1.0 + jnp.abs(h))).astype(P.dtype)
    n, m2 = P.shape[0], G.shape[0]
    if method == "reduced":
        ng = A.shape[0] - n if n_gap is None else n_gap
        m = jax.lax.stop_gradient(_independent_rows(jax.lax.stop_gradient(G) * m[:, None], m, kkt_row_order(ng, n), dep_tol))
    K = jnp.block([[P, G.T * m[None, :]],
                   [G * m[:, None], -reg * jnp.diag(m) + jnp.diag(1.0 - m)]])
    rhs = jnp.concatenate([-c, m * h])
    if method == "pinv":
        sol = mv(jnp.linalg.pinv(K, rtol=rcond), rhs)
    elif method in ("exact", "reg", "reduced"):
        sol = jnp.linalg.solve(K, rhs)
    else:
        raise ValueError(f"unknown KKT derivative method {method!r}; choose from {KKT_METHODS}")
    return sol[:n], m


def solve_hard(prob: RollingProblem, sp: StepParams, settings: QPSettings, act_tol=None, kkt_method: str = "exact",
               kkt_reg: float = 1e-10, kkt_dep_tol: float = 1e-6, kkt_rcond: float = 1e-10) -> StepSolution:
    """Reference QP for the value, KKT re-solve for the gradient (see module docstring and
    :func:`kkt_resolve` for ``kkt_method``)."""
    dtype = prob.P.dtype
    if act_tol is None:
        act_tol = 1e-7 if dtype == jnp.float64 else 1e-4
    sg = jax.lax.stop_gradient
    res = solve_box_qp(sg(prob.P), sg(prob.c), sg(prob.A), sg(prob.l), sg(prob.u), settings,
                       row_mask=prob.row_mask)
    x_qp = res.x
    x_kkt, _ = kkt_resolve(prob.P, prob.c, prob.A, prob.l, prob.u, x_qp, act_tol, reg=kkt_reg, method=kkt_method,
                           rcond=kkt_rcond, dep_tol=kkt_dep_tol, n_gap=prob.normals.shape[0])
    kkt_gap = jnp.max(jnp.abs(sg(x_kkt) - x_qp))
    x = x_qp + (x_kkt - sg(x_kkt))
    x = jnp.where(res.success, x, jnp.zeros_like(x))
    sol = _finish(prob, x, res.success, kkt_gap, sp)
    return sol


def dtheta_achieved(prob: RollingProblem, dq, r_s):
    """Rotation implied by the achieved contact motion (reference ``solve_rotation_step``):
    damped least squares of ``r_s [n_c]x dtheta = J_surf,c dq`` over the valid lanes."""
    Cn = prob.normals.shape[0]
    J = prob.n_dq
    M = prob.M_roll[:, :J] * r_s                               # J_surf rows (dimensional)
    achieved = mv(M, dq).reshape(Cn, 3)                        # [Cn,3]
    A = jax.vmap(skew)(prob.normals).reshape(3 * Cn, 3) * r_s
    AtA = mtm(A) + 1e-4 * r_s ** 2 * jnp.eye(3, dtype=dq.dtype)
    return jnp.linalg.solve(AtA, vm(achieved.reshape(-1), A))


__all__ = ["RollingProblem", "StepSolution", "assemble", "solve_soft", "solve_hard", "kkt_resolve", "KKT_METHODS",
           "kkt_row_order", "dtheta_achieved"]
