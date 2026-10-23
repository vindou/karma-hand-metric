"""Batched dense QP solver for the rolling-contact steps.

Solves, for ONE problem (``vmap`` for a batch),

    minimize   1/2 x'Px + c'x
    subject to l <= Ax <= u

with ``P`` symmetric positive definite and all bounds finite. This is the form
the reference hands to OSQP (``karma/rolling_qp.py``). The problems are tiny
(``n <= 13`` variables, ``A = [A_gap; I]`` with ``n + 2`` rows), so a dense
primal-dual interior-point method with a fixed iteration budget is the right
tool: every lane does identical work, convergence to high accuracy takes ~10
to 20 iterations regardless of conditioning, and the Newton system is an
``n x n`` Cholesky solve.

Implementation notes
--------------------
* Two-sided bounds become ``Gx + s = h, s >= 0`` with ``G = [A; -A]``,
  ``h = [u; -l]``.
* Mehrotra predictor-corrector with fraction-to-boundary 0.99, starting from
  ``x = 0, s = z = 1``. Infeasible starts are fine.
* The Newton step uses the reduced normal equations
  ``(P + G' diag(z/s) G) dx = rhs``; see :func:`_newton`.
* A one-shot Jacobi equilibration ``x = E x'`` with ``E = diag(P_ii^-1/2)``
  gives ``P'`` a unit diagonal. The rolling QP has a ``1e5`` penalty block on
  the rotation variables, so without this the Newton matrix would be
  ill-conditioned in fp32.
* Lanes stop updating once converged (``done`` mask), which keeps fp32 from
  drifting after convergence. ``success`` requires the final residuals to be
  within ``infeas_tol``; a primal-infeasible problem (gap band vs joint box)
  stalls with a large primal residual and is reported as a failure, matching
  the reference's treatment of OSQP's non-solved statuses.
* Padded variables (``row_mask`` / ``var_mask``): the rolling QPs carry inert
  variables for padded active-joint lanes (zero cost, unit curvature, a
  ``[-1, 1]`` box). They decouple from the real variables in the Newton system
  but would still enter the complementarity average ``mu``, the step-length
  ratio tests and the relative tolerances, shifting the central path by a
  tolerance-sized amount. The masks remove them from those scalars, so the
  solution of a padded problem equals the unpadded one up to the summation
  order of the dense factorization. With no mask (or an all-true mask) the
  arithmetic is identical to the unmasked solver.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.linalg import cho_factor, cho_solve

from .linalg import mm, mv, vm, mtm


class QPResult(NamedTuple):
    x: object          # [n]
    success: object    # bool
    prim_res: object   # max |Gx + s - h|
    dual_res: object   # max |Px + c + G'z|
    mu: object         # final complementarity
    iters: object      # iterations taken (int32)
    status: object     # int32: 0 converged, 1 iteration budget exhausted, 2 stopped on a non-finite step


class QPSettings(NamedTuple):
    iters: int = 30
    tol: float = 1e-8          # convergence on prim/dual residual and mu
    infeas_tol: float = 1e-5   # success requires residuals below this at the end
    reg: float = 1e-10         # diagonal regularization of the Newton matrix
    solve_in_fp64: bool = False  # mixed precision: cast the assembled QP to fp64 for the solve


def default_settings(dtype) -> QPSettings:
    """Precision-appropriate settings (Section 5.1 of the plan)."""
    # The objective suboptimality at exit is bounded by the duality gap, about
    # (2 * n_constraints) * tol, so tol=1e-11 gives ~1e-9 objective accuracy.
    # fp32: 1e-6 is the practical floor; 3e-7 already breaks factorizations. On
    # slip-reliant hands (Inspire) steps pass within 0.03 mm of the slip gate,
    # and 1e-6 keeps the fp32 slip error at ~0.01 mm vs 0.03 mm at 1e-5.
    if dtype == jnp.float64:
        return QPSettings(iters=30, tol=1e-11, infeas_tol=1e-6, reg=1e-11)
    # fp32 pipelines: pure fp32 QP at tol 1e-6 (the practical fp32 floor). With
    # exact fp32 products this reproduces the fp64 search results on Shadow and
    # Inspire; mixed precision (solve_in_fp64) remains available and ~1.7x slower.
    return fp32_settings()


def fp32_settings() -> QPSettings:
    """Pure-fp32 QP solve; tol 1e-6 is the practical fp32 floor."""
    return QPSettings(iters=40, tol=1e-6, infeas_tol=1e-4, reg=1e-6)


def mixed_settings() -> QPSettings:
    """QP solved in fp64 inside an fp32 pipeline (fp64 tolerances)."""
    return QPSettings(iters=30, tol=1e-11, infeas_tol=1e-6, reg=1e-11, solve_in_fp64=True)


def _step_to_boundary(v, dv, mask=None):
    """Largest alpha in [0, 1] with v + alpha dv >= 0 (over the rows in ``mask``)."""
    neg = dv < 0
    if mask is not None:
        neg = neg & mask
    ratio = jnp.where(neg, -v / jnp.where(neg, dv, -1.0), jnp.inf)
    return jnp.minimum(1.0, jnp.min(ratio))


def _masked_abs_max(r, mask):
    return jnp.max(jnp.abs(r)) if mask is None else jnp.max(jnp.where(mask, jnp.abs(r), 0.0))


def solve_box_qp(P, c, A, l, u, settings: QPSettings = QPSettings(), *, row_mask=None, var_mask=None) -> QPResult:
    """Solve one QP. Shapes: P [n,n], c [n], A [m,n], l/u [m].

    With ``settings.solve_in_fp64`` the problem is solved in fp64 and the
    result cast back to the input dtype (the QP is ~12 variables, so this is
    cheap relative to the kinematics, and it removes fp32 sensitivity of the
    slip gate and infeasibility detection on hands with many borderline steps).

    ``row_mask`` [m] / ``var_mask`` [n] (bool) mark the constraint rows and
    variables that belong to the real problem; rows and variables of padded
    lanes (see the module docstring) are left out of ``mu``, the ratio tests,
    the tolerance scales and the residual checks. ``None`` means all real.
    """
    in_dtype = P.dtype
    if settings.solve_in_fp64 and in_dtype != jnp.float64:
        P, c, A, l, u = (jnp.asarray(v, jnp.float64) for v in (P, c, A, l, u))
        res = solve_box_qp(P, c, A, l, u, settings._replace(solve_in_fp64=False),
                           row_mask=row_mask, var_mask=var_mask)
        return QPResult(jnp.asarray(res.x, in_dtype), res.success, jnp.asarray(res.prim_res, in_dtype),
                        jnp.asarray(res.dual_res, in_dtype), jnp.asarray(res.mu, in_dtype), res.iters, res.status)
    dtype = P.dtype
    n = P.shape[0]
    m2 = 2 * A.shape[0]
    if row_mask is None:
        mask2 = None
        n_m = m2

        def mu_of(s, z):
            return jnp.dot(s, z) / m2
    else:
        mask2 = jnp.concatenate([row_mask, row_mask], axis=0)      # [m2]
        n_m = jnp.sum(mask2).astype(dtype)

        def mu_of(s, z):
            # same dot (and summation order) as the unmasked case, padded terms exactly zero
            return jnp.dot(jnp.where(mask2, s, 0.0), z) / n_m

    # Jacobi equilibration on the primal variables.
    d = 1.0 / jnp.sqrt(jnp.maximum(jnp.diag(P), jnp.asarray(1e-30, dtype)))
    Ps = d[:, None] * P * d[None, :]
    cs = d * c
    As = A * d[None, :]
    G = jnp.concatenate([As, -As], axis=0)          # [m2,n]
    h = jnp.concatenate([u, -l], axis=0)            # [m2]

    x0 = jnp.zeros((n,), dtype)
    s0 = jnp.ones((m2,), dtype)
    z0 = jnp.ones((m2,), dtype)
    eye = jnp.eye(n, dtype=dtype)
    tau = jnp.asarray(0.99, dtype)

    # Relative tolerances, as OSQP's eps_abs + eps_rel * scale: a problem with a
    # cost of magnitude 1e2 cannot reach an absolute dual residual of 1e-9 and
    # would otherwise drive complementarity to zero until the Newton matrix is
    # singular.
    scale_p = 1.0 + _masked_abs_max(h, mask2)
    scale_d = 1.0 + _masked_abs_max(cs, var_mask)

    def residuals(x, s, z):
        r_d = mv(Ps, x) + cs + vm(z, G)
        r_p = mv(G, x) + s - h
        return r_d, r_p

    def body(_, state):
        x, s, z, done, iters, status = state
        r_d, r_p = residuals(x, s, z)
        mu = mu_of(s, z)
        D = z / s
        H = Ps + mm(G.T * D[None, :], G) + settings.reg * eye
        L = cho_factor(H, lower=True)

        def newton(r_c):
            rhs = -r_d + vm((r_c - z * r_p) / s, G)
            dx = cho_solve(L, rhs)
            ds = -r_p - mv(G, dx)
            dz = (-r_c - z * ds) / s
            return dx, ds, dz

        # predictor
        dx_a, ds_a, dz_a = newton(s * z)
        a_p = _step_to_boundary(s, ds_a, mask2)
        a_d = _step_to_boundary(z, dz_a, mask2)
        mu_aff = mu_of(s + a_p * ds_a, z + a_d * dz_a)
        sigma = (mu_aff / mu) ** 3
        # corrector
        r_c = s * z + ds_a * dz_a - sigma * mu
        dx, ds, dz = newton(r_c)
        a_p = jnp.minimum(1.0, tau * _step_to_boundary(s, ds, mask2))
        a_d = jnp.minimum(1.0, tau * _step_to_boundary(z, dz, mask2))

        x_n = x + a_p * dx
        s_n = s + a_p * ds
        z_n = z + a_d * dz

        conv = (_masked_abs_max(r_p, mask2) <= settings.tol * scale_p) & \
               (_masked_abs_max(r_d, var_mask) <= settings.tol * scale_d) & (mu <= settings.tol)
        # A failed factorization (non-PD Newton matrix) yields NaNs; never apply
        # such a step. The lane keeps its last finite iterate and stops.
        finite_step = jnp.all(jnp.isfinite(x_n)) & jnp.all(jnp.isfinite(s_n)) & jnp.all(jnp.isfinite(z_n))
        upd = ~(done | conv) & finite_step
        status = jnp.where(~done & conv, 0, jnp.where(~done & ~finite_step, 2, status))
        done = done | conv | ~finite_step
        x = jnp.where(upd, x_n, x)
        s = jnp.where(upd, s_n, s)
        z = jnp.where(upd, z_n, z)
        iters = iters + upd.astype(jnp.int32)
        return x, s, z, done, iters, status

    state = (x0, s0, z0, jnp.asarray(False), jnp.asarray(0, jnp.int32), jnp.asarray(1, jnp.int32))
    x, s, z, done, iters, status = lax.fori_loop(0, settings.iters, body, state)
    r_d, r_p = residuals(x, s, z)
    prim = _masked_abs_max(r_p, mask2)
    dual = _masked_abs_max(r_d, var_mask)
    mu = mu_of(s, z)
    success = (prim <= settings.infeas_tol * scale_p) & (dual <= settings.infeas_tol * scale_d)
    return QPResult(d * x, success, prim, dual, mu, iters, status)


solve_box_qp_batched = jax.vmap(solve_box_qp, in_axes=(0, 0, 0, 0, 0, None))
