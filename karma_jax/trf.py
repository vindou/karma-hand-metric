"""Batched port of scipy's bounded Trust Region Reflective least-squares solver.

This follows ``scipy.optimize._lsq.trf.trf_bounds`` with ``tr_solver='exact'``,
``x_scale=1``, linear loss, including ``select_step`` (constrained
trust-region step vs reflected step vs constrained Cauchy step), Moré's
secular iteration for the trust-region subproblem on an SVD of the augmented
Jacobian, scipy's trust-region radius update and its termination tests. The
reference seed IK calls scipy with exactly these settings, so running this
port on the same residual and Jacobian in fp64 reproduces its iterates up to
roundoff (SVD sign conventions do not matter: the step depends on ``V (U^T f)``).

Differences from scipy, all control-flow shape only:

* scipy's inner "retry with a smaller radius" loop is flattened: every
  iteration is one trial step; a rejected trial just shrinks ``Delta`` and the
  next iteration re-solves the subproblem from the unchanged Jacobian (scipy
  reuses the SVD; we recompute it, same numbers).
* All branches are ``jnp.where``; ``lax.while_loop`` runs until the lane
  terminates or ``max_nfev`` function evaluations are spent, and under
  ``vmap`` until every lane has.

``fun_jac(x) -> (f, J)`` must be a pure JAX function.
"""
from __future__ import annotations

from typing import NamedTuple, Callable

import jax
import jax.numpy as jnp
from jax import lax


class TRFResult(NamedTuple):
    x: object
    cost: object
    nfev: object
    status: object   # scipy codes: 0 max_nfev, 1 gtol, 2 ftol, 3 xtol, 4 both


# ── helpers (ports of scipy.optimize._lsq.common) ─────────────────────────────

def _find_active_constraints(x, lb, ub, rtol):
    if rtol == 0:
        return jnp.where(x <= lb, -1, jnp.where(x >= ub, 1, 0))
    lower_dist = x - lb
    upper_dist = ub - x
    lower_threshold = rtol * jnp.maximum(1.0, jnp.abs(lb))
    upper_threshold = rtol * jnp.maximum(1.0, jnp.abs(ub))
    lower_active = jnp.isfinite(lb) & (lower_dist <= jnp.minimum(upper_dist, lower_threshold))
    upper_active = jnp.isfinite(ub) & (upper_dist <= jnp.minimum(lower_dist, upper_threshold))
    return jnp.where(upper_active, 1, jnp.where(lower_active, -1, 0))


def make_strictly_feasible(x, lb, ub, rstep):
    active = _find_active_constraints(x, lb, ub, rstep)
    if rstep == 0:
        x_lo = jnp.nextafter(lb, ub)
        x_hi = jnp.nextafter(ub, lb)
    else:
        x_lo = lb + rstep * jnp.maximum(1.0, jnp.abs(lb))
        x_hi = ub - rstep * jnp.maximum(1.0, jnp.abs(ub))
    x_new = jnp.where(active == -1, x_lo, jnp.where(active == 1, x_hi, x))
    tight = (x_new < lb) | (x_new > ub)
    return jnp.where(tight, 0.5 * (lb + ub), x_new)


def cl_scaling_vector(x, g, lb, ub):
    v = jnp.ones_like(x)
    dv = jnp.zeros_like(x)
    m_hi = (g < 0) & jnp.isfinite(ub)
    v = jnp.where(m_hi, ub - x, v)
    dv = jnp.where(m_hi, -1.0, dv)
    m_lo = (g > 0) & jnp.isfinite(lb)
    v = jnp.where(m_lo, x - lb, v)
    dv = jnp.where(m_lo, 1.0, dv)
    return v, dv


def step_size_to_bound(x, s, lb, ub):
    nz = s != 0
    s_safe = jnp.where(nz, s, 1.0)
    steps = jnp.where(nz, jnp.maximum((lb - x) / s_safe, (ub - x) / s_safe), jnp.inf)
    min_step = jnp.min(steps)
    hits = jnp.where(steps == min_step, jnp.sign(s), 0.0)
    return min_step, hits


def intersect_trust_region_pos(x, s, Delta):
    """Positive root of ||x + s t|| = Delta (scipy returns both; we need the larger)."""
    a = jnp.dot(s, s)
    b = jnp.dot(x, s)
    c = jnp.dot(x, x) - Delta ** 2
    a_safe = jnp.where(a > 0, a, 1.0)
    d = jnp.sqrt(jnp.maximum(b * b - a_safe * c, 0.0))
    q = -(b + jnp.copysign(d, b))
    q_safe = jnp.where(q != 0, q, 1.0)
    t1 = q / a_safe
    t2 = jnp.where(q != 0, c / q_safe, jnp.inf)
    t_pos = jnp.maximum(t1, t2)
    return jnp.where(a > 0, t_pos, jnp.inf)


def build_quadratic_1d(J, g, s, diag, s0=None):
    v = J @ s
    a = 0.5 * (jnp.dot(v, v) + jnp.dot(s * diag, s))
    b = jnp.dot(g, s)
    if s0 is None:
        return a, b
    u = J @ s0
    b = b + jnp.dot(u, v) + jnp.dot(s0 * diag, s)
    c = 0.5 * jnp.dot(u, u) + jnp.dot(g, s0) + 0.5 * jnp.dot(s0 * diag, s0)
    return a, b, c


def minimize_quadratic_1d(a, b, lb, ub, c=0.0):
    ext = -0.5 * b / jnp.where(a != 0, a, 1.0)
    ext_ok = (a != 0) & (lb < ext) & (ext < ub)
    t = jnp.stack([lb, ub, jnp.where(ext_ok, ext, lb)])
    y = t * (a * t + b) + c
    y = y.at[2].set(jnp.where(ext_ok, y[2], jnp.inf))
    i = jnp.argmin(y)
    return t[i], y[i]


def evaluate_quadratic(J, g, s, diag):
    Js = J @ s
    return 0.5 * (jnp.dot(Js, Js) + jnp.dot(s * diag, s)) + jnp.dot(s, g)


def update_tr_radius(Delta, actual, predicted, step_norm, bound_hit):
    ratio = jnp.where(predicted > 0, actual / jnp.where(predicted > 0, predicted, 1.0),
                      jnp.where((predicted == 0) & (actual == 0), 1.0, 0.0))
    Delta_new = jnp.where(ratio < 0.25, 0.25 * step_norm,
                          jnp.where((ratio > 0.75) & bound_hit, Delta * 2.0, Delta))
    return Delta_new, ratio


def check_termination(dF, F, dx_norm, x_norm, ratio, ftol, xtol):
    f_sat = (dF < ftol * F) & (ratio > 0.25)
    x_sat = dx_norm < xtol * (xtol + x_norm)
    return jnp.where(f_sat & x_sat, 4, jnp.where(f_sat, 2, jnp.where(x_sat, 3, -1)))


def solve_lsq_trust_region(n, m, uf, s, V, Delta, initial_alpha, eps, rtol=0.01, max_iter=10):
    """Moré's method on an SVD (scipy ``solve_lsq_trust_region``); ``m`` is the residual count."""
    suf = s * uf
    full_rank = (m >= n) & (s[-1] > eps * m * s[0])
    s_safe = jnp.where(s > 0, s, 1.0)
    p_gn = -(V @ (uf / s_safe))
    gn_ok = full_rank & (jnp.linalg.norm(p_gn) <= Delta)

    def phi_and_derivative(alpha):
        denom = s ** 2 + alpha
        p_norm = jnp.linalg.norm(suf / denom)
        phi = p_norm - Delta
        phi_prime = -jnp.sum(suf ** 2 / denom ** 3) / jnp.where(p_norm > 0, p_norm, 1.0)
        return phi, phi_prime

    alpha_upper = jnp.linalg.norm(suf) / Delta
    phi0, dphi0 = phi_and_derivative(0.0)
    alpha_lower = jnp.where(full_rank, -phi0 / jnp.where(dphi0 != 0, dphi0, -1.0), 0.0)
    alpha = jnp.where(~full_rank & (initial_alpha == 0),
                      jnp.maximum(0.001 * alpha_upper, jnp.sqrt(alpha_lower * alpha_upper)), initial_alpha)

    def body(_, carry):
        alpha, alpha_lower, alpha_upper, done = carry
        a = jnp.where((alpha < alpha_lower) | (alpha > alpha_upper),
                      jnp.maximum(0.001 * alpha_upper, jnp.sqrt(alpha_lower * alpha_upper)), alpha)
        phi, phi_prime = phi_and_derivative(a)
        au = jnp.where(phi < 0, a, alpha_upper)
        ratio = phi / jnp.where(phi_prime != 0, phi_prime, -1.0)
        al = jnp.maximum(alpha_lower, a - ratio)
        a_new = a - (phi + Delta) * ratio / Delta
        conv = jnp.abs(phi) < rtol * Delta
        upd = ~done
        return (jnp.where(upd, a_new, alpha), jnp.where(upd, al, alpha_lower),
                jnp.where(upd, au, alpha_upper), done | conv)

    alpha, _, _, _ = lax.fori_loop(0, max_iter, body, (alpha, alpha_lower, alpha_upper, jnp.asarray(False)))
    p = -(V @ (suf / (s ** 2 + alpha)))
    pn = jnp.linalg.norm(p)
    p = p * (Delta / jnp.where(pn > 0, pn, 1.0))
    return jnp.where(gn_ok, p_gn, p), jnp.where(gn_ok, 0.0, alpha)


def select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta):
    """scipy ``select_step``: best of trust-region, reflected and Cauchy steps."""
    inb = jnp.all((x + p >= lb) & (x + p <= ub))
    p_value_in = evaluate_quadratic(J_h, g_h, p_h, diag_h)

    p_stride, hits = step_size_to_bound(x, p, lb, ub)
    r_h0 = jnp.where(hits != 0, -p_h, p_h)
    r0 = d * r_h0
    pb = p * p_stride
    pb_h = p_h * p_stride
    x_on_bound = x + pb
    to_tr = intersect_trust_region_pos(pb_h, r_h0, Delta)
    to_bound, _ = step_size_to_bound(x_on_bound, r0, lb, ub)
    r_stride = jnp.minimum(to_bound, to_tr)
    pos = r_stride > 0
    r_l = jnp.where(pos, (1 - theta) * p_stride / jnp.where(pos, r_stride, 1.0), 0.0)
    r_u = jnp.where(pos, jnp.where(r_stride == to_bound, theta * to_bound, to_tr), -1.0)
    a, b, c = build_quadratic_1d(J_h, g_h, r_h0, diag_h, s0=pb_h)
    r_t, r_val = minimize_quadratic_1d(a, b, r_l, r_u, c)
    valid_r = r_l <= r_u
    r_h = r_h0 * r_t + pb_h
    r = r_h * d
    r_value = jnp.where(valid_r, r_val, jnp.inf)

    pt = pb * theta
    pt_h = pb_h * theta
    p_value = evaluate_quadratic(J_h, g_h, pt_h, diag_h)

    ag_h = -g_h
    ag = d * ag_h
    ag_norm = jnp.linalg.norm(ag_h)
    to_tr_ag = Delta / jnp.where(ag_norm > 0, ag_norm, 1.0)
    to_bound_ag, _ = step_size_to_bound(x, ag, lb, ub)
    ag_stride = jnp.where(to_bound_ag < to_tr_ag, theta * to_bound_ag, to_tr_ag)
    a2, b2 = build_quadratic_1d(J_h, g_h, ag_h, diag_h)
    ag_t, ag_value = minimize_quadratic_1d(a2, b2, 0.0, ag_stride)
    ag_h = ag_h * ag_t
    ag = ag * ag_t

    pick_p = (p_value < r_value) & (p_value < ag_value)
    pick_r = (r_value < p_value) & (r_value < ag_value)
    step = jnp.where(pick_p, pt, jnp.where(pick_r, r, ag))
    step_h = jnp.where(pick_p, pt_h, jnp.where(pick_r, r_h, ag_h))
    value = jnp.where(pick_p, p_value, jnp.where(pick_r, r_value, ag_value))

    step = jnp.where(inb, p, step)
    step_h = jnp.where(inb, p_h, step_h)
    value = jnp.where(inb, p_value_in, value)
    return step, step_h, -value


# ── main loop ────────────────────────────────────────────────────────────────

class TRFState(NamedTuple):
    x: object
    f: object
    J: object
    cost: object
    g: object
    Delta: object
    alpha: object
    nfev: object     # int32
    status: object   # int32, -1 while running


def trf_init(fun_jac: Callable, x0, lb, ub) -> TRFState:
    dtype = x0.dtype
    x = make_strictly_feasible(x0, lb, ub, 1e-10)
    f, J = fun_jac(x)
    cost = 0.5 * jnp.dot(f, f)
    g = J.T @ f
    v, _ = cl_scaling_vector(x, g, lb, ub)
    Delta = jnp.linalg.norm(x / jnp.sqrt(v))
    Delta = jnp.where(Delta == 0, 1.0, Delta)
    return TRFState(x, f, J, cost, g, Delta, jnp.asarray(0.0, dtype), jnp.asarray(1, jnp.int32),
                    jnp.asarray(-1, jnp.int32))


def trf_active(state: TRFState, max_nfev: int):
    return (state.status < 0) & (state.nfev < max_nfev)


def trf_step(fun_jac: Callable, state: TRFState, lb, ub, *, ftol, xtol, gtol, max_nfev) -> TRFState:
    """One trial step (scipy's loop body, flattened). Inactive lanes are returned unchanged."""
    x, f, J, cost, g, Delta, alpha, nfev, status = state
    dtype = x.dtype
    eps = jnp.finfo(dtype).eps
    n = x.shape[0]
    m = J.shape[0]
    active = trf_active(state, max_nfev)

    v, dv = cl_scaling_vector(x, g, lb, ub)
    g_norm = jnp.max(jnp.abs(g * v))
    gconv = g_norm < gtol

    d = jnp.sqrt(v)
    diag_h = g * dv
    g_h = d * g
    J_h = J * d[None, :]
    J_aug = jnp.concatenate([J_h, jnp.diag(jnp.sqrt(diag_h))], axis=0)
    U, s, Vt = jnp.linalg.svd(J_aug, full_matrices=False)
    V = Vt.T
    uf = U.T @ jnp.concatenate([f, jnp.zeros((n,), dtype)])
    theta = jnp.maximum(0.995, 1.0 - g_norm)

    p_h, alpha_s = solve_lsq_trust_region(n, m, uf, s, V, Delta, alpha, eps)
    p = d * p_h
    step, step_h, predicted = select_step(x, J_h, diag_h, g_h, p, p_h, d, Delta, lb, ub, theta)
    x_new = make_strictly_feasible(x + step, lb, ub, 0)
    f_new, J_new = fun_jac(x_new)
    finite = jnp.all(jnp.isfinite(f_new))
    step_h_norm = jnp.linalg.norm(step_h)
    cost_new = 0.5 * jnp.dot(f_new, f_new)
    actual = cost - cost_new
    Delta_new, ratio = update_tr_radius(Delta, actual, predicted, step_h_norm, step_h_norm > 0.95 * Delta)
    term = check_termination(actual, cost, jnp.linalg.norm(step), jnp.linalg.norm(x), ratio, ftol, xtol)
    accept = finite & (actual > 0)

    Delta_out = jnp.where(finite, Delta_new, 0.25 * step_h_norm)
    alpha_out = jnp.where(finite, alpha_s * Delta / jnp.where(Delta_new > 0, Delta_new, 1.0), alpha_s)
    new = TRFState(
        x=jnp.where(accept, x_new, x), f=jnp.where(accept, f_new, f), J=jnp.where(accept, J_new, J),
        cost=jnp.where(accept, cost_new, cost), g=jnp.where(accept, J_new.T @ f_new, g),
        Delta=Delta_out, alpha=alpha_out, nfev=nfev + 1,
        status=jnp.where(finite & (term >= 0), term, status))
    # gtol fires before the trial in scipy (the trial is then never made)
    gstate = state._replace(status=jnp.asarray(1, jnp.int32))
    out = jax.tree_util.tree_map(lambda a, b: jnp.where(gconv, a, b), gstate, new)
    return jax.tree_util.tree_map(lambda a, b: jnp.where(active, a, b), out, state)


def trf_run(fun_jac: Callable, state: TRFState, lb, ub, n_iters: int, **kw) -> TRFState:
    """``n_iters`` trial steps (fixed shape; inactive lanes idle)."""
    return lax.fori_loop(0, n_iters, lambda _, st: trf_step(fun_jac, st, lb, ub, **kw), state)


def trf_bounds(fun_jac: Callable, x0, lb, ub, *, ftol=1e-8, xtol=1e-8, gtol=1e-8,
               max_nfev: int = 500) -> TRFResult:
    """Run to termination (``lax.while_loop``); see :func:`trf_init` / :func:`trf_run` for chunked use."""
    kw = dict(ftol=ftol, xtol=xtol, gtol=gtol, max_nfev=max_nfev)
    state = trf_init(fun_jac, x0, lb, ub)
    state = lax.while_loop(lambda st: trf_active(st, max_nfev),
                           lambda st: trf_step(fun_jac, st, lb, ub, **kw), state)
    status = jnp.where(state.status < 0, 0, state.status)
    return TRFResult(state.x, state.cost, state.nfev, status)
