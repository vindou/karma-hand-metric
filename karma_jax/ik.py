"""Batched seed IK: place a thumb-index pinch on the sphere (port of ``karma/seed.py``).

The reference solves a bounded nonlinear least-squares problem with scipy's
trust-region-reflective method, using a hand-written Jacobian whose collision
and antipodal rows are deliberately ZERO (to avoid finite-difference noise).
Here the same residual vector is minimized by a projected Levenberg-Marquardt
iteration (Nielsen damping schedule, fixed budget, one lane per candidate),
and by default with a complete analytic Jacobian: the antipodal row and the
hinge-collision row are differentiated too. Without them LM stalls where TRF
happens to find a path (see ``IKSettings.full_jacobian`` and the oracle test).

Residuals, in the reference's order and weights:

    r[0:2]   contact gaps / eps_g * 0.35
    r[2]     collision hinge  max(0, (eps_col - clearance) / eps_col) * 0.25
    r[3]     antipodal        (1 - cos) * 200, or 200 if a normal is degenerate
    r[4:]    (q - q_init) * 0.1

The sphere variable is the non-dimensional displacement ``dp/r_s`` with bound
``+-3 (1 + r_l/r_s)``; the gap rows' joint columns use the raw (unfolded)
Jacobian columns exactly as the reference does.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from .model import DeviceModel
from .kinematics import forward_kinematics, active_limits, get_active, set_active, apply_mimic
from .geometry import capsule_pass, contact_kinematics, check_antipodal_friction
from .rolling import StepParams

GAP_W = 0.35
COL_W = 0.25
ANTI_W = 200.0
REG_W = 0.1


class IKSettings(NamedTuple):
    method: str = "trf"          # "trf": scipy-faithful port (default); "lm": projected Levenberg-Marquardt
    iters: int = 100             # lm only
    tau: float = 1e-3            # lm only: initial damping relative to max diag(J'J)
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8           # trf only
    max_nfev: int = 500          # trf only (reference setting)
    full_jacobian: bool = False  # False: the reference's zero rows for collision/antipodal


class IKResult(NamedTuple):
    q: object           # [J_pad] full configuration, mimic-consistent
    p: object           # [3] sphere centre
    max_gap: object     # scalar
    feasible: object    # bool
    cost: object        # final 0.5 * |r|^2
    iters: object       # int32


def _seg_projector_apply(cp, c, v):
    """``M v`` with ``M = I - seg seg^T / |seg|^2`` in the interior regime, else ``v``."""
    seg = cp.b[c] - cp.a[c]
    ssq = jnp.dot(seg, seg)
    t = cp.t[c]
    interior = (t > 1e-12) & (t < 1.0 - 1e-12) & (ssq > 1e-18)
    proj = (jnp.dot(v, seg) / jnp.where(ssq > 1e-18, ssq, 1.0)) * seg
    return jnp.where(interior, v - proj, v)


def _dnormal(model, cp, ck, c, p):
    """Derivatives of the unit normal of capsule ``c``: ``(dn/dq [3,J_full], dn/dp [3,3])``."""
    n = ck.normal
    dist = jnp.maximum(cp.dist[c], 1e-12)
    Pn = (jnp.eye(3, dtype=n.dtype) - jnp.outer(n, n)) / dist
    dn_dq = -(Pn @ ck.Jv)                       # axis point moves with q, sphere fixed
    M = jax.vmap(lambda e: _seg_projector_apply(cp, c, e))(jnp.eye(3, dtype=n.dtype)).T  # columns M e_k
    dn_dp = Pn @ M
    return dn_dq, dn_dp


def _residuals_and_jac(model: DeviceModel, x, q_init, centre, pair, params: StepParams, full: bool):
    """Residual vector [4+M] and Jacobian [4+M, M+3] at x = [q_active, dp/r_s]."""
    M = model.q_ids.shape[0]
    r_s, r_l = params.sphere_radius, params.link_radius
    eps_g, eps_col = params.eps_g, params.eps_col
    dtype = x.dtype
    valid = model.active_valid
    validf = valid.astype(dtype)

    qa = x[:M]
    dp = x[M:]
    q = set_active(model, q_init, qa)
    p = centre + r_s * dp
    fk = forward_kinematics(model, q)
    cp = capsule_pass(model, fk, p, r_s, r_l)
    ck_t = contact_kinematics(model, fk, cp, pair[0], p, r_s)
    ck_i = contact_kinematics(model, fk, cp, pair[1], p, r_s)

    # gaps
    scale = GAP_W / eps_g
    r_gaps = jnp.stack([ck_t.g, ck_i.g]) * scale

    # collision hinge on the closest non-pair capsule
    C = cp.gap.shape[0]
    idx = jnp.arange(C)
    cmask = model.cap_valid & (idx != pair[0]) & (idx != pair[1])
    gaps_m = jnp.where(cmask, cp.gap, jnp.inf)
    c_min = jnp.argmin(gaps_m)
    clearance = gaps_m[c_min]
    active_col = clearance < eps_col
    r_col = jnp.where(active_col, (eps_col - clearance) / eps_col, 0.0) * COL_W

    # antipodal
    n_t, n_i = ck_t.normal, ck_i.normal
    nt_norm = jnp.linalg.norm(n_t)
    ni_norm = jnp.linalg.norm(n_i)
    good = (nt_norm > 1e-9) & (ni_norm > 1e-9)
    cos_a = jnp.dot(-n_t / jnp.where(good, nt_norm, 1.0), n_i / jnp.where(good, ni_norm, 1.0))
    r_anti = jnp.where(good, (1.0 - cos_a) * ANTI_W, ANTI_W)

    r_reg = (qa - get_active(model, q_init)) * REG_W * validf
    r = jnp.concatenate([r_gaps, r_col[None], r_anti[None], r_reg])

    # ── Jacobian ──
    take_q = lambda Jf: jnp.take(Jf, model.q_ids, axis=1) * validf[None, :]  # noqa: E731
    row_t = jnp.concatenate([-scale * (n_t @ take_q(ck_t.Jv)), scale * _seg_projector_apply(cp, pair[0], n_t) * r_s])
    row_i = jnp.concatenate([-scale * (n_i @ take_q(ck_i.Jv)), scale * _seg_projector_apply(cp, pair[1], n_i) * r_s])
    J_reg = jnp.concatenate([jnp.diag(jnp.full((M,), REG_W, dtype)), jnp.zeros((M, 3), dtype)], axis=1)

    if full:
        ck_c = contact_kinematics(model, fk, cp, c_min, p, r_s)
        n_c = ck_c.normal
        d_clear_dq = -(n_c @ take_q(ck_c.Jv))
        d_clear_dp = _seg_projector_apply(cp, c_min, n_c) * r_s
        row_c = jnp.where(active_col, -(COL_W / eps_col) * jnp.concatenate([d_clear_dq, d_clear_dp]), 0.0)
        dnt_dq, dnt_dp = _dnormal(model, cp, ck_t, pair[0], p)
        dni_dq, dni_dp = _dnormal(model, cp, ck_i, pair[1], p)
        dcos_dq = -(n_i @ take_q(dnt_dq) + n_t @ take_q(dni_dq))
        dcos_dp = -(n_i @ dnt_dp + n_t @ dni_dp) * r_s
        row_a = jnp.where(good, -ANTI_W * jnp.concatenate([dcos_dq, dcos_dp]), 0.0)
    else:
        row_c = jnp.zeros((M + 3,), dtype)
        row_a = jnp.zeros((M + 3,), dtype)

    J = jnp.concatenate([row_t[None], row_i[None], row_c[None], row_a[None], J_reg], axis=0)
    return r, J, (q, p, ck_t, ck_i, cp, c_min)


def solve_pinch_ik(model: DeviceModel, q_init, centre, pair, params: StepParams,
                   settings: IKSettings = IKSettings()) -> IKResult:
    """Projected Levenberg-Marquardt on the reference's seed-IK residual, one candidate."""
    dtype = q_init.dtype
    M = model.q_ids.shape[0]
    r_s, r_l = params.sphere_radius, params.link_radius
    q_lo, q_hi = active_limits(model)
    valid = model.active_valid
    margin = 3.0 * (r_s + r_l) / r_s
    lo = jnp.concatenate([jnp.where(valid, q_lo, -1.0), jnp.full((3,), -margin, dtype)])
    hi = jnp.concatenate([jnp.where(valid, q_hi, 1.0), jnp.full((3,), margin, dtype)])
    mask = jnp.concatenate([valid, jnp.ones((3,), bool)]).astype(dtype)
    full = settings.full_jacobian

    x0 = jnp.clip(jnp.concatenate([get_active(model, q_init), jnp.zeros((3,), dtype)]), lo, hi)

    if settings.method == "trf":
        from .trf import trf_bounds
        res = trf_bounds(lambda xx: _residuals_and_jac(model, xx, q_init, centre, pair, params, full)[:2],
                         x0, lo, hi, ftol=settings.ftol, xtol=settings.xtol, gtol=settings.gtol,
                         max_nfev=settings.max_nfev)
        return _finish(model, res.x, q_init, centre, pair, params, full, res.cost, res.nfev)

    r0, J0, _ = _residuals_and_jac(model, x0, q_init, centre, pair, params, full)
    cost0 = 0.5 * jnp.dot(r0, r0)
    lam0 = settings.tau * jnp.max(jnp.diag(J0.T @ J0))

    def body(_, state):
        x, cost, lam, nu, done, iters = state
        r, J, _ = _residuals_and_jac(model, x, q_init, centre, pair, params, full)
        JtJ = J.T @ J
        g = J.T @ r
        H = JtJ + lam * jnp.eye(M + 3, dtype=dtype)
        delta = -jnp.linalg.solve(H, g) * mask
        x_new = jnp.clip(x + delta, lo, hi)
        step = x_new - x
        r_new, _, _ = _residuals_and_jac(model, x_new, q_init, centre, pair, params, full)
        cost_new = 0.5 * jnp.dot(r_new, r_new)
        pred = -(jnp.dot(step, g) + 0.5 * step @ (JtJ @ step))
        rho = (cost - cost_new) / jnp.where(pred > 0, pred, 1e-30)
        accept = (cost_new < cost) & jnp.isfinite(cost_new)
        # Nielsen's update
        lam_acc = lam * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
        lam = jnp.where(accept, lam_acc, lam * nu)
        nu = jnp.where(accept, 2.0, nu * 2.0)
        lam = jnp.clip(lam, 1e-15, 1e15)
        conv = accept & (((cost - cost_new) <= settings.ftol * cost) |
                         (jnp.linalg.norm(step) <= settings.xtol * (settings.xtol + jnp.linalg.norm(x))))
        upd = ~done
        x = jnp.where(upd & accept, x_new, x)
        cost = jnp.where(upd & accept, cost_new, cost)
        iters = iters + upd.astype(jnp.int32)
        done = done | conv
        return x, cost, lam, nu, done, iters

    state = (x0, cost0, lam0, jnp.asarray(2.0, dtype), jnp.asarray(False), jnp.asarray(0, jnp.int32))
    x, cost, lam, nu, done, iters = lax.fori_loop(0, settings.iters, body, state)
    return _finish(model, x, q_init, centre, pair, params, full, cost, iters)


def _finish(model, x, q_init, centre, pair, params, full, cost, iters) -> IKResult:
    r, _, (q, p, ck_t, ck_i, cp, c_min) = _residuals_and_jac(model, x, q_init, centre, pair, params, full)
    max_gap = jnp.maximum(jnp.abs(ck_t.g), jnp.abs(ck_i.g))
    C = cp.gap.shape[0]
    idx = jnp.arange(C)
    cmask = model.cap_valid & (idx != pair[0]) & (idx != pair[1])
    col_ok = jnp.min(jnp.where(cmask, cp.gap, jnp.inf)) >= params.eps_col
    grasp_ok, _ = check_antipodal_friction(ck_t.normal, ck_i.normal, params.friction_coeff)
    feasible = (max_gap <= params.eps_g) & col_ok & grasp_ok
    return IKResult(apply_mimic(model, q), p, max_gap, feasible, cost, iters)


def make_batched_ik(model: DeviceModel, params: StepParams, settings: IKSettings = IKSettings()):
    """``jit``-compiled batched IK: (q_init[B,J], centre[B,3], pair[B,2]) -> IKResult[B].

    Every lane runs until the slowest lane in the batch terminates.
    """
    def one(q_init, centre, pair):
        return solve_pinch_ik(model, q_init, centre, pair, params, settings)
    return jax.jit(jax.vmap(one))


class ChunkedIK:
    """Batched TRF IK with host-side compaction of finished lanes.

    The reference's TRF needs a median of ~15 evaluations but up to 500; a
    plain ``vmap`` of ``while_loop`` makes every lane pay for the slowest. Here
    the batch advances ``chunk`` trial steps at a time, finished lanes are
    dropped, and the remainder continues, so the cost is proportional to the
    total number of evaluations actually needed.
    """

    def __init__(self, model: DeviceModel, params: StepParams, settings: IKSettings = IKSettings(),
                 chunk: int = 16):
        if settings.method != "trf":
            raise ValueError("ChunkedIK supports method='trf' only")
        from .trf import trf_init, trf_run
        self.model, self.params, self.settings, self.chunk = model, params, settings, chunk
        full = settings.full_jacobian
        M = model.q_ids.shape[0]
        r_s, r_l = params.sphere_radius, params.link_radius
        q_lo, q_hi = active_limits(model)
        valid = model.active_valid
        margin = 3.0 * (r_s + r_l) / r_s
        dtype = q_lo.dtype
        lo = jnp.concatenate([jnp.where(valid, q_lo, -1.0), jnp.full((3,), -margin, dtype)])
        hi = jnp.concatenate([jnp.where(valid, q_hi, 1.0), jnp.full((3,), margin, dtype)])
        kw = dict(ftol=settings.ftol, xtol=settings.xtol, gtol=settings.gtol, max_nfev=settings.max_nfev)

        def fj(q_init, centre, pair):
            return lambda xx: _residuals_and_jac(model, xx, q_init, centre, pair, params, full)[:2]

        def init(q_init, centre, pair):
            x0 = jnp.clip(jnp.concatenate([get_active(model, q_init), jnp.zeros((3,), dtype)]), lo, hi)
            return trf_init(fj(q_init, centre, pair), x0, lo, hi)

        def run(state, q_init, centre, pair):
            return trf_run(fj(q_init, centre, pair), state, lo, hi, chunk, **kw)

        def finish(state, q_init, centre, pair):
            return _finish(model, state.x, q_init, centre, pair, params, full, state.cost, state.nfev)

        self._init = jax.jit(jax.vmap(init))
        self._run = jax.jit(jax.vmap(run))
        self._finish = jax.jit(jax.vmap(finish))
        self._active = jax.jit(jax.vmap(lambda st: (st.status < 0) & (st.nfev < settings.max_nfev)))

    def __call__(self, q_init, centre, pair) -> IKResult:
        import numpy as np
        B = q_init.shape[0]
        args = (q_init, centre, pair)
        # Lane counts are padded to power-of-two buckets so compaction does not
        # trigger a compile per distinct batch size.
        def bucket(n):
            b = 64
            while b < n:
                b *= 2
            return b

        def padded_idx(idx):
            nb = bucket(idx.size)
            return np.concatenate([idx, np.repeat(idx[-1:], nb - idx.size)]) if nb > idx.size else idx

        idx = np.arange(B)                     # lanes still running (indices into the full batch)
        pidx = padded_idx(idx)
        state = self._init(*[a[pidx] for a in args])
        final = None                           # host-side (writable) per-lane final states
        while True:
            state = self._run(state, *[a[pidx] for a in args])
            active = np.asarray(self._active(state))[: idx.size]
            host = jax.tree_util.tree_map(lambda a: np.asarray(a)[: idx.size], state)
            if final is None:
                final = jax.tree_util.tree_map(lambda a: np.array(a), host)
            else:
                for leaf_f, leaf_s in zip(jax.tree_util.tree_leaves(final), jax.tree_util.tree_leaves(host)):
                    leaf_f[idx] = leaf_s
            if not active.any():
                break
            keep = np.nonzero(active)[0]
            idx = idx[keep]
            pidx = padded_idx(idx)
            pkeep = np.concatenate([keep, np.repeat(keep[-1:], pidx.size - keep.size)])
            state = jax.tree_util.tree_map(lambda a: a[pkeep], state)
        fstate = jax.tree_util.tree_map(lambda a: jnp.asarray(a), final)
        return self._finish(fstate, *args)
