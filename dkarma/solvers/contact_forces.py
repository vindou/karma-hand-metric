"""Contact-force allocation (SPEC.md §5.2-§5.5): the smooth feasibility margin solved inside
the rollout, and the exact convex allocation and wrench capacity solved on the host for
validation. Never mix them: the smooth value is a surrogate, the exact one decides.

Contact model
-------------
Per contact lane ``c``: a force ``f_c`` (3) applied by the finger on the object and a
torsional moment ``m_c n_c`` about the contact normal, bounded by the soft-finger limit
``|m_c| <= gamma f_n,c`` with ``gamma = mu * a`` for a contact patch of radius ``a``
(``torsion_radius_m``, default 3 mm; ``0`` recovers the pure point contact of SPEC §5.1).
Without torsion a two-finger pinch is generically infeasible: the weight at the sphere
centre has a moment about the contact line of order ``m g r_s sin(angle between the normals)``,
micro-N·m that no pair of point forces can produce. Variables ``x = [f (3Cn), m (Cn)]``.

    equilibrium   G_f f + G_m m + w_ext = 0        G_f = [I..; [r_c]x..],  G_m = [0..; n_c..]
    torque        tau = Jc^T f + Jw^T (n_c m_c) + tau_grav,   |tau_j| <= tau_bar_j
    cones         |f_t,c| <= mu f_n,c,  f_n,c >= 0,  |m_c| <= gamma f_n,c

Smooth problem (differentiable; Levenberg-Marquardt with adaptive Marquardt-scaled damping
``lam * diag(J^T J)`` on a sum of squared smooth residuals, fixed iteration budget; the
torsion columns are ~1/gamma times the force columns, and the unscaled ``lam * I`` damping
of Phase 4 left some three-contact states at ``E ~ 1e-3`` after 200 iterations, Phase 8.1):

    E_F* = min_x  w_w |G x + w_ext|^2 / F0^2 + w_tau sum_j hinge((|tau_j| - tau_bar_j)/tau_bar_j)^2
                + w_mu sum_c hinge((|f_t,c| - mu f_n,c)/F0)^2 + w_n sum_c hinge(-f_n,c/F0)^2
                + w_mu sum_c hinge((|m_c| - gamma f_n,c)/(gamma F0))^2

Exact problem (host, SLSQP on a convex smooth formulation with independent equality rows):

    min 1/2 |x|^2  s.t.  the constraints above (cones as mu f_n - sqrt(|f_t|^2 + eps^2) >= 0)

and the wrench capacity ``alpha*(d) = max alpha`` s.t. ``G x + w_0 + alpha d = 0`` and the
same constraints, over standardized 6-D directions.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..core import geometry as geo


class ForceWeights(NamedTuple):
    w: float = 1.0            # equilibrium residual
    tau: float = 1.0          # torque ceiling
    mu: float = 1.0           # friction cone and torsion limit
    n: float = 1.0            # unilateral normal force
    eps_hinge: float = 0.005  # hinge smoothing (dimensionless); Phase 8.1 calibration (0.02 in Phase 4)
    eps_abs: float = 1e-3     # |x| smoothing (relative to the scale)
    iters: int = 50           # Phase 8.1: 50 with Marquardt scaling (25 with lam * I left three-contact states unconverged)
    torsion_radius_m: float = 0.003   # soft-finger patch radius a; gamma = mu * a
    grad_mode: str = "envelope"       # "envelope": dE*/dphi = dE/dphi at the converged x* (exact at a minimizer, Phase 8.1);
                                      # "unrolled": reverse-mode through the LM iterations (Phase 4; unstable once converged)


class ForceResult(NamedTuple):
    x: object               # [4Cn] = [f (3Cn), m (Cn)]
    E: object               # smooth energy at the solution
    eq_residual: object     # |G x + w_ext| (N, force rows dominate)
    torque_usage: object    # max_j |tau_j| / tau_bar_j
    cone_margin: object     # min_c (mu f_n - |f_t|) / F0  (>= 0 inside the cone)
    fn_min: object          # min_c f_n / F0
    torsion_margin: object  # min_c (gamma f_n - |m|) / (gamma F0)
    tau: object             # [J]


class ForceProblem(NamedTuple):
    """Everything the solvers need at one state (pytree)."""
    G: object          # [6, 4Cn]  equilibrium map on x = [f, m]
    Jt: object         # [J, 4Cn]  torque map on x
    tau_grav: object   # [J]
    normals: object    # [Cn, 3]
    valid: object      # bool [Cn]
    tau_bar: object    # [J]
    mu: object
    gamma: object      # torsion coefficient mu * a (m)


def build_problem(G_f, Jt_f, Jw, tau_grav, normals, valid, tau_bar, mu, torsion_radius_m) -> ForceProblem:
    """Assemble the ``[f, m]`` problem from the force-only maps and the angular Jacobians.
    ``G_f [6, 3Cn]``, ``Jt_f [J, 3Cn]``, ``Jw [Cn, 3, J]``."""
    Cn = normals.shape[0]
    dtype = G_f.dtype
    vf = valid.astype(dtype)
    n_cols = normals * vf[:, None]                                            # [Cn,3]
    G_m = jnp.concatenate([jnp.zeros((3, Cn), dtype), n_cols.T], axis=0)      # [6, Cn]
    Jt_m = jnp.stack([Jw[c].T @ n_cols[c] for c in range(Cn)], axis=1)        # [J, Cn]
    return ForceProblem(G=jnp.concatenate([G_f, G_m], axis=1), Jt=jnp.concatenate([Jt_f, Jt_m], axis=1),
                        tau_grav=tau_grav, normals=normals, valid=valid, tau_bar=tau_bar, mu=jnp.asarray(mu, dtype),
                        gamma=jnp.asarray(mu * torsion_radius_m, dtype))


def _split(x, Cn):
    return x[: 3 * Cn].reshape(Cn, 3), x[3 * Cn:]


def _decompose(F, normals, valid):
    fn = jnp.sum(F * normals, axis=-1)
    ft = F - fn[:, None] * normals
    return fn * valid, jnp.linalg.norm(ft, axis=-1) * valid


def residuals(x, pb: ForceProblem, w_ext, F0, wts: ForceWeights):
    """Stacked smooth residual vector ``r(x)`` with ``E = |r|^2``."""
    Cn = pb.normals.shape[0]
    e = wts.eps_abs
    F, m = _split(x, Cn)
    vf = pb.valid.astype(x.dtype)
    r_w = jnp.sqrt(wts.w) * (pb.G @ x + w_ext) / F0
    tau = pb.Jt @ x + pb.tau_grav
    tau_abs = jnp.sqrt(tau ** 2 + (e * pb.tau_bar) ** 2)
    r_tau = jnp.sqrt(wts.tau) * geo.hinge((tau_abs - pb.tau_bar) / pb.tau_bar, wts.eps_hinge)
    fn, ft = _decompose(F, pb.normals, vf)
    ft_s = jnp.sqrt(ft ** 2 + (e * F0) ** 2)
    r_mu = jnp.sqrt(wts.mu) * geo.hinge((ft_s - pb.mu * fn) / F0, wts.eps_hinge) * vf
    r_n = jnp.sqrt(wts.n) * geo.hinge(-fn / F0, wts.eps_hinge) * vf
    g0 = jnp.maximum(pb.gamma, 1e-9)
    m_s = jnp.sqrt(m ** 2 + (e * g0 * F0) ** 2)
    r_m = jnp.sqrt(wts.mu) * geo.hinge((m_s - pb.gamma * fn) / (g0 * F0), wts.eps_hinge) * vf
    return jnp.concatenate([r_w, r_tau, r_mu, r_n, r_m])


def initial_point(pb: ForceProblem, w_ext, F0):
    """Minimum-norm equilibrium plus an inward normal preload (inside the cones, zero torsion)."""
    Cn = pb.normals.shape[0]
    vf = pb.valid.astype(w_ext.dtype)
    x0 = -(jnp.linalg.pinv(pb.G) @ w_ext)
    preload = jnp.concatenate([(0.5 * F0 * pb.normals * vf[:, None]).reshape(-1), jnp.zeros((Cn,), w_ext.dtype)])
    mask = jnp.concatenate([jnp.repeat(vf, 3), vf])
    return (x0 + preload) * mask, mask


def solve_soft(pb: ForceProblem, w_ext, wts: ForceWeights = ForceWeights(), x_init=None) -> ForceResult:
    """Fixed-budget damped Newton (Nielsen damping, Marquardt scaling, the full Hessian of the
    convex cost) on the smooth residuals in the rescaled variables ``y = [f, m / gamma]`` (every
    column O(1); Phase 8.1: Gauss-Newton converged only linearly at infeasible states).
    Differentiable: ``grad_mode="envelope"`` returns the energy with ``x*`` held constant, so
    ``dE*/dphi = dE/dphi|_{x*}`` (exact where ``x*`` is a minimizer; validated against finite
    differences in ``dkarma/tests/test_actuation.py``); ``"unrolled"`` differentiates through the
    iterations."""
    dtype = pb.G.dtype
    Cn = pb.normals.shape[0]
    F0 = jnp.maximum(jnp.linalg.norm(w_ext), 1.0)
    x0, mask = initial_point(pb, w_ext, F0)
    if x_init is not None:
        x0 = x_init * mask
    g0 = jnp.maximum(pb.gamma, 1e-9)
    scale = jnp.concatenate([jnp.ones((3 * Cn,), dtype), jnp.full((Cn,), g0, dtype)])
    n = x0.shape[0]
    eye = jnp.eye(n, dtype=dtype)

    def r_of(y):
        return residuals(y * scale, pb, w_ext, F0, wts)

    def lm(pb_, w_):
        def r_y(y):
            return residuals(y * scale, pb_, w_, F0, wts)
        y0 = x0 / scale
        r0 = r_y(y0)

        def cost_y(y):
            r = r_y(y)
            return jnp.sum(r * r)

        def body(_, st):
            y, cost, lam, nu = st
            # damped Newton on the convex cost (Gauss-Newton alone converges linearly where the
            # residual at the optimum is large, i.e. at every infeasible state): H = J^T J + sum r_i d2 r_i
            g = 0.5 * jax.grad(cost_y)(y) * mask                          # J^T r
            Hn = 0.5 * jax.hessian(cost_y)(y) * (mask[:, None] * mask[None, :])
            D = jnp.diag(jnp.maximum(jnp.diag(Hn), 1e-12))               # Marquardt scaling: scale-invariant damping
            H = Hn + lam * D + (1.0 - mask)[:, None] * eye
            step = -jnp.linalg.solve(H, g) * mask
            y_new = y + step
            cost_new = cost_y(y_new)
            pred = -(jnp.dot(step, 2.0 * g) + step @ (Hn @ step))
            rho = (cost - cost_new) / jnp.where(pred > 0, pred, 1e-30)
            accept = (cost_new < cost) & jnp.isfinite(cost_new)
            lam_acc = lam * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
            lam = jnp.clip(jnp.where(accept, lam_acc, lam * nu), 1e-15, 1e15)
            nu = jnp.where(accept, 2.0, nu * 2.0)
            y = jnp.where(accept, y_new, y)
            cost = jnp.where(accept, cost_new, cost)
            return y, cost, lam, nu

        y, cost, _, _ = jax.lax.fori_loop(0, wts.iters, body, (y0, jnp.sum(r0 ** 2), jnp.asarray(1e-3, dtype), jnp.asarray(2.0, dtype)))
        return y, cost

    if wts.grad_mode == "envelope":
        sg = jax.lax.stop_gradient
        y_star, _ = lm(jax.tree_util.tree_map(sg, pb), sg(w_ext))
        y_star = sg(y_star)
        cost = jnp.sum(r_of(y_star) ** 2)
    else:
        y_star, cost = lm(pb, w_ext)
    x = y_star * scale
    F, m = _split(x, Cn)
    vf = pb.valid.astype(dtype)
    fn, ft = _decompose(F, pb.normals, vf)
    tau = pb.Jt @ x + pb.tau_grav
    return ForceResult(x=x, E=cost, eq_residual=jnp.linalg.norm(pb.G @ x + w_ext),
                       torque_usage=jnp.max(jnp.abs(tau) / pb.tau_bar),
                       cone_margin=jnp.min(jnp.where(pb.valid, (pb.mu * fn - ft) / F0, jnp.inf)),
                       fn_min=jnp.min(jnp.where(pb.valid, fn / F0, jnp.inf)),
                       torsion_margin=jnp.min(jnp.where(pb.valid, (pb.gamma * fn - jnp.abs(m)) / (g0 * F0), jnp.inf)),
                       tau=tau)


# ── exact (host) ─────────────────────────────────────────────────────────────

def _np(x):
    return np.asarray(x, dtype=float)


def _host(pb: ForceProblem):
    return dict(G=_np(pb.G), Jt=_np(pb.Jt), tau_grav=_np(pb.tau_grav), normals=_np(pb.normals),
                valid=np.asarray(pb.valid, bool), tau_bar=_np(pb.tau_bar), mu=float(pb.mu), gamma=float(pb.gamma))


def _reduced_equality(Gx, w, rel_tol=1e-9):
    """Independent rows of ``Gx x + w = 0`` (SVD); returns ``(A, b, |load outside range(Gx)|)``."""
    U, S, _ = np.linalg.svd(Gx)
    r = int(np.sum(S > rel_tol * max(S[0], 1e-300)))
    Ur, Up = U[:, :r], U[:, r:]
    return Ur.T @ Gx, Ur.T @ w, float(np.linalg.norm(Up.T @ w)) if Up.shape[1] else 0.0


def _constraints(h, F0, n_extra=0):
    """Convex smooth SLSQP constraints on ``x = [f, m, extra]`` with analytic Jacobians."""
    normals, valid, Jt, tau_grav, tau_bar, mu, gamma = (h[k] for k in ("normals", "valid", "Jt", "tau_grav", "tau_bar", "mu", "gamma"))
    Cn = normals.shape[0]
    n = 4 * Cn
    N = n + n_extra
    eps = 1e-9 * F0
    cons = []
    for c in range(Cn):
        E = np.zeros((3, N)); E[:, 3 * c:3 * c + 3] = np.eye(3)
        em = np.zeros(N); em[3 * Cn + c] = 1.0
        nc = normals[c]
        if valid[c]:
            cons.append(dict(type="ineq", fun=lambda x, E=E, nc=nc: float(nc @ (E @ x)) / F0, jac=lambda x, E=E, nc=nc: (nc @ E) / F0))

            def cone(x, E=E, nc=nc):
                f = E @ x; fnv = nc @ f; ft = f - fnv * nc
                return (mu * fnv - np.sqrt(ft @ ft + eps * eps)) / F0

            def cone_jac(x, E=E, nc=nc):
                f = E @ x; fnv = nc @ f; ft = f - fnv * nc
                P = np.eye(3) - np.outer(nc, nc)
                return ((mu * nc - (P @ ft) / np.sqrt(ft @ ft + eps * eps)) @ E) / F0
            cons.append(dict(type="ineq", fun=cone, jac=cone_jac))
            if gamma > 0:
                for sgn in (1.0, -1.0):
                    cons.append(dict(type="ineq",
                                     fun=lambda x, E=E, nc=nc, em=em, sgn=sgn: (gamma * float(nc @ (E @ x)) - sgn * float(em @ x)) / (gamma * F0),
                                     jac=lambda x, E=E, nc=nc, em=em, sgn=sgn: (gamma * (nc @ E) - sgn * em) / (gamma * F0)))
            else:
                cons.append(dict(type="eq", fun=lambda x, em=em: float(em @ x) / F0, jac=lambda x, em=em: em / F0))
        else:
            cons.append(dict(type="eq", fun=lambda x, E=E: (E @ x) / F0, jac=lambda x, E=E: E / F0))
            cons.append(dict(type="eq", fun=lambda x, em=em: float(em @ x) / F0, jac=lambda x, em=em: em / F0))
    Jx = np.zeros((Jt.shape[0], N)); Jx[:, :n] = Jt
    cons.append(dict(type="ineq", fun=lambda x: (tau_bar - (Jx @ x + tau_grav)) / tau_bar, jac=lambda x: -Jx / tau_bar[:, None]))
    cons.append(dict(type="ineq", fun=lambda x: (tau_bar + (Jx @ x + tau_grav)) / tau_bar, jac=lambda x: Jx / tau_bar[:, None]))
    return cons


def violation(pb: ForceProblem, w, x) -> float:
    """Maximum constraint violation of ``x`` (N; torque rows scaled by F0/tau_bar, torsion by 1/gamma)."""
    h = _host(pb)
    x = _np(x); w = _np(w)
    F0 = max(np.linalg.norm(w), 1.0)
    Cn = h["normals"].shape[0]
    F, m = x[: 3 * Cn].reshape(Cn, 3), x[3 * Cn:]
    fn = np.sum(F * h["normals"], axis=1)
    ft = F - fn[:, None] * h["normals"]
    v = [np.max(np.abs(h["G"] @ x + w))]
    for c in np.nonzero(h["valid"])[0]:
        v.append(max(0.0, np.linalg.norm(ft[c]) - h["mu"] * fn[c]))
        v.append(max(0.0, -fn[c]))
        if h["gamma"] > 0:
            v.append(max(0.0, abs(m[c]) - h["gamma"] * fn[c]) / h["gamma"])
        else:
            v.append(abs(m[c]))
    v.append(np.max(np.maximum(np.abs(h["Jt"] @ x + h["tau_grav"]) - h["tau_bar"], 0.0) / h["tau_bar"]) * F0)
    return float(max(v))


def solve_exact(pb: ForceProblem, w_ext, tol: float = 1e-5):
    """Minimum-norm feasible allocation by SLSQP. ``feasible`` certifies ``violation <= tol * F0``
    (10 µN on a 1 N scale; SLSQP's attainable precision on these problems is ~1e-6 relative)."""
    from scipy.optimize import minimize
    h = _host(pb)
    w = _np(w_ext)
    F0 = max(np.linalg.norm(w), 1.0)
    Cn = h["normals"].shape[0]
    n = 4 * Cn
    A_eq, b_eq, w_perp = _reduced_equality(h["G"], w)
    if w_perp > tol * F0:
        x0 = np.zeros(n)
        return dict(feasible=False, x=x0, tau=h["Jt"] @ x0 + h["tau_grav"], viol=w_perp, success=False, unresisted=w_perp)
    cons = [dict(type="eq", fun=lambda x: (A_eq @ x + b_eq) / F0, jac=lambda x: A_eq / F0)] + _constraints(h, F0)
    x_pre, _ = initial_point(pb, jnp.asarray(w), F0)
    best = None
    for x0 in (np.asarray(x_pre), np.zeros(n)):
        res = minimize(lambda x: 0.5 * np.dot(x, x) / F0 ** 2, x0, jac=lambda x: x / F0 ** 2, constraints=cons,
                       method="SLSQP", options=dict(maxiter=1000, ftol=1e-14))
        viol = violation(pb, w, res.x)
        cand = dict(feasible=bool(viol <= tol * F0), x=res.x, tau=h["Jt"] @ res.x + h["tau_grav"], viol=viol,
                    success=bool(res.success), unresisted=w_perp)
        if best is None or viol < best["viol"]:
            best = cand
        if best["feasible"]:
            break
    return best


def wrench_directions(force_scale: float, moment_scale: float, n: int = 26) -> np.ndarray:
    """Standardized 6-D wrench directions: the 12 signed axes, then deterministic random unit
    directions up to ``n``; scaled by ``(force_scale, moment_scale)``."""
    dirs = []
    for i in range(6):
        for s in (1.0, -1.0):
            d = np.zeros(6); d[i] = s; dirs.append(d)
    rng = np.random.RandomState(0)
    while len(dirs) < n:
        d = rng.randn(6); d /= np.linalg.norm(d); dirs.append(d)
    return np.stack(dirs[:n]) * np.array([force_scale] * 3 + [moment_scale] * 3)[None, :]


def wrench_capacity(pb: ForceProblem, w_0, directions, alpha_max: float = 50.0, tol: float = 1e-5):
    """``alpha*(d)`` per direction (SLSQP on ``(x, alpha)``); 0 where the base load is infeasible."""
    from scipy.optimize import minimize
    h = _host(pb)
    w0 = _np(w_0)
    base = solve_exact(pb, w0)
    out = np.zeros(directions.shape[0])
    if not base["feasible"]:
        return out
    F0 = max(np.linalg.norm(w0), 1.0)
    Cn = h["normals"].shape[0]
    n = 4 * Cn
    cons_fixed = _constraints(h, F0, n_extra=1)
    for k, d in enumerate(directions):
        Gx = np.concatenate([h["G"], d[:, None]], axis=1)
        A_eq, b_eq, w_perp = _reduced_equality(Gx, w0)
        if w_perp > tol * F0:
            continue
        cons = [dict(type="eq", fun=lambda x, A=A_eq, b=b_eq: (A @ x + b) / F0, jac=lambda x, A=A_eq: A / F0)] + cons_fixed
        x0 = np.concatenate([base["x"], [0.0]])
        res = minimize(lambda x: -x[n] / 10.0, x0, jac=lambda x: np.concatenate([np.zeros(n), [-0.1]]), constraints=cons,
                       bounds=[(None, None)] * n + [(0.0, alpha_max)], method="SLSQP", options=dict(maxiter=1000, ftol=1e-12))
        a = res.x[n]
        out[k] = a if violation(pb, w0 + a * d, res.x[:n]) <= tol * F0 else 0.0
    return out


__all__ = ["ForceWeights", "ForceResult", "ForceProblem", "build_problem", "residuals", "initial_point", "solve_soft",
           "violation", "solve_exact", "wrench_directions", "wrench_capacity"]
