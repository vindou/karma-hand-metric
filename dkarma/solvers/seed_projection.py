"""Seeds in physical units for a :class:`~dkarma.core.morphology.HandModel` (SPEC.md §4.6).

Two-finger pinch seeds, found deterministically with the reference's recipe on the
``dkarma`` model and the task's physical parameters (no URDF, no ``L_ref``):

1. joint samples: neutral, a per-joint linspace, unscrambled Sobol points, and
   ``RandomState(42)`` draws (``karma_jax.seeds.sample_joint_configs`` logic);
2. reach filter on the two seed capsules and candidate sphere centres at the
   reference's offsets along both capsule axes;
3. batched IK (``karma_jax.ik.solve_pinch_ik``, the reference's residual) on every
   candidate, projected Levenberg-Marquardt by default (pure JAX, one compiled
   program), feasibility = gaps within ``eps_g``, collision-free, friction cone;
4. dedup at ``dedup_rad`` in joint space, sort by ``(max_gap, q)``, nullspace
   polish toward the joint-range centre (kept only if still feasible), the
   manipulability frame;
5. a **fixed budget** of ``S`` seeds per design (SPEC §4.6): the first ``S`` after
   sorting; a design with fewer feasible seeds gets invalid lanes that score
   zero, never a different number of evaluations.

With mixed capsule radii (D19, Phase 11) the reference IK's feasibility and the polish, which apply one
radius to every capsule, are followed by the same gates with each capsule's own radius
(:func:`dkarma.core.contacts.reference_gates_per_capsule`).

:func:`reproject` is the differentiable per-morphology re-projection of a stored
seed (Gauss-Newton on the contact gaps with the sphere centre free to translate
along the pinch axis is *not* needed: the reference keeps the centre fixed and
moves the joints; we do the same with :func:`dkarma.solvers.projection.project`).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats.qmc import Sobol

from karma.math_utils import closest_points_on_segments
from karma_jax.ik import IKSettings, solve_pinch_ik
from karma_jax.kinematics import forward_kinematics
from karma_jax.rolling import StepParams
from karma_jax.seeds import _polish_one, _object_jacobian, _axis_segment

from ..core.morphology import HandModel
from ..core.topology import Topology
from ..core.contacts import ContactAssignment, mixed_radii, reference_gates_per_capsule
from ..core.kinematics import capsule_endpoints
from ..metrics.commands import seed_frame
from .projection import project


@dataclass(frozen=True)
class SeedConfig:
    n_linspace: int = 5
    n_sobol: int = 256
    n_random: int = 64
    random_seed: int = 42
    offsets_m: tuple[float, ...] = (0.0, -0.005, -0.01, 0.005, 0.01)
    dedup_rad: float = 0.1
    budget: int = 16                      # S: seeds per design (fixed)
    polish_iters: int = 5
    polish_step: float = 0.3
    ik: IKSettings = IKSettings(method="lm", iters=100, full_jacobian=True)
    max_batch: int = 4096


class SeedSet(NamedTuple):
    q: object          # [S, J]
    p: object          # [S, 3]
    frame: object      # [S, 3, 3]  manipulability frame (columns = principal directions)
    valid: object      # bool [S]
    n_feasible: int    # before the budget cut (host int)
    stats: dict


def joint_samples(hm: HandModel, cfg: SeedConfig) -> np.ndarray:
    """Deterministic joint configurations ``[N, J]`` (reference sampling)."""
    lo, hi = np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper)
    J = lo.shape[0]
    out = [np.zeros(J)]
    for i in range(J):
        for v in np.linspace(lo[i], hi[i], cfg.n_linspace):
            q = np.zeros(J); q[i] = v; out.append(q)
    if cfg.n_sobol > 0:
        for s in Sobol(d=J, scramble=False).random(cfg.n_sobol):
            out.append(lo + s * (hi - lo))
    rng = np.random.RandomState(cfg.random_seed)
    for _ in range(cfg.n_random):
        out.append(lo + rng.rand(J) * (hi - lo))
    return np.stack(out)


def candidates(hm: HandModel, topo: Topology, sp: StepParams, cfg: SeedConfig, configs=None):
    """``(q [K, J], centres [K, 3], stats)``: reachable samples × candidate centres."""
    if configs is None:
        configs = joint_samples(hm, cfg)
    ct, ci = topo.seed_caps()
    r_s, r_l = float(sp.sphere_radius), float(sp.link_radius)
    offsets = list(cfg.offsets_m)
    margin = max(abs(o) for o in offsets) * 2.0 if offsets else 0.02
    max_reach = 2.0 * (r_s + r_l) + margin

    @jax.jit
    def ends(qs):
        fk = jax.vmap(forward_kinematics, in_axes=(None, 0))(hm.dm, qs)
        a, b = jax.vmap(lambda f: capsule_endpoints(hm, f))(fk)
        return a[:, ct], b[:, ct], a[:, ci], b[:, ci]

    a_t, b_t, a_i, b_i = (np.asarray(x, float) for x in ends(jnp.asarray(configs, hm.dm.t_joint.dtype)))
    qs, cs = [], []
    n_reach = 0
    for k in range(configs.shape[0]):
        c0, d0, h0 = _axis_segment(a_t[k], b_t[k])
        c1, d1, h1 = _axis_segment(a_i[k], b_i[k])
        p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        if np.linalg.norm(p1 - p0) > max_reach:
            continue
        n_reach += 1
        mid = 0.5 * (p0 + p1)
        if np.linalg.norm(p1 - p0) < 1e-9:
            cents = [mid]
        else:
            cents = []
            for off in offsets:
                cents.append(mid + off * d0)
                if abs(off) > 1e-9:
                    cents.append(mid + off * d1)
        for c in cents:
            qs.append(configs[k]); cs.append(c)
    stats = dict(n_joint_configs=int(configs.shape[0]), n_reachable=n_reach, n_candidates=len(qs))
    if not qs:
        return np.zeros((0, configs.shape[1])), np.zeros((0, 3)), stats
    return np.stack(qs), np.stack(cs), stats


def solve_candidates(hm: HandModel, topo: Topology, sp: StepParams, cfg: SeedConfig, q0, centres, dtype=jnp.float64):
    """Batched IK on all candidates. Returns host arrays ``(q [K,J], p [K,3], max_gap [K], feasible [K])``."""
    pair = jnp.asarray(topo.seed_caps(), jnp.int32)
    f = jax.jit(jax.vmap(lambda q, c: solve_pinch_ik(hm.dm, q, c, pair, sp, cfg.ik)))
    K = q0.shape[0]
    qs, ps, gs, fs = [], [], [], []
    for s in range(0, K, cfg.max_batch):
        e = min(K, s + cfg.max_batch)
        res = f(jnp.asarray(q0[s:e], dtype), jnp.asarray(centres[s:e], dtype))
        qs.append(np.asarray(res.q)); ps.append(np.asarray(res.p)); gs.append(np.asarray(res.max_gap)); fs.append(np.asarray(res.feasible))
    if not qs:
        J = hm.dm.q_ids.shape[0]
        return np.zeros((0, J)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, bool)
    return np.concatenate(qs), np.concatenate(ps), np.concatenate(gs), np.concatenate(fs)


def envelope_ok(hm: HandModel, topo: Topology, sp: StepParams, q, p, dtype=jnp.float64) -> np.ndarray:
    """bool ``[K]``: the seed pair's states pass the reference gates with each capsule's own radius (D19)."""
    pair = jnp.asarray(topo.seed_caps(), jnp.int32)
    f = jax.jit(jax.vmap(lambda qq, pp: jnp.stack(reference_gates_per_capsule(hm, forward_kinematics(hm.dm, qq), pp, sp.sphere_radius, pair,
                                                                              sp.eps_g, sp.eps_col))))
    if np.asarray(q).shape[0] == 0:
        return np.zeros(0, bool)
    return np.all(np.asarray(f(jnp.asarray(q, dtype), jnp.asarray(p, dtype))), axis=1)


def select(q, p, max_gap, feasible, cfg: SeedConfig) -> np.ndarray:
    """Indices of the budgeted seeds: feasible, deduplicated, sorted by ``(max_gap, q)``."""
    idx = np.nonzero(feasible)[0]
    order = sorted(idx, key=lambda i: (float(max_gap[i]), tuple(np.round(q[i], 8))))
    kept = []
    for i in order:
        if any(np.linalg.norm(q[i] - q[j]) < cfg.dedup_rad for j in kept):
            continue
        kept.append(i)
        if len(kept) >= cfg.budget:
            break
    return np.asarray(kept, dtype=int)


def polish_and_frame(hm: HandModel, topo: Topology, sp: StepParams, q, p, cfg: SeedConfig, dtype=jnp.float64):
    """Nullspace polish (kept if still feasible) and the manipulability frame, batched."""
    pair = jnp.asarray(topo.seed_caps(), jnp.int32)
    pol = jax.jit(jax.vmap(lambda qq, pp: _polish_one(hm.dm, sp, qq, pp, pair, cfg.polish_iters, cfg.polish_step)))
    q_new, gap_new, feas = pol(jnp.asarray(q, dtype), jnp.asarray(p, dtype))
    if mixed_radii(hm, sp.link_radius):            # D19: the polished state must also pass the per-capsule gates
        feas = jnp.asarray(np.asarray(feas) & envelope_ok(hm, topo, sp, q_new, p, dtype))
    q_out = jnp.where(feas[:, None], q_new, jnp.asarray(q, dtype))
    jac = jax.jit(jax.vmap(lambda qq, pp: _object_jacobian(hm.dm, sp, qq, pp, pair)))
    J = jac(q_out, jnp.asarray(p, dtype))
    frames = jax.vmap(seed_frame)(J)
    return np.asarray(q_out), np.asarray(frames), np.asarray(feas)


def find_seeds(hm: HandModel, topo: Topology, sp: StepParams, cfg: SeedConfig = SeedConfig(),
               dtype=jnp.float64) -> SeedSet:
    """The whole pipeline; deterministic for a given design and config."""
    q0, centres, stats = candidates(hm, topo, sp, cfg)
    q, p, gap, feas = solve_candidates(hm, topo, sp, cfg, q0, centres, dtype)
    stats.update(n_ik=int(q.shape[0]), n_feasible=int(feas.sum()))
    if mixed_radii(hm, sp.link_radius):            # D19: the reference feasibility with each capsule's own radius
        env = envelope_ok(hm, topo, sp, q[feas], p[feas], dtype)
        f_idx = np.nonzero(feas)[0]
        feas = feas.copy(); feas[f_idx[~env]] = False
        stats.update(n_envelope_rejected=int((~env).sum()), n_feasible=int(feas.sum()))
    keep = select(q, p, gap, feas, cfg)
    S, J = cfg.budget, hm.dm.q_ids.shape[0]
    q_s = np.zeros((S, J)); p_s = np.zeros((S, 3)); frame = np.tile(np.eye(3), (S, 1, 1)); valid = np.zeros(S, bool)
    if keep.size:
        qk, fk, polished = polish_and_frame(hm, topo, sp, q[keep], p[keep], cfg, dtype)
        n = keep.size
        q_s[:n], p_s[:n], frame[:n], valid[:n] = qk, p[keep], fk, True
        stats["n_polished"] = int(polished.sum())
    stats["n_selected"] = int(keep.size)
    return SeedSet(jnp.asarray(q_s, dtype), jnp.asarray(p_s, dtype), jnp.asarray(frame, dtype), jnp.asarray(valid),
                   int(feas.sum()), stats)


def assignment(topo: Topology, n_lanes: int = 2, fingers: tuple[int, ...] | None = None) -> ContactAssignment:
    """Distal capsules of ``fingers`` (default: the pair, thumb-like first) as contact lanes, padded."""
    if fingers is None:
        caps_list = list(topo.seed_caps())
    else:
        caps_list = [int(topo.finger_caps[f, 0]) for f in fingers]
    n_lanes = max(n_lanes, len(caps_list))
    caps = np.zeros(n_lanes, np.int32); valid = np.zeros(n_lanes, bool)
    caps[: len(caps_list)] = caps_list; valid[: len(caps_list)] = True
    return ContactAssignment(jnp.asarray(caps), jnp.asarray(valid))


# ── three-contact seeds (SPEC §6.3) ──────────────────────────────────────────

def _multi_ik_residual(hm: HandModel, x, q_init, centre0, asg: ContactAssignment, sp: StepParams, beta: float):
    """``[q, dp/r_s] -> residuals``: gaps of the assigned capsules, soft object and self clearance
    hinges, and joint regularization (the reference's weights 0.35 / 0.25 / 0.1)."""
    from ..core.contacts import contact_pass as _cp, object_clearance as _oc, self_clearance as _sc
    from ..core import geometry as _geo
    from karma_jax.kinematics import forward_kinematics as _fk
    J = q_init.shape[0]
    q = x[:J]
    p = centre0 + sp.sphere_radius * x[J:]
    fkr = _fk(hm.dm, q)
    cp = _geo.capsule_pass(hm, fkr, p, sp.sphere_radius)
    cs = _cp(hm, fkr, p, sp.sphere_radius, asg)
    r_gap = cs.gap / sp.eps_g * 0.35 * asg.valid.astype(x.dtype)
    oc = _oc(hm, cp, asg, beta=beta)
    scl = _sc(hm, cp, beta=beta)
    r_col = jnp.stack([_geo.hinge((sp.eps_col - oc) / sp.eps_col, 0.05) * 0.25,
                       _geo.hinge((sp.eps_col - scl) / sp.eps_col, 0.05) * 0.25])
    return jnp.concatenate([r_gap, r_col, 0.1 * (q - q_init)])


def solve_multi_ik(hm: HandModel, q_init, centre0, asg: ContactAssignment, sp: StepParams, iters: int = 60,
                   beta: float = 500.0):
    """Projected Levenberg-Marquardt (joint box, sphere box ±3(1 + r_l/r_s)) on the multi-contact
    residual. Returns ``(q, p, max_gap, cost)``."""
    dtype = q_init.dtype
    J = q_init.shape[0]
    lo = jnp.concatenate([hm.dm.q_lower, jnp.full((3,), -3.0 * (1.0 + sp.link_radius / sp.sphere_radius), dtype)])
    hi = jnp.concatenate([hm.dm.q_upper, jnp.full((3,), 3.0 * (1.0 + sp.link_radius / sp.sphere_radius), dtype)])
    x0 = jnp.clip(jnp.concatenate([q_init, jnp.zeros((3,), dtype)]), lo, hi)
    r_of = lambda x: _multi_ik_residual(hm, x, q_init, centre0, asg, sp, beta)   # noqa: E731
    n = x0.shape[0]
    eye = jnp.eye(n, dtype=dtype)
    r0 = r_of(x0); J0 = jax.jacfwd(r_of)(x0)
    lam0 = 1e-3 * jnp.max(jnp.diag(J0.T @ J0)) + 1e-12

    def body(_, st):
        x, cost, lam, nu = st
        r = r_of(x); Jr = jax.jacfwd(r_of)(x)
        JtJ = Jr.T @ Jr; g = Jr.T @ r
        step = -jnp.linalg.solve(JtJ + lam * eye, g)
        x_new = jnp.clip(x + step, lo, hi)
        step = x_new - x
        cost_new = jnp.sum(r_of(x_new) ** 2)
        pred = -(jnp.dot(step, 2.0 * g) + step @ (JtJ @ step))
        rho = (cost - cost_new) / jnp.where(pred > 0, pred, 1e-30)
        accept = (cost_new < cost) & jnp.isfinite(cost_new)
        lam = jnp.clip(jnp.where(accept, lam * jnp.maximum(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3), lam * nu), 1e-12, 1e12)
        nu = jnp.where(accept, 2.0, nu * 2.0)
        return jnp.where(accept, x_new, x), jnp.where(accept, cost_new, cost), lam, nu

    x, cost, _, _ = jax.lax.fori_loop(0, iters, body, (x0, jnp.sum(r0 ** 2), lam0, jnp.asarray(2.0, dtype)))
    q = x[:J]
    p = centre0 + sp.sphere_radius * x[J:]
    from ..core.contacts import contact_pass as _cp
    from karma_jax.kinematics import forward_kinematics as _fk
    cs = _cp(hm, _fk(hm.dm, q), p, sp.sphere_radius, asg)
    return q, p, jnp.max(jnp.abs(cs.gap) * asg.valid), cost


def multi_candidates(hm: HandModel, topo: Topology, fingers: tuple[int, ...], sp: StepParams, cfg: SeedConfig,
                     configs=None, centre_caps: str = "tips", n_restarts: int = 0, restart_scale: float = 0.2,
                     restart_seed: int = 7):
    """Joint samples whose listed fingertips can all meet around one sphere: centre proposals at
    the centroid of the fingertip capsules' closest points plus offsets along the triangle normal.
    Phase 8.5 options: ``centre_caps="all"`` also proposes centres from every triple of phalanges
    (one capsule per finger) that can meet; ``n_restarts`` adds perturbed joint initializations
    (``restart_scale`` rad, ``RandomState(restart_seed)``) for every candidate."""
    if configs is None:
        configs = joint_samples(hm, cfg)
    tips = [int(topo.finger_caps[f, 0]) for f in fingers]
    if centre_caps == "all":
        per_finger = [[int(c) for c in topo.finger_caps[f] if int(c) >= 0] for f in fingers]
    else:
        per_finger = [[c] for c in tips]
    import itertools
    triples = list(itertools.product(*per_finger))
    all_caps = sorted({c for t in triples for c in t})
    col = {c: i for i, c in enumerate(all_caps)}
    r_s = float(sp.sphere_radius)
    rad = {c: float(np.asarray(hm.cap_radius)[c]) for c in all_caps}       # each capsule's own radius (the tips': the contact radius)
    margin = 2.0 * max(abs(o) for o in cfg.offsets_m)

    @jax.jit
    def ends(qs):
        fk = jax.vmap(forward_kinematics, in_axes=(None, 0))(hm.dm, qs)
        a, b = jax.vmap(lambda f: capsule_endpoints(hm, f))(fk)
        return a[:, all_caps], b[:, all_caps]

    a, b = (np.asarray(x, float) for x in ends(jnp.asarray(configs, hm.dm.t_joint.dtype)))
    qs, cs = [], []
    n_reach = 0
    rng = np.random.RandomState(restart_seed)
    lo, hi = np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper)
    for k in range(configs.shape[0]):
        segs = {c: _axis_segment(a[k, col[c]], b[k, col[c]]) for c in all_caps}
        reached = False
        for tri_caps in triples:
            pts = []
            ok = True
            for i in range(len(tri_caps)):
                for j in range(i + 1, len(tri_caps)):
                    p0, p1 = closest_points_on_segments(*segs[tri_caps[i]], *segs[tri_caps[j]])
                    if np.linalg.norm(p1 - p0) > (r_s + rad[tri_caps[i]]) + (r_s + rad[tri_caps[j]]) + margin:
                        ok = False
                    pts += [p0, p1]
            if not ok:
                continue
            reached = True
            mid = np.mean(pts, axis=0)
            tri = np.cross(pts[1] - pts[0], pts[2] - pts[0]) if len(pts) >= 3 else np.array([0.0, 0.0, 1.0])
            nrm = np.linalg.norm(tri)
            tri = tri / nrm if nrm > 1e-12 else np.array([0.0, 0.0, 1.0])
            for off in (0.0, 0.5 * r_s, -0.5 * r_s):
                c0 = mid + off * tri
                qs.append(configs[k]); cs.append(c0)
                for _ in range(n_restarts):
                    qs.append(np.clip(configs[k] + restart_scale * rng.randn(configs.shape[1]), lo, hi)); cs.append(c0)
        n_reach += int(reached)
    stats = dict(n_joint_configs=int(configs.shape[0]), n_reachable=n_reach, n_candidates=len(qs), n_triples=len(triples),
                 n_restarts=n_restarts)
    if not qs:
        return np.zeros((0, configs.shape[1])), np.zeros((0, 3)), stats
    return np.stack(qs), np.stack(cs), stats


def find_multi_seeds(hm: HandModel, topo: Topology, fingers: tuple[int, ...], sp: StepParams, act,
                     cfg: SeedConfig = SeedConfig(), dtype=jnp.float64, n_lanes: int = 3, centre_caps: str = "tips",
                     n_restarts: int = 0, candidates=None, keep_all: bool = False) -> SeedSet:
    """Three-contact seeds (SPEC §6.3): candidates -> multi-contact IK -> exact verification
    (gaps, clearances, limits, frictional equilibrium with unlimited torque) -> dedup -> budget.
    ``centre_caps`` / ``n_restarts``: see :func:`multi_candidates`; ``candidates=(q0, centres)``
    replaces the candidate generation (warm starts, Phase 8.5); ``keep_all`` verifies every
    deduplicated kinematic solution instead of stopping at the budget (counts in ``stats``)."""
    from ..core.contacts import contact_pass as _cp, object_clearance as _oc, self_clearance as _sc
    from ..core.actuation import grasp_matrix, contact_torque_map, gravity_torque, angular_jacobians
    from .contact_forces import build_problem, solve_exact, ForceWeights
    from karma_jax.seeds import _object_jacobian as _oj
    asg = assignment(topo, n_lanes, fingers)
    if candidates is None:
        q0, centres, stats = multi_candidates(hm, topo, fingers, sp, cfg, centre_caps=centre_caps, n_restarts=n_restarts)
    else:
        q0, centres = (np.asarray(candidates[0], float), np.asarray(candidates[1], float))
        stats = dict(n_joint_configs=int(q0.shape[0]), n_reachable=int(q0.shape[0]), n_candidates=int(q0.shape[0]))
    S, J = cfg.budget, hm.dm.q_ids.shape[0]
    out_q = np.zeros((S, J)); out_p = np.zeros((S, 3)); out_f = np.tile(np.eye(3), (S, 1, 1)); valid = np.zeros(S, bool)
    if q0.shape[0] == 0:
        return SeedSet(jnp.asarray(out_q, dtype), jnp.asarray(out_p, dtype), jnp.asarray(out_f, dtype), jnp.asarray(valid), 0, stats)
    f = jax.jit(jax.vmap(lambda q, c: solve_multi_ik(hm, q, c, asg, sp)))
    qs, ps, gaps = [], [], []
    for s in range(0, q0.shape[0], cfg.max_batch):
        e = min(q0.shape[0], s + cfg.max_batch)
        q, p, g, _ = f(jnp.asarray(q0[s:e], dtype), jnp.asarray(centres[s:e], dtype))
        qs.append(np.asarray(q)); ps.append(np.asarray(p)); gaps.append(np.asarray(g))
    q, p, gap = np.concatenate(qs), np.concatenate(ps), np.concatenate(gaps)
    # exact verification
    check = jax.jit(jax.vmap(lambda qq, pp: _verify_multi(hm, qq, pp, asg, sp)))
    gmax, oc, scl = (np.asarray(x) for x in check(jnp.asarray(q, dtype), jnp.asarray(p, dtype)))
    kin_ok = (gmax <= float(sp.eps_g)) & (oc >= float(sp.eps_col)) & (scl >= float(sp.eps_col))
    stats.update(n_ik=int(q.shape[0]), n_kinematic_ok=int(kin_ok.sum()))
    order = [i for i in sorted(np.nonzero(kin_ok)[0], key=lambda i: (float(gap[i]), tuple(np.round(q[i], 8))))]
    kept = []
    rejected_force = []
    wts = ForceWeights()
    n_force_checked = 0
    n_dedup = 0
    seen = np.zeros((0, q.shape[1]))
    for i in order:
        if seen.shape[0] and np.any(np.linalg.norm(seen - q[i][None, :], axis=1) < cfg.dedup_rad):
            continue
        seen = np.concatenate([seen, q[i][None, :]])
        n_dedup += 1
        fkr = forward_kinematics(hm.dm, jnp.asarray(q[i], dtype))
        cs = _cp(hm, fkr, jnp.asarray(p[i], dtype), sp.sphere_radius, asg)
        pb = build_problem(grasp_matrix(cs.p_sphere, jnp.asarray(p[i], dtype), asg.valid), contact_torque_map(cs.J_surf, asg.valid),
                           angular_jacobians(cs.Jw, asg.valid), gravity_torque(hm, fkr, act), cs.normal, asg.valid,
                           1e3 * act.tau_bar, act.mu, wts.torsion_radius_m)
        n_force_checked += 1
        if not solve_exact(pb, act.w_ext)["feasible"]:
            rejected_force.append(i)
            continue
        kept.append(i)
        if len(kept) >= S and not keep_all:
            break
    stats.update(n_force_checked=n_force_checked, n_distinct_kinematic=n_dedup, n_equilibrium_ok=len(kept),
                 n_equilibrium_rejected=len(rejected_force), n_selected=min(len(kept), S))
    kept = kept[:S]
    if kept:
        qk = jnp.asarray(q[kept], dtype); pk = jnp.asarray(p[kept], dtype)
        pair = jnp.asarray([int(asg.caps[0]), int(asg.caps[1])], jnp.int32)
        Jobj = jax.vmap(lambda qq, pp: _oj(hm.dm, sp, qq, pp, pair))(qk, pk)   # pinch pair frame (shared convention)
        frames = jax.vmap(seed_frame)(Jobj)
        n = len(kept)
        out_q[:n], out_p[:n], out_f[:n], valid[:n] = np.asarray(qk), np.asarray(pk), np.asarray(frames), True
    return SeedSet(jnp.asarray(out_q, dtype), jnp.asarray(out_p, dtype), jnp.asarray(out_f, dtype), jnp.asarray(valid),
                   int(kin_ok.sum()), stats)


def _verify_multi(hm, q, p, asg, sp):
    from ..core.contacts import contact_pass as _cp, object_clearance as _oc, self_clearance as _sc
    from ..core import geometry as _geo
    fkr = forward_kinematics(hm.dm, q)
    cp = _geo.capsule_pass(hm, fkr, p, sp.sphere_radius)
    cs = _cp(hm, fkr, p, sp.sphere_radius, asg)
    return jnp.max(jnp.abs(cs.gap) * asg.valid), _oc(hm, cp, asg, beta=None), _sc(hm, cp, beta=None)


def reproject(hm: HandModel, q, p, asg: ContactAssignment, sp: StepParams, iters: int = 10):
    """Differentiable re-projection of a stored seed onto the current morphology's contact
    manifold (joints move, centre fixed). Returns ``(q_proj, ok, gap_max)``."""
    pr = project(hm, q, p, asg, sp, max_iters=iters, backtrack=False)
    return pr.q, pr.ok, pr.gap_max


__all__ = ["SeedConfig", "SeedSet", "joint_samples", "candidates", "solve_candidates", "select",
           "polish_and_frame", "envelope_ok", "find_seeds", "assignment", "reproject", "solve_multi_ik", "multi_candidates",
           "find_multi_seeds"]
