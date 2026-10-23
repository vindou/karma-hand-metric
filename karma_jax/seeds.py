"""Seed generation, polishing and grid frames (ports of ``karma/seed.py`` and the
non-search parts of ``karma/seed_selection.py``).

Joint-space sampling, candidate centres, dedup and sorting follow the
reference exactly (same Sobol/random streams, same offsets, same keys), so the
candidate list is identical; only the IK solver differs (:mod:`karma_jax.ik`).
Seeds are returned as :class:`karma.seed.SeedResult` so the experiment scripts
that pass seeds around keep working.
"""
from __future__ import annotations

import logging

import numpy as np
import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol

from karma.seed import SeedResult
from karma.math_utils import closest_points_on_segments, compute_seed_frame

from .model import FlatModel, DeviceModel
from .kinematics import (forward_kinematics, fold_active, active_limits, get_active, set_active,
                         apply_mimic, skew, fk_batched)
from .geometry import (capsule_pass, capsule_endpoints, contact_kinematics, check_collisions,
                       check_self_collisions)
from .rolling import StepParams
from .ik import make_batched_ik, ChunkedIK, IKSettings
from .search import step_params_from_cfg, _bucket

logger = logging.getLogger(__name__)

N_LINSPACE_PER_JOINT = 5
N_SOBOL = 512
N_RANDOM = 200
RANDOM_SEED = 42
POLISH_ITERS = 5
POLISH_STEP = 0.3


# ── Sampling and candidates ─────────────────────────────────────────────────

def sample_joint_configs(flat: FlatModel, n_linspace: int = N_LINSPACE_PER_JOINT,
                         n_sobol: int = N_SOBOL, n_random: int = N_RANDOM) -> np.ndarray:
    """Reference sampling: neutral, per-joint linspace, unscrambled Sobol, RandomState(42)."""
    nq, m = flat.n_joints, flat.n_active
    q_ids = flat.q_ids[:m]
    q_lo = flat.q_lower[q_ids]
    q_hi = flat.q_upper[q_ids]
    q0 = np.zeros(nq)
    configs = [q0.copy()]
    for i in range(m):
        for v in np.linspace(q_lo[i], q_hi[i], n_linspace):
            q = q0.copy()
            q[q_ids[i]] = v
            configs.append(q)
    if n_sobol > 0:
        for s in Sobol(d=m, scramble=False).random(n_sobol):
            q = q0.copy()
            q[q_ids] = q_lo + s * (q_hi - q_lo)
            configs.append(q)
    rng = np.random.RandomState(RANDOM_SEED)
    for _ in range(n_random):
        q = q0.copy()
        q[q_ids] = q_lo + rng.rand(m) * (q_hi - q_lo)
        configs.append(q)
    return np.stack(configs)


def _pad_q(flat: FlatModel, qs: np.ndarray) -> np.ndarray:
    out = np.zeros((qs.shape[0], flat.J_pad))
    out[:, : flat.n_joints] = qs
    return out


def seed_capsule_endpoints(flat: FlatModel, dm: DeviceModel, configs: np.ndarray, dtype):
    """World endpoints of the two seed capsules for every config: ``(a_t, b_t, a_i, b_i)`` each ``[N,3]``."""
    ct, ci = flat.seed_thumb_cap, flat.seed_index_cap

    @jax.jit
    def f(q):
        fk = fk_batched(dm, q)
        a, b = jax.vmap(lambda fk1: capsule_endpoints(dm, fk1))(fk)
        return a[:, ct], b[:, ct], a[:, ci], b[:, ci]

    out = f(jnp.asarray(_pad_q(flat, configs), dtype))
    return tuple(np.asarray(o, float) for o in out)


def _axis_segment(a, b):
    c = 0.5 * (a + b)
    seg = b - a
    L = float(np.linalg.norm(seg))
    if L < 1e-12:
        return c, np.array([0.0, 0.0, 1.0]), 0.0
    return c, seg / L, 0.5 * L


def candidate_centres(a_t, b_t, a_i, b_i, offsets: list[float]) -> list[np.ndarray]:
    """Reference ``_candidate_sphere_centres`` from the two seed capsules' endpoints."""
    c0, d0, h0 = _axis_segment(a_t, b_t)
    c1, d1, h1 = _axis_segment(a_i, b_i)
    pt0, pt1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
    midpoint = 0.5 * (pt0 + pt1)
    if np.linalg.norm(pt1 - pt0) < 1e-9:
        return [midpoint]
    out = []
    for off in offsets:
        out.append(midpoint + off * d0)
        if abs(off) > 1e-9:
            out.append(midpoint + off * d1)
    return out


def generate_candidates(flat: FlatModel, dm: DeviceModel, cfg, dtype, configs: np.ndarray | None = None,
                        offsets: list[float] | None = None):
    """``(q_samples [K,nq], centres [K,3], stats)`` for all reachable configs, as the reference."""
    if configs is None:
        configs = sample_joint_configs(flat)
    offsets = list(cfg.seed_search_offsets_m) if offsets is None else offsets
    seed_margin = max(abs(o) for o in offsets) * 2.0 if offsets else 0.02
    max_reach = 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m) + seed_margin
    a_t, b_t, a_i, b_i = seed_capsule_endpoints(flat, dm, configs, dtype)
    qs, cs = [], []
    n_reach = 0
    for k in range(configs.shape[0]):
        c0, d0, h0 = _axis_segment(a_t[k], b_t[k])
        c1, d1, h1 = _axis_segment(a_i[k], b_i[k])
        p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        if np.linalg.norm(p1 - p0) > max_reach:
            continue
        n_reach += 1
        for c in candidate_centres(a_t[k], b_t[k], a_i[k], b_i[k], offsets):
            qs.append(configs[k])
            cs.append(c)
    stats = dict(n_joint_configs=int(configs.shape[0]), n_reachable=n_reach, n_candidates=len(qs))
    if not qs:
        return np.zeros((0, flat.n_joints)), np.zeros((0, 3)), stats
    return np.stack(qs), np.stack(cs), stats


# ── IK over candidates ───────────────────────────────────────────────────────

def solve_candidates(flat: FlatModel, dm: DeviceModel, params: StepParams, dtype, q_samples, centres,
                     settings: IKSettings = IKSettings(), max_batch: int = 1 << 14) -> list[SeedResult]:
    """Batched IK on every candidate; returns a SeedResult per candidate (reference semantics)."""
    K = q_samples.shape[0]
    if K == 0:
        return []
    # Chunked TRF with compaction: ~2.7x faster than a plain vmapped while_loop
    # on LEAP (lanes need a median of 15 but up to 500 evaluations).
    ik = ChunkedIK(dm, params, settings) if settings.method == "trf" else make_batched_ik(dm, params, settings)
    names = flat.capsule_names
    pair_names = (names[flat.seed_thumb_cap], names[flat.seed_index_cap])
    pair = np.tile([[flat.seed_thumb_cap, flat.seed_index_cap]], (K, 1)).astype(np.int32)
    qp = _pad_q(flat, q_samples)
    out: list[SeedResult] = []
    for s in range(0, K, max_batch):
        e = min(K, s + max_batch)
        n = e - s
        nb = _bucket(n, hi=max_batch)
        pad = nb - n
        padded = lambda a: np.concatenate([a[s:e], np.repeat(a[e - 1:e], pad, axis=0)], axis=0) if pad else a[s:e]  # noqa: E731
        res = jax.device_get(ik(jnp.asarray(padded(qp), dtype), jnp.asarray(padded(centres), dtype),
                                jnp.asarray(padded(pair), jnp.int32)))
        for i in range(n):
            out.append(SeedResult(
                q=np.asarray(res.q[i], float)[: flat.n_joints],
                sphere_centre_world=np.asarray(res.p[i], float),
                contact_pair=pair_names,
                max_gap=float(res.max_gap[i]),
                feasible=bool(res.feasible[i]),
            ))
    return out


def deduplicate_seeds(seeds: list[SeedResult], q_ids: np.ndarray, min_dist: float = 0.1) -> list[SeedResult]:
    kept, kept_qs = [], []
    for s in seeds:
        qa = s.q[q_ids]
        if any(np.linalg.norm(qa - kq) < min_dist for kq in kept_qs):
            continue
        kept.append(s)
        kept_qs.append(qa)
    return kept


def _sort_key(r: SeedResult):
    return (r.max_gap, tuple(r.q.round(8)))


def generate_feasible_seeds(flat: FlatModel, dm: DeviceModel, cfg, dtype,
                            ik_settings: IKSettings = IKSettings()) -> tuple[list[SeedResult], dict]:
    """Port of ``_generate_feasible_seeds``: all feasible seeds after light dedup, sorted by gap."""
    params = step_params_from_cfg(cfg, dtype)
    q_ids = flat.q_ids[: flat.n_active]
    q_samples, centres, stats = generate_candidates(flat, dm, cfg, dtype)
    logger.info("Seed selection: sampling %d joint configurations", stats["n_joint_configs"])
    logger.info("Seed selection: %d reachable configs, %d candidates to solve",
                stats["n_reachable"], stats["n_candidates"])
    results = solve_candidates(flat, dm, params, dtype, q_samples, centres, ik_settings)
    feasible = [r for r in results if r.feasible]
    nonfeasible = [r for r in results if not r.feasible]
    logger.info("Seed selection: %d valid, %d feasible, %d non-feasible",
                len(results), len(feasible), len(nonfeasible))
    stats.update(n_ik_solutions=len(results), n_feasible=len(feasible), n_after_dedup=0)
    if feasible:
        feasible = deduplicate_seeds(feasible, q_ids, min_dist=0.01)
        feasible.sort(key=_sort_key)
        stats["n_after_dedup"] = len(feasible)
        return feasible, stats
    if nonfeasible:
        nonfeasible.sort(key=_sort_key)
        stats["n_after_dedup"] = 1
        return nonfeasible[:1], stats
    # last resort: neutral configuration, sphere between the seed capsules
    a_t, b_t, a_i, b_i = seed_capsule_endpoints(flat, dm, np.zeros((1, flat.n_joints)), dtype)
    c0, d0, h0 = _axis_segment(a_t[0], b_t[0])
    c1, d1, h1 = _axis_segment(a_i[0], b_i[0])
    p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
    names = flat.capsule_names
    stats["n_after_dedup"] = 1
    return [SeedResult(q=np.zeros(flat.n_joints), sphere_centre_world=0.5 * (p0 + p1),
                       contact_pair=(names[flat.seed_thumb_cap], names[flat.seed_index_cap]),
                       max_gap=float("inf"), feasible=False)], stats


# ── Object Jacobian, polishing, grid frame ───────────────────────────────────

def _object_jacobian(dm: DeviceModel, params: StepParams, q, p, pair):
    """``J_obj = 0.5 (J_surf_t + J_surf_i)`` on the active joints, [3,M] (mimic folded)."""
    r_s, r_l = params.sphere_radius, params.link_radius
    fk = forward_kinematics(dm, q)
    cp = capsule_pass(dm, fk, p, r_s, r_l)
    J = 0.0
    for c in (pair[0], pair[1]):
        ck = contact_kinematics(dm, fk, cp, c, p, r_s)
        p_capsule = ck.axis_point + r_l * ck.normal
        r = p_capsule - ck.origin
        J = J + fold_active(dm, ck.Jv_origin) - skew(r) @ fold_active(dm, ck.Jw)
    return 0.5 * J


def _polish_one(dm: DeviceModel, params: StepParams, q, p, pair, n_iters: int, step_size: float):
    q_lo, q_hi = active_limits(dm)
    valid = dm.active_valid
    q_center = 0.5 * (q_lo + q_hi)

    def body(_, q):
        J = _object_jacobian(dm, params, q, p, pair)
        M = J.shape[1]
        N = jnp.eye(M, dtype=J.dtype) - jnp.linalg.pinv(J) @ J
        qa = get_active(dm, q)
        dq = step_size * (N @ ((q_center - qa) * valid.astype(J.dtype)))
        qa_new = jnp.clip(qa + dq, q_lo, q_hi) * valid.astype(J.dtype)
        return set_active(dm, q, qa_new)

    q = jax.lax.fori_loop(0, n_iters, body, q)
    # re-validate
    r_s, r_l = params.sphere_radius, params.link_radius
    fk = forward_kinematics(dm, q)
    cp = capsule_pass(dm, fk, p, r_s, r_l)
    ck_t = contact_kinematics(dm, fk, cp, pair[0], p, r_s)
    ck_i = contact_kinematics(dm, fk, cp, pair[1], p, r_s)
    max_gap = jnp.maximum(jnp.abs(ck_t.g), jnp.abs(ck_i.g))
    feasible = max_gap <= params.eps_g
    col_ok, _ = check_collisions(dm, cp, pair, params.eps_col)
    self_ok, _ = check_self_collisions(dm, cp, r_l, params.eps_col)
    feasible = feasible & col_ok & self_ok
    return apply_mimic(dm, q), max_gap, feasible


def polish_seeds(flat: FlatModel, dm: DeviceModel, cfg, dtype, seeds: list[SeedResult],
                 n_iters: int = POLISH_ITERS, step_size: float = POLISH_STEP) -> list[SeedResult]:
    """Port of ``_polish_seeds``: nullspace push toward joint-range centre, keep only if still feasible."""
    if not seeds:
        return seeds
    params = step_params_from_cfg(cfg, dtype)
    names = flat.capsule_names
    f = jax.jit(jax.vmap(lambda q, p, pair: _polish_one(dm, params, q, p, pair, n_iters, step_size)))
    B = len(seeds)
    q = _pad_q(flat, np.stack([s.q for s in seeds]))
    p = np.stack([s.sphere_centre_world for s in seeds])
    pair = np.array([[names.index(s.contact_pair[0]), names.index(s.contact_pair[1])] for s in seeds], np.int32)
    q_new, max_gap, feas = jax.device_get(f(jnp.asarray(q, dtype), jnp.asarray(p, dtype), jnp.asarray(pair)))
    out = []
    n_improved = 0
    for i, s in enumerate(seeds):
        if bool(feas[i]):
            out.append(SeedResult(q=np.asarray(q_new[i], float)[: flat.n_joints],
                                  sphere_centre_world=s.sphere_centre_world.copy(),
                                  contact_pair=s.contact_pair, max_gap=float(max_gap[i]), feasible=True))
            n_improved += 1
        else:
            out.append(s)
    logger.info("Seed polishing: %d/%d seeds improved", n_improved, B)
    return out


def trial_frames(flat: FlatModel, dm: DeviceModel, cfg, dtype, seeds: list[SeedResult]) -> list[np.ndarray]:
    """``R_seed`` per seed from the manipulability ellipsoid (reference ``_compute_trial_frame``)."""
    if not seeds:
        return []
    params = step_params_from_cfg(cfg, dtype)
    names = flat.capsule_names
    f = jax.jit(jax.vmap(lambda q, p, pair: _object_jacobian(dm, params, q, p, pair)))
    q = _pad_q(flat, np.stack([s.q for s in seeds]))
    p = np.stack([s.sphere_centre_world for s in seeds])
    pair = np.array([[names.index(s.contact_pair[0]), names.index(s.contact_pair[1])] for s in seeds], np.int32)
    J = np.asarray(f(jnp.asarray(q, dtype), jnp.asarray(p, dtype), jnp.asarray(pair)), float)
    m = flat.n_active
    return [compute_seed_frame(J[i][:, :m]) for i in range(len(seeds))]
