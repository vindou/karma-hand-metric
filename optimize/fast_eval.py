"""Fast KaRMA proxy for use inside an optimization loop.

Reuses the library's own kernels (seed IK, nullspace polish, manipulability
frame, rolling-QP translation BFS, Phase-2 rotation exploration) unchanged, so
the proxy measures the same physics as the official metric. It is cheaper only
because it (a) samples fewer candidate pinches, (b) ranks seeds with a short
BFS and runs the full BFS only on a few finalists, and (c) optionally skips or
subsamples Phase 2. It runs single-threaded so a pool of these can evaluate a
whole CMA-ES population in parallel.

Official numbers for any design must come from ``karma.metric.compute_metric``;
see ``evaluate_full.py``.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
from scipy.stats.qmc import Sobol

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from karma.config import load_config, MetricConfig                       # noqa: E402
from karma.robot import (                                                  # noqa: E402
    RobotContext, load_robot, update_kinematics, update_positions_only,
    active_position_indices, active_velocity_indices, neutral_configuration,
)
from karma.seed import SeedResult, _solve_for_centre, _candidate_sphere_centres   # noqa: E402
from karma.seed_selection import (                                         # noqa: E402
    _deduplicate_seeds, _polish_seeds, _compute_trial_frame, _run_trial_bfs,
)
from karma.contacts import contact_kinematics, _capsule_axis_segment       # noqa: E402
from karma.orientation import (                                            # noqa: E402
    compute_motion_primitives_frame, healpix_npix_lines, contact_line_axis_body,
    axis_to_healpix_bin,
)
from karma.math_utils import closest_points_on_segments                    # noqa: E402
from karma import metric as karma_metric                                   # noqa: E402

from optimize.hand_family import HandSpec                                  # noqa: E402

METRIC_CFG = REPO / "karma_config.yaml"


@dataclass
class FastEvalConfig:
    # seed sampling (library default: 1 + 5/joint + 512 Sobol + 200 random)
    n_linspace_per_joint: int = 3
    n_sobol: int = 128
    n_random: int = 32
    n_offsets: int = 3            # use the first k of cfg.seed_search_offsets_m (k=3 -> 5 centres)
    dedup_rad: float = 0.20       # library uses 0.01 (near-identical only)
    max_seeds: int = 24
    # two-stage seed ranking
    stage1_budget: int = 15
    n_finalists: int = 2
    final_budget: int = 1200
    # Phase 2 subsample (0 = skip rotation entirely)
    rotation_voxels: int = 0

    @staticmethod
    def loop() -> "FastEvalConfig":
        """Settings used inside the optimizer loop (fast)."""
        return FastEvalConfig()

    @staticmethod
    def rich() -> "FastEvalConfig":
        """GPU-loop setting: more seeds and the official voxel cap. On the GPU this costs
        about the same as loop() because seeds are batched; on the CPU it is ~4x slower."""
        return FastEvalConfig(n_linspace_per_joint=5, n_sobol=256, n_random=64, n_offsets=5,
                              dedup_rad=0.1, max_seeds=48, stage1_budget=15, n_finalists=3,
                              final_budget=10000)

    @staticmethod
    def careful() -> "FastEvalConfig":
        """Closer to the official pipeline: more seeds, deeper ranking, 3 finalists."""
        return FastEvalConfig(n_linspace_per_joint=5, n_sobol=256, n_random=64, n_offsets=5,
                              dedup_rad=0.1, max_seeds=60, stage1_budget=40, n_finalists=4,
                              final_budget=10000)


@dataclass
class FastEvalResult:
    karma_t: float
    karma_r: float | None
    top3_voxels: list[int]
    stage1_counts: list[int]
    n_seeds: int
    l_ref_m: float
    voxel_m: float
    reach_slack_m: float        # min over sampled poses of (tip-tip axis distance - 2(r_s + r_l)); <0 means tips can meet
    best_gap_m: float           # best IK gap among candidates (inf if none solved)
    gen_stats: dict
    elapsed_s: float
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ── Seed generation (reduced-sample version of karma.seed_selection._generate_feasible_seeds) ──

def generate_seeds(cfg: MetricConfig, ctx: RobotContext, fe: FastEvalConfig) -> tuple[list[SeedResult], dict]:
    q0 = neutral_configuration(ctx.model)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]
    n_active = len(q_ids)
    thumb_link, index_link = cfg.seed_thumb_link, cfg.seed_index_link
    offsets = list(cfg.seed_search_offsets_m[: fe.n_offsets]) or [0.0]

    seed_margin = max(abs(o) for o in cfg.seed_search_offsets_m) * 2.0
    max_reach = 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m) + seed_margin

    configs = [q0.copy()]
    for i in range(n_active):
        for v in np.linspace(q_lower[i], q_upper[i], fe.n_linspace_per_joint):
            q = q0.copy(); q[q_ids[i]] = v; configs.append(q)
    if fe.n_sobol > 0:
        for s in Sobol(d=n_active, scramble=False).random(fe.n_sobol):
            q = q0.copy(); q[q_ids] = q_lower + s * (q_upper - q_lower); configs.append(q)
    rng = np.random.RandomState(42)
    for _ in range(fe.n_random):
        q = q0.copy(); q[q_ids] = q_lower + rng.rand(n_active) * (q_upper - q_lower); configs.append(q)

    reach_slack = float("inf")
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for q in configs:
        update_positions_only(ctx, q)
        c0, d0, h0 = _capsule_axis_segment(ctx, thumb_link)
        c1, d1, h1 = _capsule_axis_segment(ctx, index_link)
        p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        dist = float(np.linalg.norm(p1 - p0))
        reach_slack = min(reach_slack, dist - 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m))
        if dist > max_reach:
            continue
        for centre in _candidate_sphere_centres(ctx, q, thumb_link, index_link, offsets):
            candidates.append((q.copy(), centre))

    results: list[SeedResult] = []
    for q, centre in candidates:
        r = _solve_for_centre(cfg, ctx, q, centre, thumb_link, index_link, q_ids, v_ids, q_lower, q_upper)
        if r is not None:
            results.append(r)
    best_gap = min((r.max_gap for r in results), default=float("inf"))
    feasible = [r for r in results if r.feasible]
    n_feasible = len(feasible)
    if feasible:
        feasible = _deduplicate_seeds(feasible, q_ids, min_dist=fe.dedup_rad)
        feasible.sort(key=lambda r: (r.max_gap, tuple(r.q.round(8))))
        feasible = feasible[: fe.max_seeds]

    stats = {
        "n_joint_configs": len(configs), "n_candidates": len(candidates),
        "n_ik_solutions": len(results), "n_feasible": n_feasible, "n_after_dedup": len(feasible),
        "reach_slack_m": reach_slack, "best_gap_m": best_gap,
    }
    return feasible, stats


# ── Phase-2 subsample ────────────────────────────────────────────────────────

def _rotation_coverage_subsample(cfg, ctx, discovered: dict, k: int) -> float:
    """Top-5-voxel mean orientation coverage over a k-voxel core subsample.

    Picks the k voxels nearest the centroid of the reachable set (the seed's
    workspace core is where the hand has the most joint slack, so this is a
    reasonable stand-in for the official top-5-over-all-voxels)."""
    voxels = list(discovered.keys())
    if len(voxels) > k:
        P = np.array([discovered[v].p for v in voxels])
        c = P.mean(axis=0)
        order = np.argsort(np.linalg.norm(P - c, axis=1))
        voxels = [voxels[i] for i in order[:k]]

    # _explore_rotations_at_voxel reloads the robot; hand it our context instead.
    orig = karma_metric.load_robot
    karma_metric.load_robot = lambda *a, **kw: ctx
    try:
        total = healpix_npix_lines(cfg.healpix_nside)
        covs = []
        for v in voxels:
            node = discovered[v]
            update_kinematics(ctx, node.q)
            ck_t = contact_kinematics(ctx, node.q, node.p, node.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m)
            ck_i = contact_kinematics(ctx, node.q, node.p, node.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m)
            ori_bin = axis_to_healpix_bin(contact_line_axis_body(ck_t.normal_world, ck_i.normal_world, node.R_sphere), cfg.healpix_nside)
            vc = karma_metric.VoxelConfig(voxel=v, centre_world=node.p.copy(), q=node.q.copy(),
                                          R_sphere=karma_metric._reorthogonalize(node.R_sphere),
                                          contact_pair=node.contact_pair, ori_bin=ori_bin)
            _, ori_configs, _ = karma_metric._explore_rotations_at_voxel(vc, cfg, None)
            covs.append(len(ori_configs) / total)
    finally:
        karma_metric.load_robot = orig
    covs.sort(reverse=True)
    return float(np.mean(covs[:5])) if covs else 0.0


# ── Main entry ───────────────────────────────────────────────────────────────

def evaluate(hand: HandSpec, workdir: str | Path, fe: FastEvalConfig | None = None,
             metric_cfg_path: str | Path = METRIC_CFG) -> FastEvalResult:
    """Write the design, load it with the CPU reference, run the proxy."""
    fe = fe or FastEvalConfig()
    t0 = time.perf_counter()
    try:
        _, yaml_path = hand.write(workdir)
        cfg = load_config(metric_cfg_path, yaml_path)
        ctx = load_robot(cfg.urdf_path, cfg)
    except Exception as e:  # malformed design
        return FastEvalResult(karma_t=0.0, karma_r=None, top3_voxels=[], stage1_counts=[], n_seeds=0,
                              l_ref_m=0.0, voxel_m=0.0, reach_slack_m=float("inf"), best_gap_m=float("inf"),
                              gen_stats={}, elapsed_s=time.perf_counter() - t0, error=f"load: {type(e).__name__}: {e}")
    return evaluate_loaded(cfg, ctx, fe)


def evaluate_loaded(cfg: MetricConfig, ctx: RobotContext, fe: FastEvalConfig | None = None) -> FastEvalResult:
    """The proxy on an already-loaded (cfg, ctx) pair, CPU reference kernels."""
    fe = fe or FastEvalConfig()
    t0 = time.perf_counter()
    empty = dict(karma_t=0.0, karma_r=None, top3_voxels=[], stage1_counts=[], n_seeds=0,
                 l_ref_m=cfg.l_ref_m, voxel_m=cfg.voxel_size_m, reach_slack_m=float("inf"), best_gap_m=float("inf"),
                 gen_stats={}, elapsed_s=0.0)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lower = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_upper = np.array(ctx.model.upperPositionLimit)[q_ids]

    try:
        seeds, gstats = generate_seeds(cfg, ctx, fe)
    except Exception as e:
        empty.update(elapsed_s=time.perf_counter() - t0, error=f"seeds: {type(e).__name__}: {e}")
        return FastEvalResult(**empty)
    empty.update(gen_stats=gstats, reach_slack_m=gstats["reach_slack_m"], best_gap_m=gstats["best_gap_m"])
    if not seeds:
        empty.update(elapsed_s=time.perf_counter() - t0)
        return FastEvalResult(**empty)

    try:
        seeds = _polish_seeds(cfg, ctx, seeds, q_ids, v_ids, q_lower, q_upper)

        # stage 1: short BFS on every seed
        stage1 = []
        for s in seeds:
            R = _compute_trial_frame(cfg, ctx, s, v_ids)
            prims = compute_motion_primitives_frame(R, cfg.voxel_size_m)
            st = _run_trial_bfs(cfg, ctx, s, R, fe.stage1_budget, q_ids, v_ids, q_lower, q_upper, prims)
            # Tiebreak seeds that all hit the stage-1 cap by their primitive
            # failure rate: a seed whose steps rarely fail has more slack.
            fail_rate = st.fail_moves / max(st.attempts, 1)
            stage1.append((st.n_voxels, fail_rate, s, R, prims))
        stage1.sort(key=lambda t: (-t[0], t[1]))

        # stage 2: full BFS on the finalists
        finals = []
        for n1, _fr, s, R, prims in stage1[: fe.n_finalists]:
            if n1 < fe.stage1_budget:          # frontier exhausted: count is already exact
                finals.append((n1, s, R, prims, None))
                continue
            st, disc = _run_trial_bfs(cfg, ctx, s, R, fe.final_budget, q_ids, v_ids, q_lower, q_upper,
                                      prims, return_configs=True)
            finals.append((st.n_voxels, s, R, prims, disc))
        finals.sort(key=lambda t: -t[0])
        top3 = [int(t[0]) for t in finals[:3]]
        karma_t = float(np.mean(top3)) * cfg.voxel_size_m ** 3 / cfg.l_ref_m ** 3

        karma_r = None
        if fe.rotation_voxels > 0:
            n, s, R, prims, disc = finals[0]
            if disc is None:
                _, disc = _run_trial_bfs(cfg, ctx, s, R, fe.final_budget, q_ids, v_ids, q_lower, q_upper,
                                         prims, return_configs=True)
            karma_r = _rotation_coverage_subsample(cfg, ctx, disc, fe.rotation_voxels)
    except Exception as e:
        empty.update(elapsed_s=time.perf_counter() - t0, error=f"bfs: {type(e).__name__}: {e}")
        return FastEvalResult(**empty)

    return FastEvalResult(
        karma_t=karma_t, karma_r=karma_r, top3_voxels=top3,
        stage1_counts=[int(t[0]) for t in stage1], n_seeds=len(seeds),
        l_ref_m=cfg.l_ref_m, voxel_m=cfg.voxel_size_m,
        reach_slack_m=gstats["reach_slack_m"], best_gap_m=gstats["best_gap_m"],
        gen_stats=gstats, elapsed_s=time.perf_counter() - t0,
    )


def _evaluate_jax(hand: HandSpec, workdir, fe: FastEvalConfig, metric_cfg_path) -> FastEvalResult:
    """Same proxy policy on the JAX backend (see karma_jax/fast.py)."""
    from karma_jax.model import build_flat_model
    from karma_jax import fast as jfast
    t0 = time.perf_counter()
    empty = dict(karma_t=0.0, karma_r=None, top3_voxels=[], stage1_counts=[], n_seeds=0,
                 l_ref_m=0.0, voxel_m=0.0, reach_slack_m=float("inf"), best_gap_m=float("inf"),
                 gen_stats={}, elapsed_s=0.0)
    try:
        _, yaml_path = hand.write(workdir)
        cfg = load_config(metric_cfg_path, yaml_path)
        flat = build_flat_model(load_robot(cfg.urdf_path, cfg), cfg)
    except Exception as e:  # malformed design
        empty.update(elapsed_s=time.perf_counter() - t0, error=f"load: {type(e).__name__}: {e}")
        return FastEvalResult(**empty)
    fc = jfast.FastConfig(n_linspace_per_joint=fe.n_linspace_per_joint, n_sobol=fe.n_sobol, n_random=fe.n_random,
                          n_offsets=fe.n_offsets, dedup_rad=fe.dedup_rad, max_seeds=fe.max_seeds,
                          stage1_budget=fe.stage1_budget, n_finalists=fe.n_finalists,
                          final_budget=fe.final_budget, rotation_voxels=fe.rotation_voxels)
    r = jfast.evaluate(cfg, flat, fc, precision=os.environ.get("KARMA_PRECISION", "fp32"))
    return FastEvalResult(**r.to_dict())


def objective_value(r: FastEvalResult, mode: str = "T") -> float:
    """Scalar to MAXIMIZE. Shaped so infeasible designs still get a gradient
    toward 'the two fingertips can meet around the sphere'."""
    if r.karma_t <= 0.0:
        # no feasible pinch: reward being close to reach, in sphere-radius units
        if np.isfinite(r.reach_slack_m) and r.voxel_m > 0:
            slack = r.reach_slack_m / (r.voxel_m / 0.05 * 0.05)  # voxel = 0.05 L_ref; r_s = 0.05 L_ref
            return -1e-3 * (1.0 + max(slack, 0.0))
        return -1e-2
    if mode == "T":
        return r.karma_t
    if mode == "R":
        return float(r.karma_r or 0.0)
    if mode == "TR":
        # both normalized by the best published hands (LEAP T=0.0969, Allegro R=0.335)
        return r.karma_t / 0.0969 + float(r.karma_r or 0.0) / 0.335
    raise ValueError(mode)
