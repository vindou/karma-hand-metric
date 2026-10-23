"""Reduced-fidelity KaRMA proxy on the JAX backend (counterpart of ``optimize/fast_eval.py``).

Same policy as the CPU proxy: fewer joint samples and sphere offsets, light
dedup, a cap on the number of seeds, a short stage-1 BFS on every seed, the
full BFS on a few finalists, and an optional Phase-2 subsample on the best
finalist. Every stage runs on the batched kernels, so the whole proxy is a
handful of GPU calls per design. Official numbers must still come from
:func:`karma_jax.metric.compute_metric` (or the reference).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict

import numpy as np
import jax.numpy as jnp

from karma.config import MetricConfig
from karma.orientation import healpix_npix_lines
from karma.math_utils import closest_points_on_segments

from .model import FlatModel
from .search import TranslationSearch, RotationSearch, SeedSpec, VoxelStart, step_params_from_cfg
from .seeds import (sample_joint_configs, seed_capsule_endpoints, _axis_segment, candidate_centres,
                    solve_candidates, deduplicate_seeds, polish_seeds, trial_frames, _sort_key)
from .ik import IKSettings


@dataclass
class FastConfig:
    n_linspace_per_joint: int = 3
    n_sobol: int = 128
    n_random: int = 32
    n_offsets: int = 3
    dedup_rad: float = 0.20
    max_seeds: int = 24
    stage1_budget: int = 15
    n_finalists: int = 2
    final_budget: int = 1200
    rotation_voxels: int = 0


@dataclass
class FastResult:
    karma_t: float
    karma_r: float | None
    top3_voxels: list[int]
    stage1_counts: list[int]
    n_seeds: int
    l_ref_m: float
    voxel_m: float
    reach_slack_m: float
    best_gap_m: float
    gen_stats: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def generate_seeds(flat: FlatModel, dm, cfg: MetricConfig, fc: FastConfig, dtype=jnp.float64):
    """Reduced-sample seed generation (mirror of ``optimize.fast_eval.generate_seeds``)."""
    configs = sample_joint_configs(flat, fc.n_linspace_per_joint, fc.n_sobol, fc.n_random)
    offsets = list(cfg.seed_search_offsets_m[: fc.n_offsets]) or [0.0]
    seed_margin = max(abs(o) for o in cfg.seed_search_offsets_m) * 2.0
    max_reach = 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m) + seed_margin
    a_t, b_t, a_i, b_i = seed_capsule_endpoints(flat, dm, configs, dtype)
    reach_slack = float("inf")
    qs, cs = [], []
    for k in range(configs.shape[0]):
        c0, d0, h0 = _axis_segment(a_t[k], b_t[k])
        c1, d1, h1 = _axis_segment(a_i[k], b_i[k])
        p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        dist = float(np.linalg.norm(p1 - p0))
        reach_slack = min(reach_slack, dist - 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m))
        if dist > max_reach:
            continue
        for c in candidate_centres(a_t[k], b_t[k], a_i[k], b_i[k], offsets):
            qs.append(configs[k])
            cs.append(c)
    params = step_params_from_cfg(cfg, dtype)
    results = solve_candidates(flat, dm, params, dtype, np.stack(qs) if qs else np.zeros((0, flat.n_joints)),
                               np.stack(cs) if cs else np.zeros((0, 3)))
    best_gap = min((r.max_gap for r in results), default=float("inf"))
    feasible = [r for r in results if r.feasible]
    n_feasible = len(feasible)
    if feasible:
        feasible = deduplicate_seeds(feasible, flat.q_ids[: flat.n_active], min_dist=fc.dedup_rad)
        feasible.sort(key=_sort_key)
        feasible = feasible[: fc.max_seeds]
    stats = {"n_joint_configs": int(configs.shape[0]), "n_candidates": len(qs), "n_ik_solutions": len(results),
             "n_feasible": n_feasible, "n_after_dedup": len(feasible), "reach_slack_m": reach_slack,
             "best_gap_m": best_gap}
    return feasible, stats


def _subsample_core(bfs, k: int) -> list[int]:
    """Indices of the k voxels nearest the centroid of the reachable set."""
    n = bfs.n_voxels
    if n <= k:
        return list(range(n))
    c = bfs.p.mean(axis=0)
    return list(np.argsort(np.linalg.norm(bfs.p - c, axis=1))[:k])


def evaluate(cfg: MetricConfig, flat: FlatModel, fc: FastConfig | None = None, *,
             precision: str = "fp32") -> FastResult:
    """Fast proxy for one design. ``cfg``/``flat`` come from ``load_config`` + ``build_flat_model``."""
    fc = fc or FastConfig()
    t0 = time.perf_counter()
    dtype = jnp.float64 if precision == "fp64" else jnp.float32
    empty = dict(karma_t=0.0, karma_r=None, top3_voxels=[], stage1_counts=[], n_seeds=0,
                 l_ref_m=cfg.l_ref_m, voxel_m=cfg.voxel_size_m, reach_slack_m=float("inf"),
                 best_gap_m=float("inf"), gen_stats={}, elapsed_s=0.0)
    dm64 = flat.to_device(jnp.float64)
    try:
        seeds, gstats = generate_seeds(flat, dm64, cfg, fc)
    except Exception as e:  # noqa: BLE001 - a malformed design must not kill the optimizer
        empty.update(elapsed_s=time.perf_counter() - t0, error=f"seeds: {type(e).__name__}: {e}")
        return FastResult(**empty)
    empty.update(gen_stats=gstats, reach_slack_m=gstats["reach_slack_m"], best_gap_m=gstats["best_gap_m"])
    if not seeds:
        empty.update(elapsed_s=time.perf_counter() - t0)
        return FastResult(**empty)

    try:
        seeds = polish_seeds(flat, dm64, cfg, jnp.float64, seeds)
        frames = trial_frames(flat, dm64, cfg, jnp.float64, seeds)
        names = flat.capsule_names
        specs = [SeedSpec(q=s.q, p=s.sphere_centre_world,
                          pair=(names.index(s.contact_pair[0]), names.index(s.contact_pair[1])), R_seed=R)
                 for s, R in zip(seeds, frames)]
        search = TranslationSearch(flat, cfg, dtype=dtype)

        # stage 1: short BFS on every seed, tie-break by failure rate
        st1 = search.run(specs, fc.stage1_budget)
        order = sorted(range(len(specs)),
                       key=lambda i: (-st1[i].n_voxels, st1[i].fail_moves / max(st1[i].attempts, 1)))
        stage1_counts = [int(st1[i].n_voxels) for i in order]

        # stage 2: full BFS on the finalists that hit the cap, in one batch
        finalists = order[: fc.n_finalists]
        todo = [i for i in finalists if st1[i].n_voxels >= fc.stage1_budget]
        full = dict(zip(todo, search.run([specs[i] for i in todo], fc.final_budget))) if todo else {}
        finals = sorted(((full[i].n_voxels if i in full else st1[i].n_voxels), i) for i in finalists)[::-1]
        top3 = [int(n) for n, _ in finals[:3]]
        karma_t = float(np.mean(top3)) * cfg.voxel_size_m ** 3 / cfg.l_ref_m ** 3

        karma_r = None
        if fc.rotation_voxels > 0:
            _, ibest = finals[0]
            bfs = full.get(ibest) or search.run([specs[ibest]], fc.final_budget)[0]
            core = _subsample_core(bfs, fc.rotation_voxels)
            starts = [VoxelStart(bfs.q[i], bfs.p[i], bfs.R[i], (int(bfs.pair[i, 0]), int(bfs.pair[i, 1]))) for i in core]
            rot = RotationSearch(flat, cfg, dtype=dtype).run(starts)
            total = healpix_npix_lines(cfg.healpix_nside)
            covs = sorted((len(r.bins) / total for r in rot), reverse=True)
            karma_r = float(np.mean(covs[:5])) if covs else 0.0
    except Exception as e:  # noqa: BLE001
        empty.update(elapsed_s=time.perf_counter() - t0, error=f"bfs: {type(e).__name__}: {e}")
        return FastResult(**empty)

    return FastResult(karma_t=karma_t, karma_r=karma_r, top3_voxels=top3, stage1_counts=stage1_counts,
                      n_seeds=len(seeds), l_ref_m=cfg.l_ref_m, voxel_m=cfg.voxel_size_m,
                      reach_slack_m=gstats["reach_slack_m"], best_gap_m=gstats["best_gap_m"],
                      gen_stats=gstats, elapsed_s=time.perf_counter() - t0)
