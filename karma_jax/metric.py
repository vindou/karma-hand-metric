"""The KaRMA metric on the JAX backend (port of ``karma/metric.py`` and
``select_seed_and_frame``), producing the reference's :class:`MetricResult`.

Pipeline (same as the reference, batched):

0. seeds: sample, IK every candidate at once, dedup, polish, grid frames
1. seed ranking: one multi-seed translation BFS over ALL seeds at ``max_states``
2. best seed + ensemble seeds #2, #3: their BFS results are reused (the
   reference re-runs an identical Phase 1 for the best seed)
3. Phase 2: one batched rotation exploration over every voxel of the three seeds
4. scoring and ``MetricResult`` assembly, identical fields and keys
"""
from __future__ import annotations

import logging
import time

import numpy as np
import jax.numpy as jnp

from karma.config import MetricConfig
from karma.results import MetricResult, StateNode, VoxelIdx, State
from karma.orientation import healpix_npix_lines
from karma.seed import SeedResult

from .model import FlatModel
from .steps import Constraints
from .search import TranslationSearch, RotationSearch, SeedSpec, VoxelStart, BFSResult
from .seeds import generate_feasible_seeds, polish_seeds, trial_frames
from .ik import IKSettings

logger = logging.getLogger(__name__)


def _dtype(precision: str):
    return jnp.float64 if precision in ("fp64", "float64") else jnp.float32


def select_seed_and_frame(cfg: MetricConfig, flat: FlatModel, *, precision: str = "fp32",
                          cons: Constraints = Constraints(), ik_settings: IKSettings = IKSettings(),
                          _return_bfs: bool = False):
    """JAX port of ``karma.seed_selection.select_seed_and_frame``.

    Returns ``(best_seed, R_seed, seed_sensitivity, top_seeds)`` like the
    reference; with ``_return_bfs`` also the per-seed BFS results and frames,
    which :func:`compute_metric` reuses.
    """
    dtype = _dtype(precision)
    dm = flat.to_device(dtype)
    # The seed IK always runs in fp64: it is a small fraction of the runtime and
    # the scipy-faithful TRF port then reproduces the reference's seed set.
    dm64 = flat.to_device(jnp.float64)
    t0 = time.perf_counter()
    seeds, gen_stats = generate_feasible_seeds(flat, dm64, cfg, jnp.float64, ik_settings)
    logger.info("Seed generation stats: %d joint configs -> %d reachable -> %d candidates -> "
                "%d IK solutions -> %d feasible -> %d after dedup (%.1fs)",
                gen_stats["n_joint_configs"], gen_stats["n_reachable"], gen_stats["n_candidates"],
                gen_stats["n_ik_solutions"], gen_stats["n_feasible"], gen_stats["n_after_dedup"],
                time.perf_counter() - t0)

    if len(seeds) == 1:
        R_seed = trial_frames(flat, dm, cfg, dtype, seeds)[0]
        if _return_bfs:
            return seeds[0], R_seed, None, [(0.0, seeds[0])], None, {id(seeds[0]): R_seed}
        return seeds[0], R_seed, None, [(0.0, seeds[0])]

    candidates = polish_seeds(flat, dm, cfg, dtype, seeds)
    frames = trial_frames(flat, dm, cfg, dtype, candidates)
    names = flat.capsule_names
    specs = [SeedSpec(q=s.q, p=s.sphere_centre_world,
                      pair=(names.index(s.contact_pair[0]), names.index(s.contact_pair[1])), R_seed=R)
             for s, R in zip(candidates, frames)]
    logger.info("Seed selection: evaluating %d seeds at max_states=%d", len(specs), cfg.max_states)
    t1 = time.perf_counter()
    search = TranslationSearch(flat, cfg, dtype=dtype, cons=cons)
    bfs = search.run(specs, cfg.max_states)
    logger.info("Seed ranking BFS done in %.1fs", time.perf_counter() - t1)

    scored = sorted(zip([float(b.n_voxels) for b in bfs], candidates, bfs, frames), key=lambda x: -x[0])
    voxel_counts = [v for v, _, _, _ in scored]
    best_v = voxel_counts[0]
    median_v = voxel_counts[len(voxel_counts) // 2]
    top_k = min(3, len(voxel_counts))
    top3 = voxel_counts[:top_k]
    seed_sensitivity = {
        "n_seeds_evaluated": len(scored),
        "voxel_counts": [int(v) for v in voxel_counts],
        "best": int(best_v),
        "top3_mean": round(float(np.mean(top3)), 2),
        "top3_voxels": [int(v) for v in top3],
        "median": int(median_v),
        "worst": int(voxel_counts[-1]),
        "mean": round(float(np.mean(voxel_counts)), 1),
        "std": round(float(np.std(voxel_counts)), 1),
        "karma_s": round(median_v / best_v, 4) if best_v > 0 else 0.0,
    }
    n_at_cap = sum(1 for v in voxel_counts if v >= cfg.max_states * 0.95)
    logger.info("Seed selection: best voxels=%d, median=%d, worst=%d (%d/%d at cap)",
                int(best_v), int(median_v), int(voxel_counts[-1]), n_at_cap, len(scored))
    best = scored[0][1]
    R_best = scored[0][3]
    top_seeds = [(v, s) for v, s, _, _ in scored[:top_k]]
    if _return_bfs:
        bfs_by_seed = {id(s): b for _, s, b, _ in scored}
        frame_by_seed = {id(s): R for _, s, _, R in scored}
        return best, R_best, seed_sensitivity, top_seeds, bfs_by_seed, frame_by_seed
    return best, R_best, seed_sensitivity, top_seeds


def _voxel_starts(bfs: BFSResult) -> list[VoxelStart]:
    return [VoxelStart(n.q, n.p, n.R, n.pair) for n in bfs.nodes]


def compute_metric(cfg: MetricConfig, flat: FlatModel, *, precision: str = "fp32",
                   max_voxels: int | None = None, precomputed_seed: tuple | None = None,
                   cons: Constraints = Constraints(), ik_settings: IKSettings = IKSettings()) -> MetricResult:
    """JAX port of ``karma.metric.compute_metric``. Same return type."""
    dtype = _dtype(precision)
    if max_voxels is None:
        max_voxels = cfg.max_states
    names = flat.capsule_names
    to_idx = lambda pair: (names.index(pair[0]), names.index(pair[1]))  # noqa: E731
    t_start = time.perf_counter()

    bfs_by_seed: dict = {}
    frame_by_seed: dict = {}
    if precomputed_seed is not None:
        seed, R_seed, seed_sensitivity, top_seeds = precomputed_seed
        logger.info("Using precomputed seed (skipping seed selection)")
    else:
        seed, R_seed, seed_sensitivity, top_seeds, bfs_by_seed, frame_by_seed = select_seed_and_frame(
            cfg, flat, precision=precision, cons=cons, ik_settings=ik_settings, _return_bfs=True)
        bfs_by_seed = bfs_by_seed or {}
    if not seed.feasible:
        logger.warning("Seed is not fully feasible (max_gap=%.4f mm)", seed.max_gap * 1000)
    logger.info("Seed: contact_pair=%s, max_gap=%.4f mm, feasible=%s",
                seed.contact_pair, seed.max_gap * 1000, seed.feasible)

    # ── Phase 1 for the best seed and the ensemble seeds ──
    extra = [(v, s) for v, s in top_seeds[1:] if s is not seed]
    phase1_seeds = [seed] + [s for _, s in extra]
    need = [s for s in phase1_seeds if id(s) not in bfs_by_seed or bfs_by_seed[id(s)].n_voxels < min(max_voxels, cfg.max_states)
            and bfs_by_seed[id(s)].frontier_size > 0]
    # Reuse ranking BFS results when they ran at the same budget; otherwise (precomputed
    # seed, or a different max_voxels) run Phase 1 now.
    if precomputed_seed is not None or max_voxels != cfg.max_states or need:
        dm = flat.to_device(dtype)
        frames = []
        for s in phase1_seeds:
            if id(s) in frame_by_seed:
                frames.append(frame_by_seed[id(s)])
            elif s is seed:
                frames.append(R_seed)
            else:
                frames.append(trial_frames(flat, dm, cfg, dtype, [s])[0])
        specs = [SeedSpec(q=s.q, p=s.sphere_centre_world, pair=to_idx(s.contact_pair), R_seed=R)
                 for s, R in zip(phase1_seeds, frames)]
        logger.info("Phase 1: translation BFS for %d seed(s) (max_voxels=%d)...", len(specs), max_voxels)
        search = TranslationSearch(flat, cfg, dtype=dtype, cons=cons)
        results = search.run(specs, max_voxels)
        bfs_by_seed = {id(s): b for s, b in zip(phase1_seeds, results)}
    bfs_best = bfs_by_seed[id(seed)]
    fail_reasons = dict(bfs_best.fail_reasons)
    if bfs_best.revisits:
        fail_reasons["duplicate_voxel"] = bfs_best.revisits
    logger.info("Phase 1 complete: attempted=%d, voxels=%d", bfs_best.attempts, bfs_best.n_voxels)
    logger.info("Phase 1 failure breakdown: %s", fail_reasons)

    # ── Phase 2: rotation exploration over every voxel of all phase-1 seeds ──
    all_starts: list[VoxelStart] = []
    owner: list[int] = []
    for k, s in enumerate(phase1_seeds):
        b = bfs_by_seed[id(s)]
        all_starts.extend(_voxel_starts(b))
        owner.extend([k] * b.n_voxels)
    logger.info("Phase 2: rotation exploration (%d voxels over %d seeds)...", len(all_starts), len(phase1_seeds))
    t2 = time.perf_counter()
    rot = RotationSearch(flat, cfg, dtype=dtype, cons=cons).run(all_starts)
    logger.info("Phase 2 complete in %.1fs", time.perf_counter() - t2)

    total_bins = healpix_npix_lines(cfg.healpix_nside)
    per_seed_rot: list[list] = [[] for _ in phase1_seeds]
    for k, r in zip(owner, rot):
        per_seed_rot[k].append(r)

    phase2_fail: dict[str, int] = {}
    for r in per_seed_rot[0]:
        for name, c in r.fail_reasons.items():
            phase2_fail[name] = phase2_fail.get(name, 0) + c

    # ── Best seed: result structures ──
    voxel_ori: dict[VoxelIdx, set[int]] = {}
    rot_cov: dict[VoxelIdx, float] = {}
    nodes: dict[State, StateNode] = {}
    for node, r in zip(bfs_best.nodes, per_seed_rot[0]):
        voxel_ori[node.voxel] = set(r.bins)
        rot_cov[node.voxel] = len(r.bins) / total_bins
        parent_state: State | None = None
        if node.parent >= 0:
            pnode = bfs_best.nodes[node.parent]
            parent_state = (pnode.voxel, per_seed_rot[0][node.parent].initial_bin)
        for b, (q, R, pair) in r.configs.items():
            nodes[(node.voxel, b)] = StateNode(
                voxel=node.voxel, ori_bin=b, centre_world=node.p.copy(), q=np.asarray(q, float).copy(),
                R_sphere=np.asarray(R, float).copy(), contact_pair=(names[pair[0]], names[pair[1]]),
                parent=parent_state)

    def _top5(rs) -> float:
        covs = sorted((len(r.bins) / total_bins for r in rs), reverse=True)[:5]
        return float(np.mean(covs)) if covs else 0.0

    ensemble_t = [float(bfs_by_seed[id(s)].n_voxels) for s in phase1_seeds]
    ensemble_r = [_top5(rs) for rs in per_seed_rot]
    mean_t = float(np.mean(ensemble_t))
    mean_r = float(np.mean(ensemble_r))
    voxel_vol = cfg.voxel_size_m ** 3
    l_ref = cfg.l_ref_m
    translational_score = (mean_t * voxel_vol) / (l_ref ** 3) if l_ref > 0 else 0.0
    if seed_sensitivity is not None:
        seed_sensitivity["ensemble_t"] = [round(v, 2) for v in ensemble_t]
        seed_sensitivity["ensemble_r"] = [round(v, 4) for v in ensemble_r]
        seed_sensitivity["top3_mean"] = round(mean_t, 2)

    n_voxels = bfs_best.n_voxels
    logger.info("Two-phase metric complete in %.1fs: voxels=%d, ensemble_T=[%s] mean=%.1f, "
                "ensemble_R=[%s] mean=%.4f, KaRMA-T=%.6f",
                time.perf_counter() - t_start, n_voxels,
                ", ".join(f"{v:.0f}" for v in ensemble_t), mean_t,
                ", ".join(f"{v:.4f}" for v in ensemble_r), mean_r, translational_score)

    return MetricResult(
        voxel_ori_bins=voxel_ori,
        state_nodes=nodes,
        n_voxels_reached=n_voxels,
        n_states_reached=len(nodes),
        total_ori_bins=total_bins,
        translational_score=translational_score,
        translational_volume_m3=n_voxels * voxel_vol,
        l_ref_m=l_ref,
        rotational_coverage_per_voxel=rot_cov,
        global_rotational_score=mean_r,
        seed_centre=np.asarray(seed.sphere_centre_world, float),
        seed_q=np.asarray(seed.q, float),
        seed_contact_pair=tuple(seed.contact_pair),
        seed_frame=np.asarray(R_seed, float),
        seed_sensitivity=seed_sensitivity,
        phase1_fail_reasons=fail_reasons,
        phase2_fail_reasons=phase2_fail,
        backend="jax",
        precision="fp64" if dtype == jnp.float64 else "fp32",
    )
