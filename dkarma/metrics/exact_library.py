"""Exact counterpart of Diff-KaRMA on the same trajectory library (SPEC.md §4.7, CLAUDE.md rule 2).

Every command of every seed is replayed with the reference step kernels
(:mod:`dkarma.solvers.exact`), trajectories are credited only up to the first
failed primitive, and coverage is **counted**: distinct task cells reached by
exactly valid states (fraction of the cube) and distinct pinch-axis line bins
(reference HEALPix binning, fraction of 228). These are discrete numbers of the
*library*, not the BFS of the original KaRMA; the original metric is run through
:func:`official_karma` on the exported design for the headline validation.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from karma_jax.kinematics import forward_kinematics
from karma_jax.rolling import StepParams
from karma_jax.steps import pinch_axis_body

from ..core.morphology import HandModel, Topology
from ..core.object_model import TaskSpec
from ..core.contacts import contact_pass
from ..solvers.rollout import GraspState
from ..solvers.exact import validate_translation, validate_rotation
from ..solvers.seed_projection import SeedSet, assignment
from .commands import build_library
from .config import MetricConfig
from .orientation import pinch_axis_bins


class ExactSeedResult(NamedTuple):
    cells: set
    bins: set
    ok_T: np.ndarray        # [U_T, K] primitive verdicts
    ok_R: np.ndarray        # [U_R, H_R]
    codes_T: np.ndarray
    codes_R: np.ndarray
    pair_switch_T: np.ndarray   # [U_T] bool: the reference changed the contact pair
    pair_switch_R: np.ndarray


class ExactResult(NamedTuple):
    K_T: float              # distinct cells / cube cells, mean over valid seeds
    K_R: float              # distinct bins / 228, mean over valid seeds
    per_seed: list


def _cell_index(p_rel, task: TaskSpec):
    h = task.task.grid_m
    n = task.task.n_per_axis
    i = np.floor(p_rel / h + n / 2.0).astype(int)
    return i


# Phase 9: the per-seed exact validators through one jitted program per sub-step count (model, state and commands are
# arguments) instead of an eager ``vmap`` of a fresh closure per seed; bit-identical outputs (docs/phase9_report.md
# §9.4). ``CACHED_VALIDATORS = False`` restores the Phase 8 code path.
CACHED_VALIDATORS = True
_VALIDATOR_PROGRAMS: dict = {}


def _validate_commands(hm, st0, lib, asg, sp, n_sub):
    if not CACHED_VALIDATORS:
        return (jax.vmap(lambda a: validate_translation(hm, st0, a, asg, sp, n_sub))(lib.dp_T),
                jax.vmap(lambda a: validate_rotation(hm, st0, a, asg, sp))(lib.dth_R))
    f = _VALIDATOR_PROGRAMS.get(n_sub)
    if f is None:
        f = (jax.jit(lambda hm_, st_, d_, asg_, sp_: jax.vmap(lambda a: validate_translation(hm_, st_, a, asg_, sp_, n_sub))(d_)),
             jax.jit(lambda hm_, st_, d_, asg_, sp_: jax.vmap(lambda a: validate_rotation(hm_, st_, a, asg_, sp_))(d_)))
        _VALIDATOR_PROGRAMS[n_sub] = f
    return f[0](hm, st0, lib.dp_T, asg, sp), f[1](hm, st0, lib.dth_R, asg, sp)


def evaluate_exact(hm: HandModel, topo: Topology, seeds: SeedSet, sp: StepParams, task: TaskSpec,
                   cfg: MetricConfig) -> ExactResult:
    asg = assignment(topo, max(2, cfg.contacts))
    n_sub = cfg.commands.n_sub
    n = task.task.n_per_axis
    per_seed = []
    for s in range(int(np.asarray(seeds.valid).sum())):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        fk0 = forward_kinematics(hm.dm, q)
        n0 = contact_pass(hm, fk0, p, sp.sphere_radius, asg).normal[0]
        lib = build_library(frame, -n0, cfg.commands)
        st0 = GraspState(q, p, jnp.eye(3, dtype=q.dtype))
        exT, exR = _validate_commands(hm, st0, lib, asg, sp, n_sub)
        okT, okR = np.asarray(exT.ok), np.asarray(exR.ok)
        cells, bins = set(), set()
        axis0 = np.asarray(pinch_axis_body(n0, jnp.eye(3, dtype=q.dtype)))
        p0 = np.asarray(p)
        cells.add(tuple(_cell_index(np.zeros(3), task)))
        bins.add(int(pinch_axis_bins(axis0[None])[0]))
        pT = np.asarray(exT.p); RT = np.asarray(exT.R)
        for u in range(okT.shape[0]):
            for k in range(okT.shape[1]):
                if not okT[u, k]:
                    break
                idx = _cell_index(pT[u, k + 1] - p0, task)
                if np.all(idx >= 0) and np.all(idx < n):
                    cells.add(tuple(idx))
                # the pinch axis at the end of the primitive: thumb normal at that state
                fkk = forward_kinematics(hm.dm, jnp.asarray(exT.q[u, k + 1]))
                nk = contact_pass(hm, fkk, jnp.asarray(pT[u, k + 1]), sp.sphere_radius,
                                  asg._replace(caps=jnp.asarray(exT.pair[u, k + 1]))).normal[0]
                ax = np.asarray(pinch_axis_body(nk, jnp.asarray(RT[u, k + 1])))
                bins.add(int(pinch_axis_bins(ax[None])[0]))
        axR = np.asarray(exR.axis_body)
        for u in range(okR.shape[0]):
            for k in range(okR.shape[1]):
                if not okR[u, k]:
                    break
                bins.add(int(pinch_axis_bins(axR[u, k][None])[0]))
        pairT = np.asarray(exT.pair); pairR = np.asarray(exR.pair)
        per_seed.append(ExactSeedResult(
            cells=cells, bins=bins, ok_T=okT, ok_R=okR, codes_T=np.asarray(exT.code), codes_R=np.asarray(exR.code),
            pair_switch_T=np.any(pairT != pairT[:, :1], axis=(1, 2)), pair_switch_R=np.any(pairR != pairR[:, :1], axis=(1, 2))))
    if not per_seed:
        return ExactResult(0.0, 0.0, [])
    K_T = float(np.mean([len(r.cells) for r in per_seed])) / task.task.n_cells
    K_R = float(np.mean([len(r.bins) for r in per_seed])) / 228.0
    return ExactResult(K_T, K_R, per_seed)


def official_karma(topo: Topology, params, workdir, name: str = "design", backend: str = "jax", precision: str = "fp32",
                   self_collision_all: bool = False):
    """The original KaRMA (reference semantics, L_ref-scaled object) on the exported design,
    through ``karma_jax.metric.compute_metric`` or the reference. Returns the summary row.
    ``self_collision_all`` (Phase 9.2, D18): the self-collision gate tests every pair of fingers
    (JAX backend only)."""
    from ..export import load_reference
    from ..fixtures import scores_from_result
    import time
    cfg, ctx, flat = load_reference(topo, params, workdir, name=name)
    if self_collision_all:
        from karma_jax.model import build_flat_model
        flat = build_flat_model(ctx, cfg, self_collision_all=True)
    t0 = time.perf_counter()
    if backend == "jax":
        from karma_jax.metric import compute_metric
        res = compute_metric(cfg, flat, precision=precision)
    else:
        from karma.metric import compute_metric
        res = compute_metric(cfg, ctx)
    return scores_from_result(name, res, time.perf_counter() - t0)


def physical_metric_config(l_ref_m: float, task: TaskSpec, workdir, base_cfg=None):
    """Write a metric YAML whose *nominal* lengths, once the reference scales them by
    ``L_ref / 0.2``, equal the physical task values (sphere 15 mm, capsule 11.8 mm, 10 mm
    voxels, the dkarma tolerances). Running the original BFS with it gives exact KaRMA on
    the physical task of SPEC.md §2 for a design of characteristic length ``l_ref_m``."""
    import yaml
    from pathlib import Path
    from ..fixtures import METRIC_CFG
    raw = yaml.safe_load(Path(base_cfg or METRIC_CFG).read_text())
    k = 0.2 / float(l_ref_m)
    r = task.rolling
    raw.update(sphere_radius_m=task.obj.radius_m * k, link_radius_m=r.link_radius_m * k,
               voxel_size_m=task.task.grid_m * k, step_size_m=r.step_size_m * k, eps_g_m=r.eps_g_m * k,
               eps_col_m=r.eps_col_m * k, rotation_slip_threshold_mm=r.slip_gate_m * 1e3 * k,
               seed_search_offsets_m=[o * k for o in (0.0, -0.005, -0.01, 0.005, 0.01)],
               qp_regularization=r.qp_regularization * k * k, friction_coeff=task.obj.friction_coefficient,
               max_joint_step_deg=r.max_joint_step_deg)
    out = Path(workdir) / "karma_config_physical.yaml"
    out.write_text(yaml.safe_dump(raw, sort_keys=False))
    return out


def official_karma_physical(topo: Topology, params, task: TaskSpec, workdir, name: str = "design",
                            backend: str = "jax", precision: str = "fp32", self_collision_all: bool = False):
    """Original KaRMA search on the **physical** task (object, voxel and tolerances of SPEC §2)
    for this design. Returns the summary row plus ``voxels_top3_mean`` and ``cube_fraction``
    (top-3 mean voxel count over the 8000 cells of the task cube: the discrete analogue of
    ``K_T~``). ``self_collision_all`` (Phase 9.2, D18): the self-collision gate tests every pair of
    fingers, not only the contact pair (JAX backend only)."""
    from ..export import load_reference, write_design
    from ..fixtures import scores_from_result
    from karma.config import load_config
    from karma.robot import load_robot
    from karma_jax.model import build_flat_model
    import dataclasses, time
    from pathlib import Path
    cfg0, _, _ = load_reference(topo, params, workdir, name=name)
    mcfg = physical_metric_config(cfg0.l_ref_m, task, workdir)
    _, yaml_path, _ = write_design(topo, params, workdir, name)
    cfg = dataclasses.replace(load_config(mcfg, yaml_path), out_dir=Path(workdir))
    ctx = load_robot(cfg.urdf_path, cfg)
    t0 = time.perf_counter()
    if backend == "jax":
        from karma_jax.metric import compute_metric
        res = compute_metric(cfg, build_flat_model(ctx, cfg, self_collision_all=self_collision_all), precision=precision)
    else:
        from karma.metric import compute_metric
        res = compute_metric(cfg, ctx)
    row = scores_from_result(name, res, time.perf_counter() - t0)
    sens = res.seed_sensitivity or {}
    top3 = sens.get("top3_mean", res.n_voxels_reached)
    row["voxels_top3_mean"] = float(top3)
    row["cube_fraction"] = float(top3) / task.task.n_cells
    row["sphere_radius_scaled_m"] = float(cfg.sphere_radius_m)
    row["self_collision_all"] = bool(self_collision_all)
    return row


__all__ = ["ExactSeedResult", "ExactResult", "evaluate_exact", "official_karma", "physical_metric_config",
           "official_karma_physical"]
