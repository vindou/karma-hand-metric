"""Diff-KaRMA: smooth, differentiable two-finger rolling reachability (SPEC.md §4).

For each seed ``s`` (fixed budget, invalid lanes masked) and each command ``u`` of its
library, :func:`dkarma.solvers.rollout.rollout` produces the trajectory and its
per-step residuals; :func:`energies` turns them into prefix log-weights
``log W_k``; the coverage kernels of :mod:`smooth_coverage` aggregate every
(seed, command, step) state into

* ``K_T~``: smooth occupancy of the task cube centred on the seed grasp (fraction),
* ``K_R~``: smooth occupancy of the 228 pinch-axis line bins by the body-frame
  pinch axis over all states (fraction), plus region-restricted coverage at the
  seed cell and the end cells of the translation commands;

per-seed score ``K_s = lambda_T K_T,s / ref_T + lambda_R K_R,s / ref_R`` and the
design score ``(1 - beta) mean_s K_s + beta softmin_s K_s`` over valid seeds.

:func:`evaluate` takes a built model and a seed set; :func:`objective` takes the
unconstrained ``Phi``, decodes it, re-projects the stored seeds onto the new
morphology (differentiably) and evaluates. Everything after the seed set is a
single ``jit``-able function of the morphology.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from karma_jax.kinematics import forward_kinematics
from karma_jax.qp import QPSettings
from karma_jax.rolling import StepParams
from karma_jax.steps import pinch_axis_body

from ..core.morphology import HandModel, Topology, Phi, Bounds, FixedParams, decode, build_hand_model
from ..core.object_model import TaskSpec
from ..core.contacts import ContactAssignment, contact_pass
from ..solvers.rollout import GraspState, rollout, energies, exact_gates, force_penalty, seed_force_energy
from ..solvers.seed_projection import SeedSet, assignment, reproject
from .commands import build_library, CommandLibrary
from .config import MetricConfig
from .orientation import pinch_axis_bin_centres
from . import smooth_coverage as SC


class SeedEval(NamedTuple):
    K_T: object            # scalar
    K_R: object            # scalar (global pinch-axis coverage)
    K_R_regions: object    # [n_regions]
    occ_T: object          # [V] cell occupancies
    occ_R: object          # [228] bin occupancies
    e_total_T: object      # [U_T] total energy per translation command
    e_total_R: object      # [U_R]
    e_worst_T: object      # [U_T]
    e_worst_R: object      # [U_R]
    gates_T: object        # [U_T, H_T, 7] bool exact gates of the smooth states
    gates_R: object        # [U_R, H_R, 7]
    end_logw_T: object     # [U_T] log-credit at the end of each translation command
    end_logw_R: object     # [U_R]
    p_T: object            # [U_T, H_T+1, 3] positions (for exact counting / plotting)
    axis_T: object         # [U_T, H_T, 3] body pinch axes
    axis_R: object         # [U_R, H_R, 3]
    reproj_ok: object      # bool
    reproj_gap: object     # m
    force_T: object        # [U_T, H_T, 4]: (E_F, torque usage, cone margin, fn_min) per step (zeros without actuation)
    force_R: object        # [U_R, H_R, 4]
    seed_force_E: object   # scalar: E_F* of the seed state itself (0 without actuation); gates the seed's credit (Phase 8.1)


class Diagnostics(NamedTuple):
    score: object
    K_T: object            # [S]
    K_R: object            # [S]
    K_s: object            # [S]
    valid: object          # [S]
    per_seed: SeedEval     # stacked over S


def _flatten_states(states_p, logw0, logw):
    """Concatenate initial states (log-weight 0) with the stepped states."""
    p = jnp.concatenate([states_p[:, :1], states_p[:, 1:]], axis=1)      # [U, H+1, 3]
    lw = jnp.concatenate([jnp.broadcast_to(logw0, (p.shape[0], 1)), logw], axis=1)
    return p.reshape(-1, 3), lw.reshape(-1)


def evaluate_seed(hm: HandModel, q, p, frame, asg: ContactAssignment, sp: StepParams, task: TaskSpec,
                  cfg: MetricConfig, cube_centres, bin_centres, qp_settings: QPSettings | None = None,
                  act=None) -> SeedEval:
    dtype = q.dtype
    sm = cfg.smooth
    # re-project the seed onto this morphology (differentiable), then the pinch axis and library
    q0, ok, gap0 = reproject(hm, q, p, asg, sp)
    fk0 = forward_kinematics(hm.dm, q0)
    cs0 = contact_pass(hm, fk0, p, sp.sphere_radius, asg)
    n0 = cs0.normal[0]
    lib: CommandLibrary = build_library(frame, -n0, cfg.commands)
    st0 = GraspState(q0, p, jnp.eye(3, dtype=dtype))

    roll_T = cfg.roll._replace(kind="translation")
    roll_R = cfg.roll._replace(kind="rotation")
    run_T = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_T, qp_settings, act))
    run_R = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_R, qp_settings, act))
    states_T, diags_T = run_T(lib.dp_T, lib.dth_T)
    states_R, diags_R = run_R(lib.dp_R, lib.dth_R)
    en_T = jax.vmap(lambda d: energies(d, sp, cfg.energy))(diags_T)
    en_R = jax.vmap(lambda d: energies(d, sp, cfg.energy))(diags_R)

    # a seed whose re-projection failed earns nothing; with actuation on, the seed state is gated by
    # its own force energy exactly as every step (Phase 8.1: the hold is required at every state)
    if cfg.roll.actuation and act is not None:
        seed_E = seed_force_energy(hm, fk0, cs0, p, asg, cfg.roll, act)
    else:
        seed_E = jnp.zeros((), dtype)
    seed_logw = jnp.where(ok, -force_penalty(seed_E, cfg.energy) / cfg.energy.sigma, -1e9)
    lw_T = en_T.prefix_logw + seed_logw
    lw_R = en_R.prefix_logw + seed_logw

    # translation coverage: all states of all commands (initial states carry the seed credit)
    p_all, lw_all = _flatten_states(states_T.p, seed_logw, lw_T)
    p_rot, lw_rot = _flatten_states(states_R.p, seed_logw, lw_R)
    p_cat = jnp.concatenate([p_all, p_rot]); lw_cat = jnp.concatenate([lw_all, lw_rot])
    cov_T = SC.translation_coverage(p_cat, lw_cat, cube_centres + p[None, :], sm.sigma_p_m)

    # orientation coverage: body-frame pinch axes of all stepped states plus the seed axis
    axis0 = pinch_axis_body(n0, jnp.eye(3, dtype=dtype))
    axes = jnp.concatenate([axis0[None], diags_T.axis_body.reshape(-1, 3), diags_R.axis_body.reshape(-1, 3)])
    lw_ax = jnp.concatenate([seed_logw[None], lw_T.reshape(-1), lw_R.reshape(-1)])
    p_ax = jnp.concatenate([p[None], states_T.p[:, 1:].reshape(-1, 3), states_R.p[:, 1:].reshape(-1, 3)])
    logk_R = SC.pinch_axis_log_kernel(axes, bin_centres, sm.sigma_r_rad)
    cov_R = SC.pinch_axis_coverage(axes, lw_ax, bin_centres, sm.sigma_r_rad)
    regions = jnp.concatenate([p[None], states_T.p[:, -1]])                 # seed cell + translation ends
    K_R_regions = SC.region_orientation_coverage(p_ax, lw_ax, regions, sm.sigma_region_m, logk_R)

    def gates(d):
        g = exact_gates(d, sp)
        return jnp.stack([g[k] for k in ("qp", "slip", "gap", "collision", "self_collision", "grasp", "limits")], -1)

    force = lambda d: jnp.stack([d.force_E, d.torque_usage, d.cone_margin, d.fn_min], -1)   # noqa: E731
    return SeedEval(force_T=force(diags_T), force_R=force(diags_R), seed_force_E=seed_E,
                    K_T=cov_T.fraction, K_R=cov_R.fraction, K_R_regions=K_R_regions, occ_T=cov_T.occupancy,
                    occ_R=cov_R.occupancy, e_total_T=en_T.total, e_total_R=en_R.total, e_worst_T=en_T.worst,
                    e_worst_R=en_R.worst, gates_T=jax.vmap(gates)(diags_T), gates_R=jax.vmap(gates)(diags_R),
                    end_logw_T=lw_T[:, -1], end_logw_R=lw_R[:, -1], p_T=states_T.p, axis_T=diags_T.axis_body,
                    axis_R=diags_R.axis_body, reproj_ok=ok, reproj_gap=gap0)


def cube_offsets(task: TaskSpec, dtype=jnp.float64):
    """Cell centres of the task cube relative to its centre, ``[V, 3]`` (constant)."""
    return jnp.asarray(task.task.cell_centres(np.zeros(3)), dtype)


def evaluate(hm: HandModel, topo: Topology, seeds: SeedSet, sp: StepParams, task: TaskSpec, cfg: MetricConfig,
             qp_settings: QPSettings | None = None, act=None) -> Diagnostics:
    """Smooth score and diagnostics for a built model and a seed set (``jit``-able). ``act`` is an
    ``ActuationParams`` for the actuation-aware variants (``cfg.roll.actuation`` must be set)."""
    asg = assignment(topo, max(2, cfg.contacts))
    cube = cube_offsets(task, hm.dm.t_joint.dtype)
    bins = jnp.asarray(pinch_axis_bin_centres(), hm.dm.t_joint.dtype)
    per_seed = jax.vmap(lambda q, p, f: evaluate_seed(hm, q, p, f, asg, sp, task, cfg, cube, bins, qp_settings, act))(
        seeds.q, seeds.p, seeds.frame)
    sm = cfg.smooth
    K_s = sm.lambda_T * per_seed.K_T / sm.ref_T + sm.lambda_R * per_seed.K_R / sm.ref_R
    valid = seeds.valid
    K_masked = jnp.where(valid, K_s, 0.0)
    n_valid = jnp.maximum(jnp.sum(valid), 1)
    mean = jnp.sum(K_masked) / n_valid
    # soft minimum over valid seeds (invalid lanes excluded by a large value)
    tau = sm.tau_seed
    smin = -tau * jax.scipy.special.logsumexp(jnp.where(valid, -K_s / tau, -jnp.inf)) + tau * jnp.log(n_valid)
    score = (1.0 - sm.beta_seed) * mean + sm.beta_seed * smin
    # a design with no valid seed scores zero
    score = jnp.where(jnp.sum(valid) > 0, score, 0.0)
    return Diagnostics(score=score, K_T=per_seed.K_T, K_R=per_seed.K_R, K_s=K_s, valid=valid, per_seed=per_seed)


def objective(phi: Phi, topo: Topology, bounds: Bounds, fixed: FixedParams, seeds: SeedSet, sp: StepParams,
              task: TaskSpec, cfg: MetricConfig, qp_settings: QPSettings | None = None, act=None):
    """Scalar smooth objective of the unconstrained parameters (seeds re-projected inside)."""
    hm = build_hand_model(topo, decode(phi, topo, bounds, fixed))
    return evaluate(hm, topo, seeds, sp, task, cfg, qp_settings, act).score


def summarize(d: Diagnostics) -> dict:
    """Host-side summary of a Diagnostics (for logs and reports)."""
    v = np.asarray(d.valid)
    ps = d.per_seed
    gates_T = np.asarray(ps.gates_T)[v]          # [S', U, H, 7]
    out = dict(score=float(d.score), n_valid=int(v.sum()),
               K_T=np.asarray(d.K_T)[v].tolist(), K_R=np.asarray(d.K_R)[v].tolist(),
               K_R_regions_mean=np.asarray(ps.K_R_regions)[v].mean(axis=1).tolist() if v.any() else [],
               reproj_ok=np.asarray(ps.reproj_ok)[v].tolist(), reproj_gap_mm=(np.asarray(ps.reproj_gap)[v] * 1e3).tolist(),
               e_worst_T=np.asarray(ps.e_worst_T)[v].tolist(), e_total_T=np.asarray(ps.e_total_T)[v].tolist(),
               e_worst_R=np.asarray(ps.e_worst_R)[v].tolist(),
               end_credit_T=np.exp(np.asarray(ps.end_logw_T)[v]).tolist(), end_credit_R=np.exp(np.asarray(ps.end_logw_R)[v]).tolist(),
               gate_pass_rate_T={k: float(gates_T[..., i].mean()) if gates_T.size else None
                                 for i, k in enumerate(("qp", "slip", "gap", "collision", "self_collision", "grasp", "limits"))})
    return out


__all__ = ["SeedEval", "Diagnostics", "evaluate_seed", "evaluate", "objective", "summarize", "cube_offsets"]
