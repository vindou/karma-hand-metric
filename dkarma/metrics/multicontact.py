"""MC-KaRMA and Full KaRMA: simultaneous multi-contact rolling with full SO(3) orientation
coverage, and the pairwise baseline (SPEC.md §6).

* :func:`evaluate_pairs`: the two-contact pipeline per permitted finger pair
  (``K_pair`` = mean, plus min and max), each pair with its own seeds.
* :func:`evaluate_fixed_contacts`: one object state, one joint configuration, one shared
  force equilibrium for all active contacts (SPEC §6.2). The rolling problem, the
  projection and the contact pass are the masked-lane kernels of Phases 2-4 with three
  valid lanes; the grasp gate is the smooth frictional equilibrium of all contacts
  (``RollConfig.grasp_mode = "equilibrium"``); with ``cfg.roll.actuation`` the torque
  ceilings and the object load gate every step too (Full KaRMA). Orientation coverage is
  over the 576 Hopf targets on SO(3) (D16), translation coverage as in Diff-KaRMA.
* :func:`evaluate_fixed_contacts_exact`: the exact counterpart on the same library:
  hard-mode replay with every exact gate (slip, gap, clearances, limits, exact frictional
  equilibrium by SLSQP at the primitive ends, exact torque feasibility when actuated),
  counting cells and 4608-bin SO(3) occupancy. There is no reference kernel for three
  contacts; this fixed-assignment validator is the exact path (documented, SPEC §6.5).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from karma_jax.kinematics import forward_kinematics
from karma_jax.rolling import StepParams

from ..core.morphology import HandModel
from ..core.topology import Topology
from ..core.object_model import TaskSpec
from ..core.contacts import contact_pass
from ..core.actuation import ActuationParams, grasp_matrix, contact_torque_map, gravity_torque, angular_jacobians
from ..solvers.rollout import GraspState, rollout, energies, exact_gates, force_penalty, seed_force_energy
from ..solvers.seed_projection import SeedSet, SeedConfig, assignment, find_seeds, reproject
from ..solvers.contact_forces import build_problem, solve_exact, ForceWeights
from .commands import CommandConfig, build_library
from .config import MetricConfig, DIFF_KARMA, MC_KARMA
from .orientation import hopf_rotations, so3_bin_index
from . import smooth_coverage as SC
from . import diff_karma as DK


# ── pairwise baseline ────────────────────────────────────────────────────────

class PairResult(NamedTuple):
    K_pair: float
    K_min: float
    K_max: float
    per_pair: dict


def evaluate_pairs(hm: HandModel, topo: Topology, pairs, sp: StepParams, task: TaskSpec, cfg: MetricConfig = DIFF_KARMA,
                   seed_cfg: SeedConfig = SeedConfig()) -> PairResult:
    """Two-contact Diff-KaRMA per finger pair (names), each with its own seeds. The hand model is
    shared; only the topology's ``pair`` (which fingers play thumb/index) changes."""
    per = {}
    for a, b in pairs:
        t2 = Topology(fingers=topo.fingers, n_joints=topo.n_joints, pair=(a, b))
        hm2 = hm._replace(dm=hm.dm._replace(thumb_caps=hm.finger_caps[t2.thumb_finger], index_caps=hm.finger_caps[t2.index_finger]))
        seeds = find_seeds(hm2, t2, sp, seed_cfg)
        if int(seeds.valid.sum()) == 0:
            per[(a, b)] = dict(score=0.0, n_seeds=0)
            continue
        d = jax.jit(lambda h: DK.evaluate(h, t2, seeds, sp, task, cfg))(hm2)
        per[(a, b)] = dict(score=float(d.score), K_T=float(np.asarray(d.K_T)[np.asarray(d.valid)].mean()),
                           K_R=float(np.asarray(d.K_R)[np.asarray(d.valid)].mean()), n_seeds=int(seeds.valid.sum()))
    scores = [v["score"] for v in per.values()]
    return PairResult(float(np.mean(scores)), float(np.min(scores)), float(np.max(scores)), per)


# ── simultaneous contacts ────────────────────────────────────────────────────

class MultiSeedEval(NamedTuple):
    K_T: object
    K_SO3: object
    e_total_T: object
    e_total_R: object
    end_logw_T: object
    end_logw_R: object
    gates_T: object      # [U_T, H_T, 8] (the 7 kinematic gates + force)
    gates_R: object
    force_T: object
    force_R: object
    grasp_T: object      # [U_T, H_T] smooth equilibrium margin (<= 0)
    reproj_ok: object
    p_T: object
    R_T: object          # [U_T, H_T+1, 3, 3]
    R_R: object
    seed_force_E: object # scalar: E_F* of the seed state (0 without actuation); gates the seed's credit (Phase 8.1)


def multi_library(frame, cfg: CommandConfig):
    """Translations along the seed frame (6) and rotations about the three frame axes (6)."""
    dtype = jnp.asarray(frame).dtype
    e = frame.T
    dirs = jnp.stack([e[0], -e[0], e[1], -e[1], e[2], -e[2]])
    dp_T = jnp.broadcast_to((cfg.step_m * dirs)[:, None, :], (6, cfg.H_T, 3))
    dth_R = jnp.broadcast_to((cfg.rotation_step_rad * dirs)[:, None, :], (6, cfg.H_R, 3))
    return dp_T, jnp.zeros_like(dp_T), jnp.zeros_like(dth_R), dth_R


def evaluate_seed_multi(hm, q, p, frame, asg, sp, task, cfg: MetricConfig, cube, targets, act: ActuationParams,
                        qp_settings=None) -> MultiSeedEval:
    dtype = q.dtype
    sm = cfg.smooth
    q0, ok, _ = reproject(hm, q, p, asg, sp)
    st0 = GraspState(q0, p, jnp.eye(3, dtype=dtype))
    dp_T, dth_T, dp_R, dth_R = multi_library(frame, cfg.commands)
    if cfg.roll.actuation:
        fk0 = forward_kinematics(hm.dm, q0)
        seed_E = seed_force_energy(hm, fk0, contact_pass(hm, fk0, p, sp.sphere_radius, asg), p, asg, cfg.roll, act)
    else:
        seed_E = jnp.zeros((), dtype)
    roll_T = cfg.roll._replace(kind="translation", grasp_mode="equilibrium")
    roll_R = cfg.roll._replace(kind="rotation", grasp_mode="equilibrium")
    states_T, diags_T = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_T, qp_settings, act))(dp_T, dth_T)
    states_R, diags_R = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_R, qp_settings, act))(dp_R, dth_R)
    en_T = jax.vmap(lambda d: energies(d, sp, cfg.energy))(diags_T)
    en_R = jax.vmap(lambda d: energies(d, sp, cfg.energy))(diags_R)
    seed_logw = jnp.where(ok, -force_penalty(seed_E, cfg.energy) / cfg.energy.sigma, -1e9)
    lw_T = en_T.prefix_logw + seed_logw
    lw_R = en_R.prefix_logw + seed_logw
    p_all = jnp.concatenate([states_T.p.reshape(-1, 3), states_R.p.reshape(-1, 3)])
    lw_all = jnp.concatenate([jnp.concatenate([jnp.broadcast_to(seed_logw, (6, 1)), lw_T], 1).reshape(-1),
                              jnp.concatenate([jnp.broadcast_to(seed_logw, (6, 1)), lw_R], 1).reshape(-1)])
    cov_T = SC.translation_coverage(p_all, lw_all, cube + p[None, :], sm.sigma_p_m)
    R_all = jnp.concatenate([states_T.R.reshape(-1, 3, 3), states_R.R.reshape(-1, 3, 3)])
    cov_S = SC.so3_coverage(R_all, lw_all, targets, sm.sigma_so3_rad)

    def gates(d):
        g = exact_gates(d, sp)
        return jnp.stack([g[k] for k in ("qp", "slip", "gap", "collision", "self_collision", "grasp", "limits", "force")], -1)

    force = lambda d: jnp.stack([d.force_E, d.torque_usage, d.cone_margin, d.fn_min], -1)   # noqa: E731
    return MultiSeedEval(K_T=cov_T.fraction, K_SO3=cov_S.fraction, e_total_T=en_T.total, e_total_R=en_R.total,
                         end_logw_T=lw_T[:, -1], end_logw_R=lw_R[:, -1], gates_T=jax.vmap(gates)(diags_T),
                         gates_R=jax.vmap(gates)(diags_R), force_T=force(diags_T), force_R=force(diags_R),
                         grasp_T=diags_T.grasp_margin, reproj_ok=ok, p_T=states_T.p, R_T=states_T.R, R_R=states_R.R,
                         seed_force_E=seed_E)


class MultiDiagnostics(NamedTuple):
    score: object
    K_T: object
    K_SO3: object
    K_s: object
    valid: object
    per_seed: MultiSeedEval


def evaluate_fixed_contacts(hm: HandModel, topo: Topology, fingers: tuple[int, ...], seeds: SeedSet, sp: StepParams,
                            task: TaskSpec, act: ActuationParams, cfg: MetricConfig = MC_KARMA, qp_settings=None) -> MultiDiagnostics:
    """Smooth MC-KaRMA (``cfg.roll.actuation=False``) or Full KaRMA (``True``) for ``fingers`` in
    simultaneous contact. ``act`` supplies the object load and friction for the grasp gate even
    when the torque ceilings are off."""
    asg = assignment(topo, 3, fingers)
    cube = DK.cube_offsets(task, hm.dm.t_joint.dtype)
    targets = jnp.asarray(hopf_rotations(task.orientation.so3_targets_smooth), hm.dm.t_joint.dtype)
    per = jax.vmap(lambda q, p, f: evaluate_seed_multi(hm, q, p, f, asg, sp, task, cfg, cube, targets, act, qp_settings))(
        seeds.q, seeds.p, seeds.frame)
    sm = cfg.smooth
    K_s = sm.lambda_T * per.K_T / sm.ref_T + sm.lambda_R * per.K_SO3 / sm.ref_R
    valid = seeds.valid
    n_valid = jnp.maximum(jnp.sum(valid), 1)
    mean = jnp.sum(jnp.where(valid, K_s, 0.0)) / n_valid
    tau = sm.tau_seed
    smin = -tau * jax.scipy.special.logsumexp(jnp.where(valid, -K_s / tau, -jnp.inf)) + tau * jnp.log(n_valid)
    score = jnp.where(jnp.sum(valid) > 0, (1.0 - sm.beta_seed) * mean + sm.beta_seed * smin, 0.0)
    return MultiDiagnostics(score, per.K_T, per.K_SO3, K_s, valid, per)


# ── exact counterpart (fixed assignment) ─────────────────────────────────────

class MultiExact(NamedTuple):
    K_T: float
    K_SO3: float
    per_seed: list


def _exact_equilibrium(hm, q, p, asg, sp, act, actuated: bool):
    fkr = forward_kinematics(hm.dm, q)
    cs = contact_pass(hm, fkr, p, sp.sphere_radius, asg)
    tau_bar = act.tau_bar if actuated else 1e3 * act.tau_bar
    pb = build_problem(grasp_matrix(cs.p_sphere, p, asg.valid), contact_torque_map(cs.J_surf, asg.valid),
                       angular_jacobians(cs.Jw, asg.valid), gravity_torque(hm, fkr, act), cs.normal, asg.valid,
                       tau_bar, act.mu, ForceWeights().torsion_radius_m)
    return solve_exact(pb, act.w_ext)["feasible"]


# Phase 9: the per-seed rollouts of the exact evaluator through one jitted program per roll configuration (the model,
# state and commands are arguments), instead of an eager ``vmap`` of a fresh closure per seed, which recompiled the
# scan for every seed (~3 s each on the CPU) and accumulated compiled programs (~15 GB over one 9.2 suite). The
# outputs are bit-identical to the eager form (docs/phase9_report.md §9.4, verified on whole suites); set
# ``CACHED_ROLLOUTS = False`` for the Phase 8 code path.
CACHED_ROLLOUTS = True
_ROLLOUT_PROGRAMS: dict = {}


def _command_rollouts(hm, st0, dps, dths, asg, sp, roll_cfg, act):
    """``vmap`` over commands of the hard fixed-assignment rollout from ``st0``."""
    if not CACHED_ROLLOUTS:
        return jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_cfg, None, act))(dps, dths)
    f = _ROLLOUT_PROGRAMS.get(roll_cfg)
    if f is None:
        f = jax.jit(lambda hm_, st_, a_, b_, asg_, sp_, act_: jax.vmap(
            lambda x, y: rollout(hm_, st_, x, y, asg_, sp_, roll_cfg, None, act_))(a_, b_))
        _ROLLOUT_PROGRAMS[roll_cfg] = f
    return f(hm, st0, dps, dths, asg, sp, act)


def evaluate_fixed_contacts_exact(hm: HandModel, topo: Topology, fingers: tuple[int, ...], seeds: SeedSet, sp: StepParams,
                                  task: TaskSpec, act: ActuationParams, cfg: MetricConfig = MC_KARMA) -> MultiExact:
    """Hard-mode replay with exact gates per primitive; cells and 4608 SO(3) bins are counted
    over exactly valid prefixes. Trajectories stop at the first failed primitive."""
    asg = assignment(topo, 3, fingers)
    n_sub = cfg.commands.n_sub
    n = task.task.n_per_axis
    actuated = bool(cfg.roll.actuation)
    per = []
    for s in range(int(np.asarray(seeds.valid).sum())):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        st0 = GraspState(q, p, jnp.eye(3, dtype=q.dtype))
        dp_T, dth_T, dp_R, dth_R = multi_library(frame, cfg.commands)
        roll_T = cfg.roll._replace(kind="translation", mode="hard", grasp_mode="equilibrium")
        roll_R = cfg.roll._replace(kind="rotation", mode="hard", grasp_mode="equilibrium")
        sT, dT = _command_rollouts(hm, st0, dp_T, dth_T, asg, sp, roll_T, act)
        sR, dR = _command_rollouts(hm, st0, dp_R, dth_R, asg, sp, roll_R, act)
        gT = jax.vmap(lambda d: exact_gates(d, sp))(dT)
        gR = jax.vmap(lambda d: exact_gates(d, sp))(dR)
        kin = lambda g: np.array(g["qp"] & g["slip"] & g["gap"] & g["collision"] & g["self_collision"] & g["limits"])   # noqa: E731  (copy: writable)
        okT_step, okR_step = kin(gT), kin(gR)
        # Phase 8.6: the actuated force check of every sub-step is the exact solver's, not the smooth
        # allocation's gate (whose 1e-6 N equilibrium tolerance rejects exactly feasible states near the
        # torque margin); it is evaluated lazily below, up to the first failed sub-step of each command
        cells, bins = set(), set()
        okT = np.zeros((6, dp_T.shape[1] // n_sub), bool)
        okR = np.zeros((6, dth_R.shape[1]), bool)
        # the seed state itself must hold the object under the torque ceilings when actuated (Phase 8.1)
        if actuated and not _exact_equilibrium(hm, q, p, asg, sp, act, True):
            per.append(dict(cells=cells, bins=bins, ok_T=okT, ok_R=okR, seed_force_ok=False))
            continue
        cells.add(tuple(np.zeros(3, int) + n // 2))
        bins.add(int(so3_bin_index(np.eye(3)[None], task.orientation.so3_bins_exact)[0]))
        for u in range(6):
            for k in range(okT.shape[1]):
                sl = slice(k * n_sub, (k + 1) * n_sub)
                if not np.all(okT_step[u, sl]):
                    break
                if actuated and not all(_exact_equilibrium(hm, sT.q[u, h + 1], sT.p[u, h + 1], asg, sp, act, True)
                                        for h in range(k * n_sub, (k + 1) * n_sub - 1)):
                    break
                qk, pk = sT.q[u, (k + 1) * n_sub], sT.p[u, (k + 1) * n_sub]
                if not _exact_equilibrium(hm, qk, pk, asg, sp, act, actuated):
                    break
                okT[u, k] = True
                idx = np.floor(np.asarray(pk - p) / task.task.grid_m + n / 2.0).astype(int)
                if np.all(idx >= 0) and np.all(idx < n):
                    cells.add(tuple(idx))
                bins.add(int(so3_bin_index(np.asarray(sT.R[u, (k + 1) * n_sub])[None], task.orientation.so3_bins_exact)[0]))
        for u in range(6):
            for k in range(okR.shape[1]):
                if not okR_step[u, k]:
                    break
                if not _exact_equilibrium(hm, sR.q[u, k + 1], sR.p[u, k + 1], asg, sp, act, actuated):
                    break
                okR[u, k] = True
                bins.add(int(so3_bin_index(np.asarray(sR.R[u, k + 1])[None], task.orientation.so3_bins_exact)[0]))
        per.append(dict(cells=cells, bins=bins, ok_T=okT, ok_R=okR, seed_force_ok=True))
    if not per:
        return MultiExact(0.0, 0.0, [])
    return MultiExact(float(np.mean([len(r["cells"]) for r in per])) / task.task.n_cells,
                      float(np.mean([len(r["bins"]) for r in per])) / task.orientation.so3_bins_exact, per)


__all__ = ["PairResult", "evaluate_pairs", "MultiSeedEval", "MultiDiagnostics", "multi_library", "evaluate_seed_multi",
           "evaluate_fixed_contacts", "MultiExact", "evaluate_fixed_contacts_exact"]
