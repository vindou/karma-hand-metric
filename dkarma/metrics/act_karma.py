"""Act-KaRMA: actuation-aware two-finger rolling reachability (SPEC.md §5).

The same trajectory library as Diff-KaRMA, with the quasi-static force problem solved
at every post-projection state (``RollConfig.actuation``): object equilibrium under the
task's external wrench (the sphere's weight), Coulomb friction cones, unilateral normal
forces, and per-joint torque ceilings including the hand's own gravity torque. The
smooth force energy ``E_F*`` enters every step's energy, so trajectories that cannot be
held earn no coverage; the exact allocation (``contact_forces.solve_exact``) and the
wrench capacity are host-side diagnostics computed at chosen states.

Interface (SPEC §5 deliverable):

    act = actuation_for(hm, task)                      # ActuationParams from task_spec.yaml
    d = evaluate(hm, topo, seeds, sp, task, cfg, act)  # Diagnostics with force diagnostics
    summary = summarize(d, ...)                        # torque usage, cone margins, wrench residual, coverage
    cap = wrench_capacity_at(hm, state, asg, sp, act)  # alpha*(d) per direction (exact, host)
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from karma_jax.kinematics import forward_kinematics
from karma_jax.rolling import StepParams

from ..core.morphology import HandModel, Topology
from ..core.object_model import TaskSpec
from ..core.actuation import (ActuationSpec, ActuationParams, load_actuation_spec, actuation_params, grasp_matrix,
                              contact_torque_map, gravity_torque, angular_jacobians)
from ..core.contacts import ContactAssignment, contact_pass
from ..solvers.rollout import GraspState
from ..solvers.contact_forces import (solve_soft, solve_exact, wrench_capacity, wrench_directions, ForceWeights,
                                      build_problem, ForceProblem)
from ..solvers.seed_projection import SeedSet, assignment
from .config import MetricConfig, ACT_KARMA
from . import diff_karma as DK


def actuation_for(hm: HandModel, task: TaskSpec, spec: ActuationSpec | None = None, dtype=None) -> ActuationParams:
    spec = spec or load_actuation_spec()
    return actuation_params(hm, spec, task.obj.gravity_m_s2, task.obj.mass_kg, task.obj.friction_coefficient, dtype)


def evaluate(hm: HandModel, topo: Topology, seeds: SeedSet, sp: StepParams, task: TaskSpec,
             cfg: MetricConfig = ACT_KARMA, act: ActuationParams | None = None, qp_settings=None) -> DK.Diagnostics:
    if not cfg.roll.actuation:
        raise ValueError("Act-KaRMA needs cfg.roll.actuation = True")
    act = act if act is not None else actuation_for(hm, task, dtype=hm.dm.t_joint.dtype)
    return DK.evaluate(hm, topo, seeds, sp, task, cfg, qp_settings, act)


def force_problem_at(hm: HandModel, state: GraspState, asg: ContactAssignment, sp: StepParams, act: ActuationParams,
                     wts: ForceWeights = ForceWeights()) -> ForceProblem:
    """The ``[f, m]`` force problem at a state (post-contact-pass quantities)."""
    fkr = forward_kinematics(hm.dm, state.q)
    cs = contact_pass(hm, fkr, state.p, sp.sphere_radius, asg)
    return build_problem(grasp_matrix(cs.p_sphere, state.p, asg.valid), contact_torque_map(cs.J_surf, asg.valid),
                         angular_jacobians(cs.Jw, asg.valid), gravity_torque(hm, fkr, act), cs.normal, asg.valid,
                         act.tau_bar, act.mu, wts.torsion_radius_m)


def soft_and_exact_at(hm, state, asg, sp, act, wts: ForceWeights = ForceWeights(), load_scale: float = 1.0):
    """Both solvers at one state: ``(ForceResult, exact dict)``."""
    pb = force_problem_at(hm, state, asg, sp, act, wts)
    w = load_scale * act.w_ext
    return solve_soft(pb, w, wts), solve_exact(pb, w)


def wrench_capacity_at(hm, state, asg, sp, act, force_scale: float = 0.343, moment_scale: float = 0.343 * 0.015,
                       n_dir: int = 26, wts: ForceWeights = ForceWeights()):
    """``alpha*(d)`` for standardized directions (default scale: the 35 g object's weight in force
    and that weight times the sphere radius in moment). Returns ``(alphas, directions)``."""
    pb = force_problem_at(hm, state, asg, sp, act, wts)
    D = wrench_directions(force_scale, moment_scale, n_dir)
    return wrench_capacity(pb, act.w_ext, D), D


def summarize(d: DK.Diagnostics) -> dict:
    """Diff-KaRMA summary plus the actuation diagnostics (SPEC §5 deliverable)."""
    out = DK.summarize(d)
    v = np.asarray(d.valid)
    if not v.any():
        return out
    fT = np.asarray(d.per_seed.force_T)[v]          # [S', U, H, 4]
    fR = np.asarray(d.per_seed.force_R)[v]
    out.update(
        torque_usage_max=float(max(fT[..., 1].max(), fR[..., 1].max())),
        torque_usage_seed_state=float(np.mean(fT[:, :, 0, 1])),
        cone_margin_min=float(min(fT[..., 2].min(), fR[..., 2].min())),
        fn_min=float(min(fT[..., 3].min(), fR[..., 3].min())),
        force_E_worst_T=fT[..., 0].max(axis=(1, 2)).tolist(),
        force_feasible_step_frac=float(np.mean((fT[..., 1] <= 1.0) & (fT[..., 2] >= 0) & (fT[..., 3] >= 0))),
    )
    return out


__all__ = ["actuation_for", "evaluate", "force_problem_at", "soft_and_exact_at", "wrench_capacity_at", "summarize"]
