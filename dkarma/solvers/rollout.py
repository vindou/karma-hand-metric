"""Fixed-horizon differentiable rolling rollout (SPEC.md §3, §4.1-§4.3).

One :func:`step` moves the grasp by a commanded object displacement ``dp`` and
extra rotation ``dtheta`` (world frame): contact pass at the fixed assignment,
rolling problem, soft or hard solve, joint update, sphere update ``p += dp``,
``R <- exp(dtheta) R``, Gauss-Newton projection back onto the contact manifold,
``R <- exp(dtheta_proj) R``, then the per-step residuals that the exact gates
test (slip, gap, clearances, friction cone, joint limits) as smooth numbers.
:func:`rollout` scans ``H`` steps. :func:`energies` turns the residuals into
dimensionless per-step energies ``e_k``, the prefix weights
``W_k = exp(-sum_{j<=k} e_j / sigma)`` (soft truncation: credit decays after the
first bad step and never recovers), and the smooth worst-step energy.

Two step kinds, matching the reference's two QPs: ``kind="translation"`` keeps
the rotation free with the rolling prior (the reference translation sub-step),
``kind="rotation"`` fixes the rotation to the command (the reference Phase-2
step) and reports the achieved rotation and efficiency.

These energies are optimization surrogates. Exact feasibility of a rollout is
decided by :mod:`dkarma.solvers.exact`, never here (CLAUDE.md rule 2).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax

from karma_jax.kinematics import forward_kinematics
from karma_jax.qp import QPSettings, default_settings
from karma_jax.rolling import StepParams, LAMBDA_THETA, THETA_BOUND
from karma_jax.steps import pinch_axis_body

from ..core.morphology import HandModel
from ..core.contacts import ContactAssignment, contact_pass, object_clearance, self_clearance, antipodal_margin
from ..core import geometry as geo
from ..core.geometry import exp_so3_safe as exp_so3   # finite derivative at a zero rotation (Phase 9.1)
from .rolling_qp import assemble, solve_soft, solve_hard, dtheta_achieved
from .projection import project


class GraspState(NamedTuple):
    q: object      # [J]
    p: object      # [3]
    R: object      # [3,3]


class StepDiag(NamedTuple):
    """Per-step residuals (all smooth functions of the morphology and the state)."""
    slip_m: object          # max tangential rolling residual over lanes (m)
    spin_err: object        # |dtheta - prior| (rad), translation steps
    gap_pre: object         # max |gap| over lanes before projection (m)
    gap_post: object        # max |gap| after projection (m)
    limit_viol: object      # max joint-limit violation of q + dq before clipping (rad)
    obj_clear: object       # soft min sphere clearance over non-contact capsules (m)
    obj_clear_exact: object
    self_clear: object      # soft min capsule-capsule clearance across fingers (m)
    self_clear_exact: object
    grasp_margin: object    # cos(-n0, n1) - cos(2 atan mu) for the first two lanes
    efficiency: object      # rotation steps: achieved/target alignment in [0,1]; 1 for translation
    qp_ok: object           # bool (hard mode)
    proj_ok: object         # bool: gap_post <= eps_g
    kkt_gap: object
    dq_norm: object
    force_E: object         # smooth force-feasibility energy E_F* (0 when actuation is off)
    force_eq_res: object    # |G f + w_ext| at the soft solution (N)
    torque_usage: object    # max_j |tau_j| / tau_bar_j
    cone_margin: object     # min_c (mu f_n - |f_t|) / F0
    fn_min: object          # min_c f_n / F0
    normal0: object         # [3] lane-0 (thumb) contact normal after projection, world frame
    axis_body: object       # [3] pinch axis R^T (-normal0) in the sphere body frame (the reference's binned quantity)


class RollConfig(NamedTuple):
    """Static configuration of the smooth rollout."""
    mode: str = "hard"             # "soft" | "hard"
    kind: str = "translation"      # "translation" | "rotation"
    proj_iters: int = 10
    proj_backtrack: bool = True
    beta_clear: float = 500.0      # soft-min sharpness for clearances (1/m)
    lambda_theta: float = LAMBDA_THETA
    theta_bound: float = THETA_BOUND
    actuation: bool = False        # Act-KaRMA: solve the smooth force problem at every step
    force_weights: object = None   # dkarma.solvers.contact_forces.ForceWeights (None = defaults)
    grasp_mode: str = "antipodal"  # "antipodal": the reference's two-contact cone test on lanes 0,1;
                                   # "equilibrium": smooth frictional equilibrium of all lanes under the
                                   # task load with unlimited torque (multi-contact grasp gate, SPEC §6.2)
    kkt_method: str = "reduced"    # hard-mode derivative of the rolling QP (Phase 9.1 default; "exact" = Phase 2): "exact" | "reg" | "pinv" | "reduced"
    kkt_reg: float = 1e-10         # dual-block regularization of the KKT system ("exact" and "reg")
    kkt_dep_tol: float = 1e-6      # "reduced": relative Gram-Schmidt tolerance for a dependent active row


def step(hm: HandModel, state: GraspState, dp, dtheta_cmd, asg: ContactAssignment, sp: StepParams,
         cfg: RollConfig, qp_settings: QPSettings | None = None, act=None):
    """One smooth step. Returns ``(new_state, StepDiag)``. With ``cfg.actuation`` and ``act``
    (an ``ActuationParams``) the quasi-static force problem is solved at the post-projection state."""
    dtype = state.q.dtype
    r_s = sp.sphere_radius
    if qp_settings is None:
        qp_settings = default_settings(dtype)
    free_theta = cfg.kind == "translation"

    fkr = forward_kinematics(hm.dm, state.q)
    cs = contact_pass(hm, fkr, state.p, r_s, asg)
    prob = assemble(hm, cs, asg, dp, dtheta_cmd, state.q, sp, free_theta=free_theta,
                    lambda_theta=cfg.lambda_theta, theta_bound=cfg.theta_bound)
    sol = solve_soft(prob, sp) if cfg.mode == "soft" else solve_hard(prob, sp, qp_settings, kkt_method=cfg.kkt_method,
                                                                     kkt_reg=cfg.kkt_reg, kkt_dep_tol=cfg.kkt_dep_tol)

    q_lo, q_hi = hm.dm.q_lower, hm.dm.q_upper
    q_raw = state.q + sol.dq
    limit_viol = jnp.max(jnp.maximum(q_raw - q_hi, q_lo - q_raw))
    q1 = jnp.clip(q_raw, q_lo, q_hi)
    p1 = state.p + dp
    if free_theta:
        dtheta = sol.dtheta
        efficiency = jnp.ones((), dtype)
    else:
        dtheta = dtheta_achieved(prob, sol.dq, r_s)
        tn = jnp.linalg.norm(dtheta_cmd)
        has = tn > 1e-10
        efficiency = jnp.where(has, jnp.clip(jnp.dot(dtheta_cmd, dtheta) / jnp.where(has, tn ** 2, 1.0), 0.0, 1.0), 0.0)
    R1 = exp_so3(dtheta) @ state.R

    pr = project(hm, q1, p1, asg, sp, max_iters=cfg.proj_iters, backtrack=cfg.proj_backtrack)
    R2 = exp_so3(pr.dtheta) @ R1

    vf = asg.valid.astype(dtype)
    gap_pre = jnp.max(jnp.abs(cs.gap * vf))
    oc_soft = object_clearance(hm, pr.cp, asg, beta=cfg.beta_clear)
    oc_exact = object_clearance(hm, pr.cp, asg, beta=None)
    sc_soft = self_clearance(hm, pr.cp, beta=cfg.beta_clear)
    sc_exact = self_clearance(hm, pr.cp, beta=None)
    if cfg.grasp_mode == "antipodal" or act is None:
        margin = antipodal_margin(pr.cs.normal[:2], sp.friction_coeff)
    else:
        # multi-contact grasp gate: can the contacts hold the object at all (torque unlimited)?
        from ..core.actuation import grasp_matrix as _gm, contact_torque_map as _ctm, gravity_torque as _gt, angular_jacobians as _aj
        from .contact_forces import solve_soft as _ss, ForceWeights as _FW, build_problem as _bp
        _w = cfg.force_weights or _FW()
        _pb = _bp(_gm(pr.cs.p_sphere, p1, asg.valid), _ctm(pr.cs.J_surf, asg.valid), _aj(pr.cs.Jw, asg.valid),
                  _gt(hm, pr.fk, act), pr.cs.normal, asg.valid, 1e3 * act.tau_bar, act.mu, _w.torsion_radius_m)
        margin = -_ss(_pb, act.w_ext, _w).E          # <= 0; 0 means a feasible frictional equilibrium

    zero = jnp.zeros((), dtype)
    force = dict(force_E=zero, force_eq_res=zero, torque_usage=zero, cone_margin=jnp.ones((), dtype), fn_min=jnp.ones((), dtype))
    if cfg.actuation and act is not None:
        from ..core.actuation import grasp_matrix, contact_torque_map, gravity_torque, angular_jacobians
        from .contact_forces import solve_soft as solve_forces, ForceWeights, build_problem
        wts = cfg.force_weights or ForceWeights()
        pb = build_problem(grasp_matrix(pr.cs.p_sphere, p1, asg.valid), contact_torque_map(pr.cs.J_surf, asg.valid),
                           angular_jacobians(pr.cs.Jw, asg.valid), gravity_torque(hm, pr.fk, act), pr.cs.normal,
                           asg.valid, act.tau_bar, act.mu, wts.torsion_radius_m)
        fr = solve_forces(pb, act.w_ext, wts)
        force = dict(force_E=fr.E, force_eq_res=fr.eq_residual, torque_usage=fr.torque_usage,
                     cone_margin=jnp.minimum(fr.cone_margin, fr.torsion_margin), fn_min=fr.fn_min)

    diag = StepDiag(slip_m=sol.slip_m, spin_err=sol.spin_err, gap_pre=gap_pre, gap_post=pr.gap_max,
                    limit_viol=jnp.maximum(limit_viol, 0.0), obj_clear=oc_soft, obj_clear_exact=oc_exact,
                    self_clear=sc_soft, self_clear_exact=sc_exact, grasp_margin=margin,
                    efficiency=efficiency, qp_ok=sol.qp_ok, proj_ok=pr.ok, kkt_gap=sol.kkt_gap,
                    dq_norm=jnp.linalg.norm(sol.dq), normal0=pr.cs.normal[0],
                    axis_body=pinch_axis_body(pr.cs.normal[0], R2), **force)
    return GraspState(pr.q, p1, R2), diag


def rollout(hm: HandModel, state0: GraspState, dps, dthetas, asg: ContactAssignment, sp: StepParams,
            cfg: RollConfig, qp_settings: QPSettings | None = None, act=None):
    """Scan ``H`` steps. ``dps``/``dthetas`` are ``[H,3]``. Returns ``(states [H+1], diags [H])``
    with the initial state prepended."""
    def body(st, cmd):
        dp, dth = cmd
        st1, dg = step(hm, st, dp, dth, asg, sp, cfg, qp_settings, act)
        return st1, (st1, dg)

    _, (states, diags) = lax.scan(body, state0, (dps, dthetas))
    states = jax.tree_util.tree_map(lambda a, b: jnp.concatenate([a[None], b], axis=0), state0, states)
    return states, diags


# ── Energies and weights ─────────────────────────────────────────────────────

class EnergyWeights(NamedTuple):
    roll: float = 1.0
    gap: float = 1.0
    collision: float = 1.0
    limits: float = 1.0
    grasp: float = 1.0
    qp_fail: float = 10.0
    sigma: float = 1.0          # trajectory weight scale: W = exp(-E / sigma)
    worst_tau: float = 0.25     # temperature of the smooth worst-step energy
    efficiency: float = 1.0     # rotation steps: penalty on (1 - efficiency)
    force: float = 1.0          # Act-KaRMA: weight on the smooth force-feasibility energy E_F*
    # Phase 8.1 credit mapping (docs/phase8_report.md §8.1): extra penalty force_steep * hinge((E_F* - knee) / knee);
    # the knee is the Youden threshold of E_F* against the exact label on the held-out states (hinge 0.005)
    force_knee: float = 7.0e-7
    force_steep: float = 3.0    # penalty per knee-unit of excess energy (0 = linear mapping only)
    force_knee_eps: float = 0.125  # softplus width of that hinge, as a fraction of the knee


def step_energy(d: StepDiag, sp: StepParams, w: EnergyWeights):
    """Dimensionless per-step energy: each term is (excess over its tolerance / tolerance)^2."""
    slip_gate = jnp.where(jnp.isfinite(sp.slip_threshold), sp.slip_threshold, 1e-3)
    e_roll = (d.slip_m / slip_gate) ** 2
    e_gap = (geo.hinge(d.gap_post - sp.eps_g, 0.1 * sp.eps_g) / sp.eps_g) ** 2
    e_col = ((geo.hinge(sp.eps_col - d.obj_clear, 0.1 * sp.eps_col) / sp.eps_col) ** 2
             + (geo.hinge(sp.eps_col - d.self_clear, 0.1 * sp.eps_col) / sp.eps_col) ** 2)
    e_lim = (geo.hinge(d.limit_viol, 0.01) / sp.max_joint_step) ** 2
    cone = 1.0 - jnp.cos(2.0 * jnp.arctan(sp.friction_coeff))
    e_grasp = (geo.hinge(-d.grasp_margin, 0.02) / cone) ** 2
    e_qp = jnp.where(d.qp_ok, 0.0, 1.0)
    e_eff = (1.0 - d.efficiency) ** 2
    return (w.roll * e_roll + w.gap * e_gap + w.collision * e_col + w.limits * e_lim + w.grasp * e_grasp
            + w.qp_fail * e_qp + w.efficiency * e_eff + force_penalty(d.force_E, w))


def force_penalty(E_F, w: EnergyWeights):
    """The force term of the step energy: ``force * E_F + force_steep * hinge((E_F - knee) / knee)``
    (Phase 8.1 credit mapping; the hinge is ``knee_eps * softplus(x / knee_eps)``). Applied to every
    rollout step and, for actuation-aware variants, to the seed state itself."""
    e = w.force * E_F
    if w.force_steep > 0:
        e = e + w.force_steep * geo.hinge((E_F - w.force_knee) / w.force_knee, w.force_knee_eps)
    return e


def seed_force_energy(hm: HandModel, fkr, cs, p, asg: ContactAssignment, cfg: RollConfig, act):
    """Smooth force energy ``E_F*`` of a (seed) state from its contact set (Phase 8.1: the seed
    state is gated like every step; SPEC §5 requires the hold at every state)."""
    from ..core.actuation import grasp_matrix, contact_torque_map, gravity_torque, angular_jacobians
    from .contact_forces import solve_soft as solve_forces, ForceWeights, build_problem
    wts = cfg.force_weights or ForceWeights()
    pb = build_problem(grasp_matrix(cs.p_sphere, p, asg.valid), contact_torque_map(cs.J_surf, asg.valid),
                       angular_jacobians(cs.Jw, asg.valid), gravity_torque(hm, fkr, act), cs.normal, asg.valid,
                       act.tau_bar, act.mu, wts.torsion_radius_m)
    return solve_forces(pb, act.w_ext, wts).E


class TrajectoryEnergy(NamedTuple):
    per_step: object      # [H]
    prefix_logw: object   # [H] log W_k = -cumsum(e)/sigma
    worst: object         # smooth max over steps
    total: object         # sum


def energies(diags: StepDiag, sp: StepParams, w: EnergyWeights) -> TrajectoryEnergy:
    e = jax.vmap(lambda d: step_energy(d, sp, w))(diags)
    logw = -jnp.cumsum(e) / w.sigma
    worst = geo.softmax(e, 1.0 / w.worst_tau)
    return TrajectoryEnergy(e, logw, worst, jnp.sum(e))


def exact_gates(d: StepDiag, sp: StepParams):
    """The reference's per-step gates evaluated on the smooth step's own (exact) quantities;
    for reporting, not for scoring (that is ``dkarma.solvers.exact``)."""
    return dict(
        qp=d.qp_ok,
        slip=d.slip_m <= sp.slip_threshold,
        gap=d.proj_ok,
        collision=d.obj_clear_exact >= sp.eps_col,
        self_collision=d.self_clear_exact >= sp.eps_col,
        grasp=d.grasp_margin >= 0.0,
        limits=d.limit_viol <= 1e-6,
        force=(d.torque_usage <= 1.0 + 1e-6) & (d.cone_margin >= -1e-6) & (d.fn_min >= -1e-6) & (d.force_eq_res <= 1e-6),
    )


__all__ = ["GraspState", "StepDiag", "RollConfig", "step", "rollout", "EnergyWeights", "step_energy", "force_penalty",
           "seed_force_energy", "TrajectoryEnergy", "energies", "exact_gates"]
