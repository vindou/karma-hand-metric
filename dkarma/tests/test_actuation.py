"""Phase 4 physical sanity tests (SPEC §10.1-§10.2): torque sign on the planar toy, grasp
matrix, exact allocation feasibility certificates, torque/friction/load monotonicity, the
ideal-actuation limit, soft-vs-exact agreement away from the margin, gradients of the
force energy through the morphology, and the Act-KaRMA evaluator."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from karma_jax.kinematics import forward_kinematics

from dkarma.core import build_hand_model, load_task_spec, Bounds, encode, decode, fixed_from_params
from dkarma.core.actuation import (ActuationSpec, load_actuation_spec, actuation_params, grasp_matrix,
                                   contact_torque_map, gravity_torque)
from dkarma.core.contacts import ContactAssignment, contact_pass
from dkarma.core.kinematics import fk
from dkarma.solvers.contact_forces import solve_soft, solve_exact, wrench_capacity, wrench_directions, ForceWeights, violation
from dkarma.solvers.rollout import GraspState, RollConfig, rollout, energies, EnergyWeights
from dkarma.solvers.seed_projection import find_seeds, SeedConfig, assignment
from dkarma.metrics.config import ACT_KARMA, MetricConfig
from dkarma.metrics.commands import CommandConfig
from dkarma.metrics import act_karma as AK
from .conftest import load_design, planar_toy

jax.config.update("jax_enable_x64", True)

TASK = load_task_spec()
SP = TASK.step_params(jnp.float64)
SPEC = load_actuation_spec()
G_VEC = np.array([0.0, 0.0, -9.81])

_cache = {}


def baseline_with_seeds():
    if "b" not in _cache:
        topo, params, _ = load_design("baseline")
        hm = build_hand_model(topo, params)
        seeds = find_seeds(hm, topo, SP, SeedConfig(n_linspace=3, n_sobol=32, n_random=8, budget=3))
        _cache["b"] = (topo, params, hm, seeds)
    return _cache["b"]


def test_actuation_spec_values():
    assert SPEC.servo == "XL330-M077-T" and SPEC.stall_torque_nm == 0.215 and SPEC.torque_limit_nm == 0.108
    assert SPEC.link_mass_kg == 0.028 and SPEC.terminal_link_mass_kg == 0.010


def test_gravity_torque_planar_closed_form():
    """Horizontal two-link finger under gravity along -y (in-plane): the base joint torque that
    holds the links up is sum_i m_i g x_com,i; the terminal link's torque about joint 2 is
    m_2 g (l2 - r)/2."""
    topo, params, r = planar_toy(l1=0.05, l2=0.04, r=0.01)
    hm = build_hand_model(topo, params)
    spec = ActuationSpec(link_mass_kg=0.028, terminal_link_mass_kg=0.010)
    act = actuation_params(hm, spec, [0.0, -9.81, 0.0], 0.035, 0.6)
    tau = gravity_torque(hm, fk(hm, jnp.zeros(3)), act)
    l1, axis2 = 0.05, 0.04 - r
    expected_j2 = 0.010 * 9.81 * (axis2 / 2)
    expected_j1 = 0.028 * 9.81 * (l1 / 2) + 0.010 * 9.81 * (l1 + axis2 / 2)
    np.testing.assert_allclose(float(tau[1]), expected_j2, rtol=1e-10)
    np.testing.assert_allclose(float(tau[0]), expected_j1, rtol=1e-10)
    # the dummy finger's joint carries only its own link
    np.testing.assert_allclose(float(tau[2]), 0.010 * 9.81 * (0.03 - r) / 2, rtol=1e-10)


def test_contact_torque_planar_closed_form():
    """A force on the fingertip of a straight horizontal finger: joint torques are the cross
    products f x lever (planar: tau_1 = f_y * x_tip, tau_2 = f_y * (x_tip - l1))."""
    topo, params, r = planar_toy(l1=0.05, l2=0.04, r=0.01)
    hm = build_hand_model(topo, params)
    q = jnp.zeros(3)
    fkr = fk(hm, q)
    centre = jnp.array([0.05 + 0.03 - r, 0.0 - (0.015 + r), 0.0])       # sphere below the terminal capsule
    asg = ContactAssignment(caps=jnp.array([1, 2], jnp.int32), valid=jnp.array([True, False]))
    cs = contact_pass(hm, fkr, centre, 0.015, asg)
    Jt = contact_torque_map(cs.J_surf, asg.valid)
    f = jnp.array([0.0, -1.0, 0.0, 0.0, 0.0, 0.0])                      # 1 N pushing down on the object
    tau = Jt @ f
    x_c = float(cs.p_capsule[0][0])
    np.testing.assert_allclose(float(tau[0]), -1.0 * x_c, rtol=1e-9)
    np.testing.assert_allclose(float(tau[1]), -1.0 * (x_c - 0.05), rtol=1e-9)
    assert abs(float(tau[2])) < 1e-12


def test_grasp_matrix_equilibrium_structure():
    p = jnp.array([0.1, 0.2, 0.3])
    pc = p[None, :] + 0.015 * jnp.array([[1.0, 0, 0], [-1.0, 0, 0]])
    G = grasp_matrix(pc, p, jnp.array([True, True]))
    assert G.shape == (6, 6)
    # equal and opposite normal forces along the pinch axis are internal: G f = 0
    f = jnp.array([-1.0, 0, 0, 1.0, 0, 0])
    np.testing.assert_allclose(np.asarray(G @ f), 0.0, atol=1e-15)
    # a single force produces a moment r x f
    f1 = jnp.array([0, 0, 1.0, 0, 0, 0])
    np.testing.assert_allclose(np.asarray(G @ f1)[3:], np.cross([0.015, 0, 0], [0, 0, 1.0]), atol=1e-15)


def _pb(hm, seeds, s, asg, act):
    return AK.force_problem_at(hm, GraspState(seeds.q[s], seeds.p[s], jnp.eye(3)), asg, SP, act)


def test_exact_allocation_is_a_certificate_and_monotone():
    topo, params, hm, seeds = baseline_with_seeds()
    asg = assignment(topo)
    act = AK.actuation_for(hm, TASK)
    n_feas = 0
    for s in range(int(seeds.valid.sum())):
        pb = _pb(hm, seeds, s, asg, act)
        ex = solve_exact(pb, act.w_ext)
        if ex["feasible"]:
            n_feas += 1
            x = ex["x"]
            assert violation(pb, act.w_ext, x) <= 1e-5
            np.testing.assert_allclose(np.asarray(pb.G) @ x + np.asarray(act.w_ext), 0.0, atol=2e-5)
            assert np.all(np.abs(ex["tau"]) <= np.asarray(act.tau_bar) * (1 + 1e-5))
            D = wrench_directions(0.343, 0.343 * 0.015, 12)
            a_lo = wrench_capacity(pb, act.w_ext, D)
            a_hi = wrench_capacity(pb._replace(tau_bar=2.0 * pb.tau_bar), act.w_ext, D)
            assert np.all(a_hi >= a_lo - 1e-4), (a_lo, a_hi)
            a_mu = wrench_capacity(pb._replace(mu=1.5 * pb.mu, gamma=1.5 * pb.gamma), act.w_ext, D)
            assert np.all(a_mu >= a_lo - 1e-4), (a_lo, a_mu)
            ex2 = solve_exact(pb, 2.0 * act.w_ext)
            if ex2["feasible"]:
                assert (np.max(np.abs(ex2["tau"]) / np.asarray(act.tau_bar))
                        >= np.max(np.abs(ex["tau"]) / np.asarray(act.tau_bar)) - 1e-6)
    assert n_feas >= 1, "no seed grasp could hold the 35 g sphere under the primary ceiling"


def test_soft_matches_exact_away_from_margin_and_ideal_limit():
    topo, params, hm, seeds = baseline_with_seeds()
    asg = assignment(topo)
    act = AK.actuation_for(hm, TASK)
    checked = 0
    for s in range(int(seeds.valid.sum())):
        pb = _pb(hm, seeds, s, asg, act)
        for scale in (1.0, 3.0, 10.0, 30.0):
            w = scale * act.w_ext
            ex = solve_exact(pb, w)
            so = solve_soft(pb, w)
            assert bool(jnp.isfinite(so.E))
            # feasible loads give a near-zero energy; loads violated by more than 10 % of their
            # magnitude give an energy well above the feasible level (observed: 1e-7..1e-5 vs >1e-2)
            if ex["feasible"]:
                assert float(so.E) < 2e-3, (s, scale, float(so.E))
                checked += 1
            elif ex["viol"] > 0.1 * max(float(jnp.linalg.norm(w)), 1.0):
                assert float(so.E) > 2e-3, (s, scale, float(so.E), ex["viol"])
                checked += 1
        so_ideal = solve_soft(pb._replace(tau_bar=1e3 * pb.tau_bar, mu=jnp.asarray(50.0), gamma=jnp.asarray(50.0 * 0.003)), act.w_ext)
        assert float(so_ideal.E) < 1e-6 and float(so_ideal.eq_residual) < 1e-5
    assert checked >= 2


def test_force_energy_gradient_wrt_morphology():
    """Phase 8.1 regime: the damped-Newton solve converges (E at 50 and 100 iterations agree) and
    the envelope gradient ``dE*/dphi`` matches finite differences at a feasible load, near the
    margin and far past it (Phase 4 checked the unrolled derivative of an unconverged 25-iteration
    map, which had the wrong sign relative to the converged energy's gradient)."""
    topo, params, hm, seeds = baseline_with_seeds()
    asg = assignment(topo)
    bounds = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
    fixed = fixed_from_params(params)
    flat0, unravel = jax.flatten_util.ravel_pytree(encode(params, topo, bounds))
    st = GraspState(seeds.q[0], seeds.p[0], jnp.eye(3))
    act = AK.actuation_for(hm, TASK)
    rng = np.random.RandomState(0)
    idx = rng.choice(flat0.shape[0], 6, replace=False)
    checked = 0
    for scale in (1.0, 2.5, 6.0):
        w = scale * act.w_ext

        def E(x, wts=ForceWeights()):
            m = build_hand_model(topo, decode(unravel(x), topo, bounds, fixed))
            return solve_soft(AK.force_problem_at(m, st, asg, SP, act), w, wts).E

        e50, e100 = float(E(flat0)), float(E(flat0, ForceWeights(iters=100)))
        assert abs(e50 - e100) <= 1e-6 * max(e50, 1e-12) + 1e-14, (scale, e50, e100)
        g = np.asarray(jax.grad(E)(flat0))
        assert np.all(np.isfinite(g))
        if np.linalg.norm(g) < 1e-9:          # a feasible interior state: E ~ 0 and so is its gradient
            continue
        for i in idx:
            h = 1e-6
            e = jnp.zeros_like(flat0).at[i].set(h)
            fd = (float(E(flat0 + e)) - float(E(flat0 - e))) / (2 * h)
            assert abs(fd - g[i]) <= 1e-3 * max(abs(fd), abs(g[i])) + 1e-9, (scale, i, fd, g[i])
            checked += 1
    assert checked >= 6


def test_act_karma_evaluate_and_gating():
    topo, params, hm, seeds = baseline_with_seeds()
    cfg = MetricConfig(commands=CommandConfig(n_cells=1, n_sub=4, n_rot_steps=2), roll=RollConfig(mode="hard", actuation=True))
    act = AK.actuation_for(hm, TASK)
    d = AK.evaluate(hm, topo, seeds, SP, TASK, cfg, act)
    s = AK.summarize(d)
    assert np.isfinite(s["score"]) and 0 <= s["force_feasible_step_frac"] <= 1
    assert d.per_seed.force_T.shape == (3, 6, 4, 4)
    # more permissive actuation never lowers the smooth score; a very heavy object lowers it
    act_ideal = act._replace(tau_bar=1e3 * act.tau_bar, mu=jnp.asarray(50.0))
    d_ideal = AK.evaluate(hm, topo, seeds, SP, TASK, cfg, act_ideal)
    assert float(d_ideal.score) >= float(d.score) - 1e-9
    act_heavy = act._replace(w_ext=50.0 * act.w_ext)
    d_heavy = AK.evaluate(hm, topo, seeds, SP, TASK, cfg, act_heavy)
    assert float(d_heavy.score) <= float(d.score) + 1e-9
    # the kinematic-only evaluation (actuation off) equals the ideal limit up to the kernel scale
    from dkarma.metrics import diff_karma as DK
    d_kin = DK.evaluate(hm, topo, seeds, SP, TASK, cfg._replace(roll=RollConfig(mode="hard")) if hasattr(cfg, "_replace") else
                        MetricConfig(commands=cfg.commands, roll=RollConfig(mode="hard")))
    assert abs(float(d_kin.score) - float(d_ideal.score)) <= 0.05 * max(float(d_kin.score), 1e-6) + 1e-6
    # gradient finite
    bounds = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
    fixed = fixed_from_params(params); phi = encode(params, topo, bounds)
    from dkarma.metrics import diff_karma as DK2
    g = jax.grad(lambda ph: DK2.objective(ph, topo, bounds, fixed, seeds, SP, TASK, cfg, None, act))(phi)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))


def test_wrench_capacity_sanity():
    topo, params, hm, seeds = baseline_with_seeds()
    asg = assignment(topo)
    act = AK.actuation_for(hm, TASK)
    for s in range(int(seeds.valid.sum())):
        st = GraspState(seeds.q[s], seeds.p[s], jnp.eye(3))
        pb = AK.force_problem_at(hm, st, asg, SP, act)
        if not solve_exact(pb, act.w_ext)["feasible"]:
            continue
        a, D = AK.wrench_capacity_at(hm, st, asg, SP, act, n_dir=12)
        assert a.shape == (12,) and np.all(a >= 0) and np.any(a > 0)
        # the moment about the pinch axis is resisted only by torsional friction: its capacity is
        # bounded by 2 gamma f_n / moment_scale, and vanishes with gamma = 0 (pure point contacts)
        n0 = np.array(pb.normals[0]); n0 = n0 / np.linalg.norm(n0)
        Dm = np.zeros((2, 6)); Dm[0, 3:] = 0.343 * 0.015 * n0; Dm[1, 3:] = -0.343 * 0.015 * n0
        am = wrench_capacity(pb, act.w_ext, Dm)
        pb0 = pb._replace(gamma=jnp.asarray(0.0))
        am0 = wrench_capacity(pb0, act.w_ext, Dm)
        assert np.all(am0 <= 1e-3 + 1e-9) or not solve_exact(pb0, act.w_ext)["feasible"]
        assert np.all(am >= am0 - 1e-9)
        return
    pytest.skip("no feasible base grasp among the small seed set")


def test_force_credit_mapping_knee():
    """Phase 8.1 credit mapping: with ``force_steep = 0`` the step energy is linear in E_F
    (unchanged behaviour); with a knee the extra penalty is ~0 well below the knee, grows
    linearly with slope ``force_steep`` above it, and stays monotone and differentiable."""
    from dkarma.solvers.rollout import StepDiag, step_energy
    one = jnp.ones(()); zero = jnp.zeros(())

    def diag(E):
        return StepDiag(slip_m=zero, spin_err=zero, gap_pre=zero, gap_post=zero, limit_viol=zero, obj_clear=one, obj_clear_exact=one,
                        self_clear=one, self_clear_exact=one, grasp_margin=one, efficiency=one, qp_ok=jnp.asarray(True),
                        proj_ok=jnp.asarray(True), kkt_gap=zero, dq_norm=zero, force_E=jnp.asarray(E), force_eq_res=zero,
                        torque_usage=zero, cone_margin=one, fn_min=one, normal0=jnp.array([1.0, 0, 0]), axis_body=jnp.array([1.0, 0, 0]))

    e0 = float(step_energy(diag(0.0), SP, EnergyWeights(force_steep=0.0)))
    lin = EnergyWeights(force=20.0, force_steep=0.0)
    np.testing.assert_allclose(float(step_energy(diag(0.01), SP, lin)) - e0, 0.2, rtol=1e-9)
    knee = EnergyWeights(force=0.0, force_knee=1e-3, force_steep=2.0, force_knee_eps=0.125)
    f = lambda E: float(step_energy(diag(E), SP, knee)) - e0   # noqa: E731
    assert f(0.0) < 1e-3 and f(5e-4) < 5e-3               # softplus floor 2 * 0.125 * softplus(-8) ~ 1e-4 per step
    assert abs((f(5e-3) - f(3e-3)) / 2.0 - 2.0) < 1e-6      # slope = force_steep per knee-unit above the knee
    Es = np.linspace(0.0, 0.02, 41)
    vals = np.array([f(E) for E in Es])
    assert np.all(np.diff(vals) >= 0.0)
    g = jax.grad(lambda E: step_energy(diag(E), SP, knee))(jnp.asarray(1e-3))
    assert np.isfinite(float(g)) and float(g) > 0.0
    two = EnergyWeights(force=1.0, force_knee=1e-3, force_steep=2.0, force_knee_eps=0.125)
    np.testing.assert_allclose(float(step_energy(diag(0.01), SP, two)) - e0, f(0.01) + 0.01, rtol=1e-9)
    # the adopted defaults (Phase 8.1): knee 7e-7, slope 3 per knee-unit: E_F = 1e-5 costs > 40 per step
    d = EnergyWeights()
    assert d.force_knee == 7.0e-7 and d.force_steep == 3.0
    assert float(step_energy(diag(1e-5), SP, d)) - float(step_energy(diag(0.0), SP, d)) > 30.0
    assert float(step_energy(diag(1e-7), SP, d)) - float(step_energy(diag(0.0), SP, d)) < 0.05
