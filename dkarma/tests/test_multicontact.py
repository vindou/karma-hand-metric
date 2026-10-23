"""Phase 5 tests (SPEC §6, §10): two-contact reduction of the masked-lane kernels, the extra-contact
monotonicity of the rolling problem, three-contact seeds on the 3x4 anthropomorphic hand,
three-contact consistency, the MC/Full evaluators (shapes, finiteness, jit, gradient), the exact
fixed-assignment counterpart, SO(3) coverage, and the pairwise baseline."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from karma_jax.kinematics import forward_kinematics

from dkarma.core import Topology, build_hand_model, load_task_spec, anthropomorphic, Bounds, encode, decode, fixed_from_params
from dkarma.core.contacts import ContactAssignment, contact_pass
from dkarma.solvers.rolling_qp import assemble, solve_soft, solve_hard
from dkarma.solvers.projection import project
from dkarma.solvers.rollout import GraspState, RollConfig
from dkarma.solvers.seed_projection import find_multi_seeds, find_seeds, SeedConfig, assignment, solve_multi_ik
from dkarma.metrics.config import MC_KARMA, FULL_KARMA, DIFF_KARMA, MetricConfig
from dkarma.metrics.commands import CommandConfig
from dkarma.metrics import multicontact as MC
from dkarma.metrics import act_karma as AK
from .conftest import load_design
from karma_jax.qp import default_settings

jax.config.update("jax_enable_x64", True)

TASK = load_task_spec()
SP = TASK.step_params(jnp.float64)
SMALL_CMDS = CommandConfig(n_cells=1, n_sub=4, n_rot_steps=2)

_cache = {}


def tri_with_seeds():
    """3x4 anthropomorphic hand with three-contact seeds (thumb, index, middle)."""
    if "tri" not in _cache:
        topo = Topology.three_finger(4)
        params = anthropomorphic(topo)
        hm = build_hand_model(topo, params)
        act = AK.actuation_for(hm, TASK)
        fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
        seeds = find_multi_seeds(hm, topo, fingers, SP, act, SeedConfig(n_linspace=3, n_sobol=64, n_random=16, budget=2))
        _cache["tri"] = (topo, params, hm, act, fingers, seeds)
    return _cache["tri"]


def test_two_contact_reduction_of_masked_lanes():
    """With the third lane masked, the 3-lane assembly/solve/projection equal the 2-lane ones."""
    topo, params, _ = load_design("baseline")
    hm = build_hand_model(topo, params)
    seeds = find_seeds(hm, topo, SP, SeedConfig(n_linspace=3, n_sobol=16, n_random=4, budget=1))
    assert int(seeds.valid.sum()) >= 1
    q, p = seeds.q[0], seeds.p[0]
    asg2 = assignment(topo, 2)
    asg3 = ContactAssignment(caps=jnp.concatenate([asg2.caps, jnp.array([0], jnp.int32)]), valid=jnp.array([True, True, False]))
    fkr = forward_kinematics(hm.dm, q)
    dp = jnp.array([0.5, 0.2, -0.3]) * 1e-3
    s2 = solve_hard(assemble(hm, contact_pass(hm, fkr, p, SP.sphere_radius, asg2), asg2, dp, jnp.zeros(3), q, SP), SP, default_settings(jnp.float64))
    s3 = solve_hard(assemble(hm, contact_pass(hm, fkr, p, SP.sphere_radius, asg3), asg3, dp, jnp.zeros(3), q, SP), SP, default_settings(jnp.float64))
    np.testing.assert_allclose(np.asarray(s3.dq), np.asarray(s2.dq), atol=1e-10)
    np.testing.assert_allclose(float(s3.slip_m), float(s2.slip_m), atol=1e-12)
    q_off = q + 0.01
    p2, p3 = project(hm, q_off, p, asg2, SP), project(hm, q_off, p, asg3, SP)
    np.testing.assert_allclose(np.asarray(p3.q), np.asarray(p2.q), atol=1e-12)
    assert bool(p2.ok) == bool(p3.ok)


def test_extra_contact_never_reduces_rolling_residual():
    """A mandatory third contact adds residual rows: the optimal (soft) rolling residual cannot
    decrease and the max slip cannot decrease (SPEC §6.5 expectation)."""
    topo, params, hm, act, fingers, seeds = tri_with_seeds()
    if int(seeds.valid.sum()) == 0:
        pytest.skip("no three-contact seed found with the small budget")
    q, p = seeds.q[0], seeds.p[0]
    asg3 = assignment(topo, 3, fingers)
    asg2 = asg3._replace(valid=jnp.array([True, True, False]))
    fkr = forward_kinematics(hm.dm, q)
    worse = 0
    for dp in (jnp.array([1.0, 0, 0]), jnp.array([0, 1.0, 0]), jnp.array([0, 0, 1.0]), jnp.array([-1.0, 0.5, 0.2])):
        dp = dp * 1e-3
        pr2 = assemble(hm, contact_pass(hm, fkr, p, SP.sphere_radius, asg2), asg2, dp, jnp.zeros(3), q, SP)
        pr3 = assemble(hm, contact_pass(hm, fkr, p, SP.sphere_radius, asg3), asg3, dp, jnp.zeros(3), q, SP)
        s2, s3 = solve_soft(pr2, SP), solve_soft(pr3, SP)
        obj2 = float(jnp.sum(s2.residual ** 2)); obj3 = float(jnp.sum(s3.residual ** 2))
        assert obj3 >= obj2 - 1e-12
        worse += obj3 > obj2 + 1e-12
    assert worse >= 1


def test_three_contact_seeds_consistent():
    topo, params, hm, act, fingers, seeds = tri_with_seeds()
    assert seeds.q.shape == (2, topo.J)
    if int(seeds.valid.sum()) == 0:
        pytest.skip("no three-contact seed found with the small budget")
    asg = assignment(topo, 3, fingers)
    for s in range(int(seeds.valid.sum())):
        cs = contact_pass(hm, forward_kinematics(hm.dm, seeds.q[s]), seeds.p[s], SP.sphere_radius, asg)
        assert bool(jnp.all(jnp.abs(cs.gap) <= SP.eps_g + 1e-12)), np.asarray(cs.gap)
        assert bool(jnp.all(seeds.q[s] >= hm.dm.q_lower - 1e-9)) and bool(jnp.all(seeds.q[s] <= hm.dm.q_upper + 1e-9))
        pb = AK.force_problem_at(hm, GraspState(seeds.q[s], seeds.p[s], jnp.eye(3)), asg, SP, act)
        from dkarma.solvers.contact_forces import solve_exact
        assert solve_exact(pb._replace(tau_bar=1e3 * pb.tau_bar), act.w_ext)["feasible"]


def test_mc_and_full_evaluate_and_exact():
    topo, params, hm, act, fingers, seeds = tri_with_seeds()
    if int(seeds.valid.sum()) == 0:
        pytest.skip("no three-contact seed found with the small budget")
    cfg_mc = MetricConfig(name="mc", contacts=3, orientation_space="SO3", commands=SMALL_CMDS, roll=RollConfig(mode="hard"))
    d = MC.evaluate_fixed_contacts(hm, topo, fingers, seeds, SP, TASK, act, cfg_mc)
    assert np.isfinite(float(d.score)) and d.per_seed.gates_T.shape == (2, 6, SMALL_CMDS.H_T, 8)
    v = np.asarray(d.valid)
    assert np.all((np.asarray(d.K_SO3)[v] >= 0) & (np.asarray(d.K_SO3)[v] <= 1))
    dj = jax.jit(lambda h: MC.evaluate_fixed_contacts(h, topo, fingers, seeds, SP, TASK, act, cfg_mc))(hm)
    np.testing.assert_allclose(float(dj.score), float(d.score), rtol=1e-10)
    cfg_full = MetricConfig(name="full", contacts=3, orientation_space="SO3", commands=SMALL_CMDS,
                            roll=RollConfig(mode="hard", actuation=True))
    df = MC.evaluate_fixed_contacts(hm, topo, fingers, seeds, SP, TASK, act, cfg_full)
    assert np.isfinite(float(df.score)) and float(df.score) <= float(d.score) + 1e-9    # torque gates only remove credit
    ex = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, seeds, SP, TASK, act, cfg_mc)
    assert 0 <= ex.K_T <= 1 and 0 <= ex.K_SO3 <= 1 and len(ex.per_seed) == int(v.sum())
    exf = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, seeds, SP, TASK, act, cfg_full)   # actuated gates
    assert exf.K_T <= ex.K_T + 1e-12 and exf.K_SO3 <= ex.K_SO3 + 1e-12
    # gradient of the Full objective through the morphology is finite
    bounds = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
    fixed = fixed_from_params(params); phi = encode(params, topo, bounds)

    def obj(ph):
        m = build_hand_model(topo, decode(ph, topo, bounds, fixed))
        return MC.evaluate_fixed_contacts(m, topo, fingers, seeds, SP, TASK, act, cfg_full).score
    g = jax.grad(obj)(phi)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))


def test_pairwise_baseline_on_three_fingers():
    topo = Topology.three_finger(4)
    hm = build_hand_model(topo, anthropomorphic(topo))
    cfg = MetricConfig(commands=SMALL_CMDS, roll=RollConfig(mode="hard"))
    res = MC.evaluate_pairs(hm, topo, [("thumb", "index"), ("thumb", "middle"), ("index", "middle")], SP, TASK, cfg,
                            SeedConfig(n_linspace=2, n_sobol=16, n_random=4, budget=2))
    assert res.K_min <= res.K_pair <= res.K_max and len(res.per_pair) == 3
    assert res.per_pair[("thumb", "index")]["n_seeds"] >= 1


@pytest.mark.parametrize("actuated", [False, True])
def test_exact_evaluator_cached_rollouts_identical(actuated):
    """Phase 9.4: the exact three-contact evaluator's cached jitted rollouts give results identical to the Phase 8
    per-seed eager path (cells, SO(3) bins and per-primitive verdicts of every seed)."""
    topo, params, hm, act, fingers, seeds = tri_with_seeds()
    if int(seeds.valid.sum()) == 0:
        pytest.skip("no three-contact seed found with the small budget")
    cfg = MetricConfig(name="full" if actuated else "mc", contacts=3, orientation_space="SO3", commands=SMALL_CMDS,
                       roll=RollConfig(mode="hard", actuation=actuated))
    old = MC.CACHED_ROLLOUTS
    try:
        MC.CACHED_ROLLOUTS = False
        a = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, seeds, SP, TASK, act, cfg)
        MC.CACHED_ROLLOUTS = True
        b = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, seeds, SP, TASK, act, cfg)
    finally:
        MC.CACHED_ROLLOUTS = old
    assert a.K_T == b.K_T and a.K_SO3 == b.K_SO3 and len(a.per_seed) == len(b.per_seed)
    for x, y in zip(a.per_seed, b.per_seed):
        assert x["cells"] == y["cells"] and x["bins"] == y["bins"]
        assert np.array_equal(x["ok_T"], y["ok_T"]) and np.array_equal(x["ok_R"], y["ok_R"])
