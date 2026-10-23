"""Phase 3 unit tests: the physical-unit seed pipeline, the smooth evaluator (shapes, bounds,
determinism, jit, gradient vs FD in soft mode), the exact library counterpart, and the
continuation schedule. Small configurations so the file runs on CPU in a few minutes."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from karma_jax.qp import QPSettings

from dkarma.core import build_hand_model, load_task_spec, Bounds, encode, decode, fixed_from_params
from dkarma.solvers.seed_projection import find_seeds, SeedConfig, reproject, assignment
from dkarma.solvers.rollout import RollConfig
from dkarma.metrics.config import DIFF_KARMA, MetricConfig
from dkarma.metrics.commands import CommandConfig
from dkarma.metrics import diff_karma as DK
from dkarma.metrics.exact_library import evaluate_exact
from dkarma.metrics.continuation import schedule
from .conftest import load_design

jax.config.update("jax_enable_x64", True)

TASK = load_task_spec()
SP = TASK.step_params(jnp.float64)
SMALL_SEEDS = SeedConfig(n_linspace=3, n_sobol=32, n_random=8, budget=3)
SMALL_CMDS = CommandConfig(n_cells=1, n_sub=4, n_rot_steps=2)
CFG = MetricConfig(commands=SMALL_CMDS, roll=RollConfig(mode="hard"))
BOUNDS = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
TIGHT = QPSettings(iters=80, tol=1e-13, infeas_tol=1e-6, reg=1e-13)

_cache = {}


def baseline_with_seeds():
    if "b" not in _cache:
        topo, params, _ = load_design("baseline")
        hm = build_hand_model(topo, params)
        seeds = find_seeds(hm, topo, SP, SMALL_SEEDS)
        _cache["b"] = (topo, params, hm, seeds)
    return _cache["b"]


def test_task_spec_physical_params():
    assert TASK.n_sub == 8
    assert float(SP.sphere_radius) == 0.015 and float(SP.link_radius) == 0.0118
    assert abs(float(SP.slip_threshold) - 0.0005) < 1e-12 and abs(float(SP.max_joint_step) - np.radians(10)) < 1e-12


def test_seed_pipeline_deterministic_and_feasible():
    topo, params, hm, seeds = baseline_with_seeds()
    assert int(seeds.valid.sum()) >= 1
    assert seeds.q.shape == (3, topo.J) and seeds.frame.shape == (3, 3, 3)
    dets = np.linalg.det(np.asarray(seeds.frame)[np.asarray(seeds.valid)])
    np.testing.assert_allclose(dets, 1.0, atol=1e-10)
    # feasibility of the selected seeds under the exact gates: gaps within eps_g, cone ok
    asg = assignment(topo)
    for s in range(int(seeds.valid.sum())):
        q, ok, gap = reproject(hm, seeds.q[s], seeds.p[s], asg, SP)
        assert bool(ok) and float(gap) <= float(SP.eps_g) + 1e-12
    # deterministic
    seeds2 = find_seeds(hm, topo, SP, SMALL_SEEDS)
    np.testing.assert_array_equal(np.asarray(seeds2.q), np.asarray(seeds.q))
    np.testing.assert_array_equal(np.asarray(seeds2.p), np.asarray(seeds.p))


def test_fixed_budget_pads_with_invalid_lanes():
    topo, params, hm, _ = baseline_with_seeds()
    seeds = find_seeds(hm, topo, SP, SeedConfig(n_linspace=2, n_sobol=4, n_random=0, budget=6))
    assert seeds.q.shape[0] == 6 and int(seeds.valid.sum()) <= 6
    assert not bool(seeds.valid[-1]) or int(seeds.valid.sum()) == 6


def test_evaluate_shapes_bounds_and_jit():
    topo, params, hm, seeds = baseline_with_seeds()
    d = DK.evaluate(hm, topo, seeds, SP, TASK, CFG)
    assert d.K_T.shape == (3,) and d.per_seed.gates_T.shape == (3, 6, SMALL_CMDS.H_T, 7)
    v = np.asarray(d.valid)
    assert np.all((np.asarray(d.K_T)[v] >= 0) & (np.asarray(d.K_T)[v] <= 1))
    assert np.all((np.asarray(d.K_R)[v] >= 0) & (np.asarray(d.K_R)[v] <= 1))
    assert np.all(np.asarray(d.per_seed.K_R_regions)[v] <= np.asarray(d.K_R)[v][:, None] + 1e-9)
    assert float(d.score) > 0
    dj = jax.jit(lambda h: DK.evaluate(h, topo, seeds, SP, TASK, CFG))(hm)
    np.testing.assert_allclose(float(dj.score), float(d.score), rtol=1e-10)
    s = DK.summarize(d)
    assert s["n_valid"] == int(v.sum()) and 0 <= s["gate_pass_rate_T"]["gap"] <= 1
    # no valid seeds -> zero score, finite
    d0 = DK.evaluate(hm, topo, seeds._replace(valid=jnp.zeros(3, bool)), SP, TASK, CFG)
    assert float(d0.score) == 0.0


def test_objective_gradient_soft_mode_vs_fd():
    topo, params, hm, seeds = baseline_with_seeds()
    cfg = MetricConfig(commands=CommandConfig(n_cells=1, n_sub=2, n_rot_steps=1), roll=RollConfig(mode="soft", proj_backtrack=False))
    fixed = fixed_from_params(params)
    flat0, unravel = jax.flatten_util.ravel_pytree(encode(params, topo, BOUNDS))
    f = jax.jit(lambda x: DK.objective(unravel(x), topo, BOUNDS, fixed, seeds, SP, TASK, cfg, TIGHT))
    g = np.asarray(jax.jit(jax.grad(lambda x: DK.objective(unravel(x), topo, BOUNDS, fixed, seeds, SP, TASK, cfg, TIGHT)))(flat0))
    assert np.all(np.isfinite(g)) and np.linalg.norm(g) > 0
    rng = np.random.RandomState(0)
    idx = rng.choice(flat0.shape[0], size=8, replace=False)
    h = 1e-6
    for i in idx:
        e = jnp.zeros_like(flat0).at[i].set(h)
        fd = (float(f(flat0 + e)) - float(f(flat0 - e))) / (2 * h)
        assert abs(fd - g[i]) <= 1e-3 * max(abs(fd), abs(g[i])) + 1e-6, (i, fd, g[i])


def test_hard_mode_gradient_finite_and_ascent():
    topo, params, hm, seeds = baseline_with_seeds()
    fixed = fixed_from_params(params)
    phi = encode(params, topo, BOUNDS)
    f = jax.jit(jax.value_and_grad(lambda ph: DK.objective(ph, topo, BOUNDS, fixed, seeds, SP, TASK, CFG)))
    v, g = f(phi)
    gn = jnp.sqrt(sum(jnp.sum(x ** 2) for x in jax.tree_util.tree_leaves(g)))
    assert bool(jnp.isfinite(v)) and bool(jnp.isfinite(gn)) and float(gn) > 0
    improved = False
    for s in (1e-3, 3e-3, 1e-2):
        v2, _ = f(jax.tree_util.tree_map(lambda a, gg: a + s * gg / gn, phi, g))
        improved |= float(v2) > float(v)
    assert improved, "no ascent direction found among three step sizes"


def test_exact_library_counts():
    topo, params, hm, seeds = baseline_with_seeds()
    ex = evaluate_exact(hm, topo, seeds, SP, TASK, CFG)
    assert 0 <= ex.K_T <= 1 and 0 <= ex.K_R <= 1
    assert len(ex.per_seed) == int(seeds.valid.sum())
    for r in ex.per_seed:
        assert r.ok_T.shape == (6, 1) and r.ok_R.shape == (4, 2)
        assert len(r.cells) >= 1 and len(r.bins) >= 1


def test_continuation_schedule():
    st = schedule(DIFF_KARMA, n_stages=3)
    assert len(st) == 3 and st[0].roll.mode == "soft" and st[-1].roll.mode == "hard"
    assert st[0].smooth.sigma_p_m > st[1].smooth.sigma_p_m > st[2].smooth.sigma_p_m == DIFF_KARMA.smooth.sigma_p_m
    assert st[0].energy.sigma > st[-1].energy.sigma == DIFF_KARMA.energy.sigma


def test_exact_library_cached_validators_identical():
    """Phase 9.4: the two-finger exact library's cached jitted validators give results identical to the Phase 8
    per-seed eager path (cells, pinch-axis bins, verdicts and gate codes of every seed)."""
    from dkarma.metrics import exact_library as EL
    topo, params, hm, seeds = baseline_with_seeds()
    old = EL.CACHED_VALIDATORS
    try:
        EL.CACHED_VALIDATORS = False
        a = evaluate_exact(hm, topo, seeds, SP, TASK, CFG)
        EL.CACHED_VALIDATORS = True
        b = evaluate_exact(hm, topo, seeds, SP, TASK, CFG)
    finally:
        EL.CACHED_VALIDATORS = old
    assert a.K_T == b.K_T and a.K_R == b.K_R and len(a.per_seed) == len(b.per_seed)
    for x, y in zip(a.per_seed, b.per_seed):
        assert x.cells == y.cells and x.bins == y.bins
        assert np.array_equal(x.ok_T, y.ok_T) and np.array_equal(x.ok_R, y.ok_R)
        assert np.array_equal(x.codes_T, y.codes_T) and np.array_equal(x.codes_R, y.codes_R)
