"""Phase 6 tests: the objective wrapper, a short Adam run that raises the smooth objective, a few
CMA-ES generations, the hybrid driver, exact validation selection, and the artifact round trip.
Tiny configurations so the file runs on CPU in a few minutes."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import Topology, Bounds, anthropomorphic, encode, load_task_spec
from dkarma.solvers.seed_projection import SeedConfig
from dkarma.solvers.rollout import RollConfig
from dkarma.metrics.config import MetricConfig
from dkarma.metrics.commands import CommandConfig
from dkarma.optimization import make_objective, adam, cmaes, hybrid, save_run, load_run
from dkarma.optimization.objectives import exact_validation, Constraints, reach_penalty
from .conftest import load_design

jax.config.update("jax_enable_x64", True)

TASK = load_task_spec()
BOUNDS = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
CFG = MetricConfig(commands=CommandConfig(n_cells=1, n_sub=2, n_rot_steps=1), roll=RollConfig(mode="soft", proj_backtrack=False))
SEEDS = SeedConfig(n_linspace=3, n_sobol=32, n_random=8, budget=2)

_cache = {}


def objective():
    if "o" not in _cache:
        topo, params, _ = load_design("baseline")
        _cache["o"] = (topo, params, make_objective(topo, params, BOUNDS, TASK, CFG, SEEDS))
    return _cache["o"]


def test_objective_value_grad_and_reach_penalty():
    topo, params, obj = objective()
    phi = encode(params, topo, BOUNDS)
    f = obj.value_and_grad()
    v, g = f(phi)
    assert np.isfinite(float(v)) and all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree_util.tree_leaves(g))
    assert float(reach_penalty(params, topo, Constraints(reach_max_m=0.25))) < 1e-6
    assert float(reach_penalty(params, topo, Constraints(reach_max_m=0.05))) > 0.0


def test_adam_increases_smooth_objective_and_selects_by_validation():
    topo, params, obj = objective()
    phi = encode(params, topo, BOUNDS)
    calls = []

    def validate(p):
        calls.append(1)
        return {"exact_score": float(len(calls))}         # monotone: the last validated design wins

    res = adam(obj, phi, n_iters=6, lr=0.02, stages=2, validate_every=3, validation_fn=validate, log=lambda s: None)
    sm = [h["smooth"] for h in res.training_history]
    assert len(sm) == 6 and np.all(np.isfinite(sm))
    assert max(sm[3:]) >= sm[0] - 1e-6 or max(sm) > sm[0]        # ascent within a stage (the stage switch rescales)
    # validations at the start, at both stage ends (it 3, it 6); the periodic one at it 3 coincides with a stage end
    assert [v["iter"] for v in res.validation_history] == [0, 3, 6] and res.best_validation["exact_score"] == 3.0
    st = res.meta["stages_log"]
    assert len(st) == 2 and st[0]["exact_start"] == 1.0 and st[0]["exact_end_raw"] == 2.0 and st[1]["exact_end_raw"] == 3.0
    assert "gate_fail_share" in st[0]["diag_start"] and res.meta["final_iterate_exact_raw"] == 3.0 and res.meta["best_intermediate_exact"] == 2.0


def test_adam_hygiene_stage_select_and_seed_policies():
    """Phase 8.2/8.3 options: non-finite gradients make no update when skipped, clipping bounds the
    step, stage selection warm-starts from the best validated design of the stage, and the seed
    policies produce seed sets of the fixed budget."""
    topo, params, obj = objective()
    phi = encode(params, topo, BOUNDS)
    import jax.flatten_util
    scores = iter([0.5, 0.9, 0.2, 0.4, 0.3])          # start, it2 (periodic), stage-0 end (it3) ... -> warm start from it 2

    def validate(p):
        return {"exact_score": next(scores, 0.1)}

    ilog = []
    res = adam(obj, phi, n_iters=6, lr=0.02, stages=2, validate_every=2, validation_fn=validate, log=lambda s: None,
               grad_clip=1e-3, skip_nonfinite=True, reset_moments=True, stage_select=True, iterate_log=ilog)
    h = res.training_history
    its = [r for r in ilog if "iter" in r]                                            # Phase 9.1 trace log: one record per iteration
    assert [r["iter"] for r in its] == [hh["iter"] for hh in h] and all(r["x"].shape == jax.flatten_util.ravel_pytree(phi)[0].shape for r in its)
    assert [r for r in ilog if "seeds" in r][0]["seed_id"] == 0 and all(r["finite"] == hh["grad_finite"] for r, hh in zip(its, h))
    assert all(hh["clipped"] for hh in h if hh["grad_finite"])                      # a tiny clip bound clips every finite gradient
    assert all(hh["step_norm"] <= 0.02 * np.sqrt(len(jax.flatten_util.ravel_pytree(phi)[0])) + 1e-9 for hh in h)
    assert res.meta["stages_log"][0]["selected_iter"] == 2 and res.meta["stages_log"][0]["exact_end_selected"] == 0.9
    assert res.meta["options"]["stage_select"] and res.meta["options"]["grad_clip"] == 1e-3
    # seed policies
    S = obj.seeds.q.shape[0]
    fresh = obj.regenerate_seeds(params)
    assert fresh.q.shape == obj.seeds.q.shape and int(fresh.valid.sum()) >= 1
    rep = obj.reproject_seeds(params, obj.seeds)
    assert rep.q.shape == obj.seeds.q.shape and bool(jnp.all(rep.valid == obj.seeds.valid))
    uni, info = obj.union_seeds(params, obj.seeds)
    assert uni.q.shape == (S, topo.J) and int(uni.valid.sum()) <= S and info["kept_from_reprojected"] + info["kept_from_fresh"] == int(uni.valid.sum())
    sc = obj.exact_seed_scores(params, obj.seeds)
    assert sc.shape == (S,) and np.all(sc >= 0)
    for pol in ("regenerate_stage", "union_stage"):
        s2, inf2 = obj.seeds_for_policy(params, pol, obj.seeds, at_stage_boundary=True, it=3)
        assert s2.q.shape == obj.seeds.q.shape and inf2["policy"] == pol
    s3, inf3 = obj.seeds_for_policy(params, "regenerate_every_k", obj.seeds, at_stage_boundary=False, it=5)
    assert inf3 is None
    # the jitted objective accepts a different seed set without a shape change
    f = obj.value_and_grad()
    v1, _ = f(phi); v2, _ = f(phi, fresh)
    assert np.isfinite(float(v1)) and np.isfinite(float(v2))


def test_cmaes_and_hybrid_run():
    topo, params, obj = objective()
    phi = encode(params, topo, BOUNDS)
    r = cmaes(obj, phi, popsize=4, n_gens=2, validate_every_gens=1, validation_fn=lambda p: {"exact_score": 0.0}, log=lambda s: None)
    assert len(r.training_history) == 2 and np.isfinite(r.training_history[-1]["best"])
    h = hybrid(obj, phi, n_starts=2, sobol_scale=0.1, adam_iters=2, validation_fn=lambda p: {"exact_score": float(np.random.rand())},
               log=lambda s: None, stages=1)
    assert h.meta["method"] == "hybrid" and len(h.meta["start_criteria"]) == 2


def test_exact_validation_and_artifacts(tmp_path):
    topo, params, obj = objective()
    validate = exact_validation(obj, tmp_path / "val", official=False)
    v = validate(params)
    assert {"n_seeds", "exact_K_T", "exact_K_R", "exact_score"} <= set(v) and 0 <= v["exact_K_T"] <= 1
    phi = encode(params, topo, BOUNDS)
    res = adam(obj, phi, n_iters=2, stages=1, validate_every=2, validation_fn=validate, log=lambda s: None)
    out = save_run(tmp_path / "run", topo, params, res, {"seed_config": SEEDS.__dict__}, actuator_spec={"torque_limit_nm": 0.108}, seeds=obj.seeds)
    for f in ("config.yaml", "actuator_spec.yaml", "parameter_history.npz", "objective_history.csv", "validation_history.csv",
              "random_seeds.json", "results_summary.json", "constraint_diagnostics.json", "seeds.npz",
              "initial_morphology/initial_morphology.urdf", "optimized_morphology/robot_optimized_morphology.yaml"):
        assert (out / f).exists(), f
    topo2, params2, summary = load_run(out)
    assert topo2 == topo and "best_validation" in summary
    np.testing.assert_allclose(np.asarray(params2.link_len)[topo.joint_mask], np.asarray(res.best_params.link_len)[topo.joint_mask], atol=1e-12)
