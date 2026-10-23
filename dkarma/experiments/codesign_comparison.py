"""Experiments III and IV (SPEC.md §12), simulation legs.

III: optimize 3x4 hands from the same start, bounds and budget with (1) thumb-index
Diff-KaRMA, (2) mean pairwise KaRMA, (3) simultaneous three-contact MC-KaRMA, (4) Full
(actuation-aware three-contact) KaRMA; evaluate every result on the common three-contact
suite: exact MC counts (cells, SO(3) bins), exact Full counts (actuated), wrench capacity
at the seeds, and two-finger exact coverage. Also report the pairwise-vs-simultaneous gap.

IV (design A vs B): A = CMA-ES against the original KaRMA (the fast proxy of ``karma_jax``
as in ``optimize/``, final scoring by the original metric) on the same 3x4 topology; B = the
Full-KaRMA design of III. Both scored on the physical three-contact suite and by the
original metric on the physical task. Hardware realization is out of scope.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.codesign_comparison --out workspace/phase7/exp34 --iters 40
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import Topology, Bounds, anthropomorphic, encode, decode, load_task_spec, build_hand_model, fixed_from_params  # noqa: E402
from dkarma.core.actuation import load_actuation_spec  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig, find_seeds, find_multi_seeds, assignment  # noqa: E402
from dkarma.solvers.rollout import GraspState  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, PAIRWISE_KARMA, MC_KARMA, FULL_KARMA  # noqa: E402
from dkarma.metrics.commands import CommandConfig  # noqa: E402
from dkarma.metrics import multicontact as MC, act_karma as AK  # noqa: E402
from dkarma.metrics.exact_library import evaluate_exact, official_karma_physical  # noqa: E402
from dkarma.optimization import make_objective, adam, save_run  # noqa: E402
from dkarma.optimization.objectives import exact_validation  # noqa: E402
from dkarma.export import write_design  # noqa: E402

BOUNDS = Bounds(link_len_min_m=0.012, link_len_max_m=0.090, tip_len_min_m=0.012, tip_len_max_m=0.090, base_xyz_max_m=0.10)


def suite(topo, params, task, sp, budget, workdir, official=True):
    """Common three-contact task suite plus two-finger exact coverage for one design."""
    hm = build_hand_model(topo, params)
    act = AK.actuation_for(hm, task)
    fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
    out = {}
    s3 = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=budget))
    out["n_seeds_3"] = int(s3.valid.sum())
    cfg_mc = replace(MC_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10))
    cfg_full = replace(FULL_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10))
    if out["n_seeds_3"]:
        ex_mc = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, s3, sp, task, act, cfg_mc)
        ex_full = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, s3, sp, task, act, cfg_full)
        out.update(mc_exact_K_T=ex_mc.K_T, mc_exact_K_SO3=ex_mc.K_SO3, full_exact_K_T=ex_full.K_T, full_exact_K_SO3=ex_full.K_SO3,
                   mc_cells=[len(r["cells"]) for r in ex_mc.per_seed], mc_bins=[len(r["bins"]) for r in ex_mc.per_seed])
        d_mc = jax.jit(lambda h: MC.evaluate_fixed_contacts(h, topo, fingers, s3, sp, task, act, cfg_mc))(hm)
        d_full = jax.jit(lambda h: MC.evaluate_fixed_contacts(h, topo, fingers, s3, sp, task, act, cfg_full))(hm)
        out.update(mc_smooth=float(d_mc.score), full_smooth=float(d_full.score))
        caps = []
        asg3 = assignment(topo, 3, fingers)
        for s in range(out["n_seeds_3"]):
            a, _ = AK.wrench_capacity_at(hm, GraspState(s3.q[s], s3.p[s], jnp.eye(3)), asg3, sp, act, n_dir=14)
            caps.append(dict(mean=float(a.mean()), min=float(a.min()), frac_ge_1=float(np.mean(a >= 1.0))))
        out["wrench_capacity_3"] = caps
    s2 = find_seeds(hm, topo, sp, SeedConfig(budget=budget))
    out["n_seeds_2"] = int(s2.valid.sum())
    if out["n_seeds_2"]:
        ex2 = evaluate_exact(hm, topo, s2, sp, task, replace(DIFF_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10)))
        out.update(two_exact_K_T=ex2.K_T, two_exact_K_R=ex2.K_R)
        asg2 = assignment(topo)
        caps = []
        for s in range(out["n_seeds_2"]):
            a, _ = AK.wrench_capacity_at(hm, GraspState(s2.q[s], s2.p[s], jnp.eye(3)), asg2, sp, act, n_dir=14)
            caps.append(dict(mean=float(a.mean()), min=float(a.min())))
        out["wrench_capacity_2"] = caps
    if official:
        out["official_physical"] = official_karma_physical(topo, params, task, workdir, name="suite")
    return out


def cmaes_original_karma(topo, params0, workdir, popsize=8, n_gens=12, sigma0=0.3, seed=0, log=print):
    """Arm A of Experiment IV: CMA-ES against the original KaRMA via the fast proxy (as the
    ``optimize/`` study), on the unconstrained coordinates of the 3x4 family."""
    import cma
    import jax.flatten_util
    from karma.config import load_config
    from karma.robot import load_robot
    from karma_jax.model import build_flat_model
    from karma_jax.fast import evaluate as fast_eval, FastConfig
    from dkarma.fixtures import METRIC_CFG
    fixed = fixed_from_params(params0)
    x0, unravel = jax.flatten_util.ravel_pytree(encode(params0, topo, BOUNDS))
    fc = FastConfig(n_linspace_per_joint=5, n_sobol=256, n_random=64, n_offsets=5, dedup_rad=0.1, max_seeds=48,
                    stage1_budget=15, n_finalists=3, final_budget=10000)

    def fitness(x, k):
        params = decode(unravel(jnp.asarray(x)), topo, BOUNDS, fixed)
        wd = Path(workdir) / f"cma_{k:05d}"
        _, yaml_path, _ = write_design(topo, params, wd, "cand")
        cfg = load_config(METRIC_CFG, yaml_path)
        try:
            ctx = load_robot(cfg.urdf_path, cfg)
            r = fast_eval(cfg, build_flat_model(ctx, cfg), fc)
            return float(r.karma_t) if r.error is None else 0.0
        except Exception as e:  # noqa: BLE001
            log(f"    candidate {k} failed: {e}")
            return 0.0

    es = cma.CMAEvolutionStrategy(np.asarray(x0), sigma0, dict(popsize=popsize, seed=seed, verbose=-9))
    k = 0
    hist = []
    for g in range(n_gens):
        X = es.ask()
        vals = []
        for x in X:
            vals.append(fitness(x, k)); k += 1
        es.tell(X, [-v for v in vals])
        hist.append(dict(gen=g, best=float(max(vals)), mean=float(np.mean(vals))))
        log(f"  CMA gen {g}: best proxy KaRMA-T {max(vals):.4f} mean {np.mean(vals):.4f}")
    return decode(unravel(jnp.asarray(es.result.xbest)), topo, BOUNDS, fixed), hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase7/exp34")
    ap.add_argument("--iters", type=int, default=40)
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--validate-every", type=int, default=20)
    ap.add_argument("--cma-gens", type=int, default=12)
    ap.add_argument("--cma-pop", type=int, default=8)
    ap.add_argument("--skip-official", action="store_true")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    task = load_task_spec(); sp = task.step_params()
    topo = Topology.three_finger(4)
    params0 = anthropomorphic(topo)
    results = {"start": None, "arms": {}, "armA": None}
    log_path = out / "log.txt"

    def log(s):
        print(s, flush=True)
        with open(log_path, "a") as fh:
            fh.write(s + "\n")

    t0 = time.perf_counter()
    results["start"] = suite(topo, params0, task, sp, args.budget, out / "suite_start", official=not args.skip_official)
    log(f"start suite ({time.perf_counter() - t0:.0f}s): {results['start']}")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))

    arms = {"diff": DIFF_KARMA, "pairwise": PAIRWISE_KARMA, "mc": MC_KARMA, "full": FULL_KARMA}
    cmds = CommandConfig(n_cells=4, n_rot_steps=10)
    for name, cfg in arms.items():
        cfg = replace(cfg, commands=cmds)
        t0 = time.perf_counter()
        obj = make_objective(topo, params0, BOUNDS, task, cfg, SeedConfig(budget=args.budget))
        log(f"[{name}] seeds {int(obj.seeds.valid.sum())} ({obj.seeds.stats})")
        validate = exact_validation(obj, out / f"val_{name}", official=False)
        res = adam(obj, encode(params0, topo, BOUNDS), n_iters=args.iters, validate_every=args.validate_every, validation_fn=validate,
                   log=log, seed=0, stages=2)
        save_run(out / f"run_{name}", topo, params0, res, dict(metric=name, iters=args.iters), actuator_spec=load_actuation_spec().__dict__, seeds=obj.seeds)
        t_opt = time.perf_counter() - t0
        t0 = time.perf_counter()
        st = suite(topo, res.best_params, task, sp, args.budget, out / f"suite_{name}", official=not args.skip_official)
        results["arms"][name] = dict(optimize_s=t_opt, suite_s=time.perf_counter() - t0, best_validation=res.best_validation,
                                     smooth_history=[h["smooth"] for h in res.training_history], suite=st)
        log(f"[{name}] done opt {t_opt:.0f}s suite {time.perf_counter() - t0:.0f}s: {st}")
        (out / "results.json").write_text(json.dumps(results, indent=1, default=float))

    # Experiment IV arm A
    t0 = time.perf_counter()
    paramsA, hist = cmaes_original_karma(topo, params0, out / "armA", popsize=args.cma_pop, n_gens=args.cma_gens, log=log)
    write_design(topo, paramsA, out / "armA_design", "armA")
    stA = suite(topo, paramsA, task, sp, args.budget, out / "suite_armA", official=not args.skip_official)
    results["armA"] = dict(history=hist, suite=stA, cma_s=time.perf_counter() - t0)
    log(f"[armA] done ({time.perf_counter() - t0:.0f}s): {stA}")
    (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
