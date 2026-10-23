"""Co-design driver (SPEC.md §8, §13): optimize a hand morphology against a metric variant with
Adam, CMA-ES or the hybrid, validate exactly, and write the §13 artifact layout.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.optimize_morphology \\
        --metric diff --topology 2x4 --optimizer adam --iters 60 --out workspace/phase6/diff_adam

Metrics: diff | act | mc | full. Topologies: 2x4 (the Phase 0 family, thumb + index) | 3x4
(thumb + index + middle, D2). Starts: anthropomorphic (default), a design JSON, or the
Phase 0 fixtures (baseline, T, A1 for 2x4). Everything is logged so the run regenerates.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import Topology, Bounds, anthropomorphic, encode, load_task_spec  # noqa: E402
from dkarma.core.actuation import load_actuation_spec  # noqa: E402
from dkarma.export import design_from_dict  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, ACT_KARMA, MC_KARMA, FULL_KARMA, PAIRWISE_KARMA  # noqa: E402
from dkarma.metrics.commands import CommandConfig  # noqa: E402
from dkarma.optimization import make_objective, adam, cmaes, hybrid, save_run  # noqa: E402
from dkarma.optimization.objectives import exact_validation, Constraints  # noqa: E402

METRICS = {"diff": DIFF_KARMA, "act": ACT_KARMA, "mc": MC_KARMA, "full": FULL_KARMA, "pairwise": PAIRWISE_KARMA}
BOUNDS = Bounds(link_len_min_m=0.012, link_len_max_m=0.090, tip_len_min_m=0.012, tip_len_max_m=0.090, base_xyz_max_m=0.10)


def start_design(args):
    if args.topology == "2x4":
        topo = Topology.two_finger(4, 4)
    elif args.topology == "3x4":
        topo = Topology.three_finger(4)
    else:
        raise SystemExit(f"unknown topology {args.topology}")
    if args.start == "anthropomorphic":
        return topo, anthropomorphic(topo)
    if args.start in ("baseline", "T", "A1"):
        from dkarma.tests.conftest import load_design
        t, p, _ = load_design(args.start)
        if t.n_joints != topo.n_joints:
            raise SystemExit("fixture topology differs from --topology")
        return t, p
    t, p, _ = design_from_dict(json.loads(Path(args.start).read_text()))
    return t, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", choices=list(METRICS), default="diff")
    ap.add_argument("--topology", default="2x4")
    ap.add_argument("--start", default="anthropomorphic")
    ap.add_argument("--optimizer", choices=["adam", "cmaes", "hybrid"], default="adam")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--stages", type=int, default=3)
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--gens", type=int, default=30)
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--budget", type=int, default=8, help="seeds per design")
    ap.add_argument("--cells", type=int, default=5)
    ap.add_argument("--rot-steps", type=int, default=12)
    ap.add_argument("--validate-every", type=int, default=10)
    ap.add_argument("--no-official", action="store_true", help="skip the original-KaRMA validation runs")
    ap.add_argument("--objective", choices=["balanced", "translation", "rotation"], default="balanced")
    ap.add_argument("--torque", type=float, default=None, help="override the torque ceiling (N·m)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    task = load_task_spec()
    topo, params0 = start_design(args)
    cfg = METRICS[args.metric]
    from dataclasses import replace
    lam_T, lam_R = {"balanced": (1.0, 1.0), "translation": (1.0, 0.0), "rotation": (0.0, 1.0)}[args.objective]
    cfg = replace(cfg, commands=CommandConfig(n_cells=args.cells, n_rot_steps=args.rot_steps),
                  smooth=replace(cfg.smooth, lambda_T=lam_T, lambda_R=lam_R))
    seed_cfg = SeedConfig(budget=args.budget)
    obj = make_objective(topo, params0, BOUNDS, task, cfg, seed_cfg)
    if args.torque is not None and obj.act is not None:
        obj.act = obj.act._replace(tau_bar=jnp.full_like(obj.act.tau_bar, args.torque))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    validate = exact_validation(obj, out / "validation_runs", official=not args.no_official)
    phi0 = encode(params0, topo, BOUNDS)
    log_lines = []

    def log(s):
        print(s, flush=True); log_lines.append(s)

    log(f"metric={args.metric} topology={args.topology} optimizer={args.optimizer} seeds={int(obj.seeds.valid.sum())} "
        f"stats={obj.seeds.stats}")
    v0 = validate(params0)
    log(f"start validation: {v0}")
    t0 = time.perf_counter()
    if args.optimizer == "adam":
        res = adam(obj, phi0, n_iters=args.iters, lr=args.lr, stages=args.stages, validate_every=args.validate_every,
                   validation_fn=validate, log=log, seed=args.seed)
    elif args.optimizer == "cmaes":
        res = cmaes(obj, phi0, popsize=args.popsize, n_gens=args.gens, validate_every_gens=max(args.validate_every // 2, 1),
                    validation_fn=validate, log=log, seed=args.seed)
    else:
        res = hybrid(obj, phi0, n_starts=args.starts, adam_iters=args.iters, validation_fn=validate, log=log, seed=args.seed,
                     lr=args.lr, stages=args.stages)
    res.meta.update(wall_clock_s=time.perf_counter() - t0, start_validation=v0)
    config = dict(args=vars(args), metric=cfg.name, commands=cfg.commands.__dict__, smooth=cfg.smooth.__dict__,
                  energy=cfg.energy._asdict(), roll=cfg.roll._asdict(), seed_config=seed_cfg.__dict__,
                  bounds=BOUNDS.__dict__, task=dict(object=task.obj.__dict__, task=task.task.__dict__, rolling=task.rolling.__dict__))
    save_run(out, topo, params0, res, config, actuator_spec=load_actuation_spec().__dict__, seeds=obj.seeds)
    (out / "log.txt").write_text("\n".join(log_lines))
    log(f"done: best validation {res.best_validation} wall {res.meta['wall_clock_s']:.0f}s -> {out}")


if __name__ == "__main__":
    main()
