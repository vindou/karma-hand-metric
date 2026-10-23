"""Phase 8.4 (SPEC.md §11.1): statistically meaningful optimizer comparison.

For one objective (``diff2``: two-finger Diff-KaRMA from the Phase 0 baseline; ``full3``: 3x4
Full KaRMA from the anthropomorphic start), >= 8 independent starts (the nominal start plus
Sobol-perturbed designs that have seeds), three optimizers at matched wall clock on this GPU:

1. Adam with the Phase 8.2 schedule and the Phase 8.3 seed policy (``ADAM_DEFAULTS``); its
   optimizer time per start (compile excluded by a warm-up, validations excluded and run on
   every checkpoint) sets the budget ``B`` of that start;
2. CMA-ES (population 8, the smooth objective batched on the GPU) until its optimizer time
   reaches ``B``; the incumbent is validated every two generations;
3. the hybrid (Sobol starts around the start design, short Adam refinements) until ``B``.

Every returned design (the best by exact validation, SPEC §8.5) is then evaluated with fresh
seeds by the exact library and by the original KaRMA on the physical task. The summary reports
mean / median / best / IQR per optimizer and the paired comparison per start (differences,
sign counts, Wilcoxon signed-rank) for the exact score and the physical cube fraction.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.optimizer_comparison --objective diff2 --starts 8 --out workspace/phase8/8.4/diff2
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.optimizer_comparison --summary workspace/phase8/8.4/diff2
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

from dkarma.core import decode, fixed_from_params, load_task_spec, build_hand_model  # noqa: E402
from dkarma.core.actuation import load_actuation_spec  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig  # noqa: E402
from dkarma.optimization import make_objective, adam, cmaes, hybrid, save_run  # noqa: E402
from dkarma.optimization.optimizers import ADAM_DEFAULTS  # noqa: E402
from dkarma.optimization.objectives import exact_validation  # noqa: E402
from dkarma.metrics.exact_library import official_karma_physical  # noqa: E402
from dkarma.experiments.continuation_study import objective_spec, make_starts, BOUNDS  # noqa: E402


def final_evaluation(obj, params, workdir):
    """Fresh-seed exact library (the validation criterion) and the original KaRMA on the physical task."""
    v = exact_validation(obj, workdir, official=False)(params)
    row = official_karma_physical(obj.topo, params, obj.task, Path(workdir) / "official", name="final")
    v.update(official_cube_fraction=row["cube_fraction"], official_KaRMA_R=row["KaRMA_R"], official_voxels=row["n_voxels"],
             official_L_ref_mm=row["L_ref_mm"])
    return v


def run_start(spec, task, name, phi, out, log, methods, budget_s=None, cma_pop=8, hybrid_starts=4):
    topo, cfg = spec["topo"], spec["cfg"]
    params = decode(phi, topo, BOUNDS, fixed_from_params(spec["params0"]))
    obj = make_objective(topo, params, BOUNDS, task, cfg, SeedConfig(budget=spec["budget"]))
    validate = exact_validation(obj, out / "val", official=False)
    recs = {}
    v0 = final_evaluation(obj, params, out / "final_start")
    recs["start"] = dict(final=v0)
    log(f"[{name}] start: exact {v0['exact_score']:.4f} official {v0['official_cube_fraction']:.5f}")
    B = budget_s
    for method in methods:
        t0 = time.perf_counter()
        if method == "adam":
            res = adam(obj, phi, n_iters=spec["n_iters"], stages=spec["stages"], validate_every=spec["validate_every"], validation_fn=validate,
                       log=log, seed=0, instrument=False, warmup=True)
            B = res.meta["optimizer_time_s"] if budget_s is None else budget_s
        elif method == "cmaes":
            # the incumbent is validated every 16 generations (128 evaluations): ~8 checkpoints per run at the measured
            # budget, like Adam's 7; start 0 of diff2 ran with every 2 generations before this was set (phase8_report §8.4)
            res = cmaes(obj, phi, popsize=cma_pop, n_gens=10_000, validate_every_gens=16, validation_fn=validate, log=log, seed=1,
                        time_budget_s=B, warmup=True)
        else:
            per = max(spec["n_iters"] // hybrid_starts, 5)
            res = hybrid(obj, phi, n_starts=hybrid_starts, sobol_scale=0.25, adam_iters=per, validation_fn=validate, log=log, seed=2,
                         time_budget_s=B, warmup=True, stages=max(spec["stages"] - 1, 1))
        wall = time.perf_counter() - t0
        rdir = out / method; rdir.mkdir(parents=True, exist_ok=True)
        save_run(rdir, topo, params, res, dict(method=method, start=name, budget_s=B, objective=cfg.name, adam_defaults=ADAM_DEFAULTS,
                                                seed_config=SeedConfig(budget=spec["budget"]).__dict__, optimizer_seed=res.meta.get("seed")),
                 actuator_spec=load_actuation_spec().__dict__, seeds=obj.seeds)
        fin = final_evaluation(obj, res.best_params, rdir / "final")
        recs[method] = dict(final=fin, best_validation=res.best_validation, optimizer_time_s=res.meta["optimizer_time_s"],
                            validation_time_s=res.meta["validation_time_s"], wall_clock_s=wall, budget_s=B,
                            n_evaluations=(len(res.training_history) if method != "cmaes" else sum(1 for _ in res.training_history) * cma_pop),
                            meta={k: v for k, v in res.meta.items() if k not in ("stages_log", "per_start")})
        log(f"[{name}/{method}] exact {fin['exact_score']:.4f} official {fin['official_cube_fraction']:.5f} "
            f"opt {res.meta['optimizer_time_s']:.0f}s (budget {B:.0f}s) val {res.meta['validation_time_s']:.0f}s wall {wall:.0f}s")
        (out / "run.json").write_text(json.dumps(recs, indent=1, default=float))
    return recs


def summarize(root: Path):
    from scipy.stats import wilcoxon
    root = Path(root)
    runs = {p.parent.name: json.loads(p.read_text()) for p in sorted(root.glob("start*/run.json"))}
    methods = [m for m in ("adam", "cmaes", "hybrid") if all(m in r for r in runs.values())]
    out = dict(n_starts=len(runs), methods=methods, metrics={})
    lines = [f"starts: {len(runs)} ({', '.join(runs)})", ""]
    for metric in ("exact_score", "official_cube_fraction"):
        start = np.array([r["start"]["final"][metric] for r in runs.values()])
        vals = {m: np.array([r[m]["final"][metric] for r in runs.values()]) for m in methods}
        rec = dict(start=dict(mean=float(start.mean()), median=float(np.median(start))), per_method={}, paired={})
        lines += [f"### {metric}", "", "| method | mean | median | best | IQR | beats start | mean gain vs start |", "| --- | --- | --- | --- | --- | --- | --- |"]
        for m, v in vals.items():
            q1, q3 = np.percentile(v, [25, 75])
            rec["per_method"][m] = dict(mean=float(v.mean()), median=float(np.median(v)), best=float(v.max()), iqr=[float(q1), float(q3)],
                                        beats_start=int(np.sum(v > start)), values=v.tolist(), gain_vs_start=(v - start).tolist())
            lines.append(f"| {m} | {v.mean():.4f} | {np.median(v):.4f} | {v.max():.4f} | [{q1:.4f}, {q3:.4f}] | {int(np.sum(v > start))}/{len(v)} | {np.mean(v - start):+.4f} |")
        lines += ["", "| pair | mean diff | median diff | wins / ties / losses | Wilcoxon p |", "| --- | --- | --- | --- | --- |"]
        for i, a in enumerate(methods):
            for b in methods[i + 1:]:
                d = vals[a] - vals[b]
                w = int(np.sum(d > 1e-12)); l_ = int(np.sum(d < -1e-12)); t = len(d) - w - l_
                try:
                    pv = float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0
                except ValueError:
                    pv = float("nan")
                rec["paired"][f"{a}-{b}"] = dict(mean=float(d.mean()), median=float(np.median(d)), wins=w, ties=t, losses=l_, wilcoxon_p=pv, diffs=d.tolist())
                lines.append(f"| {a} - {b} | {d.mean():+.4f} | {np.median(d):+.4f} | {w} / {t} / {l_} | {pv:.3f} |")
        lines.append("")
        out["metrics"][metric] = rec
    budgets = {m: [r[m]["optimizer_time_s"] for r in runs.values()] for m in methods}
    out["optimizer_time_s"] = {m: dict(mean=float(np.mean(v)), values=v) for m, v in budgets.items()}
    out["budget_s"] = [r[methods[0]]["budget_s"] for r in runs.values()]
    lines += ["optimizer time per start (s, validations and compile excluded): " + ", ".join(f"{m} {np.mean(v):.0f}" for m, v in budgets.items())]
    (root / "summary.json").write_text(json.dumps(out, indent=1, default=float))
    (root / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["diff2", "full3"], default="diff2")
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--methods", nargs="*", default=["adam", "cmaes", "hybrid"])
    ap.add_argument("--budget-s", type=float, default=None, help="fixed optimizer budget (default: Adam's measured time per start)")
    ap.add_argument("--only", type=int, default=None, help="run only this start index (one process per start keeps host memory bounded)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    args = ap.parse_args()
    if args.summary:
        summarize(Path(args.summary)); return
    out = Path(args.out or f"workspace/phase8/8.4/{args.objective}"); out.mkdir(parents=True, exist_ok=True)
    task = load_task_spec()
    spec = objective_spec(args.objective)
    log_path = out / "log.txt"

    def log(s):
        print(s, flush=True)
        with open(log_path, "a") as fh:
            fh.write(s + "\n")

    starts, _ = make_starts(spec, task, args.starts, 0.25, out, seed=321, log=log, scales=[0.25] * 4 + [0.5] * 16)
    log(f"objective {args.objective}: {len(starts)} starts; ADAM_DEFAULTS {ADAM_DEFAULTS}")
    for k, (name, phi) in enumerate(starts):
        if args.only is not None and k != args.only:
            continue
        rdir = out / name
        if (rdir / "run.json").exists() and all((rdir / m).exists() for m in args.methods):
            log(f"[{name}] exists, skipping"); continue
        rdir.mkdir(parents=True, exist_ok=True)
        run_start(spec, task, name, phi, rdir, log, args.methods, budget_s=args.budget_s)
    if args.only is None or args.only == len(starts) - 1:
        summarize(out)
    log("done")


if __name__ == "__main__":
    main()
