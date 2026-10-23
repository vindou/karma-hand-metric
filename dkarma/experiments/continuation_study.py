"""Phase 8.2 / 8.3 (SPEC.md §11.1): continuation-schedule repair and seed-policy study.

Runs the instrumented Adam (``dkarma.optimization.optimizers.adam``) from the same starts
under several schedule / optimizer-hygiene / seed-policy variants, with exact-library
validation (fresh seeds) at every stage boundary and every ``validate_every`` iterations,
and records per run: the exact score of the start, of every validated iterate, of the raw
final iterate and of the stage-selected final iterate; the smooth-vs-exact gap and the gate
shares at every stage boundary; the number of non-finite and clipped gradients; the seed
events; wall clock with the validation time separated.

Objectives: ``diff2`` = two-finger Diff-KaRMA from the Phase 0 baseline (Experiment I's
setting: 8 seeds, 5 cells, 12 rotation steps, 60 iterations, 3 stages); ``full3`` = 3x4 Full
KaRMA from the anthropomorphic start (Experiment III's setting: 6 three-contact seeds, 4 cells,
10 rotation steps, 40 iterations, 2 stages). Starts: the nominal start plus Sobol-perturbed
designs (scale 0.25 in the unconstrained coordinates) that have seeds.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.continuation_study --objective diff2 \\
        --variants phase7 hygiene --starts 5 --out workspace/phase8/8.2/diff2
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.continuation_study --summary workspace/phase8/8.2/diff2
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import Topology, Bounds, anthropomorphic, encode, decode, load_task_spec, fixed_from_params  # noqa: E402
from dkarma.core.actuation import load_actuation_spec  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, FULL_KARMA  # noqa: E402
from dkarma.metrics.commands import CommandConfig  # noqa: E402
from dkarma.optimization import make_objective, adam, save_run  # noqa: E402
from dkarma.optimization.objectives import exact_validation  # noqa: E402
from dkarma.tests.conftest import load_design  # noqa: E402

BOUNDS = Bounds(link_len_min_m=0.012, link_len_max_m=0.090, tip_len_min_m=0.012, tip_len_max_m=0.090, base_xyz_max_m=0.10)

HYGIENE = dict(grad_clip=100.0, skip_nonfinite=True, reset_moments=True)
HYGIENE2 = dict(grad_clip=None, grad_spike=10.0, skip_nonfinite=True, reset_moments=True)   # skip spikes instead of clipping them
VARIANTS = {
    "hygiene2": dict(**HYGIENE2, stage_select=False, seed_policy="reproject"),
    "hygiene2_select": dict(**HYGIENE2, stage_select=True, seed_policy="reproject"),
    "hygiene2_select_reseed": dict(**HYGIENE2, stage_select=True, seed_policy="regenerate_stage"),
    "hygiene2_select_skipstrict": dict(**HYGIENE2, stage_select=True, seed_policy="reproject", schedule_kw=dict(skip_strict=True)),
    "phase7": dict(grad_clip=None, skip_nonfinite=False, reset_moments=False, stage_select=False, seed_policy="reproject"),
    "hygiene": dict(**HYGIENE, stage_select=False, seed_policy="reproject"),
    "hygiene_select": dict(**HYGIENE, stage_select=True, seed_policy="reproject"),
    "hygiene_gentle": dict(**HYGIENE, stage_select=False, seed_policy="reproject", schedule_kw=dict(sigma_energy_end=2.0)),
    "hygiene_skipstrict": dict(**HYGIENE, stage_select=False, seed_policy="reproject", schedule_kw=dict(skip_strict=True)),
    "hygiene_reseed": dict(**HYGIENE, stage_select=False, seed_policy="regenerate_stage"),
    "hygiene_select_reseed": dict(**HYGIENE, stage_select=True, seed_policy="regenerate_stage"),
    "hygiene_select_skipstrict": dict(**HYGIENE, stage_select=True, seed_policy="reproject", schedule_kw=dict(skip_strict=True)),
    # 8.3 seed policies on top of the adopted 8.2 schedule (set --base-variant)
    "policy_reproject": dict(seed_policy="reproject"),
    "policy_regenerate_stage": dict(seed_policy="regenerate_stage"),
    "policy_regenerate_every_k": dict(seed_policy="regenerate_every_k"),
    "policy_union_stage": dict(seed_policy="union_stage"),
}


def objective_spec(name):
    if name == "diff2":
        topo, params0, _ = load_design("baseline")
        cfg = replace(DIFF_KARMA, commands=CommandConfig(n_cells=5, n_rot_steps=12))
        return dict(topo=topo, params0=params0, cfg=cfg, budget=8, n_iters=60, stages=3, validate_every=10, min_seeds=4)
    if name == "full3":
        topo = Topology.three_finger(4)
        cfg = replace(FULL_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10))
        return dict(topo=topo, params0=anthropomorphic(topo), cfg=cfg, budget=6, n_iters=40, stages=2, validate_every=10, min_seeds=3)
    raise SystemExit(f"unknown objective {name}")


def make_starts(spec, task, n_starts, scale, out, seed=123, log=print, scales=None):
    """Nominal start plus Sobol perturbations with seeds; cached in ``out/starts.json``.
    ``scales``: per-Sobol-point perturbation scale (default ``scale`` for every point)."""
    import jax.flatten_util
    from scipy.stats.qmc import Sobol
    topo, params0, cfg = spec["topo"], spec["params0"], spec["cfg"]
    cache = Path(out) / "starts.json"
    phi0 = encode(params0, topo, BOUNDS)
    x0, unravel = jax.flatten_util.ravel_pytree(phi0)
    if cache.exists():
        rec = json.loads(cache.read_text())
        return [(r["name"], unravel(jnp.asarray(r["x"]))) for r in rec["starts"]], unravel
    starts = [("start0_nominal", x0)]
    pts = Sobol(d=x0.shape[0], scramble=True, seed=seed).random(8 * n_starts)
    fixed = fixed_from_params(params0)
    k = 0
    for j, p in enumerate(pts):
        if len(starts) >= n_starts:
            break
        sc = scales[j] if scales is not None and j < len(scales) else scale
        x = x0 + sc * jnp.asarray(2.0 * p - 1.0)
        params = decode(unravel(x), topo, BOUNDS, fixed)
        obj = make_objective(topo, params, BOUNDS, task, cfg, SeedConfig(budget=spec["budget"]))
        n = int(obj.seeds.valid.sum())
        log(f"  sobol point {j}: {n} seeds")
        if n >= spec["min_seeds"]:
            k += 1
            starts.append((f"start{k}_sobol{j}_s{sc:g}", x))
    cache.write_text(json.dumps(dict(seed=seed, scale=scale, scales=scales, starts=[dict(name=n, x=np.asarray(x).tolist()) for n, x in starts]), indent=1))
    return [(n, unravel(x)) for n, x in starts], unravel


def run_one(spec, task, name, phi, variant, out, log, base_kw, kkt_method: str | None = None):
    topo, cfg = spec["topo"], spec["cfg"]
    if kkt_method is not None:                       # Phase 9.1: the hard-QP derivative under study
        cfg = replace(cfg, roll=cfg.roll._replace(kkt_method=kkt_method))
    params = decode(phi, topo, BOUNDS, fixed_from_params(spec["params0"]))
    kw = dict(base_kw); kw.update(VARIANTS[variant])
    obj = make_objective(topo, params, BOUNDS, task, cfg, SeedConfig(budget=spec["budget"]), seed_policy=kw.pop("seed_policy", "reproject"),
                         reseed_every=kw.pop("reseed_every", 10))
    validate_raw = exact_validation(obj, out / "val", official=False)
    cache_path = out.parent.parent / "start_validation.json"      # the start's validation is identical for every variant
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    p0 = np.asarray(jax.flatten_util.ravel_pytree(params)[0])

    def validate(p):
        if np.allclose(np.asarray(jax.flatten_util.ravel_pytree(p)[0]), p0, atol=1e-12):
            if name not in cache:
                cache[name] = validate_raw(p); cache_path.write_text(json.dumps(cache, indent=1, default=float))
            return dict(cache[name])
        return validate_raw(p)
    t0 = time.perf_counter()
    res = adam(obj, phi, n_iters=spec["n_iters"], stages=spec["stages"], validate_every=spec["validate_every"], validation_fn=validate,
               log=log, seed=0, **kw)
    wall = time.perf_counter() - t0
    config = dict(objective=cfg.name, variant=variant, adam=res.meta["options"], n_iters=spec["n_iters"], stages=spec["stages"], kkt_method=cfg.roll.kkt_method,
                  commands=cfg.commands.__dict__, smooth=cfg.smooth.__dict__, energy=cfg.energy._asdict(), roll=cfg.roll._asdict(),
                  seed_config=SeedConfig(budget=spec["budget"]).__dict__, bounds=BOUNDS.__dict__, start=name, optimizer_seed=0)
    save_run(out, topo, params, res, config, actuator_spec=load_actuation_spec().__dict__, seeds=obj.seeds)
    rec = dict(start=name, variant=variant, wall_clock_s=wall, meta=res.meta, best_validation=res.best_validation,
               training_history=res.training_history, validation_history=res.validation_history, n_seeds_start=int(obj.seeds.valid.sum()),
               kkt_method=cfg.roll.kkt_method)
    (out / "run.json").write_text(json.dumps(rec, indent=1, default=float))
    return rec


def summarize(root: Path):
    root = Path(root)
    rows = []
    for rj in sorted(root.glob("*/*/run.json")):
        r = json.loads(rj.read_text())
        m = r["meta"]
        st = m["stages_log"]
        start_exact = st[0].get("exact_start") if st else None
        th = r.get("training_history", [])
        rows.append(dict(variant=r["variant"], start=r["start"], kkt_method=r.get("kkt_method", "exact"), start_exact=start_exact, best_intermediate=m["best_intermediate_exact"],
                         n_big_grad=sum(1 for h in th if h.get("grad_finite") and h.get("grad_norm", 0) > 100.0),
                         n_big_grad_hard=sum(1 for h in th if h.get("grad_finite") and h.get("grad_norm", 0) > 100.0 and h.get("stage", 0) > 0),
                         final_raw=m["final_iterate_exact_raw"], final_selected=m["final_iterate_exact_selected"],
                         returned=r["best_validation"].get("exact_score"), best_iter=m["best_iter"],
                         n_nonfinite=sum(s["n_nonfinite"] for s in st), n_clipped=sum(s["n_clipped"] for s in st),
                         wall=r["wall_clock_s"], val_time=m["validation_time_s"],
                         stage_ends=[(s.get("exact_start"), s.get("exact_end_raw")) for s in st]))
    by_var = {}
    for r in rows:
        by_var.setdefault(r["variant"], []).append(r)
    lines = ["| variant | starts | final raw >= best-2% | final selected >= best-2% | returned > start | mean start -> returned | mean final raw | nonfinite / clipped | wall (val) s |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    out = {}
    for v, rs in by_var.items():
        def ok(key):
            return sum(1 for r in rs if r[key] is not None and r["best_intermediate"] is not None and r[key] >= 0.98 * r["best_intermediate"])
        imp = sum(1 for r in rs if r["returned"] is not None and r["start_exact"] is not None and r["returned"] > r["start_exact"])
        ms = np.mean([r["start_exact"] for r in rs]); mr = np.mean([r["returned"] for r in rs]); mf = np.mean([r["final_raw"] for r in rs])
        out[v] = dict(n=len(rs), final_raw_ok=ok("final_raw"), final_selected_ok=ok("final_selected"), improved=imp, mean_start=ms, mean_returned=mr,
                      mean_final_raw=mf, n_nonfinite=sum(r["n_nonfinite"] for r in rs), n_clipped=sum(r["n_clipped"] for r in rs),
                      mean_wall=np.mean([r["wall"] for r in rs]), mean_val=np.mean([r["val_time"] for r in rs]), rows=rs)
        lines.append(f"| {v} | {len(rs)} | {ok('final_raw')}/{len(rs)} | {ok('final_selected')}/{len(rs)} | {imp}/{len(rs)} | {ms:.4f} -> {mr:.4f} | {mf:.4f} | "
                     f"{out[v]['n_nonfinite']} / {out[v]['n_clipped']} | {out[v]['mean_wall']:.0f} ({out[v]['mean_val']:.0f}) |")
    (root / "study.json").write_text(json.dumps(out, indent=1, default=float))
    (root / "study.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--objective", choices=["diff2", "full3"], default="diff2")
    ap.add_argument("--variants", nargs="*", default=["phase7", "hygiene"])
    ap.add_argument("--base-variant", default=None, help="variant whose options the policy_* variants build on")
    ap.add_argument("--starts", type=int, default=5)
    ap.add_argument("--start-scale", type=float, default=0.25)
    ap.add_argument("--reseed-every", type=int, default=10)
    ap.add_argument("--out", default=None)
    ap.add_argument("--summary", default=None)
    ap.add_argument("--kkt-method", default=None, help="Phase 9.1: hard-QP derivative (exact | reg | pinv | reduced); default = RollConfig's")
    ap.add_argument("--starts-from", default=None, help="reuse the starts.json of another study directory (same starts, e.g. the Phase 8 runs)")
    args = ap.parse_args()
    if args.summary:
        summarize(Path(args.summary)); return
    out = Path(args.out or f"workspace/phase8/8.2/{args.objective}"); out.mkdir(parents=True, exist_ok=True)
    task = load_task_spec()
    spec = objective_spec(args.objective)
    log_path = out / "log.txt"

    def log(s):
        print(s, flush=True)
        with open(log_path, "a") as fh:
            fh.write(s + "\n")

    if args.starts_from and not (out / "starts.json").exists():
        import shutil
        shutil.copy(Path(args.starts_from) / "starts.json", out / "starts.json")
        if (Path(args.starts_from) / "start_validation.json").exists():       # the start's exact validation is design-only: reusable
            shutil.copy(Path(args.starts_from) / "start_validation.json", out / "start_validation.json")
    starts, _ = make_starts(spec, task, args.starts, args.start_scale, out, log=log)
    log(f"objective {args.objective}: {len(starts)} starts: {[n for n, _ in starts]}")
    base_kw = dict(VARIANTS[args.base_variant]) if args.base_variant else {}
    base_kw["reseed_every"] = args.reseed_every
    for variant in args.variants:
        for name, phi in starts:
            rdir = out / variant / name
            if (rdir / "run.json").exists():
                log(f"[{variant}/{name}] exists, skipping"); continue
            rdir.mkdir(parents=True, exist_ok=True)
            log(f"[{variant}/{name}] start")
            rec = run_one(spec, task, name, phi, variant, rdir, log, base_kw, kkt_method=args.kkt_method)
            m = rec["meta"]
            log(f"[{variant}/{name}] done: start {m['stages_log'][0].get('exact_start'):.4f} best-int {m['best_intermediate_exact']:.4f} "
                f"final raw {m['final_iterate_exact_raw']:.4f} selected {m['final_iterate_exact_selected']:.4f} returned {rec['best_validation'].get('exact_score'):.4f} "
                f"wall {rec['wall_clock_s']:.0f}s (val {m['validation_time_s']:.0f}s) nonfinite {sum(s['n_nonfinite'] for s in m['stages_log'])}")
        summarize(out)
    log("done")


if __name__ == "__main__":
    main()
