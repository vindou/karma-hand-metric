"""Experiment I (SPEC.md §12, Phase 3 exit criterion): is Diff-KaRMA a useful surrogate?

For a dataset of two-finger designs (the three Phase 0 designs plus perturbations of
the baseline), compute

* smooth: ``K_T~``, ``K_R~``, score (``dkarma.metrics.diff_karma``), gradient norm;
* exact library: counted cells / bins over the same commands (``evaluate_exact``);
* official KaRMA on the physical task (``official_karma_physical``: the original BFS
  with the SPEC §2 object and tolerances): top-3 voxel fraction of the cube, KaRMA-R;
* official KaRMA as published (L_ref-scaled nominal task): KaRMA-T, KaRMA-R;

then Spearman rank correlations, and a local-improvement trial: from several
starts, normalized gradient ascent steps on the smooth objective (accepted when
the smooth value increases), re-scored by the exact evaluators.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.surrogate_validation --out workspace/phase3/surrogate.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import build_hand_model, load_task_spec, Bounds, encode, decode, fixed_from_params, HandParams  # noqa: E402
from dkarma.solvers.seed_projection import find_seeds, SeedConfig  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA  # noqa: E402
from dkarma.metrics import diff_karma as DK  # noqa: E402
from dkarma.metrics.exact_library import evaluate_exact, official_karma_physical, official_karma  # noqa: E402
from dkarma.tests.conftest import load_design  # noqa: E402

BOUNDS = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)


def spearman(a, b):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    return float(spearmanr(a[ok], b[ok]).correlation)


def dataset(n_perturb: int, seed: int):
    topo, params, _ = load_design("baseline")
    out = [("baseline", topo, params)]
    for nm in ("T", "A1"):
        t, p, _ = load_design(nm)
        out.append((nm, t, p))
    fixed = fixed_from_params(params)
    phi0 = encode(params, topo, BOUNDS)
    rng = np.random.RandomState(seed)
    for i in range(n_perturb):
        scale = 0.15 + 0.35 * rng.rand()
        phi = jax.tree_util.tree_map(lambda a: a + scale * rng.randn(*a.shape), phi0)
        out.append((f"pert{i:02d}", topo, decode(phi, topo, BOUNDS, fixed)))
    return out


def score_design(name, topo, params, task, sp, cfg, seed_cfg, workdir, do_official=True):
    rec = {"name": name}
    hm = build_hand_model(topo, params)
    t0 = time.perf_counter()
    seeds = find_seeds(hm, topo, sp, seed_cfg)
    rec["seed_stats"] = seeds.stats
    rec["n_valid_seeds"] = int(seeds.valid.sum())
    rec["t_seeds"] = time.perf_counter() - t0
    if rec["n_valid_seeds"] == 0:
        rec.update(smooth=None, exact=None)
        return rec, None, None
    fixed = fixed_from_params(params)
    phi = encode(params, topo, BOUNDS)
    f = jax.jit(jax.value_and_grad(lambda ph: DK.objective(ph, topo, BOUNDS, fixed, seeds, sp, task, cfg)))
    t0 = time.perf_counter()
    val, grad = f(phi)
    d = jax.jit(lambda h: DK.evaluate(h, topo, seeds, sp, task, cfg))(hm)
    rec["t_smooth"] = time.perf_counter() - t0
    v = np.asarray(d.valid)
    rec["smooth"] = dict(score=float(val), K_T=float(np.asarray(d.K_T)[v].mean()), K_R=float(np.asarray(d.K_R)[v].mean()),
                         K_T_best=float(np.asarray(d.K_T)[v].max()),
                         grad_norm=float(jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grad)))))
    t0 = time.perf_counter()
    ex = evaluate_exact(hm, topo, seeds, sp, task, cfg)
    rec["t_exact"] = time.perf_counter() - t0
    rec["exact"] = dict(K_T=ex.K_T, K_R=ex.K_R, cells_per_seed=[len(r.cells) for r in ex.per_seed],
                        ok_T_frac=float(np.mean([r.ok_T.mean() for r in ex.per_seed])),
                        pair_switch_frac=float(np.mean([r.pair_switch_T.mean() for r in ex.per_seed])))
    if do_official:
        t0 = time.perf_counter()
        rec["official_physical"] = official_karma_physical(topo, params, task, workdir, name=f"{name}_phys")
        rec["official_nominal"] = official_karma(topo, params, workdir, name=f"{name}_nom")
        rec["t_official"] = time.perf_counter() - t0
    return rec, (seeds, fixed, phi, f), d


def local_improvement(name, topo, params, task, sp, cfg, seed_cfg, workdir, n_steps, step):
    rec0, ctx, _ = score_design(name, topo, params, task, sp, cfg, seed_cfg, workdir)
    if ctx is None:
        return {"name": name, "skipped": "no seeds"}
    seeds, fixed, phi, f = ctx
    val, grad = f(phi)
    hist = [float(val)]
    for _ in range(n_steps):
        gn = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grad)))
        s = step
        for _try in range(4):
            phi2 = jax.tree_util.tree_map(lambda a, g: a + s * g / (gn + 1e-12), phi, grad)
            v2, g2 = f(phi2)
            if float(v2) > float(val):
                phi, val, grad = phi2, v2, g2
                break
            s *= 0.5
        hist.append(float(val))
    params2 = decode(phi, topo, BOUNDS, fixed)
    rec1, _, _ = score_design(name + "_improved", topo, params2, task, sp, cfg, seed_cfg, workdir)
    return {"name": name, "smooth_history": hist, "before": rec0, "after": rec1}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase3/surrogate.json")
    ap.add_argument("--n-perturb", type=int, default=9)
    ap.add_argument("--local-starts", type=int, default=4)
    ap.add_argument("--local-steps", type=int, default=6)
    ap.add_argument("--local-step", type=float, default=0.05)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-official", action="store_true")
    ap.add_argument("--local-only", action="store_true", help="skip the dataset; run the local-improvement trial only")
    ap.add_argument("--resume", default=None, help="existing JSON to extend (skips designs already scored)")
    args = ap.parse_args()
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    task = load_task_spec(); sp = task.step_params(jnp.float64)
    cfg = DIFF_KARMA
    seed_cfg = SeedConfig(n_sobol=128, n_random=32, budget=args.budget)
    results = {"designs": [], "local": [], "config": dict(n_perturb=args.n_perturb, budget=args.budget, seed=args.seed)}
    if args.resume and Path(args.resume).exists():
        results = json.loads(Path(args.resume).read_text())
    done = {r["name"] for r in results["designs"]}
    work = Path(tempfile.mkdtemp(prefix="surrogate_"))
    for name, topo, params in ([] if args.local_only else dataset(args.n_perturb, args.seed)):
        if name in done:
            continue
        t0 = time.perf_counter()
        rec, _, _ = score_design(name, topo, params, task, sp, cfg, seed_cfg, work / name, do_official=not args.no_official)
        rec["t_total"] = time.perf_counter() - t0
        results["designs"].append(rec)
        print(f"[{name}] seeds={rec['n_valid_seeds']} smooth={rec.get('smooth')} exact={rec.get('exact')} "
              f"phys={rec.get('official_physical', {}).get('cube_fraction') if rec.get('official_physical') else None} "
              f"nomT={rec.get('official_nominal', {}).get('KaRMA_T') if rec.get('official_nominal') else None} ({rec['t_total']:.0f}s)", flush=True)
        out.write_text(json.dumps(results, indent=1, default=float))
    rows = [r for r in results["designs"] if r.get("smooth") and r.get("exact")]
    corr = {}
    if rows:
        sT = [r["smooth"]["K_T"] for r in rows]; sR = [r["smooth"]["K_R"] for r in rows]; sc = [r["smooth"]["score"] for r in rows]
        corr["smooth_KT_vs_exactlib_KT"] = spearman(sT, [r["exact"]["K_T"] for r in rows])
        corr["smooth_KR_vs_exactlib_KR"] = spearman(sR, [r["exact"]["K_R"] for r in rows])
        if not args.no_official:
            corr["smooth_KT_vs_official_physical_cube_fraction"] = spearman(sT, [r["official_physical"]["cube_fraction"] for r in rows])
            corr["smooth_KR_vs_official_physical_KaRMA_R"] = spearman(sR, [r["official_physical"]["KaRMA_R"] for r in rows])
            corr["smooth_KT_vs_official_nominal_KaRMA_T"] = spearman(sT, [r["official_nominal"]["KaRMA_T"] for r in rows])
            corr["exactlib_KT_vs_official_physical_cube_fraction"] = spearman([r["exact"]["K_T"] for r in rows], [r["official_physical"]["cube_fraction"] for r in rows])
            corr["score_vs_official_physical_cube_fraction"] = spearman(sc, [r["official_physical"]["cube_fraction"] for r in rows])
    results["correlations"] = corr
    print("correlations:", json.dumps(corr, indent=1), flush=True)
    out.write_text(json.dumps(results, indent=1, default=float))
    starts = [d for d in dataset(args.n_perturb, args.seed) if d[0] in ("baseline",) or d[0].startswith("pert")][: args.local_starts]
    for name, topo, params in starts:
        rec = local_improvement(name, topo, params, task, sp, cfg, seed_cfg, work / ("loc_" + name), args.local_steps, args.local_step)
        results["local"].append(rec)
        if "after" in rec:
            b, a = rec["before"], rec["after"]
            print(f"[local {name}] smooth {rec['smooth_history'][0]:.4f} -> {rec['smooth_history'][-1]:.4f}; "
                  f"exactlib K_T {b['exact']['K_T']:.5f} -> {a['exact']['K_T']:.5f}; "
                  f"phys cube {b['official_physical']['cube_fraction']:.5f} -> {a['official_physical']['cube_fraction']:.5f}; "
                  f"nominal T {b['official_nominal']['KaRMA_T']:.5f} -> {a['official_nominal']['KaRMA_T']:.5f}", flush=True)
        out.write_text(json.dumps(results, indent=1, default=float))
    print("done", flush=True)


if __name__ == "__main__":
    main()
