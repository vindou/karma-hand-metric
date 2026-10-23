"""Optimize a two-finger hand's kinematic tree AGAINST the KaRMA metric with CMA-ES.

    python optimize/run_optimization.py --out optimize/runs/T --objective T --hours 2.5

Each generation: CMA-ES proposes `popsize` genotypes -> HandFamily.decode ->
URDF + YAML -> fast_eval.evaluate (the library's own kernels at reduced seed
fidelity) in a process pool -> objective -> CMA-ES tell. Every evaluation is
appended to evals.jsonl; the best design so far is kept in best.json; the
optimizer state is checkpointed each generation so --resume continues a run.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import logging
import multiprocessing as mp
import pickle
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from optimize.hand_family import HandFamily, FamilyConfig            # noqa: E402
from optimize.fast_eval import FastEvalConfig, objective_value  # noqa: E402
from optimize import backend as BK  # noqa: E402

log = logging.getLogger("optimize")

# ── worker ───────────────────────────────────────────────────────────────────

_W = {}


def _init_worker(fc_dict: dict, fe_dict: dict, designs_dir: str, mode: str, backend: str):
    logging.disable(logging.CRITICAL)
    _W["fam"] = HandFamily(FamilyConfig(**fc_dict))
    _W["fe"] = FastEvalConfig(**fe_dict)
    _W["dir"] = Path(designs_dir)
    _W["mode"] = mode
    _W["backend"] = backend


def _eval_one(task):
    from types import SimpleNamespace
    gen, idx, x = task
    fam, fe = _W["fam"], _W["fe"]
    hand = fam.decode(np.asarray(x))
    wd = _W["dir"] / f"g{gen:03d}_{idx:02d}"
    t0 = time.perf_counter()
    try:
        _, yaml_path = hand.write(wd)
        r = BK.Model(BK.METRIC_CFG, yaml_path, _W["backend"]).proxy(fe).to_dict()
    except Exception as e:  # noqa: BLE001 - a bad design must not kill the run
        r = dict(karma_t=0.0, karma_r=None, top3_voxels=[], stage1_counts=[], n_seeds=0, l_ref_m=0.0, voxel_m=0.0,
                 reach_slack_m=float("inf"), best_gap_m=float("inf"), gen_stats={},
                 elapsed_s=time.perf_counter() - t0, error=f"{type(e).__name__}: {e}")
    return {"gen": gen, "idx": idx, "x": list(map(float, x)), "objective": objective_value(SimpleNamespace(**r), _W["mode"]),
            "result": r, "workdir": str(wd)}


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--objective", default="T", choices=["T", "R", "TR"])
    ap.add_argument("--popsize", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--sigma0", type=float, default=1.0, help="multiplier on the family's per-coordinate stds")
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--max-gens", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n-thumb", type=int, default=4)
    ap.add_argument("--n-index", type=int, default=4)
    ap.add_argument("--x0", default=None, help="JSON file with a genotype list or a design .json (HandSpec) to warm-start from; "
                    "or 'screen:<seed>:<n>' to evaluate n random designs and start from the best feasible one")
    ap.add_argument("--lim-max-deg", type=float, default=75.0, help="per-side joint-limit cap (total range cap = 2x)")
    ap.add_argument("--lim-min-deg", type=float, default=15.0)
    ap.add_argument("--rotation-voxels", type=int, default=None, help="Phase-2 subsample size for R/TR objectives (default 12)")
    ap.add_argument("--final-budget", type=int, default=None)
    ap.add_argument("--n-finalists", type=int, default=None, help="seeds given the full BFS in the proxy (default 2)")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--backend", choices=["jax", "reference"], default=None, help="default: jax if importable")
    ap.add_argument("--proxy", choices=["loop", "rich"], default=None, help="proxy preset (default: rich on jax, loop on reference)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    designs_dir = out / "designs"; designs_dir.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(out / "run.log")])

    import math
    fc = FamilyConfig(n_thumb=a.n_thumb, n_index=a.n_index,
                      lim_max_rad=math.radians(a.lim_max_deg), lim_min_rad=math.radians(a.lim_min_deg))
    fam = HandFamily(fc)
    backend = a.backend or BK.default_backend()
    proxy = a.proxy or ("rich" if backend == "jax" else "loop")
    fe = FastEvalConfig.rich() if proxy == "rich" else FastEvalConfig.loop()
    log.info("backend=%s proxy=%s", backend, proxy)
    if a.objective in ("R", "TR"):
        fe.rotation_voxels = a.rotation_voxels if a.rotation_voxels is not None else 12
    elif a.rotation_voxels:
        fe.rotation_voxels = a.rotation_voxels
    if a.final_budget:
        fe.final_budget = a.final_budget
    if a.n_finalists:
        fe.n_finalists = a.n_finalists

    import cma
    state_path = out / "cma_state.pkl"
    meta_path = out / "meta.json"
    if a.resume and state_path.exists():
        es = pickle.loads(state_path.read_bytes())
        meta = json.loads(meta_path.read_text())
        gen0 = meta["generations_done"]
        log.info("resumed at generation %d", gen0)
    else:
        es = None
        if a.x0 and a.x0.startswith("screen:"):
            x0 = None
        elif a.x0:
            p = Path(a.x0)
            d = json.loads(p.read_text())
            if isinstance(d, dict) and "thumb" in d:
                from optimize.hand_family import HandSpec
                x0 = fam.encode(HandSpec.from_json(p.read_text()))
            elif isinstance(d, dict) and "x" in d:
                x0 = np.array(d["x"], dtype=float)
            else:
                x0 = np.array(d, dtype=float)
        else:
            x0 = fam.anthropomorphic_x0()
        opts = {"bounds": fam.cma_bounds(), "CMA_stds": list(fam.stds), "popsize": a.popsize,
                "seed": a.seed, "verb_disp": 0, "verb_log": 0, "verbose": -9}
        if x0 is not None:
            es = cma.CMAEvolutionStrategy(list(x0), a.sigma0, opts)
        gen0 = 0
        meta = {"args": vars(a), "family": asdict(fc), "fast_eval": asdict(fe), "names": fam.names,
                "backend": backend, "proxy": proxy, "generations_done": 0, "best": None}
        meta_path.write_text(json.dumps(meta, indent=2))

    best = meta.get("best")
    if backend == "jax":
        BK.gpu_env()
    ctx = mp.get_context("forkserver")
    t_start = time.time()
    with ctx.Pool(a.workers, initializer=_init_worker,
                  initargs=(asdict(fc), asdict(fe), str(designs_dir), a.objective, backend),
                  maxtasksperchild=50) as pool, open(out / "evals.jsonl", "a") as evals_f, \
            open(out / "progress.csv", "a") as prog_f:
        if es is None:
            # random screen: uniform in the bounded coordinates, uniform angles in [-pi, pi]
            _, sseed, sn = a.x0.split(":")
            rng = np.random.RandomState(int(sseed))
            X = []
            for _ in range(int(sn)):
                x = np.empty(fam.dim)
                for i, (lo, hi) in enumerate(zip(fam.lower, fam.upper)):
                    x[i] = rng.uniform(lo, hi) if lo is not None else rng.uniform(-np.pi, np.pi)
                X.append(x)
            res = pool.map(_eval_one, [(-1, i, x) for i, x in enumerate(X)], chunksize=1)
            for r in res:
                evals_f.write(json.dumps(r) + "\n")
            objs = [r["objective"] for r in res]
            ib = int(np.argmax(objs))
            log.info("screen: %d random designs, %d feasible, best objective %.5f", len(res),
                     sum(1 for o in objs if o > 0), objs[ib])
            x0 = np.array(res[ib]["x"])
            es = cma.CMAEvolutionStrategy(list(x0), a.sigma0, opts)
            meta["screen"] = {"seed": int(sseed), "n": int(sn), "best_objective": objs[ib],
                              "n_feasible": sum(1 for o in objs if o > 0)}
            meta_path.write_text(json.dumps(meta, indent=2))
        if gen0 == 0:
            prog_f.write("gen,best_obj_gen,mean_obj_gen,best_obj_all,best_T_all,sigma,gen_s,wall_s\n")
        gen = gen0
        while gen < a.max_gens and (time.time() - t_start) < a.hours * 3600:
            tg = time.time()
            X = es.ask()
            tasks = [(gen, i, x) for i, x in enumerate(X)]
            results = pool.map(_eval_one, tasks, chunksize=1)
            objs = np.array([r["objective"] for r in results])
            es.tell(X, list(-objs))  # cma minimizes
            for r in results:
                evals_f.write(json.dumps(r) + "\n")
            evals_f.flush()

            ib = int(np.argmax(objs))
            if best is None or objs[ib] > best["objective"]:
                best = dict(results[ib])
                best["design"] = json.loads(fam.decode(np.array(best["x"])).to_json())
                (out / "best.json").write_text(json.dumps(best, indent=2))
                fam.decode(np.array(best["x"])).write(out / "best_design")
            gen += 1
            meta["generations_done"] = gen; meta["best"] = best
            meta_path.write_text(json.dumps(meta, indent=2))
            state_path.write_bytes(es.pickle_dumps())

            dt = time.time() - tg
            bT = best["result"]["karma_t"]
            n_err = sum(1 for r in results if r["result"]["error"])
            prog_f.write(f"{gen},{objs[ib]:.6f},{objs.mean():.6f},{best['objective']:.6f},{bT:.6f},"
                         f"{es.sigma:.4f},{dt:.1f},{time.time()-t_start:.0f}\n"); prog_f.flush()
            vox = [r["result"]["top3_voxels"][:1] for r in results]
            log.info("gen %3d | obj best %.5f mean %.5f | best-ever %.5f (T=%.5f, vox=%s) | sigma %.3f | "
                     "%d/%d infeasible, %d errors | %.0fs (wall %.1f min)",
                     gen, objs[ib], objs.mean(), best["objective"], bT, best["result"]["top3_voxels"],
                     es.sigma, int((objs <= 0).sum()), len(objs), n_err, dt, (time.time() - t_start) / 60)
    log.info("done: %d generations, best objective %.6f", gen, best["objective"] if best else float("nan"))


if __name__ == "__main__":
    main()
