"""Phase 10.3 (SPEC.md §11.3): the task-feasible grasp subset ``S_task = {s : min_d alpha_s(d) >= 1}`` at the primary
ceiling (definitions pre-registered in ``docs/phase10_report.md`` "P10.3").

``stask --design <key>``: for one design and every seed configuration, the three-contact seeds (``find_multi_seeds``,
budget 12), their wrench capacities in the 14 standard directions and their per-seed exact Full counts
(``MC.evaluate_fixed_contacts_exact``, exact SLSQP force check at every sub-step). The per-seed counts must equal the
design's suite (``suite.json``) where it stored them, which proves that ``S_task`` is a subset of the suite's seeds.
Phase 10 suites written with ``--per-seed-detail`` already hold the capacities and are read directly.

``report``: proportions, capacity distributions, the exact Full score on ``S_task`` beside the all-seeds score, and the
10.2 primary endpoint on ``S_task`` (export seeds matched to suite seeds by joint configuration).

    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.robust_grasps stask --design armB
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.robust_grasps report

Phase 11 (another phase directory): ``report --suite-root <dir> --sim-root <dir>/sim --starts <starts.json> --arm full_d19``
reads the suites (with per-seed detail) of the start design and of every ``<arm>/<start>`` from ``<suite-root>/suite/`` and
the replays from ``<sim-root>/<config>/<model>/`` (design names ``candidate_selection.sim_name``).
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

OUT = Path("workspace/phase10/10.3")
P9 = Path("workspace/phase9/9.2")
P10 = Path("workspace/phase10/10.2")
CONFIGS = ("default", "sobol512_rs7", "lin7_rs13")
SUITE_ROOTS = None          # ``--suite-root``: searched for suite/<key>/suite.json instead of the Phase 9/10 roots
SIM_ROOT = None             # ``--sim-root``: <root>/<config>/<model>/simulation_<controller>__<name>.json, exports in <root>/<config>/
KEYS = None                 # the designs of ``report`` (default: ``designs_listed()``)
ALPHA_MIN = 1.0
N_CELLS, N_BINS, REF_T, REF_R = 8000, 4608, 0.10, 0.30


def seed_score(cells, bins):
    """Per-seed exact Full score (the suite's formula with lambda = 1, per seed)."""
    return np.asarray(cells, float) / N_CELLS / REF_T + np.asarray(bins, float) / N_BINS / REF_R


def suite_path(key: str) -> Path:
    """Phase 10 designs (``armAprime/start8_...``) live in workspace/phase10/10.2, the others in workspace/phase9/9.2."""
    if SUITE_ROOTS is not None:
        cands = [Path(r) / "suite" / key.replace("/", "__") / "suite.json" for r in SUITE_ROOTS]
        return next((c for c in cands if c.exists()), cands[0])
    p10 = P10 / "suite" / key.replace("/", "__") / "suite.json"
    return p10 if p10.exists() else P9 / "suite" / key.replace("/", "__") / "suite.json"


def run_stask(key: str, out: Path = OUT, log=print):
    """Seeds, capacities and per-seed exact Full counts of one design under the three seed configurations."""
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from dkarma.core import load_task_spec, build_hand_model
    from dkarma.solvers.seed_projection import SeedConfig, find_multi_seeds, assignment
    from dkarma.solvers.rollout import GraspState
    from dkarma.metrics.config import FULL_KARMA
    from dkarma.metrics import multicontact as MC, act_karma as AK
    from dkarma.experiments import admissible_baseline as AB
    fn = out / "stask" / f"{key.replace('/', '__')}.json"
    if fn.exists():
        log(f"[stask/{key}] exists"); return
    fn.parent.mkdir(parents=True, exist_ok=True)
    k = int(key.split("/")[1].split("_")[0].replace("start", "")) if "/" in key else -1
    phase10 = k >= 8
    AB.STARTS_DIR, AB.N_STARTS = (P10, 33) if phase10 else (AB.PHASE84_FULL3, 8)
    topo, params, src = AB.design_by_key(key, P10 if phase10 else P9)
    task = load_task_spec(); sp = task.step_params()
    hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
    fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
    asg3 = assignment(topo, 3, fingers)
    cfg_full = replace(FULL_KARMA, commands=AB.CMDS)
    sp_ = suite_path(key)
    suite = json.loads(sp_.read_text()) if sp_.exists() else None
    rec = dict(design=key, source=src["source"], suite=(str(sp_) if suite else None), task="fixed physical (SPEC §2), primary ceiling 0.108 N·m", per_config={})
    t0 = time.perf_counter()
    for cname in CONFIGS:
        s3 = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=AB.BUDGET_3, **AB.SEED_CONFIGS[cname]))
        n = int(s3.valid.sum())
        alphas = [np.asarray(AK.wrench_capacity_at(hm, GraspState(s3.q[s], s3.p[s], jnp.eye(3)), asg3, sp, act, n_dir=14)[0]).tolist() for s in range(n)]
        r = dict(n_seeds_3=n, seed_q_3=np.asarray(s3.q)[:n].tolist(), seed_p_3=np.asarray(s3.p)[:n].tolist(), wrench_alpha_3=alphas)
        if n:
            ex = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, s3, sp, task, act, cfg_full)
            r.update(full_cells=[len(x["cells"]) for x in ex.per_seed], full_bins=[len(x["bins"]) for x in ex.per_seed],
                     full_seed_force_ok=[bool(x.get("seed_force_ok", True)) for x in ex.per_seed])
        else:
            r.update(full_cells=[], full_bins=[], full_seed_force_ok=[])
        sc = (suite or {}).get("per_seed_config", {}).get(cname)
        if sc is not None:
            r["matches_suite"] = bool(sc.get("n_seeds_3") == n and list(sc.get("full_cells", [])) == r["full_cells"] and list(sc.get("full_bins", [])) == r["full_bins"])
        rec["per_config"][cname] = r
        log(f"  [stask/{key}] {cname}: {n} seeds, |S_task| {sum(min(a) >= ALPHA_MIN for a in alphas)}, matches suite {r.get('matches_suite')}")
    if suite and "wrench_capacity_3" in suite:
        rec["default_capacity_matches_suite"] = bool([round(c["min"], 12) for c in suite["wrench_capacity_3"]] ==
                                                     [round(min(a), 12) for a in rec["per_config"]["default"]["wrench_alpha_3"]])
    rec["wall_s"] = time.perf_counter() - t0; rec["platform"] = jax.default_backend()
    fn.write_text(json.dumps(rec, indent=1, default=float))
    log(f"[stask/{key}] done in {rec['wall_s']:.0f}s")


def per_seed_data(key: str, out: Path = OUT):
    """{config: dict(alpha_min [S], cells [S], bins [S], q [S, J])} from the stask record, else the Phase 10 suite."""
    fn = out / "stask" / f"{key.replace('/', '__')}.json"
    if fn.exists():
        src = json.loads(fn.read_text())["per_config"]; origin = str(fn)
    else:
        sp_ = suite_path(key)
        if not sp_.exists():
            return None, None
        s = json.loads(sp_.read_text())["per_seed_config"]
        if not all("wrench_alpha_3" in c for c in s.values() if c.get("n_seeds_3")):
            return None, None
        src = s; origin = str(sp_)
    res = {}
    for c, r in src.items():
        n = r.get("n_seeds_3", 0)
        res[c] = dict(n=n, alpha_min=[min(a) for a in r.get("wrench_alpha_3", [])], cells=r.get("full_cells", [])[:n], bins=r.get("full_bins", [])[:n],
                      q=r.get("seed_q_3", []), matches_suite=r.get("matches_suite"))
    return res, origin


def subset_measures(d):
    """S_task of one seed configuration and the exact Full score on it (beside the all-seeds score)."""
    n = d["n"]
    st = [s for s in range(n) if d["alpha_min"][s] >= ALPHA_MIN]
    sc = seed_score(d["cells"], d["bins"]) if n else np.zeros(0)
    return dict(n_found=n, n_task=len(st), frac_found=(len(st) / n if n else None), frac_budget=len(st) / 12.0, s_task=st,
                alpha_min=d["alpha_min"], full_all=(float(sc.mean()) if n else 0.0), full_task=(float(sc[st].mean()) if st else 0.0))


def match_export_seeds(design_q, export_rows_q):
    """Map export seed index -> suite seed index by joint configuration (atol 1e-9)."""
    m = {}
    for e, qe in export_rows_q.items():
        for s, qs in enumerate(design_q):
            if np.allclose(np.asarray(qe), np.asarray(qs), atol=1e-9, rtol=0):
                m[e] = s; break
    return m


def export_seed_q(npz: Path):
    """Seed joint configurations of a three-contact export (state 0 of each seed's trajectories)."""
    from dkarma.experiments.simulation_replay import load_trajectories
    out = {}
    for t in load_trajectories(npz.parent, npz.name.replace("_trajectories.npz", "")):
        if t["library"] == "three_contact" and t["seed"] not in out:
            out[t["seed"]] = t["q"][0]
    return out


def _sim_file(key, cname="default", model="20A_2mm", controller="force"):
    """(simulation JSON, design name) of a design's 12-seed three-contact replay."""
    if SIM_ROOT is not None:
        from dkarma.experiments.candidate_selection import sim_name
        name = sim_name(key)
        return Path(SIM_ROOT) / cname / model / f"simulation_{controller}__{name}.json", name
    arm, st = (key.split("/") + [None])[:2]
    if st is None:
        return Path("workspace/phase10/10.1") / model / "9.2" / f"simulation_{controller}__{key}.json", key
    k = int(st.split("_")[0].replace("start", "")); name = f"{arm}_start{k}"
    if k < 8:
        return Path("workspace/phase10/10.1") / model / "9.2" / f"simulation_{controller}__{name}.json", name
    return P10 / "sim" / cname / model / f"simulation_{controller}__{name}.json", name


def endpoint_on_subset(key, d, cname="default", model="20A_2mm", controller="force"):
    """The 10.2 primary endpoint on S_task: (E all seeds, E per robust seed, E per budgeted seed, matched seeds)."""
    from dkarma.experiments.powered_comparison import endpoint
    arm, st = (key.split("/") + [None])[:2]
    if SIM_ROOT is not None:
        sim, name = _sim_file(key, cname, model, controller)
        npz = Path(SIM_ROOT) / cname / f"{name}_trajectories.npz"
        st = "phase_dir"                                          # every design of the phase directory has its own export
    elif st is None:
        name = key; sim = Path("workspace/phase10/10.1") / model / "9.2" / f"simulation_{controller}__{name}.json"; npz = Path("workspace/phase9/9.2/sim") / f"{name}_trajectories.npz"
    else:
        k = int(st.split("_")[0].replace("start", "")); name = f"{arm}_start{k}"
        if k < 8:
            sim = Path("workspace/phase10/10.1") / model / "9.2" / f"simulation_{controller}__{name}.json"; npz = Path("workspace/phase9/9.2/sim") / f"{name}_trajectories.npz"
        else:
            sim = P10 / "sim" / cname / model / f"simulation_{controller}__{name}.json"; npz = P10 / "sim" / cname / f"{name}_trajectories.npz"
    if SIM_ROOT is None and cname != "default" and (st is None or int(st.split("_")[0].replace("start", "")) < 8):
        return None
    if not npz.exists() or not sim.exists():
        return None
    if npz.stat().st_size < 1000:
        return dict(E_all=0.0, E_task_per_robust=0.0, E_task_per_budget=0.0, n_matched=0)
    rows = json.loads(sim.read_text())[name]["rows"]
    m = match_export_seeds(d["q"], export_seed_q(npz))
    sm = subset_measures(d)
    task_export = [e for e, s in m.items() if s in set(sm["s_task"])]
    return dict(E_all=endpoint(rows), E_task_per_robust=endpoint(rows, seeds=task_export, per="subset"),
                E_task_per_budget=endpoint(rows, seeds=task_export), n_matched=len(m), n_export_seeds=len({r["seed"] for r in rows}),
                export_seeds_in_task=sorted(task_export))


def designs_listed():
    """arm B, the anthropomorphic start, the Phase 9 (pilot) designs and every 10.2 design."""
    p9 = [x["name"] for x in json.loads(Path("workspace/phase8/8.4/full3/starts.json").read_text())["starts"]]
    keys = ["armB", "start/start0_nominal"] + [f"{a}/{n}" for a in ("armAprime", "full_d18") for n in p9]
    if (P10 / "starts.json").exists():
        names = [s["name"] for s in json.loads((P10 / "starts.json").read_text())["starts"]]
        keys += [f"{a}/{names[k]}" for k in range(8, 32) for a in ("armAprime", "full_d18")]
    return keys


def report(out: Path = OUT, log=print):
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    f4 = lambda x: "–" if x is None else f"{x:.4f}"   # noqa: E731
    rec = dict(task="fixed physical (SPEC §2), primary ceiling 0.108 N·m, 14 standard wrench directions, S_task = {s : min_d alpha_s(d) >= 1}",
               definitions="docs/phase10_report.md P10.3", designs={})
    lines = [f"# {'Phase 10.3' if KEYS is None else 'Task-feasible grasps'}: task-feasible grasp subset S_task (fixed physical task, primary ceiling)", "",
             "S_task = {s : min_d α_s(d) ≥ 1} over the 14 standard wrench directions at the primary ceiling (35 g weight and weight × radius as scales). "
             "Per seed configuration: |S_task| of the found seeds (and of the 12-seed budget), the exact Full score on all seeds (unchanged, as the suite) and on "
             "S_task alone (same formula, mean over S_task seeds). The subset results are not equivalent to the original criterion.", "",
             "| design | config | seeds found | S_task | of found / of 12 | min α per seed (sorted) | exact Full, all seeds | exact Full, S_task only | per-seed counts = suite |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key in (KEYS if KEYS is not None else designs_listed()):
        d, origin = per_seed_data(key, out)
        if d is None:
            continue
        r = dict(origin=origin, per_config={}, endpoint={})
        for c in CONFIGS:
            if c not in d:
                continue
            m = subset_measures(d[c])
            r["per_config"][c] = m
            am = ", ".join(f"{x:.2f}" for x in sorted(m["alpha_min"]))
            lines.append(f"| {key} | {c} | {m['n_found']} | {m['n_task']} | {f2(m['frac_found'])} / {m['frac_budget']:.2f} | {am} | {f4(m['full_all'])} | {f4(m['full_task'])} | {d[c].get('matches_suite')} |")
        vals = [r["per_config"][c] for c in CONFIGS if c in r["per_config"]]
        if len(vals) == 3:
            r["mean_over_configs"] = dict(full_all=float(np.mean([v["full_all"] for v in vals])), full_task=float(np.mean([v["full_task"] for v in vals])),
                                          full_task_sd=float(np.std([v["full_task"] for v in vals])), n_task=float(np.mean([v["n_task"] for v in vals])),
                                          n_task_min=int(min(v["n_task"] for v in vals)))
        for c in ("default", "sobol512_rs7"):
            if c in d:
                e = endpoint_on_subset(key, d[c], c)
                if e is not None:
                    r["endpoint"][c] = e
        rec["designs"][key] = r
    lines += ["", "## Exact Full score on S_task, mean over the three seed configurations, and the 10.2 primary endpoint on S_task", "",
              "| design | mean |S_task| (min over configs) | exact Full, all seeds | exact Full, S_task | E, all seeds (default) | E on S_task, per robust seed | E on S_task, per budgeted seed | E all / S_task per budgeted seed (sobol512_rs7) |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key, r in rec["designs"].items():
        mo = r.get("mean_over_configs") or {}
        e = r["endpoint"].get("default") or {}; es = r["endpoint"].get("sobol512_rs7") or {}
        lines.append(f"| {key} | {f2(mo.get('n_task'))} ({mo.get('n_task_min', '–')}) | {f4(mo.get('full_all'))} | {f4(mo.get('full_task'))} | {f2(e.get('E_all'))} | "
                     f"{f2(e.get('E_task_per_robust'))} | {f2(e.get('E_task_per_budget'))} | {f2(es.get('E_all'))} / {f2(es.get('E_task_per_budget'))} |")
    # post hoc (after the results; not pre-registered): S_task membership vs holding in simulation (central pad, force-floor,
    # default configuration), pooled over every replayed seed of every listed design
    tab = {(True, True): 0, (True, False): 0, (False, True): 0, (False, False): 0}
    succ = {True: 0, False: 0}
    for key, r in rec["designs"].items():
        e = r["endpoint"].get("default")
        if not e or not e.get("n_export_seeds"):
            continue
        sim, name = _sim_file(key)
        rows = json.loads(sim.read_text())[name]["rows"]
        from dkarma.experiments.powered_comparison import per_seed_counts
        cnt = per_seed_counts(rows)
        for sd in {x["seed"] for x in rows}:
            hold = any(x["settle_err_m"] <= 0.004 for x in rows if x["seed"] == sd)
            task = sd in set(e["export_seeds_in_task"])
            tab[(task, hold)] += 1; succ[task] += cnt.get(sd, 0)
    rec["posthoc_hold_vs_task"] = {f"task={a},hold={b}": v for (a, b), v in tab.items()} | {"E_counts_task": succ[True], "E_counts_not_task": succ[False]}
    lines += ["", "## Post hoc: S_task membership vs holding in simulation (central pad, force-floor, default configuration; all replayed seeds)", "",
              "| | holds (settle drift <= 4 mm) | does not hold |", "| --- | --- | --- |",
              f"| seed in S_task | {tab[(True, True)]} | {tab[(True, False)]} |", f"| seed not in S_task | {tab[(False, True)]} | {tab[(False, False)]} |", "",
              f"Successful primitives (the 10.2 count) on S_task seeds: {succ[True]}; on the other seeds: {succ[False]}.", ""]
    (out / "stask_report.json").write_text(json.dumps(rec, indent=1, default=float))
    (out / "stask_report.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stask", "report"])
    ap.add_argument("--design", default=None)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--suite-root", nargs="*", default=None, help="report: directories with suite/<key>/suite.json (default: the Phase 9/10 ones)")
    ap.add_argument("--sim-root", default=None, help="report: replay root <root>/<config>/<model>/ with the exports in <root>/<config>/")
    ap.add_argument("--starts", default=None, help="report: starts.json of a phase directory; designs start/start0_nominal and <arm>/<start>")
    ap.add_argument("--arm", default="full_d19", help="report: the arm of --starts")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global SUITE_ROOTS, SIM_ROOT, KEYS
    SUITE_ROOTS, SIM_ROOT = args.suite_root, args.sim_root
    if args.starts:
        names = [s["name"] for s in json.loads(Path(args.starts).read_text())["starts"]]
        KEYS = ["start/start0_nominal"] + [f"{args.arm}/{n}" for n in names]
    if args.stage == "stask":
        run_stask(args.design, out)
    else:
        report(out)
    print("done", flush=True)


if __name__ == "__main__":
    main()
