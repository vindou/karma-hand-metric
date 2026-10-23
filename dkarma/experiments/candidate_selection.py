"""Phase 9.5 (SPEC.md §11.2): fabrication-candidate selection by the criterion of
``docs/phase8_report.md`` §8.7(b), amended for D18 and made operational in ``docs/phase9_report.md``
§9.5 (fixed before the 9.2 suites ran).

Inputs (all on the fixed physical task of SPEC §2):

* the 9.2/9.4 suites ``workspace/phase9/9.2/suite/<key>/suite.json`` (12 three-contact and 16 two-finger
  fresh seeds under three seed-search configurations, wrench capacities, original KaRMA scores);
* the robustness sweeps ``workspace/phase9/9.2/sweeps/<key>.json`` (``admissible_baseline sweep``);
* the 9.3 simulations ``workspace/phase9/9.3/simulation_<controller>[__<design>].json``.

Candidates: arm B; the arm A' design with the highest published original KaRMA-T (all-finger gate);
the Full-KaRMA D18 design with the highest primary measure. Arm A is the failure case (ineligible).

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.candidate_selection --out workspace/phase9/9.5
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.candidate_selection --out workspace/phase9/9.5 --export results/fabrication_candidates --rescore
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

SUITE_ROOT = Path("workspace/phase9/9.2/suite")
SWEEP_ROOT = Path("workspace/phase9/9.2/sweeps")
SIM_ROOTS = (Path("workspace/phase9/9.3"), Path("workspace/phase9/9.2/sim"))   # later roots override per library (9.2/sim: 12 seeds)
RUN_ROOT = Path("workspace/phase9/9.2")
GRACEFUL_MASS_RETENTION = 0.5       # Full score at 70 g >= 50 % of the 35 g score (primary ceiling, same seeds)
MIN_WRENCH_CAPACITY = 1.0           # x object weight, all 14 directions, every default-configuration seed


def sim_name(key: str) -> str:
    """Name of a design in the 9.3 simulation files: ``armB``; ``armAprime/start3_sobol3_s0.25`` -> ``armAprime_start3``."""
    if "/" not in key:
        return key
    arm, st = key.split("/")
    return f"{arm}_{st.split('_')[0]}"


def load_suites():
    return {p.parent.name.replace("__", "/"): json.loads(p.read_text()) for p in sorted(SUITE_ROOT.glob("*/suite.json"))}


def primary(s):
    cfgs = s["per_seed_config"]
    vals = [c.get("full_exact_score", 0.0) for c in cfgs.values()]
    two = [c.get("two_exact_score", 0.0) for c in cfgs.values()]
    per_seed_sd = []
    for c in cfgs.values():
        if c.get("full_cells"):
            sm_cells = np.asarray(c["full_cells"], float) / 8000.0 / 0.1 + np.asarray(c.get("full_bins", [0] * len(c["full_cells"])), float) / 4608.0 / 0.3
            per_seed_sd.append(float(np.std(sm_cells)))
    return dict(full_mean=float(np.mean(vals)), full_sd_configs=float(np.std(vals)), full_values=vals,
                full_K_T=[c.get("full_exact_K_T", 0.0) for c in cfgs.values()], full_K_SO3=[c.get("full_exact_K_SO3", 0.0) for c in cfgs.values()],
                per_seed_score_sd=per_seed_sd, two_mean=float(np.mean(two)), two_sd_configs=float(np.std(two)), two_values=two,
                n_seeds_3=[c.get("n_seeds_3", 0) for c in cfgs.values()], n_seeds_2=[c.get("n_seeds_2", 0) for c in cfgs.values()],
                configs=list(cfgs))


def eligibility(key, s):
    p = primary(s)
    reasons = []
    if key == "armA":
        reasons.append("arm A: index and middle bases 17.8 mm apart, not buildable as three fingers (failure case, SPEC §11.2)")
    if not s["admissibility"]["admissible"]:
        reasons.append("inadmissible under D18: " + "; ".join(s["admissibility"]["violations"]))
    if not all(v > 0 for v in p["two_values"]):
        reasons.append(f"no exact two-finger coverage under some seed configuration ({p['two_values']})")
    if not all(n > 0 for n in p["n_seeds_3"]):
        reasons.append(f"no three-contact seed under some seed configuration (found {p['n_seeds_3']} of 12)")
    if len(p["configs"]) < 3:
        reasons.append("scored under fewer than three seed configurations")
    return dict(eligible=not reasons, reasons=reasons)


def robustness(key, s, sweep):
    caps = s.get("wrench_capacity_3", [])
    mins = [c["min"] for c in caps]
    cap_ok = bool(mins) and all(m >= MIN_WRENCH_CAPACITY for m in mins)
    out = dict(n_seeds=len(caps), min_capacity=(min(mins) if mins else None), capacity_ok=cap_ok,
               frac_seeds_ge_1=(float(np.mean([m >= MIN_WRENCH_CAPACITY for m in mins])) if mins else None), sweep=None, mass_ok=None)
    if sweep is not None:
        rows = {(r["kind"], round(r["tau_bar"], 4), round(r["mass_kg"], 4)): r for r in sweep["rows"]}
        base = rows.get(("torque", 0.108, 0.035)); heavy = next((r for r in sweep["rows"] if r["kind"] == "mass"), None)
        cont = rows.get(("torque", 0.043, 0.035)); stall = rows.get(("torque", 0.215, 0.035))
        retention = (heavy["full_exact_score"] / base["full_exact_score"]) if base and heavy and base["full_exact_score"] > 0 else None
        mass_ok = bool(heavy and heavy["seeds_hold"] >= 1 and retention is not None and retention >= GRACEFUL_MASS_RETENTION)
        out.update(sweep=dict(stall=stall and stall["full_exact_score"], primary=base and base["full_exact_score"], continuous=cont and cont["full_exact_score"],
                              heavy=heavy and heavy["full_exact_score"], heavy_seeds_hold=heavy and heavy["seeds_hold"], seeds=sweep["n_seeds_3"],
                              continuous_seeds_hold=cont and cont["seeds_hold"], mass_retention=retention), mass_ok=mass_ok)
    out["ok"] = bool(out["capacity_ok"] and out["mass_ok"])
    return out


def load_simulations():
    """{controller: {design: {library: separated outcomes}}} from ``simulation_<controller>[__<design>].json`` under
    ``SIM_ROOTS``; a later root overrides an earlier one per library (the 9.2 replays use the 9.4 seed budget)."""
    res = {}
    for root in SIM_ROOTS:
        for ctrl in ("position", "force"):
            for p in [root / f"simulation_{ctrl}.json"] + sorted(root.glob(f"simulation_{ctrl}__*.json")):
                if not p.exists():
                    continue
                for design, r in json.loads(p.read_text()).items():
                    for lib, v in r.get("separated", {}).items():
                        res.setdefault(ctrl, {}).setdefault(design, {})[lib] = dict(v, source=str(p))
    return res


def fp_rate(sep_lib):
    """Strict false-positive rate: exactly valid primitives of holding seeds that fail pose or contact retention."""
    fr = sep_lib["frac"]["strict_rot"]
    return None if fr is None else 1.0 - fr


def agreement(key, sims):
    """Requirement 4 per controller: candidate three-contact strict FP rate < baseline two-finger strict FP rate."""
    out = {}
    for ctrl, ds in sims.items():
        base = ds.get("baseline", {}).get("two_finger")
        cand = ds.get(sim_name(key), {})
        b = fp_rate(base) if base else None
        c3 = fp_rate(cand["three_contact"]) if "three_contact" in cand else None
        c2 = fp_rate(cand["two_finger"]) if "two_finger" in cand else None
        out[ctrl] = dict(baseline_two_finger_fp=b, candidate_three_contact_fp=c3, candidate_two_finger_fp=c2,
                         candidate_three_contact_valid=(cand.get("three_contact", {}).get("exact_valid_primitives")),
                         ok=(None if (b is None or c3 is None) else bool(c3 < b)))
    oks = [v["ok"] for v in out.values() if v["ok"] is not None]
    return dict(per_controller=out, ok=(any(oks) if oks else None), evaluated=bool(oks))


def candidates(suites):
    """The three pre-registered candidates (docs/phase9_report.md §9.5)."""
    a_keys = [k for k in suites if k.startswith("armAprime/")]
    f_keys = [k for k in suites if k.startswith("full_d18/")]
    best_a = max(a_keys, key=lambda k: suites[k].get("official_published_allfinger", {}).get("KaRMA_T", -1.0)) if a_keys else None
    best_f = max(f_keys, key=lambda k: primary(suites[k])["full_mean"]) if f_keys else None
    return [k for k in ("armB", best_a, best_f) if k is not None and k in suites], dict(armAprime_by_published_KaRMA_T=best_a, full_d18_by_primary=best_f)


def select(out: Path, export: Path | None = None, rescore: bool = False, log=print):
    suites = load_suites()
    sims = load_simulations()
    sweeps = {p.stem.replace("__", "/"): json.loads(p.read_text()) for p in sorted(SWEEP_ROOT.glob("*.json"))} if SWEEP_ROOT.exists() else {}
    cands, how = candidates(suites)
    rows = {}
    for key in cands + (["armA"] if "armA" in suites else []):
        s = suites[key]
        rows[key] = dict(primary=primary(s), eligibility=eligibility(key, s), robustness=robustness(key, s, sweeps.get(key)),
                         agreement=agreement(key, sims), admissibility=s["admissibility"],
                         official=dict(physical_allfinger=s.get("official_physical_allfinger", {}).get("cube_fraction"),
                                       physical_pair=s.get("official_physical_pair", {}).get("cube_fraction"),
                                       published_allfinger_KaRMA_T=s.get("official_published_allfinger", {}).get("KaRMA_T")))
    passing = [k for k in cands if rows[k]["eligibility"]["eligible"] and rows[k]["robustness"]["ok"] and rows[k]["agreement"]["ok"]]
    decision = dict(candidates=cands, how_chosen=how, passing=passing, selected=None, runner_up=None, tie=False, note="")
    if passing:
        ranked = sorted(passing, key=lambda k: -rows[k]["primary"]["full_mean"])
        decision["ranked"] = ranked
        sel = ranked[0]
        if len(ranked) > 1:
            second = ranked[1]
            gap = rows[sel]["primary"]["full_mean"] - rows[second]["primary"]["full_mean"]
            spread = max(rows[sel]["primary"]["full_sd_configs"], rows[second]["primary"]["full_sd_configs"])
            decision.update(runner_up=second, gap=gap, spread=spread)
            if gap < spread:
                decision["tie"] = True
                if (rows[second]["robustness"]["min_capacity"] or 0) > (rows[sel]["robustness"]["min_capacity"] or 0):
                    sel, decision["runner_up"] = second, sel
                decision["note"] = "primary measures within the seed-configuration spread; minimum wrench capacity decided"
        decision["selected"] = sel
    else:
        decision["note"] = "no candidate meets eligibility, robustness and simulation agreement; nothing is selected (criterion not loosened)"
    two_all = {k: primary(s)["two_mean"] for k, s in suites.items() if not k.startswith("start/") and len(s["per_seed_config"]) >= 3}
    best_two = max(two_all, key=two_all.get) if two_all else None
    a_two = {k: v for k, v in two_all.items() if k.startswith("armAprime/")}
    ref = decision["selected"] or (max(cands, key=lambda k: rows[k]["primary"]["full_mean"]) if cands else None)
    if ref is not None:
        decision["two_finger_cost"] = dict(design=ref, its_two_score=rows[ref]["primary"]["two_mean"], best_two_score=two_all.get(best_two), best_two_design=best_two,
                                           best_armAprime_two_score=(max(a_two.values()) if a_two else None),
                                           best_armAprime_two_design=(max(a_two, key=a_two.get) if a_two else None))
    rec = dict(criterion="docs/phase8_report.md §8.7(b) amended for D18, operational form in docs/phase9_report.md §9.5", task="fixed physical (SPEC §2)",
               thresholds=dict(min_wrench_capacity=MIN_WRENCH_CAPACITY, graceful_mass_retention=GRACEFUL_MASS_RETENTION), rows=rows, decision=decision)
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(rec, indent=1, default=float))
    f4 = lambda x: "–" if x is None else f"{x:.4f}"   # noqa: E731
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    lines = ["# Phase 9.5 selection (fixed physical task; D18: base separation >= 24.1 mm, reach <= 250 mm, all-finger clearance)", "",
             f"Candidates: {cands} ({how}).", "",
             "| design | eligible | primary: Full exact score, mean ± sd over 3 seed configs | two-finger exact score | min wrench capacity (all >= 1?) | Full score at 0.215 / 0.108 / 0.043 N·m, and at 70 g (seeds holding) | mass retention | published KaRMA-T (all-finger) | requirements 1 / 3 / 4 |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in rows.items():
        p, e, rb, ag = r["primary"], r["eligibility"], r["robustness"], r["agreement"]
        sw = rb["sweep"] or {}
        lines.append(f"| {k} | {e['eligible']} | {p['full_mean']:.4f} ± {p['full_sd_configs']:.4f} | {p['two_mean']:.4f} ± {p['two_sd_configs']:.4f} | "
                     f"{f2(rb['min_capacity'])} ({rb['capacity_ok']}) | {f4(sw.get('stall'))} / {f4(sw.get('primary'))} / {f4(sw.get('continuous'))}, {f4(sw.get('heavy'))} ({sw.get('heavy_seeds_hold')}/{sw.get('seeds')}) | "
                     f"{f2(sw.get('mass_retention'))} | {f4(r['official']['published_allfinger_KaRMA_T'])} | {e['eligible']} / {rb['ok']} / {ag['ok']} |")
    lines += ["", "Simulation agreement (strict false-positive rate on exactly valid primitives of holding seeds; candidate three-contact vs baseline two-finger, same controller):", "",
              "| design | controller | candidate three-contact FP | candidate two-finger FP | baseline two-finger FP | passes |", "| --- | --- | --- | --- | --- | --- |"]
    for k, r in rows.items():
        for ctrl, v in r["agreement"]["per_controller"].items():
            lines.append(f"| {k} | {ctrl} | {f2(v['candidate_three_contact_fp'])} | {f2(v['candidate_two_finger_fp'])} | {f2(v['baseline_two_finger_fp'])} | {v['ok']} |")
    lines += ["", "Ineligibility reasons: " + "; ".join(f"{k}: {', '.join(r['eligibility']['reasons'])}" for k, r in rows.items() if r["eligibility"]["reasons"]), "",
              f"Decision: {json.dumps(decision, default=float)}"]
    (out / "selection.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    if export is not None and decision["selected"]:
        export_candidate(decision["selected"], suites[decision["selected"]], rec, export, rescore, log)
    return rec


def export_candidate(key, suite, rec, export: Path, rescore: bool, log):
    """Copy the selected design's specification (URDF, robot YAML, design JSON, actuator spec) and the criterion's
    numbers to ``results/fabrication_candidates/selected_<key>/``; verify that the copy is the scored design."""
    import yaml
    from dkarma.export import design_from_dict
    from dkarma.core.actuation import load_actuation_spec
    src_json = Path(suite["source"])
    src = src_json.parent
    dst = export / f"selected_{key.replace('/', '_')}"
    dst.mkdir(parents=True, exist_ok=True)
    for p in src.glob("*"):
        if p.is_file():
            shutil.copy(p, dst / p.name)
    run_dir = src.parent
    for extra in ("config.yaml", "random_seeds.json", "results_summary.json", "run.json"):
        if (run_dir / extra).exists():
            shutil.copy(run_dir / extra, dst / extra)
    (dst / "actuator_spec.yaml").write_text(yaml.safe_dump({k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in load_actuation_spec().__dict__.items()}, sort_keys=False))
    t0, p0, _ = design_from_dict(json.loads(src_json.read_text()))
    t1, p1, _ = design_from_dict(json.loads((dst / src_json.name).read_text()))
    same = t0 == t1 and all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(p0, p1))
    check = dict(params_identical=bool(same))
    if rescore:
        from dkarma.core import load_task_spec
        from dkarma.experiments.admissible_baseline import suite_d18, SEED_CONFIGS
        task = load_task_spec()
        r = suite_d18(t1, p1, task, task.step_params(), dst / "_rescore", seed_cfgs={"default": SEED_CONFIGS["default"]}, official=False, log=log)
        a = r["per_seed_config"]["default"]; b = suite["per_seed_config"]["default"]
        check.update(rescored_full_exact_score=a.get("full_exact_score"), suite_full_exact_score=b.get("full_exact_score"),
                     rescored_two_exact_score=a.get("two_exact_score"), suite_two_exact_score=b.get("two_exact_score"),
                     reproduces=bool(abs(a.get("full_exact_score", 0) - b.get("full_exact_score", 0)) < 1e-12 and abs(a.get("two_exact_score", 0) - b.get("two_exact_score", 0)) < 1e-12))
        shutil.rmtree(dst / "_rescore", ignore_errors=True)
    (dst / "selection_numbers.json").write_text(json.dumps(dict(candidate=key, criterion=rec["criterion"], task=rec["task"], row=rec["rows"][key],
                                                               decision=rec["decision"], export_check=check), indent=1, default=float))
    log(f"exported {key} -> {dst}: {check}")


def preserve_candidates(rec, suites, export: Path, log=print, keys=None):
    """Copy the specifications of Phase 9 candidates that were **not** selected (design JSON, URDF, robot YAML, the
    run record, the actuator specification) and their criterion numbers to ``<export>/<arm>_<start>/``, so the
    owner can act on them if the criterion is amended; arms A and B are already preserved (Phase 8.7)."""
    import yaml
    from dkarma.core.actuation import load_actuation_spec
    keys = keys or [k for k in rec["decision"]["candidates"] if k not in ("armA", "armB") and k != rec["decision"].get("selected")]
    for key in keys:
        src_json = Path(suites[key]["source"]); src = src_json.parent
        dst = export / sim_name(key)
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.glob("*"):
            if f.is_file():
                shutil.copy(f, dst / f.name)
        for extra in ("config.yaml", "random_seeds.json", "results_summary.json", "run.json"):
            if (src.parent / extra).exists():
                shutil.copy(src.parent / extra, dst / extra)
        (dst / "actuator_spec.yaml").write_text(yaml.safe_dump({k: (v.tolist() if hasattr(v, "tolist") else v)
                                                               for k, v in load_actuation_spec().__dict__.items()}, sort_keys=False))
        (dst / "candidate_numbers.json").write_text(json.dumps(dict(candidate=key, selected=False, criterion=rec["criterion"], task=rec["task"],
                                                                   row=rec["rows"][key], decision=rec["decision"]), indent=1, default=float))
        log(f"preserved (not selected) {key} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase9/9.5")
    ap.add_argument("--export", default=None, help="results/fabrication_candidates: export the selected specification")
    ap.add_argument("--rescore", action="store_true", help="re-score the exported copy (default seed configuration) and compare")
    ap.add_argument("--preserve", default=None, help="results/fabrication_candidates: also copy the candidates that were not selected")
    args = ap.parse_args()
    rec = select(Path(args.out), Path(args.export) if args.export else None, args.rescore)
    if args.preserve:
        preserve_candidates(rec, load_suites(), Path(args.preserve))


if __name__ == "__main__":
    main()
