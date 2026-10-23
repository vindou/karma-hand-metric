"""Phase 10.4 (SPEC.md §11.3): selection criterion v2 (frozen in ``docs/phase10_report.md`` "P10.4" and SPEC §16 item 12
before any candidate was evaluated against it) applied beside the original 9.5 criterion, unchanged.

Inputs (fixed physical task):
* suites: ``workspace/phase9/9.2/suite/`` (Phase 9 designs, arm B) and ``workspace/phase10/10.2/suite/`` (10.2 designs);
* ``S_task`` records: ``workspace/phase10/10.3/stask_report.json`` (``robust_grasps report``);
* load sweeps: ``workspace/phase9/9.2/sweeps/`` and ``workspace/phase10/10.2/sweeps/`` (``admissible_baseline sweep``);
* compliant replays (R5): ``workspace/phase10/10.1/20A_2mm/9.2/`` (Phase 9 libraries) and
  ``workspace/phase10/10.2/sim/default/20A_2mm/`` (10.2 libraries), force-floor controller;
  the reference value: ``workspace/phase10/10.1/analysis.json`` (``reference.force.median_9_2_eligible``);
* rigid replays for the original criterion: ``workspace/phase9/9.3``, ``workspace/phase9/9.2/sim``, ``workspace/phase10/10.2/sim/default/rigid``.

    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 --out workspace/phase10/10.4
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 --out workspace/phase10/10.4 --export results/fabrication_candidates --rescore
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from dkarma.experiments import candidate_selection as CS

P9 = Path("workspace/phase9/9.2")
P10 = Path("workspace/phase10/10.2")
OUT = Path("workspace/phase10/10.4")
R2_MIN_TASK = 6
R5_MIN_VALID = 10
R3_REFERENCE = "start/start0_nominal"


def load_suites():
    s = {p.parent.name.replace("__", "/"): json.loads(p.read_text()) for p in sorted((P9 / "suite").glob("*/suite.json"))}
    s.update({p.parent.name.replace("__", "/"): json.loads(p.read_text()) for p in sorted((P10 / "suite").glob("*/suite.json"))})
    return s


def load_sweeps():
    sw = {}
    for root in (P9 / "sweeps", P10 / "sweeps", OUT / "sweeps"):
        if root.exists():
            sw.update({p.stem.replace("__", "/"): json.loads(p.read_text()) for p in sorted(root.glob("*.json"))})
    return sw


def sim_name(key):
    return CS.sim_name(key)


def start_index(key):
    return int(key.split("/")[1].split("_")[0].replace("start", "")) if "/" in key and not key.startswith("start/") else None


def compliant_rows(key, controller="force", model="20A_2mm"):
    """The candidate's 12-seed three-contact library replayed with the central pads (Phase 9 libraries in 10.1, 10.2 libraries in 10.2)."""
    k = start_index(key); name = sim_name(key)
    if k is not None and k >= 8:
        p = P10 / "sim" / "default" / model / f"simulation_{controller}__{name}.json"
    else:
        p = Path("workspace/phase10/10.1") / model / "9.2" / f"simulation_{controller}__{name}.json"
    return json.loads(p.read_text())[name]["rows"] if p.exists() else None


def original_criterion_sims():
    """{controller: {design: {library: separated}}}: the rigid replays the 9.5 criterion uses (9.3, 9.2 12-seed libraries,
    and the 10.2 libraries replayed under rigid contact)."""
    sims = CS.load_simulations()
    root = P10 / "sim" / "default" / "rigid"
    for ctrl in ("position", "force"):
        for p in sorted(root.glob(f"simulation_{ctrl}__*.json")) if root.exists() else []:
            for design, r in json.loads(p.read_text()).items():
                for lib, v in r.get("separated", {}).items():
                    sims.setdefault(ctrl, {}).setdefault(design, {})[lib] = dict(v, source=str(p))
    return sims


def candidates(out10_2: Path = P10):
    """arm B, the 9.5 candidates, and the best design per arm from 10.2 by the primary endpoint E (ties: exact Full score)."""
    from dkarma.experiments.powered_comparison import value, NEW_STARTS
    names = [s["name"] for s in json.loads((out10_2 / "starts.json").read_text())["starts"]]
    suites = load_suites()
    best = {}
    for arm in ("armAprime", "full_d18"):
        rows = []
        for k in NEW_STARTS:
            e = value(out10_2, k, arm)
            key = f"{arm}/{names[k]}"
            if e is None or key not in suites:
                continue
            full = CS.primary(suites[key])["full_mean"]
            rows.append((e, full, key))
        if rows:
            best[arm] = max(rows)[2]
    return ["armB", "full_d18/start0_nominal", "armAprime/start6_sobol9_s0.5"] + [best[a] for a in ("armAprime", "full_d18") if a in best], best


def criterion_v2(keys, suites, stask, sweeps, reference, log=print):
    ref_r3 = stask["designs"].get(R3_REFERENCE, {}).get("mean_over_configs", {}).get("full_task")
    rows = {}
    for key in keys:
        s = suites[key]
        el = CS.eligibility(key, s)
        st = stask["designs"].get(key, {})
        pc = st.get("per_config", {})
        n_task = {c: pc[c]["n_task"] for c in pc}
        r2 = len(n_task) == 3 and all(v >= R2_MIN_TASK for v in n_task.values())
        mo = st.get("mean_over_configs") or {}
        r3 = mo.get("full_task") is not None and ref_r3 is not None and mo["full_task"] >= ref_r3
        rb = CS.robustness(key, s, sweeps.get(key))
        r4 = bool(rb["mass_ok"])
        cr = compliant_rows(key)
        from dkarma.experiments.compliant_contact_study import outcomes
        o = outcomes(cr, "three_contact") if cr is not None else None
        fp = o and o["fp_strict"]
        r5 = bool(o is not None and o["exactly_valid"] >= R5_MIN_VALID and fp is not None and reference is not None and fp <= reference)
        rows[key] = dict(R1=el["eligible"], R1_reasons=el["reasons"], R2=r2, n_task=n_task, R3=r3, full_task_mean=mo.get("full_task"), full_task_sd=mo.get("full_task_sd"),
                         full_all_mean=mo.get("full_all"), R3_reference=ref_r3, R4=r4, sweep=rb["sweep"], R5=r5,
                         compliant_three_contact=(dict(seeds_hold=o["seeds_hold"], exactly_valid=o["exactly_valid"], fp_strict=fp) if o else None),
                         R5_reference=reference, n_task_mean=mo.get("n_task"), passes=bool(el["eligible"] and r2 and r3 and r4 and r5))
    passing = [k for k in keys if rows[k]["passes"]]
    dec = dict(passing=passing, selected=None, tie=False, note="")
    if passing:
        ranked = sorted(passing, key=lambda k: -rows[k]["full_task_mean"])
        sel = ranked[0]
        if len(ranked) > 1:
            sec = ranked[1]
            gap = rows[sel]["full_task_mean"] - rows[sec]["full_task_mean"]
            spread = max(rows[sel]["full_task_sd"] or 0.0, rows[sec]["full_task_sd"] or 0.0)
            dec.update(runner_up=sec, gap=gap, spread=spread)
            if gap < spread:
                dec["tie"] = True
                if (rows[sec]["n_task_mean"] or 0) > (rows[sel]["n_task_mean"] or 0):
                    sel, dec["runner_up"] = sec, sel
                dec["note"] = "S_task Full scores within the seed-configuration spread; mean |S_task| decided"
        dec["selected"] = sel; dec["ranked"] = ranked
    else:
        dec["note"] = "no candidate meets R1-R5; nothing is selected (criterion not loosened)"
    return rows, dec


def two_finger_cost(key, suites):
    two_all = {k: CS.primary(s)["two_mean"] for k, s in suites.items() if not k.startswith("start/") and len(s["per_seed_config"]) >= 3}
    a_two = {k: v for k, v in two_all.items() if k.startswith("armAprime/")}
    best = max(two_all, key=two_all.get)
    best_a = max(a_two, key=a_two.get)
    return dict(design=key, its_two_score=two_all.get(key), best_two_score=two_all[best], best_two_design=best, best_armAprime_two_score=a_two[best_a],
                best_armAprime_two_design=best_a)


def select(out: Path = OUT, export: Path | None = None, rescore: bool = False, log=print):
    suites = load_suites()
    keys, best = candidates()
    stask = json.loads(Path("workspace/phase10/10.3/stask_report.json").read_text())
    ref = json.loads(Path("workspace/phase10/10.1/analysis.json").read_text())["reference"]["force"]["median_9_2_eligible"]
    sweeps = load_sweeps()
    rows, dec = criterion_v2(keys, suites, stask, sweeps, ref, log)
    # the original 9.5 criterion, unchanged, on the same candidates (rigid-contact simulation, both controllers)
    sims = original_criterion_sims()
    orig = {}
    for key in keys + (["armA"] if "armA" in suites else []):
        s = suites[key]
        e, rb, ag = CS.eligibility(key, s), CS.robustness(key, s, sweeps.get(key)), CS.agreement(key, sims)
        orig[key] = dict(primary=CS.primary(s), eligibility=e, robustness=rb, agreement=ag, passes=bool(e["eligible"] and rb["ok"] and ag["ok"]))
    o_pass = [k for k in keys if orig[k]["passes"]]
    orig_dec = dict(passing=o_pass, selected=(max(o_pass, key=lambda k: orig[k]["primary"]["full_mean"]) if o_pass else None))
    ref_key = dec["selected"] or max(keys, key=lambda k: rows[k]["full_task_mean"] or 0.0)
    rec = dict(criterion_v2="docs/phase10_report.md P10.4 (SPEC §16 item 12; frozen 2026-10-14 15:50 EDT, subject to the owner's review)",
               original="docs/phase9_report.md §9.5 (unchanged)", task="fixed physical (SPEC §2)", candidates=keys, best_10_2=best,
               thresholds=dict(R2_min_task=R2_MIN_TASK, R3_reference=R3_REFERENCE, R5_reference_fp=ref, R5_min_valid=R5_MIN_VALID,
                               R4=dict(graceful_mass_retention=CS.GRACEFUL_MASS_RETENTION)),
               v2=dict(rows=rows, decision=dec), original_criterion=dict(rows=orig, decision=orig_dec), two_finger_cost=two_finger_cost(ref_key, suites))
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection_v2.json").write_text(json.dumps(rec, indent=1, default=float))
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    f4 = lambda x: "–" if x is None else f"{x:.4f}"   # noqa: E731
    lines = ["# Phase 10.4: criterion v2 and the original 9.5 criterion, side by side (fixed physical task)", "",
             f"Candidates: {keys}. Best 10.2 design per arm by E: {best}.", "",
             f"Criterion v2 thresholds: R2 |S_task| >= {R2_MIN_TASK} under every configuration; R3 S_task Full score (mean over configs) >= {f4(rows[keys[0]]['R3_reference'])} "
             f"(anthropomorphic start's own S_task score); R4 the 9.5 load sweep; R5 compliant three-contact strict FP (force-floor, 20A/2 mm) <= {f2(ref)} with >= {R5_MIN_VALID} exactly valid primitives.", "",
             "| candidate | R1 | R2: |S_task| per config | R3: Full on S_task (mean ± sd) vs all seeds | R4: 70 g retention (seeds holding) | R5: compliant FP (valid) | v2 | original 9.5 (1 / 3 / 4) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in keys:
        r = rows[k]; o = orig[k]; sw = r["sweep"] or {}
        ct = r["compliant_three_contact"] or {}
        lines.append(f"| {k} | {r['R1']} | {r['n_task']} → {r['R2']} | {f4(r['full_task_mean'])} ± {f4(r['full_task_sd'])} vs {f4(r['full_all_mean'])} → {r['R3']} | "
                     f"{f2(sw.get('mass_retention'))} ({sw.get('heavy_seeds_hold')}/{sw.get('seeds')}) → {r['R4']} | {f2(ct.get('fp_strict'))} ({ct.get('exactly_valid')}) → {r['R5']} | "
                     f"**{r['passes']}** | {o['eligibility']['eligible']} / {o['robustness']['ok']} / {o['agreement']['ok']} → {o['passes']} |")
    lines += ["", f"Criterion v2 decision: {json.dumps(dec, default=float)}", f"Original criterion decision: {json.dumps(orig_dec, default=float)}",
              f"Two-finger cost (of the v2 selection, or of the best v2 candidate by R3 if none): {json.dumps(rec['two_finger_cost'], default=float)}"]
    (out / "selection_v2.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    if export is not None and dec["selected"]:
        CS.export_candidate(dec["selected"], suites[dec["selected"]], dict(criterion=rec["criterion_v2"], task=rec["task"], rows={dec["selected"]: rows[dec["selected"]]},
                                                                           decision=dec), export, rescore, log)
        write_numbers(Path(export) / f"selected_{dec['selected'].replace('/', '_')}", rec, dec["selected"])
    return rec


def write_numbers(dst: Path, rec: dict, key: str):
    """Phase 10 numbers beside the exported specification: criterion v2's row and decision, the original criterion's
    verdict, the two-finger cost, the 10.2 endpoint and the 10.3 S_task record of the selected design."""
    from dkarma.experiments.powered_comparison import value
    k = start_index(key)
    stask = json.loads(Path("workspace/phase10/10.3/stask_report.json").read_text())["designs"].get(key)
    out = dict(design=key, criterion_v2=rec["criterion_v2"], original_criterion=rec["original"], task=rec["task"], thresholds=rec["thresholds"],
               v2_row=rec["v2"]["rows"][key], v2_decision=rec["v2"]["decision"],
               original_criterion_row=rec["original_criterion"]["rows"].get(key), original_criterion_decision=rec["original_criterion"]["decision"],
               two_finger_cost=rec["two_finger_cost"],
               endpoint_10_2=(dict(E_default=value(P10, k, key.split("/")[0]), E_sobol512_rs7=value(P10, k, key.split("/")[0], config="sobol512_rs7"),
                                   E_rigid=value(P10, k, key.split("/")[0], model="rigid")) if k is not None else None),
               stask_10_3=stask, provenance="docs/phase10_report.md §10.2-10.4; workspace/phase10/10.4/selection_v2.json")
    (Path(dst) / "phase10_numbers.json").write_text(json.dumps(out, indent=1, default=float))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--export", default=None)
    ap.add_argument("--rescore", action="store_true")
    args = ap.parse_args()
    select(Path(args.out), Path(args.export) if args.export else None, args.rescore)
    print("done", flush=True)


if __name__ == "__main__":
    main()
