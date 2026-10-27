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

Phase 11 (SPEC §11.4 item 11.5, ``--phase-dir workspace/phase11``): the same criterion v2 and the same original 9.5 criterion on
every design returned by 11.3 (``<phase-dir>/11.3/full_d19/<start>``), with the 11.4 suites, ``S_task`` report, sweeps and
replays (``<phase-dir>/11.4/``), R3 from N11 (``start/start0_nominal`` in the 11.4 ``S_task`` report), R5 from 10.1 and eligibility
under D19 (the suites' admissibility records include the servo envelope). ``evaluate`` writes the 11.4 per-design table
(``E`` split into rotations and translations, drop events split into releases and > 20 mm deviations).

    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 evaluate --phase-dir workspace/phase11 --out workspace/phase11/11.4
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 select --phase-dir workspace/phase11 --out workspace/phase11/11.5 --export results/fabrication_candidates --rescore
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


def criterion_v2(keys, suites, stask, sweeps, reference, log=print, rows_of=None):
    """``rows_of(key)``: the candidate's compliant force-floor replay rows (default: the Phase 10 locations)."""
    rows_of = rows_of or compliant_rows
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
        cr = rows_of(key)
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
    best_a = max(a_two, key=a_two.get) if a_two else None          # Phase 11 has no arm A'
    return dict(design=key, its_two_score=two_all.get(key), best_two_score=two_all[best], best_two_design=best,
                best_armAprime_two_score=(a_two[best_a] if best_a else None), best_armAprime_two_design=best_a)


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


# ── another phase directory (Phase 11: the D19 box) ───────────────────────────

P10_SELECTED = Path("results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25")
P10_SELECTED_SIM = P10 / "sim" / "default"
SIM_SETTINGS = (("20A_2mm", "force"), ("rigid", "force"), ("20A_2mm", "position"), ("rigid", "position"))


class PhaseDir:
    """The layout of a phase directory (``workspace/phase11``): ``11.2`` starts and N11, ``11.3`` the optimizer runs
    (``<arm>/<start>/run.json``), ``11.4`` the suites, sweeps, ``S_task`` report and replays (``sim/default/<model>/``)."""

    def __init__(self, root: Path, arm: str = "full_d19", starts="11.2", runs="11.3", evaluation="11.4"):
        self.root = Path(root); self.arm = arm
        self.starts_dir, self.runs, self.ev = self.root / starts, self.root / runs, self.root / evaluation
        self.sim = self.ev / "sim" / "default"

    def names(self):
        return [x["name"] for x in json.loads((self.starts_dir / "starts.json").read_text())["starts"]]

    def keys(self):
        """The candidates: every design returned by the optimizer (a run with ``run.json``)."""
        return [f"{self.arm}/{n}" for n in self.names() if (self.runs / self.arm / n / "run.json").exists()]

    def run(self, key):
        p = self.runs / key / "run.json"
        return json.loads(p.read_text()) if p.exists() else None

    def suites(self):
        return {p.parent.name.replace("__", "/"): json.loads(p.read_text()) for p in sorted((self.ev / "suite").glob("*/suite.json"))}

    def sweeps(self):
        root = self.ev / "sweeps"
        return {p.stem.replace("__", "/"): json.loads(p.read_text()) for p in sorted(root.glob("*.json"))} if root.exists() else {}

    def rows(self, key, model="20A_2mm", controller="force"):
        """Replay rows of the design's default-configuration three-contact export; ``[]`` if the export found no seed
        (``E = 0``, P10.2), ``None`` if not run."""
        name = sim_name(key)
        npz = self.sim / f"{name}_trajectories.npz"
        if not npz.exists():
            return None
        if npz.stat().st_size < 1000:
            return []
        p = self.sim / model / f"simulation_{controller}__{name}.json"
        return json.loads(p.read_text())[name]["rows"] if p.exists() else None


def _sim_measures(rows):
    """``E`` and its split, the drop events, and the compliant/rigid outcomes (strict FP over exactly valid primitives of
    holding seeds) of one replay."""
    from dkarma.experiments.powered_comparison import endpoint_split, drop_events
    from dkarma.experiments.compliant_contact_study import outcomes
    if rows is None:
        return None
    if not rows:
        return dict(E=0.0, E_R=0.0, E_T=0.0, drops=None, fp_strict=None, exactly_valid=0, seeds_hold=0, n_seeds=0)
    o = outcomes(rows, "three_contact")
    return dict(endpoint_split(rows), drops=drop_events(rows), fp_strict=o["fp_strict"], exactly_valid=o["exactly_valid"],
                seeds_hold=o["seeds_hold"], n_seeds=o["n_seeds"])


def phase_r3(pd: PhaseDir, log=print):
    """Criterion v2's R3 reference for the phase: N11's (``start/start0_nominal``) exact Full score on its own ``S_task``, mean
    over the three seed configurations; written with a timestamp and the suites present at that moment (P11.4 order)."""
    import time as _t
    from dkarma.experiments import robust_grasps as RG
    RG.SUITE_ROOTS = [pd.ev]
    d, origin = RG.per_seed_data(R3_REFERENCE, pd.ev)
    if d is None or len(d) < 3:
        raise SystemExit(f"{R3_REFERENCE}: no suite with per-seed detail under three seed configurations in {pd.ev}")
    per = {c: RG.subset_measures(d[c]) for c in RG.CONFIGS}
    rec = dict(design=R3_REFERENCE, source=origin, rule="criterion v2 R3 (SPEC §16 item 12), docs/phase11_report.md P11.4",
               per_config={c: dict(n_found=m["n_found"], n_task=m["n_task"], full_all=m["full_all"], full_task=m["full_task"]) for c, m in per.items()},
               full_task_mean=float(np.mean([m["full_task"] for m in per.values()])), full_task_sd=float(np.std([m["full_task"] for m in per.values()])),
               written=_t.strftime("%Y-%m-%d %H:%M:%S %Z"), suites_present=sorted(pd.suites()))
    others = [k for k in rec["suites_present"] if k != R3_REFERENCE]
    rec["computed_before_any_optimized_design_suite"] = not others
    (pd.ev / "r3_reference.json").write_text(json.dumps(rec, indent=1, default=float))
    log(f"[r3] N11 S_task Full {rec['full_task_mean']:.5f} (sd {rec['full_task_sd']:.5f}); other suites present: {others}")
    return rec


def phase_runs(pd: PhaseDir, out: Path, log=print):
    """The 11.3 table (docs/phase11_report.md P11.3): compute, gradient hygiene, the returned iteration and its exact validation,
    and the returned design's admissibility and margins, per start; from ``<runs>/<arm>/<start>/run.json``."""
    rows = {}
    for n in pd.names():
        r = pd.run(f"{pd.arm}/{n}")
        if r is None:
            continue
        a = r["admissibility"]
        rows[n] = dict(wall_s=r["wall_s"], optimizer_time_s=r["optimizer_time_s"], validation_time_s=r["validation_time_s"], n_evaluations=r["n_evaluations"],
                       n_nonfinite=r["n_nonfinite"], n_clipped=r["n_clipped"], best_iter=r["best_iter"], n_start_seeds=r.get("n_start_seeds"),
                       returned_exact=r["best_validation"].get("exact_score"), returned_n_seeds=r["best_validation"].get("n_seeds"),
                       validations=[(v["iter"], v["validation"].get("exact_score"), v["validation"].get("admissible")) for v in r["validation_history"]],
                       start_admissible=(r.get("start_admissibility") or {}).get("admissible"), admissible=a["admissible"], violations=a["violations"],
                       min_base_separation_m=a["min_base_separation_m"], max_reach_m=a["max_reach_m"], margins_m=a.get("margins_m"), lengths_m=a.get("lengths_m"))
    vals = lambda k: np.asarray([r[k] for r in rows.values() if r[k] is not None], float)   # noqa: E731
    desc = {k: dict(mean=float(vals(k).mean()), median=float(np.median(vals(k))), min=float(vals(k).min()), max=float(vals(k).max()))
            for k in ("wall_s", "optimizer_time_s", "validation_time_s", "returned_exact") if vals(k).size}
    rec = dict(arm=pd.arm, n_runs=len(rows), rows=rows, summary=desc, n_admissible=sum(r["admissible"] for r in rows.values()),
               n_nonfinite_total=int(sum(r["n_nonfinite"] for r in rows.values())))
    out.mkdir(parents=True, exist_ok=True)
    (out / "runs.json").write_text(json.dumps(rec, indent=1, default=float))
    mm = lambda x: "–" if x is None else f"{1e3 * x:.1f}"   # noqa: E731
    lines = [f"# {pd.runs.name}: Full KaRMA + Adam under D19, {len(rows)} runs (fixed physical task)", "",
             "| start | wall s | optimizer s | validation s | non-finite / clipped | validations (iter: exact) | returned it, exact (seeds) | admissible | "
             "base sep (margin) mm | shortest servo link (margin) mm | shortest fingertip mm | longest link mm | reach mm |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for n, r in rows.items():
        m = r["margins_m"] or {}; L = r["lengths_m"] or {}
        lines.append(f"| {n} | {r['wall_s']:.0f} | {r['optimizer_time_s']:.0f} | {r['validation_time_s']:.0f} | {r['n_nonfinite']} / {r['n_clipped']} | "
                     + ", ".join(f"{i}: {v:.4f}" for i, v, _ in r["validations"]) + f" | {r['best_iter']}, {r['returned_exact']:.4f} ({r['returned_n_seeds']}) | {r['admissible']} | "
                     f"{mm(r['min_base_separation_m'])} ({mm(m.get('base_separation'))}) | {mm(L.get('shortest_servo_link'))} ({mm(m.get('servo_link_len'))}) | "
                     f"{mm(L.get('shortest_tip'))} | {mm(L.get('longest'))} | {mm(r['max_reach_m'])} |")
    lines += ["", "Summary: " + "; ".join(f"{k} mean {v['mean']:.4g} [{v['min']:.4g}, {v['max']:.4g}]" for k, v in desc.items())
              + f"; admissible {rec['n_admissible']} of {len(rows)}; non-finite gradient iterations {rec['n_nonfinite_total']}."]
    (out / "runs.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def phase10_reference():
    """The Phase 10 selection (Full-D18 start31) on the same measures, for the descriptive comparison."""
    nums = json.loads((P10_SELECTED / "phase10_numbers.json").read_text())
    out = dict(design="full_d18/start31_sobol41_s0.25", v2_row=nums["v2_row"], two_finger_cost=nums["two_finger_cost"], sim={})
    for model, ctrl in SIM_SETTINGS:
        p = P10_SELECTED_SIM / model / f"simulation_{ctrl}__full_d18_start31.json"
        if p.exists():
            out["sim"][f"{model}/{ctrl}"] = _sim_measures(json.loads(p.read_text())["full_d18_start31"]["rows"])
    return out


def phase_evaluate(pd: PhaseDir, out: Path, log=print):
    """The 11.4 table (docs/phase11_report.md P11.4): per design, compute and admissibility (11.3), the suite's exact scores,
    ``S_task``, ``E`` split into rotations and translations, the drop events split into releases and > 20 mm deviations, the
    other three replays and the load sweep. Descriptive only."""
    from dkarma.experiments import robust_grasps as RG
    RG.SUITE_ROOTS, RG.SIM_ROOT = [pd.ev], pd.ev / "sim"
    RG.KEYS = [R3_REFERENCE] + pd.keys()
    stask = RG.report(pd.ev, log=lambda *_: None)
    suites, sweeps = pd.suites(), pd.sweeps()
    rows = {}
    for key in RG.KEYS:
        s = suites.get(key)
        if s is None:
            continue
        run = pd.run(key) or {}
        st = stask["designs"].get(key, {})
        r = dict(admissibility={k: s["admissibility"].get(k) for k in ("admissible", "violations", "min_base_separation_m", "max_reach_m", "margins_m", "lengths_m")},
                 primary=CS.primary(s), stask=dict(n_task_per_config={c: v["n_task"] for c, v in st.get("per_config", {}).items()}, **(st.get("mean_over_configs") or {})),
                 robustness=CS.robustness(key, s, sweeps.get(key)), sim={f"{m}/{c}": _sim_measures(pd.rows(key, m, c)) for m, c in SIM_SETTINGS},
                 run={k: run.get(k) for k in ("wall_s", "optimizer_time_s", "validation_time_s", "best_iter", "n_nonfinite", "n_clipped", "n_evaluations")} if run else None,
                 returned_validation=(run.get("best_validation") if run else None),
                 seeds_2=[c.get("n_seeds_2") for c in s["per_seed_config"].values()])
        rows[key] = r
    p10 = phase10_reference()
    ref_E = p10["sim"]["20A_2mm/force"]
    opt = [k for k in rows if k != R3_REFERENCE and rows[k]["sim"]["20A_2mm/force"] is not None]
    E = lambda k, part: rows[k]["sim"]["20A_2mm/force"][part]   # noqa: E731
    dist = {part: np.asarray([E(k, part) for k in opt]) for part in ("E", "E_R", "E_T")}
    summary = dict(n_designs=len(opt), phase10_selection=dict(E=ref_E["E"], E_R=ref_E["E_R"], E_T=ref_E["E_T"]),
                   distribution={part: (dict(mean=float(v.mean()), median=float(np.median(v)), best=float(v.max()), worst=float(v.min()),
                                             iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))]) if v.size else None) for part, v in dist.items()},
                   n_E_T_above_phase10=int(sum(E(k, "E_T") > ref_E["E_T"] for k in opt)), n_E_R_above_phase10=int(sum(E(k, "E_R") > ref_E["E_R"] for k in opt)),
                   n_E_above_phase10=int(sum(E(k, "E") > ref_E["E"] for k in opt)))
    rec = dict(task="fixed physical (SPEC §2); D19 box (SPEC §8.1)", rules="docs/phase11_report.md P11.4 (descriptive; no comparative claim)",
               settings=dict(E="compliant 20A/2 mm, force-floor controller (9.3 parameters), strict success counted to the first failure regardless of the exact "
                               "verdict, per budgeted seed (12)", drop="centre > 20 mm from the plan or no assigned fingertip in contact; release = no fingertip "
                               "in contact, deviation = > 20 mm off the plan with a fingertip touching", position_servo="0.75 mm squeeze compliant, 0.5 mm rigid"),
               rows=rows, summary=summary, phase10_selection=p10)
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(rec, indent=1, default=float))
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    f4 = lambda x: "–" if x is None else f"{x:.4f}"   # noqa: E731
    lines = ["# Phase 11.4: evaluation of every returned design (fixed physical task, D19 box)", "",
             f"Simulation: {rec['settings']['E']}. Drop events: {rec['settings']['drop']}. Position servo: {rec['settings']['position_servo']}.", "",
             "| design | admissible (margins mm: bases / servo links) | three-contact seeds per config | exact Full, mean ± sd over configs | |S_task| per config | exact Full on S_task | "
             "two-finger exact | E (E_R + E_T) | drops T: release / dev. | drops R: release / dev. | compliant FP (valid) | E rigid force / compliant position / rigid position | "
             "70 g: retention (seeds) | continuous ceiling: score (seeds) | optimizer: wall s / opt s / best it / non-finite |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in rows.items():
        a, pr, stq, sm = r["admissibility"], r["primary"], r["stask"], r["sim"]
        mg = a.get("margins_m") or {}
        c = sm["20A_2mm/force"] or {}
        dr = c.get("drops") or {}
        sw = r["robustness"]["sweep"] or {}
        ru = r["run"] or {}
        oth = " / ".join(f2((sm[x] or {}).get("E")) for x in ("rigid/force", "20A_2mm/position", "rigid/position"))
        lines.append(f"| {k} | {a['admissible']} ({f2((mg.get('base_separation') or 0) * 1e3)} / {f2((mg.get('servo_link_len') or 0) * 1e3)}) | {pr['n_seeds_3']} | "
                     f"{pr['full_mean']:.4f} ± {pr['full_sd_configs']:.4f} | {list(stq.get('n_task_per_config', {}).values())} | {f4(stq.get('full_task'))} | {pr['two_mean']:.4f} | "
                     f"{f2(c.get('E'))} ({f2(c.get('E_R'))} + {f2(c.get('E_T'))}) | {dr.get('T', {}).get('release', '–')} / {dr.get('T', {}).get('deviation', '–')} | "
                     f"{dr.get('R', {}).get('release', '–')} / {dr.get('R', {}).get('deviation', '–')} | {f2(c.get('fp_strict'))} ({c.get('exactly_valid')}) | {oth} | "
                     f"{f2(sw.get('mass_retention'))} ({sw.get('heavy_seeds_hold')}/{sw.get('seeds')}) | {f4(sw.get('continuous'))} ({sw.get('continuous_seeds_hold')}) | "
                     f"{f2(ru.get('wall_s'))} / {f2(ru.get('optimizer_time_s'))} / {ru.get('best_iter', '–')} / {ru.get('n_nonfinite', '–')} |")
    d = summary["distribution"]
    lines += ["", f"Over the {summary['n_designs']} optimized designs: " + "; ".join(
        f"{part} mean {v['mean']:.2f}, median {v['median']:.2f}, best {v['best']:.2f}, IQR [{v['iqr'][0]:.2f}, {v['iqr'][1]:.2f}]" for part, v in d.items() if v),
        f"Phase 10 selection (Full-D18 start31, D18 box): E {ref_E['E']:.2f} = E_R {ref_E['E_R']:.2f} + E_T {ref_E['E_T']:.2f}. Designs above it: "
        f"E_T {summary['n_E_T_above_phase10']}, E_R {summary['n_E_R_above_phase10']}, E {summary['n_E_above_phase10']} of {summary['n_designs']} (descriptive)."]
    (out / "evaluation.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def phase_select(pd: PhaseDir, out: Path, export: Path | None = None, rescore: bool = False, log=print):
    """11.5: criterion v2 (frozen; R3 from N11, R5 from 10.1) and the original 9.5 criterion on every Phase 11 design."""
    suites, sweeps, keys = pd.suites(), pd.sweeps(), pd.keys()
    keys = [k for k in keys if k in suites]
    stask = json.loads((pd.ev / "stask_report.json").read_text())
    r3 = json.loads((pd.ev / "r3_reference.json").read_text())
    ref = json.loads(Path("workspace/phase10/10.1/analysis.json").read_text())["reference"]["force"]["median_9_2_eligible"]
    rows, dec = criterion_v2(keys, suites, stask, sweeps, ref, log, rows_of=lambda k: pd.rows(k))
    r3_consistent = abs(rows[keys[0]]["R3_reference"] - r3["full_task_mean"]) < 1e-12 if keys else None
    sims = CS.load_simulations()                      # the 9.3 two-finger baseline (rigid) for requirement 4
    for ctrl in ("position", "force"):
        for p in sorted((pd.sim / "rigid").glob(f"simulation_{ctrl}__*.json")):
            for design, r in json.loads(p.read_text()).items():
                for lib, v in r.get("separated", {}).items():
                    sims.setdefault(ctrl, {}).setdefault(design, {})[lib] = dict(v, source=str(p))
    orig = {}
    for key in keys:
        s = suites[key]
        e, rb, ag = CS.eligibility(key, s), CS.robustness(key, s, sweeps.get(key)), CS.agreement(key, sims)
        orig[key] = dict(primary=CS.primary(s), eligibility=e, robustness=rb, agreement=ag, passes=bool(e["eligible"] and rb["ok"] and ag["ok"]))
    o_pass = [k for k in keys if orig[k]["passes"]]
    orig_dec = dict(passing=o_pass, selected=None)
    if o_pass:                                        # the 9.5 decision step: higher mean exact Full; ties by the minimum wrench capacity
        ranked = sorted(o_pass, key=lambda k: -orig[k]["primary"]["full_mean"]); sel = ranked[0]
        if len(ranked) > 1:
            sec = ranked[1]
            if orig[sel]["primary"]["full_mean"] - orig[sec]["primary"]["full_mean"] < max(orig[sel]["primary"]["full_sd_configs"], orig[sec]["primary"]["full_sd_configs"]):
                if (orig[sec]["robustness"]["min_capacity"] or 0) > (orig[sel]["robustness"]["min_capacity"] or 0):
                    sel = sec
        orig_dec["selected"] = sel
    ref_key = dec["selected"] or (max(keys, key=lambda k: rows[k]["full_task_mean"] or 0.0) if keys else None)
    rec = dict(criterion_v2="docs/phase10_report.md P10.4 (SPEC §16 item 12; frozen 2026-10-14 15:50 EDT, subject to the owner's review); "
                            "applied as docs/phase11_report.md P11.5 (eligibility under D19)",
               original="docs/phase9_report.md §9.5 (requirements 1, 3, 4; eligibility under D19)", task="fixed physical (SPEC §2); D19 box",
               candidates=keys, thresholds=dict(R2_min_task=R2_MIN_TASK, R3_reference=R3_REFERENCE, R3_value=r3["full_task_mean"], R3_file=str(pd.ev / "r3_reference.json"),
                                                R3_consistent=r3_consistent, R5_reference_fp=ref, R5_min_valid=R5_MIN_VALID,
                                                R4=dict(graceful_mass_retention=CS.GRACEFUL_MASS_RETENTION)),
               v2=dict(rows=rows, decision=dec), original_criterion=dict(rows=orig, decision=orig_dec),
               two_finger_cost=(two_finger_cost(ref_key, {k: v for k, v in suites.items() if k in keys}) if ref_key else None),
               reference_design=ref_key, phase10_selection=phase10_reference())
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(rec, indent=1, default=float))
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    f4 = lambda x: "–" if x is None else f"{x:.4f}"   # noqa: E731
    lines = ["# Phase 11.5: criterion v2 and the original 9.5 criterion, side by side (fixed physical task, D19 box; not equivalent)", "",
             f"Candidates: the {len(keys)} designs returned by 11.3. R3 reference: N11's S_task Full {f4(r3['full_task_mean'])}; R5 reference {f4(ref)}.", "",
             "| candidate | R1 | R2: |S_task| per config | R3: Full on S_task (mean ± sd) vs all seeds | R4: 70 g retention (seeds holding) | R5: compliant FP (valid) | v2 | original 9.5 (1 / 3 / 4) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k in keys:
        r = rows[k]; o = orig[k]; sw = r["sweep"] or {}
        ct = r["compliant_three_contact"] or {}
        lines.append(f"| {k} | {r['R1']} | {r['n_task']} → {r['R2']} | {f4(r['full_task_mean'])} ± {f4(r['full_task_sd'])} vs {f4(r['full_all_mean'])} → {r['R3']} | "
                     f"{f2(sw.get('mass_retention'))} ({sw.get('heavy_seeds_hold')}/{sw.get('seeds')}) → {r['R4']} | {f2(ct.get('fp_strict'))} ({ct.get('exactly_valid')}) → {r['R5']} | "
                     f"**{r['passes']}** | {o['eligibility']['eligible']} / {o['robustness']['ok']} / {o['agreement']['ok']} → {o['passes']} |")
    lines += ["", f"Criterion v2 decision: {json.dumps(dec, default=float)}", f"Original criterion decision: {json.dumps(orig_dec, default=float)}",
              f"Two-finger cost: {json.dumps(rec['two_finger_cost'], default=float)}",
              "Ineligibility reasons: " + "; ".join(f"{k}: {', '.join(rows[k]['R1_reasons'])}" for k in keys if rows[k]["R1_reasons"])]
    (out / "selection.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    if export is not None and dec["selected"]:
        sel = dec["selected"]
        CS.export_candidate(sel, suites[sel], dict(criterion=rec["criterion_v2"], task=rec["task"], rows={sel: rows[sel]}, decision=dec), export, rescore, log)
        ev = json.loads((pd.ev / "evaluation.json").read_text())["rows"].get(sel) if (pd.ev / "evaluation.json").exists() else None
        nums = dict(design=sel, criterion_v2=rec["criterion_v2"], original_criterion=rec["original"], task=rec["task"], thresholds=rec["thresholds"],
                    v2_row=rows[sel], v2_decision=dec, original_criterion_row=orig.get(sel), original_criterion_decision=orig_dec,
                    two_finger_cost=rec["two_finger_cost"], evaluation_11_4=ev, phase10_selection=rec["phase10_selection"],
                    provenance=f"docs/phase11_report.md §11.3-11.5; {out / 'selection.json'}; {pd.ev / 'evaluation.json'}")
        (Path(export) / f"selected_{sel.replace('/', '_')}" / "phase11_numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", choices=["select", "evaluate", "r3", "runs"], default="select",
                    help="select (10.4 / 11.5); with --phase-dir also r3 (N11's R3 reference), runs (the 11.3 table) and evaluate (the 11.4 table)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--export", default=None)
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--phase-dir", default=None, help="a phase directory with the 11.2/11.3/11.4 layout (Phase 11: workspace/phase11)")
    ap.add_argument("--arm", default="full_d19", help="with --phase-dir: the optimizer arm whose designs are the candidates")
    args = ap.parse_args()
    if args.phase_dir is None:
        select(Path(args.out), Path(args.export) if args.export else None, args.rescore)
    else:
        pd = PhaseDir(Path(args.phase_dir), arm=args.arm)
        if args.stage == "r3":
            phase_r3(pd)
        elif args.stage == "runs":
            phase_runs(pd, Path(args.out))
        elif args.stage == "evaluate":
            phase_evaluate(pd, Path(args.out))
        else:
            phase_select(pd, Path(args.out), Path(args.export) if args.export else None, args.rescore)
    print("done", flush=True)


if __name__ == "__main__":
    main()
