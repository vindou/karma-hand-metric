"""Phase 10.2 (SPEC.md §11.3): arm A' vs Full-D18 on 24 new starts, judged on a common manipulation benchmark.

The endpoint, test and claim rule are pre-registered in ``docs/phase10_report.md`` "P10.2" (frozen before the first
new run). This module only computes them from the replays written by ``workspace/phase10/chain_102_cpu.sh``:

* ``workspace/phase10/10.2/sim/<seed config>/<contact model>/simulation_<controller>__<arm>_start<k>.json``
  (new starts; exports in ``sim/<seed config>/``), and for the pilot (Phase 9 starts 0-7)
  ``workspace/phase10/10.1/<contact model>/9.2/simulation_<controller>__<arm>_start<k>.json`` (the 9.2 libraries);
* the suites ``workspace/phase10/10.2/suite/<arm>__<start>/suite.json`` and the runs ``<arm>/<start>/run.json``.

    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.powered_comparison --out workspace/phase10/10.2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

BUDGET = 12
NEW_STARTS = list(range(8, 32))            # pre-registered: start8 .. start31
PILOT_STARTS = list(range(0, 8))
ARMS = ("armAprime", "full_d18")
CENTRAL = "20A_2mm"
PILOT_SIM = Path("workspace/phase10/10.1")


def _first_fail(info, ok):
    return next((k for k, i in enumerate(info) if not ok(i)), len(info))


def _duty(i):
    return all(d >= 0.5 for d in i["contact_duty"])


def per_seed_counts(rows, criterion="strict", capped=False, library="three_contact"):
    """{seed: summed per-command count}: primitives executed successfully from the command start up to the first
    failure. ``criterion``: ``strict`` (pose 4 mm / 0.2 rad + every assigned contact at the primitive end, ``n_sim_rot``),
    ``pose`` (``n_pose``), ``duty`` (pose + every lane in contact >= 50 % of the primitive). ``capped``: the 9.2 measure
    (only exactly valid primitives count)."""
    out = {}
    for r in rows:
        if r["library"] != library:
            continue
        if criterion == "strict":
            n = r["n_sim_rot"]
        elif criterion == "pose":
            n = r["n_pose"]
        elif criterion == "duty":
            n = _first_fail(r["info"], lambda i: i["pose_ok"] and _duty(i))
        else:
            raise ValueError(criterion)
        if capped:
            n = min(n, r["n_exact"])
        out[r["seed"]] = out.get(r["seed"], 0) + int(n)
    return out


def endpoint(rows, criterion="strict", capped=False, seeds=None, per="budget", budget=BUDGET):
    """P10.2 primary endpoint ``E`` (``criterion="strict"``, not capped): the summed counts over seeds and commands per
    budgeted seed. ``seeds``: restrict to a subset (P10.3 ``S_task``); ``per="subset"`` divides by the subset size."""
    c = per_seed_counts(rows, criterion, capped)
    if seeds is not None:
        c = {s: v for s, v in c.items() if s in set(seeds)}
    tot = float(sum(c.values()))
    if per == "budget":
        return tot / budget
    n = len(seeds) if seeds is not None else len(c)
    return tot / n if n else 0.0


def load_rows(path: Path, name: str):
    if not path.exists():
        return None
    return json.loads(path.read_text())[name]["rows"]


def sim_path(out: Path, k: int, arm: str, config="default", model=CENTRAL, controller="force"):
    name = f"{arm}_start{k}"
    if k in PILOT_STARTS:
        return PILOT_SIM / model / "9.2" / f"simulation_{controller}__{name}.json", name
    return out / "sim" / config / model / f"simulation_{controller}__{name}.json", name


def export_has_seeds(out: Path, k: int, arm: str, config="default"):
    """A design whose export found no three-contact seed has ``E = 0`` (P10.2)."""
    name = f"{arm}_start{k}"
    if k in PILOT_STARTS:
        p = Path("workspace/phase9/9.2/sim") / f"{name}_trajectories.npz"
    else:
        p = out / "sim" / config / f"{name}_trajectories.npz"
    if not p.exists():
        return None
    return p.stat().st_size >= 1000


def value(out, k, arm, config="default", model=CENTRAL, controller="force", criterion="strict", capped=False):
    has = export_has_seeds(out, k, arm, config)
    if has is None:
        return None                      # export not done
    if has is False:
        return 0.0
    p, name = sim_path(out, k, arm, config, model, controller)
    rows = load_rows(p, name)
    return None if rows is None else endpoint(rows, criterion, capped)


def analyze(out: Path, log=print, starts=NEW_STARTS):
    from dkarma.experiments.admissible_baseline import _paired_stats, _describe
    names = [s["name"] for s in json.loads((out / "starts.json").read_text())["starts"]]
    measures = [
        ("primary", dict(config="default", model=CENTRAL, controller="force", criterion="strict"), "E: strict (pose + every contact at the primitive end), compliant 20A/2 mm, force-floor, default seeds"),
        ("exact_valid", dict(config="default", model=CENTRAL, controller="force", criterion="strict", capped=True), "9.2 measure: exactly valid primitives succeeding (strict), same model and controller"),
        ("pose", dict(config="default", model=CENTRAL, controller="force", criterion="pose"), "pose-only E (compliant, force-floor)"),
        ("duty", dict(config="default", model=CENTRAL, controller="force", criterion="duty"), "E with duty-cycle retention (compliant, force-floor)"),
        ("rigid", dict(config="default", model="rigid", controller="force", criterion="strict"), "E under rigid contact (force-floor)"),
        ("sobol512_rs7", dict(config="sobol512_rs7", model=CENTRAL, controller="force", criterion="strict"), "E under the sobol512_rs7 seed configuration (robustness check)"),
        ("position_compliant", dict(config="default", model=CENTRAL, controller="position", criterion="strict"), "E, compliant, position servo (0.75 mm, A1) [descriptive]"),
        ("position_rigid", dict(config="default", model="rigid", controller="position", criterion="strict"), "E, rigid, position servo [descriptive]"),
    ]
    rec = dict(task="fixed physical (SPEC §2)", endpoint="docs/phase10_report.md P10.2", new_starts=[names[k] for k in starts], measures={}, pilot={}, pooled={})
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    lines = ["# Phase 10.2: arm A′ vs Full-D18 on the common manipulation benchmark", "",
             "Fixed physical task. Planner: the shared fixed-assignment hard-mode rollout (6 translation commands × 4 primitives, 6 rotation commands × 10 "
             "primitives per seed), 12 fresh three-contact seeds. Simulation: MuJoCo 8.6/9.3 scene; contact model and controller per row. Success per "
             "primitive: pose (centre within 4 mm, orientation within 0.2 rad) and every assigned fingertip in contact at the primitive end, counted along "
             "each command until its first failure, regardless of the exact verdict; E = total / 12 budgeted seeds.", ""]
    for key, kw, desc in measures:
        for label, ks in (("new", starts), ("pilot", PILOT_STARTS)):
            if label == "pilot" and kw["config"] != "default":
                continue
            vals = {k: (value(out, k, "armAprime", **kw), value(out, k, "full_d18", **kw)) for k in ks}
            done = [k for k, v in vals.items() if v[0] is not None and v[1] is not None]
            r = dict(per_start={names[k]: dict(armAprime=vals[k][0], full_d18=vals[k][1]) for k in ks}, n_complete=len(done))
            if len(done) >= 2:
                a = [vals[k][0] for k in done]; b = [vals[k][1] for k in done]
                r.update(stats=_paired_stats(a, b), armAprime=_describe(a), full_d18=_describe(b))
            rec["measures" if label == "new" else "pilot"].setdefault(key, r)
    # the frozen decision on the primary endpoint
    pm = rec["measures"]["primary"]
    st = pm.get("stats")
    decision = None
    if st:
        if pm["n_complete"] < 16:
            decision = f"not evaluable: {pm['n_complete']} complete new starts (< 16)"
        elif st["wilcoxon_p"] < 0.05 and st["wins_b"] > st["losses_b"]:
            decision = "claim made: Full KaRMA produces better simultaneous-contact manipulation than a collision-valid original-KaRMA optimum (p < 0.05, Full-D18 ahead)"
        elif st["wilcoxon_p"] < 0.05 and st["wins_b"] < st["losses_b"]:
            decision = "reverse: arm A′ ahead at p < 0.05 (not attributed to the objective without a budget-matched control)"
        else:
            decision = "no claim: p >= 0.05"
    rec["decision"] = decision
    # tables
    lines += ["## Per start (new starts; primary endpoint and secondaries)", "",
              "| start | A′ E | Full E | Full − A′ | A′ / Full: 9.2 measure | pose-only E | rigid E | E (sobol512_rs7) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    M = rec["measures"]
    for k in starts:
        n = names[k]
        g = lambda key, arm: M[key]["per_start"][n][arm]   # noqa: E731
        a, b = g("primary", "armAprime"), g("primary", "full_d18")
        lines.append(f"| {n} | {f2(a)} | {f2(b)} | {('–' if a is None or b is None else f'{b - a:+.2f}')} | {f2(g('exact_valid', 'armAprime'))} / {f2(g('exact_valid', 'full_d18'))} | "
                     f"{f2(g('pose', 'armAprime'))} / {f2(g('pose', 'full_d18'))} | {f2(g('rigid', 'armAprime'))} / {f2(g('rigid', 'full_d18'))} | "
                     f"{f2(g('sobol512_rs7', 'armAprime'))} / {f2(g('sobol512_rs7', 'full_d18'))} |")
    lines += ["", "## Paired comparisons (Full-D18 − arm A′)", "",
              "| measure | starts | A′ mean / median / best / IQR | Full mean / median / best / IQR | wins / ties / losses | mean diff [bootstrap 95 % CI] | Wilcoxon p | sign test p |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for tag, src in (("new starts", rec["measures"]), ("pilot (Phase 9 starts 0–7)", rec["pilot"])):
        for key, kw, desc in measures:
            r = src.get(key)
            if not r or "stats" not in r:
                continue
            s, a, b = r["stats"], r["armAprime"], r["full_d18"]
            lines.append(f"| {desc} — {tag} | {r['n_complete']} | {a['mean']:.2f} / {a['median']:.2f} / {a['best']:.2f} / [{a['iqr'][0]:.2f}, {a['iqr'][1]:.2f}] | "
                         f"{b['mean']:.2f} / {b['median']:.2f} / {b['best']:.2f} / [{b['iqr'][0]:.2f}, {b['iqr'][1]:.2f}] | {s['wins_b']} / {s['ties']} / {s['losses_b']} | "
                         f"{s['mean_diff']:+.2f} [{s['boot_ci95'][0]:+.2f}, {s['boot_ci95'][1]:+.2f}] | {s['wilcoxon_p']:.4f} | {s['sign_test_p']:.4f} |")
    # pooled 32 (descriptive)
    pa, pb = [], []
    for k in PILOT_STARTS + list(starts):
        a, b = value(out, k, "armAprime"), value(out, k, "full_d18")
        if a is not None and b is not None:
            pa.append(a); pb.append(b)
    if len(pa) >= 2:
        rec["pooled"] = dict(n=len(pa), stats=_paired_stats(pa, pb))
        s = rec["pooled"]["stats"]
        lines += ["", f"Pooled pilot + new starts (descriptive only, not a test of the claim): n = {len(pa)}, wins / ties / losses {s['wins_b']} / {s['ties']} / {s['losses_b']}, "
                  f"mean diff {s['mean_diff']:+.2f} [{s['boot_ci95'][0]:+.2f}, {s['boot_ci95'][1]:+.2f}], Wilcoxon p = {s['wilcoxon_p']:.4f}"]
    lines += ["", f"**Frozen decision (P10.2, primary endpoint, new starts):** {decision}", ""]
    # post hoc (after the frozen decision; not part of it): where the difference comes from
    seeds = {}
    for k in starts:
        for arm in ARMS:
            p, name = sim_path(out, k, arm)
            rows = load_rows(p, name) if export_has_seeds(out, k, arm) else []
            seeds[(k, arm)] = dict(found=len({r["seed"] for r in rows or []}), hold=len({r["seed"] for r in rows or [] if r["settle_err_m"] <= 0.004}))
    both = [k for k in starts if seeds[(k, "armAprime")]["found"] > 0 and seeds[(k, "full_d18")]["found"] > 0]
    both_hold = [k for k in starts if seeds[(k, "armAprime")]["hold"] > 0 and seeds[(k, "full_d18")]["hold"] > 0]
    ph = dict(seeds={f"{names[k]}/{arm}": v for (k, arm), v in seeds.items()},
              no_seed={arm: [names[k] for k in starts if seeds[(k, arm)]["found"] == 0] for arm in ARMS},
              no_holding_seed={arm: [names[k] for k in starts if seeds[(k, arm)]["found"] > 0 and seeds[(k, arm)]["hold"] == 0] for arm in ARMS})
    lines += ["## Post hoc (after the frozen decision): where the difference comes from", "",
              f"Three-contact seeds found by the export (default configuration, budget 12): arm A′ none on {len(ph['no_seed']['armAprime'])} of {len(starts)} new starts "
              f"({', '.join(ph['no_seed']['armAprime']) or '–'}), Full-D18 none on {len(ph['no_seed']['full_d18'])}. Seeds found but none holding in simulation: "
              f"arm A′ {len(ph['no_holding_seed']['armAprime'])} ({', '.join(ph['no_holding_seed']['armAprime']) or '–'}), Full-D18 {len(ph['no_holding_seed']['full_d18'])}.", ""]
    for label, ks in (("both arms have >= 1 three-contact seed", both), ("both arms have >= 1 seed that holds in simulation", both_hold)):
        if len(ks) >= 2:
            a = [value(out, k, "armAprime") for k in ks]; b = [value(out, k, "full_d18") for k in ks]
            st2 = _paired_stats(a, b)
            ph.setdefault("subsets", {})[label] = dict(starts=[names[k] for k in ks], stats=st2)
            lines.append(f"* Primary E on the {len(ks)} new starts where {label}: wins / ties / losses {st2['wins_b']} / {st2['ties']} / {st2['losses_b']}, "
                         f"mean diff {st2['mean_diff']:+.2f} [{st2['boot_ci95'][0]:+.2f}, {st2['boot_ci95'][1]:+.2f}], Wilcoxon p = {st2['wilcoxon_p']:.4f}")
    lines.append("")
    rec["post_hoc"] = ph
    # compute
    comp = {}
    for arm in ARMS:
        rs = [json.loads((out / arm / names[k] / "run.json").read_text()) for k in starts if (out / arm / names[k] / "run.json").exists()]
        if rs:
            comp[arm] = dict(n=len(rs), wall_s=_describe([r.get("wall_s", 0.0) for r in rs]),
                             optimizer_time_s=(_describe([r["optimizer_time_s"] for r in rs]) if all("optimizer_time_s" in r for r in rs) else None),
                             evaluations=sorted({r.get("n_evaluations") for r in rs}), n_inadmissible=[r.get("n_inadmissible_candidates") for r in rs if "n_inadmissible_candidates" in r],
                             n_nonfinite=[r.get("n_nonfinite") for r in rs if "n_nonfinite" in r], all_admissible=all(r.get("admissibility", {}).get("admissible", False) for r in rs))
    rec["compute"] = comp
    lines += ["## Compute (GPU, one process at a time; shared with another user's torch job, see the report)", "", "| arm | runs | evaluations | wall clock per run, s (mean [min, max]) | optimizer time, s | every returned design admissible |", "| --- | --- | --- | --- | --- | --- |"]
    for arm, c in comp.items():
        w = c["wall_s"]
        opt = "–" if c["optimizer_time_s"] is None else f"{c['optimizer_time_s']['mean']:.0f}"
        lines.append(f"| {arm} | {c['n']} | {c['evaluations']} | {w['mean']:.0f} [{w['worst']:.0f}, {w['best']:.0f}] | {opt} | {c['all_admissible']} |")
    (out / "comparison.json").write_text(json.dumps(rec, indent=1, default=float))
    (out / "comparison.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase10/10.2")
    args = ap.parse_args()
    analyze(Path(args.out))
    print("done", flush=True)


if __name__ == "__main__":
    main()
