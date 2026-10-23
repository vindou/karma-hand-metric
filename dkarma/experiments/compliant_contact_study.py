"""Phase 10.1 (SPEC.md §11.3): does compliant contact resolve the remaining three-contact simulation mismatch?

Replays the 9.3 archives (6 seeds per library) and the 9.2 12-seed libraries in the MuJoCo scene of 8.6/9.3 under
the rigid contact (the bound) and the compliant silicone fingertips of ``compliant_pad`` (the 3 x 3 grid, central
20A / 2 mm, and the diagnostics D1 rigid friction / D2 damping 0.3), with both 9.3 controllers (position servo,
force-floor). The pre-registered outcomes, decision rule and agreement reference are in ``docs/phase10_report.md``
"P10.1"; this module computes them.

    # one process per (contact model, archive, controller); designs in-process (CPU)
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study replay --model 20A_2mm --src 9.3 --controller force
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study stability
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study analyze
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from dkarma.experiments.compliant_pad import load_pads, CENTRAL, GRID

OUT = Path("workspace/phase10/10.1")
PAD_FILE = "dkarma/configs/compliant_pads.json"
GRID_LABELS = [f"{h:g}A_{t * 1e3:g}mm" for h, t in GRID]
# contact model -> (pad label or None for the rigid 8.6/9.3 contact, pad variant)
MODELS = {"rigid": (None, "patch"), **{lab: (lab, "patch") for lab in GRID_LABELS},
          "D1_rigid_friction": (CENTRAL, "rigid_friction"), "D2_zeta0.3": (f"{CENTRAL}_zeta0.3", "patch"),
          "D3_posthoc_patch_friction": (CENTRAL, "patch_friction_only")}     # post hoc (after the 10.1 results): rigid normal law, patch friction
SOURCES = {
    "9.3": (Path("workspace/phase9/9.3"), [("baseline", ()), ("start", ()), ("armB", ()), ("armAprime_start0", ()), ("armA", ("middle",))]),
    "9.2": (Path("workspace/phase9/9.2/sim"), [(f"{a}_start{k}", ()) for a in ("armAprime", "full_d18") for k in range(8)] + [("armB", ()), ("baseline", ())]),
}
GAP_DESIGNS = ("start", "armB", "armAprime_start0")      # three-contact designs with both libraries (9.3 archives)
RESOLVED_GAP = 0.10
NOT_RESOLVED_SHRINK = 0.5
POS_TOL = 0.004
CONTROLLER_SETTINGS = dict(kp=20.0, squeeze_mm=0.5, force_scale=1.0, k_i_m_per_Ns=0.2, k_p_m_per_N=0.0, leak=0.1, f_min_n=0.05)   # 9.3 parameters


def position_squeeze_compliant(root=OUT):
    """Addendum A1: the position-servo squeeze of every compliant replay, re-tuned on the two-finger baseline only
    (``retune.json``); the 9.3 value 0.5 mm until the re-tune exists."""
    fn = root / "retune.json"
    if not fn.exists():
        raise SystemExit("compliant position-servo replays need the A1 re-tune (workspace/phase10/10.1/retune.json)")
    return float(json.loads(fn.read_text())["chosen_squeeze_mm"])


def _print(*a):
    print(*a, flush=True)


def result_path(model, src, controller, design, root=OUT):
    return root / model / src / f"simulation_{controller}__{design}.json"


def replay(model, src, controller, designs=None, root=OUT, log=_print, controller_settings=None):
    """Replay the designs of one archive under one contact model and controller; one JSON per design."""
    from dkarma.core import load_task_spec
    from dkarma.core.actuation import load_actuation_spec
    from dkarma.experiments.simulation_replay import simulate_design, result_record
    task = load_task_spec()
    tau = load_actuation_spec().torque_limit_nm
    label, variant = MODELS[model]
    pad = load_pads(PAD_FILE)[label] if label else None
    archive, entries = SOURCES[src]
    cs = dict(CONTROLLER_SETTINGS, **(controller_settings or {}))
    out = root / model / src
    out.mkdir(parents=True, exist_ok=True)
    contact_model = dict(model="rigid 8.6/9.3 (solref 0.002 1, solimp 0.95 0.99 0.001, condim 4, torsional mu*3 mm)") if pad is None else \
        dict(model=f"compliant pad {label} ({variant} friction)", pad=pad.as_dict(), pad_variant=variant, pad_file=PAD_FILE)
    for name, drop in entries:
        if designs and name not in designs:
            continue
        fn = result_path(model, src, controller, name, root)
        if fn.exists():
            continue
        if not (archive / f"{name}_trajectories.npz").exists() or (archive / f"{name}_trajectories.npz").stat().st_size < 1000:
            log(f"[{model}/{src}/{controller}] {name}: no trajectories in {archive} (no seed found at export), skipped"); continue
        t0 = time.perf_counter()
        rows = simulate_design(out, name, task, tau, kp=cs["kp"], squeeze_mm=cs["squeeze_mm"], drop_fingers=drop, controller=controller,
                               force_scale=cs["force_scale"], k_i_m_per_Ns=cs["k_i_m_per_Ns"], k_p_m_per_N=cs["k_p_m_per_N"], leak=cs["leak"],
                               f_min_n=cs["f_min_n"], stop_at_first_failure=False, pad=pad, pad_variant=variant, archive=archive)
        settings = dict(controller=controller, **cs, settle_s=0.3, sub_s=0.05, pos_tol_m=0.004, rot_tol_rad=0.2, drop_tol_m=0.02, drop_fingers=list(drop),
                        torque_ceiling_nm=tau, contact_model=contact_model, archive=str(archive), model_key=model, wall_s=time.perf_counter() - t0)
        fn.write_text(json.dumps({name: result_record(rows, settings)}, default=float))
        log(f"[{model}/{src}/{controller}] {name}: {len(rows)} trajectories in {time.perf_counter() - t0:.0f}s")


# ── outcomes ────────────────────────────────────────────────────────────────

def _first_fail(info, ok):
    return next((k for k, i in enumerate(info) if not ok(i)), len(info))


def duty_ok(i):
    return all(d >= 0.5 for d in i["contact_duty"])


def outcomes(rows, library):
    """Pre-registered P10.1 outcomes on one library: rates over the exactly valid primitives of the seeds that hold."""
    rs = [r for r in rows if r["library"] == library]
    hold = [r for r in rs if r["settle_err_m"] <= POS_TOL]
    nv = sum(r["n_exact"] for r in hold)

    def cnt(ok):
        return sum(min(_first_fail(r["info"], ok), r["n_exact"]) for r in hold)
    c = dict(strict=cnt(lambda i: i["ok_rot"]), pose=cnt(lambda i: i["pose_ok"]), retain_end=cnt(lambda i: i["retain_ok"]),
             retain_duty=cnt(duty_ok), pose_duty=cnt(lambda i: i["pose_ok"] and duty_ok(i)), no_drop=cnt(lambda i: not i["drop"]))
    return dict(n_seeds=len({r["seed"] for r in rs}), seeds_hold=len({r["seed"] for r in hold}), exactly_valid=nv, counts=c,
                rate={k: (v / nv if nv else None) for k, v in c.items()},
                fp_strict=(1.0 - c["strict"] / nv if nv else None),
                drops_on_valid=sum(1 for r in hold for k, i in enumerate(r["info"]) if k < r["n_exact"] and i["drop"]))


def load_result(model, src, controller, design, root=OUT):
    fn = result_path(model, src, controller, design, root)
    return json.loads(fn.read_text())[design] if fn.exists() else None


def phase9_rows(src, controller, design):
    """The Phase 9 replay of the same archive (rigid contact) for the reproduction check."""
    archive = SOURCES[src][0]
    fn = archive / f"simulation_{controller}__{design}.json"
    return json.loads(fn.read_text())[design]["rows"] if fn.exists() else None


def stability(root=OUT, log=_print):
    """The pre-registered controller-stability rule on the two-finger baseline (9.3: 6 seeds; 9.2: 16 seeds), central
    parameters vs rigid contact: unstable if fewer seeds hold, or a holding seed drops the sphere during an exactly valid
    primitive where it did not under rigid contact."""
    rec = {}
    for ctrl in ("position", "force"):
        verdicts = []
        for src in ("9.3", "9.2"):
            rr, rc = load_result("rigid", src, ctrl, "baseline", root), load_result(CENTRAL, src, ctrl, "baseline", root)
            if rr is None or rc is None:
                continue
            o_r, o_c = outcomes(rr["rows"], "two_finger"), outcomes(rc["rows"], "two_finger")
            drops_r = {(r["seed"], r["kind"], r["command"]) for r in rr["rows"] if r["settle_err_m"] <= POS_TOL and any(i["drop"] for i in r["info"][: r["n_exact"]])}
            drops_c = {(r["seed"], r["kind"], r["command"]) for r in rc["rows"] if r["settle_err_m"] <= POS_TOL and any(i["drop"] for i in r["info"][: r["n_exact"]])}
            unstable = o_c["seeds_hold"] < o_r["seeds_hold"] or bool(drops_c - drops_r)
            verdicts.append(unstable)
            rec.setdefault(ctrl, {})[src] = dict(seeds_hold_rigid=o_r["seeds_hold"], seeds_hold_central=o_c["seeds_hold"], new_drops=sorted(map(list, drops_c - drops_r)),
                                                 strict_rigid=o_r["rate"]["strict"], strict_central=o_c["rate"]["strict"], unstable=unstable)
        rec.setdefault(ctrl, {})["unstable"] = any(verdicts) if verdicts else None
    (root / "stability.json").write_text(json.dumps(rec, indent=1, default=float))
    log(json.dumps(rec, indent=1, default=float))
    return rec


RETUNE_SQUEEZES = ("0.5", "0.75", "1.0", "1.25", "1.5")      # mm, as spelled in the retune/squeeze<q> directories


def retune(root=OUT, log=_print):
    """Addendum A1: the smallest position-servo squeeze passing the stability rule on both two-finger baseline libraries
    at the central parameters (candidates replayed into ``retune/squeeze<q>/``; 0.5 mm in ``retune/squeeze0.5``)."""
    def drops(rows):
        return {(r["seed"], r["kind"], r["command"]) for r in rows if r["settle_err_m"] <= POS_TOL and any(i["drop"] for i in r["info"][: r["n_exact"]])}
    res = {}
    for q in RETUNE_SQUEEZES:
        rec, ok_all = {}, True
        for src in ("9.3", "9.2"):
            rr = load_result("rigid", src, "position", "baseline", root)["rows"]
            rc = load_result(CENTRAL, src, "position", "baseline", root / "retune" / f"squeeze{q}")["rows"]
            o_r, o_c = outcomes(rr, "two_finger"), outcomes(rc, "two_finger")
            new = sorted(drops(rc) - drops(rr))
            ok = o_c["seeds_hold"] >= o_r["seeds_hold"] and not new
            ok_all &= ok
            rec[src] = dict(seeds_hold=o_c["seeds_hold"], seeds_hold_rigid=o_r["seeds_hold"], new_drops=new, strict=o_c["rate"]["strict"],
                            strict_rigid=o_r["rate"]["strict"], settle_fn_median=np.median([x["settle_fn"] for x in rc], 0).round(3).tolist(), passes=ok)
        res[q] = dict(per_library=rec, passes=bool(ok_all))
        log(f"squeeze {q} mm: {'passes' if ok_all else 'fails'} {rec}")
    chosen = next((float(q) for q in res if res[q]["passes"]), None)
    out = dict(procedure="docs/phase10_report.md addendum A1: smallest squeeze in {0.5,0.75,1.0,1.25,1.5} mm passing the stability rule on both two-finger "
                         "baseline libraries at the central parameters", candidates=res, chosen_squeeze_mm=(chosen if chosen is not None else 0.5),
               retune_succeeded=chosen is not None, rigid_squeeze_mm=0.5, data="two-finger baseline only (9.3 6 seeds, 9.2 16 seeds)")
    (root / "retune.json").write_text(json.dumps(out, indent=1, default=float))
    return out


def verdict(gaps_c, gaps_r):
    """P10.1 rule on the designs' gaps at one parameter set (dict design -> gap) against the rigid gaps."""
    if not gaps_c or any(g is None for g in gaps_c.values()):
        return None
    if all(g <= RESOLVED_GAP for g in gaps_c.values()):
        return "resolved"
    if all(gaps_c[d] > NOT_RESOLVED_SHRINK * gaps_r[d] for d in gaps_c):
        return "not resolved"
    return "partial"


def gap_of(model, ctrl, d, root=OUT):
    r = load_result(model, "9.3", ctrl, d, root)
    if r is None:
        return None, None, None
    s2 = outcomes(r["rows"], "two_finger")["rate"]["strict"] or 0.0
    s3 = outcomes(r["rows"], "three_contact")["rate"]["strict"] or 0.0
    return s2 - s3, s2, s3


def analyze(root=OUT, log=_print):
    f2 = lambda x: "–" if x is None else f"{x:.2f}"   # noqa: E731
    rec = dict(models=list(MODELS), central=CENTRAL, grid=GRID_LABELS, outcomes={}, gaps={}, verdicts={}, reference={}, reproduction={}, confusion={})
    lines = ["# Phase 10.1: compliant fingertips vs rigid capsules in the 8.6/9.3 MuJoCo scene", "",
             "Fixed physical task. Controllers with their 9.3 parameters: position servo (kp 20 N·m/rad, 0.5 mm squeeze) and force-floor "
             "(k_i 0.2 m/(N·s), leak 0.1, set-point = exact allocation, floor 0.05 N). Criteria at each primitive end: pose = centre within 4 mm and "
             "orientation within 0.2 rad; retention (end) = every assigned fingertip in contact; retention (duty) = every assigned lane in contact during "
             ">= 50 % of the primitive; drop = centre > 20 mm from the command or no assigned contact. Rates over exactly valid primitives of the seeds "
             "that hold (settle drift <= 4 mm) under that model and controller; trajectories run to their end, stop only at a drop.", ""]
    # reproduction of Phase 9 by the rigid re-run
    n_same = n_all = 0
    for src, (_, entries) in SOURCES.items():
        for name, _ in entries:
            for ctrl in ("position", "force"):
                a, b = load_result("rigid", src, ctrl, name, root), phase9_rows(src, ctrl, name)
                if a is None or b is None:
                    continue
                keys = ("n_sim", "n_sim_rot", "n_pose", "n_retain", "n_drop", "n_exact")
                same = len(a["rows"]) == len(b) and all(all(x[k] == y[k] for k in keys) and np.isclose(x["settle_err_m"], y["settle_err_m"], rtol=0, atol=1e-12)
                                                         for x, y in zip(a["rows"], b))
                rec["reproduction"][f"{src}/{name}/{ctrl}"] = same
                n_all += 1; n_same += same
    lines += [f"Rigid re-run reproduces the Phase 9 replays (first-failure indices of every criterion and settle drift, per trajectory): {n_same}/{n_all}.", ""]
    # outcome tables
    lines += ["## Separated outcomes", "", "| model | archive | design | library | controller | seeds hold | exactly valid | strict | pose | retention (end) | retention (duty) | pose + duty | no drop | strict FP |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model in MODELS:
        for src, (_, entries) in SOURCES.items():
            for name, _ in entries:
                for ctrl in ("position", "force"):
                    r = load_result(model, src, ctrl, name, root)
                    if r is None:
                        continue
                    for lib in sorted({x["library"] for x in r["rows"]}):
                        o = outcomes(r["rows"], lib)
                        rec["outcomes"].setdefault(model, {}).setdefault(src, {}).setdefault(name, {}).setdefault(lib, {})[ctrl] = o
                        if model in ("rigid", CENTRAL, "D1_rigid_friction", "D2_zeta0.3", "D3_posthoc_patch_friction"):
                            rt = o["rate"]
                            lines.append(f"| {model} | {src} | {name} | {lib} | {ctrl} | {o['seeds_hold']}/{o['n_seeds']} | {o['exactly_valid']} | {f2(rt['strict'])} | {f2(rt['pose'])} | "
                                         f"{f2(rt['retain_end'])} | {f2(rt['retain_duty'])} | {f2(rt['pose_duty'])} | {f2(rt['no_drop'])} | {f2(o['fp_strict'])} |")
    lines.append("")
    # gaps and verdicts
    lines += ["## Residual gap (two-finger strict − three-contact strict, same design, model and controller; 9.3 archives)", "",
              "| model | controller | " + " | ".join(f"{d}: 2f / 3c / gap" for d in GAP_DESIGNS) + " | verdict |", "| --- | --- | " + " | ".join("---" for _ in GAP_DESIGNS) + " | --- |"]
    for ctrl in ("force", "position"):
        rigid = {d: gap_of("rigid", ctrl, d, root)[0] for d in GAP_DESIGNS}
        for model in MODELS:
            g = {d: gap_of(model, ctrl, d, root) for d in GAP_DESIGNS}
            if any(v[0] is None for v in g.values()):
                continue
            gaps = {d: v[0] for d, v in g.items()}
            v = None if model == "rigid" or any(x is None for x in rigid.values()) else verdict(gaps, rigid)
            rec["gaps"].setdefault(ctrl, {})[model] = dict(gaps=gaps, strict_2f={d: x[1] for d, x in g.items()}, strict_3c={d: x[2] for d, x in g.items()}, verdict=v)
            lines.append(f"| {model} | {ctrl} | " + " | ".join(f"{g[d][1]:.2f} / {g[d][2]:.2f} / {g[d][0]:+.2f}" for d in GAP_DESIGNS) + f" | {v or '–'} |")
        grid_v = [rec["gaps"].get(ctrl, {}).get(m, {}).get("verdict") for m in GRID_LABELS]
        central_v = rec["gaps"].get(ctrl, {}).get(CENTRAL, {}).get("verdict")
        rec["verdicts"][ctrl] = dict(central=central_v, grid=dict(zip(GRID_LABELS, grid_v)),
                                     robust=(bool(all(x == central_v for x in grid_v)) if all(x is not None for x in grid_v) else None))
    lines += ["", "Decision rule (pre-registered, P10.1): resolved if gap <= 0.10 on every design at the central parameters; not resolved if the gap shrinks "
              "by less than half of its rigid value on every design; partial otherwise; robust if the verdict is the same at all nine grid points. "
              "Headline: force-floor controller.", ""]
    for ctrl in ("force", "position"):
        v = rec["verdicts"].get(ctrl, {})
        lines.append(f"* {ctrl}: central verdict **{v.get('central')}**; grid {v.get('grid')}; robust {v.get('robust')}")
    lines.append("")
    # compliant three-contact agreement reference
    lines += ["## Compliant three-contact agreement reference (central parameters)", "",
              "| archive | design | controller | seeds hold | exactly valid | strict FP (central) | strict FP (rigid) |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for ctrl in ("force", "position"):
        fps = {}
        for src, (_, entries) in SOURCES.items():
            for name, _ in entries:
                oc = rec["outcomes"].get(CENTRAL, {}).get(src, {}).get(name, {}).get("three_contact", {}).get(ctrl)
                orr = rec["outcomes"].get("rigid", {}).get(src, {}).get(name, {}).get("three_contact", {}).get(ctrl)
                if oc is None:
                    continue
                lines.append(f"| {src} | {name} | {ctrl} | {oc['seeds_hold']}/{oc['n_seeds']} | {oc['exactly_valid']} | {f2(oc['fp_strict'])} | {f2(orr and orr['fp_strict'])} |")
                fps[f"{src}/{name}"] = oc
        elig = {k: v for k, v in fps.items() if k.startswith("9.2/") and v["exactly_valid"] >= 10 and v["fp_strict"] is not None}
        all_rates = [v["fp_strict"] for v in fps.values() if v["fp_strict"] is not None]
        pooled_num = sum(v["exactly_valid"] - v["counts"]["strict"] for v in fps.values()); pooled_den = sum(v["exactly_valid"] for v in fps.values())
        base2 = rec["outcomes"].get(CENTRAL, {}).get("9.2", {}).get("baseline", {}).get("two_finger", {}).get(ctrl)
        rec["reference"][ctrl] = dict(per_design={k: v["fp_strict"] for k, v in fps.items()},
                                      median_9_2_eligible=(float(np.median([v["fp_strict"] for v in elig.values()])) if elig else None), n_eligible=len(elig),
                                      eligible=sorted(elig), median_all=(float(np.median(all_rates)) if all_rates else None),
                                      quartiles_all=([float(np.percentile(all_rates, 25)), float(np.percentile(all_rates, 75))] if all_rates else None),
                                      pooled=(pooled_num / pooled_den if pooled_den else None), baseline_two_finger_fp=(base2 and base2["fp_strict"]))
    lines.append("")
    for ctrl in ("force", "position"):
        r = rec["reference"][ctrl]
        lines.append(f"* {ctrl}: median strict FP over the {r['n_eligible']} 9.2 12-seed libraries with >= 10 exactly valid primitives on holding seeds: "
                     f"**{f2(r['median_9_2_eligible'])}**; all three-contact libraries: median {f2(r['median_all'])}, quartiles {r['quartiles_all'] and [round(x, 2) for x in r['quartiles_all']]}, "
                     f"pooled {f2(r['pooled'])}; baseline two-finger FP (16 seeds) {f2(r['baseline_two_finger_fp'])}")
    lines += ["", "Criterion v2 (P10.4, R5) uses the force-floor median over the eligible 9.2 libraries.", ""]
    # confusion matrices (8.6 convention), rigid vs central
    from dkarma.experiments.simulation_replay import confusion
    lines += ["## Confusion matrices (8.6 convention: all seeds, per primitive up to the first exactly rejected one; criterion pose + all assigned contacts)", "",
              "| archive | design | library | controller | rigid: valid succeed / fail | rigid: rejected succeed / fail | central: valid succeed / fail | central: rejected succeed / fail |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for src, (_, entries) in SOURCES.items():
        for name, _ in entries:
            for ctrl in ("position", "force"):
                a, b = load_result("rigid", src, ctrl, name, root), load_result(CENTRAL, src, ctrl, name, root)
                if a is None or b is None:
                    continue
                for lib in sorted({x["library"] for x in a["rows"]}):
                    ca = confusion([x for x in a["rows"] if x["library"] == lib], "n_sim_rot"); cb = confusion([x for x in b["rows"] if x["library"] == lib], "n_sim_rot")
                    rec["confusion"].setdefault(src, {}).setdefault(name, {}).setdefault(lib, {})[ctrl] = dict(rigid=ca, central=cb)
                    lines.append(f"| {src} | {name} | {lib} | {ctrl} | {ca['exact_valid_sim_success']} / {ca['exact_valid_sim_fail']} | {ca['exact_rejected_sim_success']} / {ca['exact_rejected_sim_fail']} | "
                                 f"{cb['exact_valid_sim_success']} / {cb['exact_valid_sim_fail']} | {cb['exact_rejected_sim_success']} / {cb['exact_rejected_sim_fail']} |")
    lines.append("")
    # grid summary of the three-contact strict rate (9.3 gap designs and the 9.2 pooled)
    lines += ["## Sensitivity grid: three-contact strict success, force-floor controller (9.3 gap designs; 9.2 pooled over the 12-seed libraries)", "",
              "| model | " + " | ".join(GAP_DESIGNS) + " | 9.2 pooled strict | 9.2 pooled pose | 9.2 pooled retention (end) | 9.2 pooled retention (duty) |",
              "| --- | " + " | ".join("---" for _ in GAP_DESIGNS) + " | --- | --- | --- | --- |"]
    for model in MODELS:
        o = rec["outcomes"].get(model, {})
        v3 = [o.get("9.3", {}).get(d, {}).get("three_contact", {}).get("force") for d in GAP_DESIGNS]
        p = [x["three_contact"]["force"] for n, x in o.get("9.2", {}).items() if "three_contact" in x and "force" in x["three_contact"]]
        den = sum(x["exactly_valid"] for x in p)
        pool = {k: (sum(x["counts"][k] for x in p) / den if den else None) for k in ("strict", "pose", "retain_end", "retain_duty")}
        rec.setdefault("grid_pooled", {})[model] = pool
        lines.append(f"| {model} | " + " | ".join(f2(x and x["rate"]["strict"]) for x in v3) + f" | {f2(pool['strict'])} | {f2(pool['pose'])} | {f2(pool['retain_end'])} | {f2(pool['retain_duty'])} |")
    lines.append("")
    # post hoc (after the results; not part of the rule): normal law x friction, pooled 9.2 three-contact rates by primitive kind
    lines += ["## Post hoc: normal law x friction model (9.2 12-seed three-contact libraries pooled, by primitive kind)", "",
              "Cells: rigid (rigid normal law, rigid friction: condim 4, torsional 1.8 mm), D1 (pad normal law, rigid friction), "
              "D3 (rigid normal law, patch friction: condim 6, torsional 0.32 mm, rolling 34 um; post hoc), central (pad normal law, patch friction).", "",
              "| model | controller | kind | exactly valid | strict | pose | retention (end) | retention (duty) |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for model in ("rigid", "D1_rigid_friction", "D3_posthoc_patch_friction", CENTRAL):
        for ctrl in ("force", "position"):
            for kind in ("T", "R"):
                tot = dict(valid=0, strict=0, pose=0, retain_end=0, retain_duty=0)
                for name, _ in SOURCES["9.2"][1]:
                    r = load_result(model, "9.2", ctrl, name, root)
                    if r is None:
                        continue
                    o = outcomes([x for x in r["rows"] if x["kind"] == kind], "three_contact")
                    tot["valid"] += o["exactly_valid"]
                    for k in ("strict", "pose", "retain_end", "retain_duty"):
                        tot[k] += o["counts"][k]
                if tot["valid"]:
                    rec.setdefault("posthoc_factorial", {}).setdefault(model, {}).setdefault(ctrl, {})[kind] = tot
                    v = tot["valid"]
                    lines.append(f"| {model} | {ctrl} | {kind} | {v} | {tot['strict'] / v:.2f} | {tot['pose'] / v:.2f} | {tot['retain_end'] / v:.2f} | {tot['retain_duty'] / v:.2f} |")
    lines.append("")
    (root / "analysis.json").write_text(json.dumps(rec, indent=1, default=float))
    (root / "analysis.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["replay", "stability", "retune", "analyze"])
    ap.add_argument("--model", choices=list(MODELS))
    ap.add_argument("--src", choices=list(SOURCES))
    ap.add_argument("--controller", choices=["position", "force"])
    ap.add_argument("--designs", nargs="*", default=None)
    ap.add_argument("--root", default=str(OUT))
    ap.add_argument("--squeeze-mm", type=float, default=None, help="replay: position-servo squeeze (default: 0.5 mm rigid; the A1 re-tuned value for compliant models)")
    args = ap.parse_args()
    root = Path(args.root)
    if args.stage == "replay":
        cs = {}
        if args.squeeze_mm is not None:
            cs["squeeze_mm"] = args.squeeze_mm
        elif args.controller == "position" and MODELS[args.model][0] is not None:
            cs["squeeze_mm"] = position_squeeze_compliant()
        replay(args.model, args.src, args.controller, args.designs, root, controller_settings=cs)
    elif args.stage == "stability":
        stability(root)
    elif args.stage == "retune":
        retune(root)
    else:
        analyze(root)
    print("done", flush=True)


if __name__ == "__main__":
    main()
