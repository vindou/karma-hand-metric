"""Phase 9.2 (SPEC.md §11.2): the collision-valid original-KaRMA baseline (arm A') and the key
comparison with Full KaRMA on the admissible design space (D18: finger bases >= 24.1 mm apart,
reach <= 250 mm per finger, all-finger clearance in every self-collision gate).

Sub-commands (one process per start; ``--only k``):

``armAprime``: CMA-ES against the original KaRMA (the ``optimize/`` fast proxy of ``karma_jax``,
Phase 7 settings: population 8, 12 generations, sigma0 0.3, 96 evaluations) from start ``k`` of
the Phase 8.4 ``full3`` start set, with D18 enforced throughout: a candidate violating the base
separation or the reach bound is scored zero without evaluation, and the proxy's self-collision
gate tests every pair of fingers (``build_flat_model(self_collision_all=True)``). Fresh CMA seed
per start. The returned design is the best admissible candidate by the proxy.

``full``: Adam on the Full-KaRMA objective from the same start with the D18 base-separation
penalty, the reach penalty and D18-gated exact validation (an inadmissible iterate scores zero
and is never returned); Phase 8.4 settings (40 iterations, 2 stages, 6 three-contact seeds,
``ADAM_DEFAULTS``) and the Phase 9.1 KKT derivative.

``suite``: the common three-contact physical suite of ``codesign_comparison.suite`` at the Phase
9.4 budgets (>= 16 two-finger and >= 12 three-contact fresh seeds) under three seed-search
configurations, plus the physical original KaRMA (all-finger gate) and the admissibility record,
for the starts, the arm A' and Full-D18 designs, arm A and arm B.

``summary``: tables and paired statistics (sign counts, Wilcoxon) of arm A' vs Full-D18 per start,
the seed-configuration spread and the ranking stability (``summary.md`` / ``summary.json``).

``--box d19`` (Phase 11, SPEC §11.4): the D19 servo envelope (``BOXES``): the ``full3_d19`` objective from N11
(``nominal`` stage: the nominal start and its two fallbacks, frozen in ``<starts-dir>/n11.json``), the D19 bounds,
``D19_CONSTRAINTS`` and the envelope in every hard check; the suite computes no original-KaRMA numbers (``karma_jax``
applies one radius to every capsule). The design key of a Phase 11 run is ``full_d19/<start>``.

    for K in 0 1 2 3 4 5 6 7; do ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline armAprime --only $K --out workspace/phase9/9.2; done
    for K in 0 1 2 3 4 5 6 7; do ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline full --only $K --out workspace/phase9/9.2; done
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline suite --design armAprime/start0_nominal --out workspace/phase9/9.2
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline summary --out workspace/phase9/9.2
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

from dkarma.core import Topology, decode, encode, load_task_spec, build_hand_model, fixed_from_params, anthropomorphic  # noqa: E402
from dkarma.core.actuation import load_actuation_spec  # noqa: E402
from dkarma.core.admissibility import admissibility, MIN_BASE_SEPARATION_M, REACH_MAX_M, D19_ENVELOPE, D19_MIN_BASE_SEPARATION_M  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig, find_seeds, find_multi_seeds, assignment  # noqa: E402
from dkarma.solvers.rollout import GraspState  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, MC_KARMA, FULL_KARMA  # noqa: E402
from dkarma.metrics.commands import CommandConfig  # noqa: E402
from dkarma.metrics import multicontact as MC, act_karma as AK  # noqa: E402
from dkarma.metrics.exact_library import evaluate_exact, official_karma_physical  # noqa: E402
from dkarma.optimization import make_objective, adam, save_run  # noqa: E402
from dkarma.optimization.objectives import exact_validation, D18_CONSTRAINTS, D19_CONSTRAINTS  # noqa: E402
from dkarma.optimization.optimizers import ADAM_DEFAULTS  # noqa: E402
from dkarma.export import write_design, design_from_dict  # noqa: E402
from dkarma.experiments.continuation_study import objective_spec, make_starts, nominal_d19, BOUNDS, BOUNDS_D19  # noqa: E402

PHASE84_FULL3 = Path("workspace/phase8/8.4/full3")
ARM_A = Path("results/fabrication_candidates/armA/armA.design.json")
ARM_B = Path("results/fabrication_candidates/armB/optimized_morphology.design.json")
CMDS = CommandConfig(n_cells=4, n_rot_steps=10)

# Phase 9.4: three seed-search configurations (fresh seeds; the default is the Phase 7/8 one)
SEED_CONFIGS = {
    "default": dict(),
    "sobol512_rs7": dict(n_sobol=512, n_random=96, random_seed=7),
    "lin7_rs13": dict(n_linspace=7, n_sobol=192, n_random=64, random_seed=13, offsets_m=(0.0, -0.0075, 0.0075, -0.0125, 0.0125)),
}
BUDGET_2, BUDGET_3 = 16, 12


STARTS_DIR = PHASE84_FULL3     # ``--starts-dir``: Phase 10.2 continues the 8.4 sequence in workspace/phase10/10.2
N_STARTS = 8                   # ``--n-starts``: 33 in Phase 10.2 (the eight Phase 9 starts + 24 new + the nominal's slot)

# ``--box``: the design box. d18 = Phases 9-10 (24.1 mm bases, uniform 11.8 mm capsules); d19 = Phase 11's servo envelope
BOXES = {
    "d18": dict(objective="full3", bounds=BOUNDS, constraints=D18_CONSTRAINTS, envelope=None, official=True, arm="full_d18",
                min_base_separation_m=MIN_BASE_SEPARATION_M),
    "d19": dict(objective="full3_d19", bounds=BOUNDS_D19, constraints=D19_CONSTRAINTS, envelope=D19_ENVELOPE, official=False, arm="full_d19",
                min_base_separation_m=D19_MIN_BASE_SEPARATION_M),
}
BOX = BOXES["d18"]
RUNS_DIR = None                # ``--runs-dir``: where ``<arm>/<start>/optimized_morphology`` live, if not ``--out`` (Phase 11: 11.3 runs, 11.4 evaluation)
# Phase 11 N11 fallbacks (docs/phase11_report.md P11.2): pitches, then fingertip extensions, tried in order
N11_PITCHES_M = (0.040, 0.045, 0.050, 0.055, 0.060)
N11_TIP_EXTRA_M = (0.0, 0.005, 0.010, 0.015, 0.020)
N11_MIN_SEEDS = 3


def box_spec():
    """The objective spec of the current box (``full3_d19`` reads the frozen N11 from ``STARTS_DIR/n11.json``)."""
    if BOX["objective"] == "full3_d19":
        n11 = json.loads((Path(STARTS_DIR) / "n11.json").read_text())["chosen"]
        return objective_spec("full3_d19", n11=dict(pitch_m=n11["pitch_m"], tip_extra_m=n11["tip_extra_m"]))
    return objective_spec(BOX["objective"])


def load_starts(task, log=print):
    spec = box_spec()
    starts, unravel = make_starts(spec, task, N_STARTS, 0.25, STARTS_DIR, seed=321, log=log, scales=[0.25] * 4 + [0.5] * 16)
    return spec, starts, unravel


def run_nominal(log=print):
    """Phase 11.2: N11 and its fallbacks in the frozen order (docs/phase11_report.md P11.2); writes ``STARTS_DIR/n11.json``.
    1. the pitch: the first of ``N11_PITCHES_M`` at which N11 passes the hard D19 check;
    2. the fingertips: the first of ``N11_TIP_EXTRA_M`` with >= 3 three-contact seeds (default configuration, budget 12),
       with the seed-search attempt counts at every step."""
    task = load_task_spec(); sp = task.step_params()
    topo = Topology.three_finger(4)
    fn = Path(STARTS_DIR) / "n11.json"
    if fn.exists():
        log(f"[nominal] {fn} exists, skipping"); return
    fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
    rec = dict(rule="docs/phase11_report.md P11.2", task="fixed physical (SPEC §2)", attempts=[], chosen=None)
    pitch = None
    for pm in N11_PITCHES_M:
        adm = admissibility(nominal_d19(topo, pitch_m=pm), topo, envelope=D19_ENVELOPE)
        rec["attempts"].append(dict(step="pitch", pitch_m=pm, admissible=adm["admissible"], violations=adm["violations"],
                                    min_base_separation_m=adm["min_base_separation_m"], base_separations=adm["base_separations"]))
        log(f"[nominal] pitch {pm * 1e3:.0f} mm: admissible {adm['admissible']} {adm['violations']}")
        if adm["admissible"]:
            pitch = pm; break
    if pitch is not None:
        for ex in N11_TIP_EXTRA_M:
            params = nominal_d19(topo, pitch_m=pitch, tip_extra_m=ex)
            adm = admissibility(params, topo, envelope=D19_ENVELOPE)
            hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
            t0 = time.perf_counter()
            s3 = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=BUDGET_3))
            n3 = int(s3.valid.sum())
            rec["attempts"].append(dict(step="tips", pitch_m=pitch, tip_extra_m=ex, admissible=adm["admissible"], violations=adm["violations"], n_seeds_3=n3,
                                        seed_stats_3={k: v for k, v in s3.stats.items() if isinstance(v, (int, float))},
                                        link_len_mm={f: [round(1e3 * x, 6) for x in np.asarray(params.link_len)[i, :topo.n_joints[i]].tolist()] for i, f in enumerate(topo.fingers)},
                                        seconds=time.perf_counter() - t0))
            log(f"[nominal] pitch {pitch * 1e3:.0f} mm, fingertips +{ex * 1e3:.0f} mm: admissible {adm['admissible']}, {n3} three-contact seeds; {rec['attempts'][-1]['seed_stats_3']}")
            if adm["admissible"] and n3 >= N11_MIN_SEEDS:
                rec["chosen"] = dict(pitch_m=pitch, tip_extra_m=ex, n_seeds_3=n3); break
    rec["stopped"] = rec["chosen"] is None
    rec["platform"] = jax.default_backend()
    Path(STARTS_DIR).mkdir(parents=True, exist_ok=True)
    if rec["chosen"]:                         # the start design's files (URDF with the capsule visuals, robot YAML, design JSON)
        write_design(topo, nominal_d19(topo, pitch_m=rec["chosen"]["pitch_m"], tip_extra_m=rec["chosen"]["tip_extra_m"]), Path(STARTS_DIR) / "n11", "n11")
    fn.write_text(json.dumps(rec, indent=1, default=float))
    log(f"[nominal] chosen {rec['chosen']}" if rec["chosen"] else "[nominal] neither fallback yields a start: Phase 11 stops here (P11.2)")


def run_starts(log=print):
    """Phase 10.2: the 8.4 start procedure continued past the eight Phase 9 starts (``N_STARTS`` starts into
    ``STARTS_DIR/starts.json``); the first eight must reproduce ``workspace/phase8/8.4/full3/starts.json``.
    Phase 11 (``--box d19``): the same procedure from N11 in the D19 box; the check records every start's D19
    admissibility and the rejected Sobol points instead."""
    task = load_task_spec()
    spec, starts, _ = load_starts(task, log)                 # computes and caches STARTS_DIR/starts.json
    if BOX["envelope"] is not None:
        rec = json.loads((Path(STARTS_DIR) / "starts.json").read_text())
        adm = {}
        for name, phi in starts:
            a = admissibility(decode(phi, spec["topo"], BOX["bounds"], fixed_from_params(spec["params0"])), spec["topo"], envelope=BOX["envelope"])
            adm[name] = dict(admissible=a["admissible"], violations=a["violations"], min_base_separation_m=a["min_base_separation_m"], max_reach_m=a["max_reach_m"],
                             margins_m=a["margins_m"], lengths_m=a["lengths_m"])
        rej = [p for p in rec.get("points", []) if not p["accepted"]]
        out = dict(n_starts=len(starts), names=[n for n, _ in starts], admissibility=adm, rejected_points=rej, box="d19")
        (Path(STARTS_DIR) / "starts_check.json").write_text(json.dumps(out, indent=1, default=float))
        log(f"[starts] {len(starts)} starts in {STARTS_DIR}; D19-admissible {sum(v['admissible'] for v in adm.values())}; "
            f"rejected Sobol points {[(p['sobol_index'], p['n_seeds']) for p in rej]}")
        return
    ref = json.loads((PHASE84_FULL3 / "starts.json").read_text())["starts"]
    new = json.loads((Path(STARTS_DIR) / "starts.json").read_text())["starts"]
    same = [(r["name"], s["name"], bool(r["name"] == s["name"] and np.array_equal(np.asarray(r["x"]), np.asarray(s["x"])))) for r, s in zip(ref, new)]
    rec = dict(n_starts=len(new), names=[s["name"] for s in new], phase9_reproduced=bool(len(same) == len(ref) and all(s[2] for s in same)), phase9_check=same)
    (Path(STARTS_DIR) / "starts_check.json").write_text(json.dumps(rec, indent=1))
    log(f"[starts] {len(new)} starts in {STARTS_DIR}; the eight Phase 9 starts reproduced: {rec['phase9_reproduced']} {same}")
    if not rec["phase9_reproduced"]:
        raise SystemExit("the first eight starts differ from the 8.4 start set; see starts_check.json")


# ── arm A' ───────────────────────────────────────────────────────────────────

def cmaes_original_karma_d18(topo, params0, workdir, popsize=8, n_gens=12, sigma0=0.3, seed=0, log=print):
    """Arm A' (Phase 9.2): the Phase 7 ``codesign_comparison.cmaes_original_karma`` with D18 enforced on
    every candidate (hard base-separation and reach check before evaluation, scored zero if violated;
    all-finger self-collision gate inside the proxy). Returns ``(params, history, candidates)``."""
    import cma
    from karma.config import load_config
    from karma.robot import load_robot
    from karma_jax.model import build_flat_model
    from karma_jax.fast import evaluate as fast_eval, FastConfig
    from dkarma.fixtures import METRIC_CFG
    fixed = fixed_from_params(params0)
    x0, unravel = jax.flatten_util.ravel_pytree(encode(params0, topo, BOUNDS))
    fc = FastConfig(n_linspace_per_joint=5, n_sobol=256, n_random=64, n_offsets=5, dedup_rad=0.1, max_seeds=48,
                    stage1_budget=15, n_finalists=3, final_budget=10000)
    cands = []

    def fitness(x, k):
        params = decode(unravel(jnp.asarray(x)), topo, BOUNDS, fixed)
        adm = admissibility(params, topo)
        rec = dict(k=k, admissible=adm["admissible"], min_base_separation_m=adm["min_base_separation_m"], max_reach_m=adm["max_reach_m"], proxy_karma_t=0.0, elapsed_s=0.0)
        if not adm["admissible"]:
            rec["violations"] = adm["violations"]; cands.append(rec)
            return 0.0
        wd = Path(workdir) / f"cma_{k:05d}"
        _, yaml_path, _ = write_design(topo, params, wd, "cand")
        cfg = load_config(METRIC_CFG, yaml_path)
        t0 = time.perf_counter()
        try:
            ctx = load_robot(cfg.urdf_path, cfg)
            r = fast_eval(cfg, build_flat_model(ctx, cfg, self_collision_all=True), fc)
            val = float(r.karma_t) if r.error is None else 0.0
            rec.update(proxy_karma_t=val, n_seeds=r.n_seeds, error=r.error)
        except Exception as e:  # noqa: BLE001
            log(f"    candidate {k} failed: {e}"); val = 0.0; rec["error"] = str(e)
        rec["elapsed_s"] = time.perf_counter() - t0
        cands.append(rec)
        return val

    es = cma.CMAEvolutionStrategy(np.asarray(x0), sigma0, dict(popsize=popsize, seed=seed, verbose=-9))
    k = 0
    hist = []
    for g in range(n_gens):
        X = es.ask()
        vals = []
        for x in X:
            vals.append(fitness(x, k)); k += 1
        es.tell(X, [-v for v in vals])
        n_inadm = sum(1 for c in cands[-popsize:] if not c["admissible"])
        hist.append(dict(gen=g, best=float(max(vals)), mean=float(np.mean(vals)), n_inadmissible=n_inadm))
        log(f"  CMA gen {g}: best proxy KaRMA-T {max(vals):.4f} mean {np.mean(vals):.4f} inadmissible {n_inadm}/{popsize}")
    best = decode(unravel(jnp.asarray(es.result.xbest)), topo, BOUNDS, fixed)
    return best, hist, cands


def run_armAprime(k, out: Path, log):
    task = load_task_spec()
    spec, starts, _ = load_starts(task, log)
    name, phi = starts[k]
    topo = spec["topo"]
    params0 = decode(phi, topo, BOUNDS, fixed_from_params(spec["params0"]))
    rdir = out / "armAprime" / name; rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "run.json").exists():
        log(f"[armAprime/{name}] exists, skipping"); return
    adm0 = admissibility(params0, topo)
    log(f"[armAprime/{name}] start admissible {adm0['admissible']} (min base sep {adm0['min_base_separation_m'] * 1e3:.1f} mm, reach {adm0['max_reach_m'] * 1e3:.1f} mm); CMA seed {k}")
    t0 = time.perf_counter()
    params, hist, cands = cmaes_original_karma_d18(topo, params0, rdir / "cma", seed=k, log=log)
    wall = time.perf_counter() - t0
    write_design(topo, params0, rdir / "initial_morphology", "initial_morphology")
    write_design(topo, params, rdir / "optimized_morphology", "optimized_morphology")
    adm = admissibility(params, topo)
    rec = dict(arm="armAprime", start=name, cma_seed=k, popsize=8, n_gens=12, sigma0=0.3, n_evaluations=len(cands), wall_s=wall,
               n_inadmissible_candidates=sum(1 for c in cands if not c["admissible"]), history=hist, candidates=cands,
               best_proxy_karma_t=max(c["proxy_karma_t"] for c in cands), admissibility=adm, start_admissibility=adm0,
               d18=dict(min_base_separation_m=MIN_BASE_SEPARATION_M, reach_max_m=REACH_MAX_M, self_collision_all=True),
               proxy=dict(n_linspace_per_joint=5, n_sobol=256, n_random=64, n_offsets=5, dedup_rad=0.1, max_seeds=48, stage1_budget=15, n_finalists=3, final_budget=10000))
    (rdir / "run.json").write_text(json.dumps(rec, indent=1, default=float))
    log(f"[armAprime/{name}] done: best proxy {rec['best_proxy_karma_t']:.4f}, {rec['n_inadmissible_candidates']} inadmissible of {len(cands)}, wall {wall:.0f}s, returned admissible {adm['admissible']}")


# ── Full KaRMA under D18 ─────────────────────────────────────────────────────

def run_full(k, out: Path, log, kkt_method: str = "exact"):
    task = load_task_spec()
    spec, starts, _ = load_starts(task, log)
    name, phi = starts[k]
    topo = spec["topo"]
    cfg = spec["cfg"]
    cfg = replace(cfg, roll=cfg.roll._replace(kkt_method=kkt_method))
    bounds, cons, env, arm = BOX["bounds"], BOX["constraints"], BOX["envelope"], BOX["arm"]
    params0 = decode(phi, topo, bounds, fixed_from_params(spec["params0"]))
    rdir = out / arm / name; rdir.mkdir(parents=True, exist_ok=True)
    if (rdir / "run.json").exists():
        log(f"[{arm}/{name}] exists, skipping"); return
    obj = make_objective(topo, params0, bounds, task, cfg, SeedConfig(budget=spec["budget"]), constraints=cons)
    validate = exact_validation(obj, rdir / "val", official=False, enforce_d18=True, envelope=env)
    adm0 = admissibility(params0, topo, envelope=env)
    log(f"[{arm}/{name}] seeds {int(obj.seeds.valid.sum())}; kkt_method {kkt_method}; constraints {cons}; start admissible {adm0['admissible']} {adm0['violations']}")
    t0 = time.perf_counter()
    res = adam(obj, phi, n_iters=spec["n_iters"], stages=spec["stages"], validate_every=spec["validate_every"], validation_fn=validate,
               log=log, seed=0, instrument=False, warmup=True)
    wall = time.perf_counter() - t0
    adm = admissibility(res.best_params, topo, envelope=env)
    config = dict(arm=arm, start=name, objective=cfg.name, adam=res.meta["options"], kkt_method=kkt_method, n_iters=spec["n_iters"], stages=spec["stages"],
                  commands=cfg.commands.__dict__, seed_config=SeedConfig(budget=spec["budget"]).__dict__, bounds=bounds.__dict__,
                  constraints=cons.__dict__, optimizer_seed=0, adam_defaults=ADAM_DEFAULTS)
    if env is not None:
        config.update(envelope=env.__dict__, cap_radius_m=np.asarray(params0.cap_radius).tolist())
    save_run(rdir, topo, params0, res, config, actuator_spec=load_actuation_spec().__dict__, seeds=obj.seeds)
    nonfinite = sum(1 for h in res.training_history if not h["grad_finite"])
    rec = dict(arm=arm, start=name, wall_s=wall, optimizer_time_s=res.meta["optimizer_time_s"], validation_time_s=res.meta["validation_time_s"],
               best_validation=res.best_validation, best_iter=res.meta["best_iter"], n_nonfinite=nonfinite,
               n_clipped=sum(1 for h in res.training_history if h["clipped"]), validation_history=res.validation_history,
               admissibility=adm, kkt_method=kkt_method, n_evaluations=len(res.training_history))
    if env is not None:
        rec.update(start_admissibility=adm0, n_start_seeds=int(obj.seeds.valid.sum()), platform=jax.default_backend())
    (rdir / "run.json").write_text(json.dumps(rec, indent=1, default=float))
    log(f"[{arm}/{name}] done: returned exact {res.best_validation.get('exact_score', float('nan')):.4f} (it {res.meta['best_iter']}), admissible {adm['admissible']}, "
        f"non-finite {nonfinite}, wall {wall:.0f}s (opt {res.meta['optimizer_time_s']:.0f}s)")


# ── suite ────────────────────────────────────────────────────────────────────

def design_by_key(key: str, out: Path):
    """``armAprime/<start>`` | ``full_d18/<start>`` | ``full_d19/<start>`` | ``start/<start>`` | ``armA`` | ``armB``."""
    task = load_task_spec()
    if key == "armA":
        t, p, _ = design_from_dict(json.loads(ARM_A.read_text())); return t, p, dict(source=str(ARM_A))
    if key == "armB":
        t, p, _ = design_from_dict(json.loads(ARM_B.read_text())); return t, p, dict(source=str(ARM_B))
    arm, name = key.split("/")
    spec, starts, _ = load_starts(task, log=lambda s: None)
    if arm == "start":
        phi = dict(starts)[name]
        return spec["topo"], decode(phi, spec["topo"], BOX["bounds"], fixed_from_params(spec["params0"])), dict(source=str(Path(STARTS_DIR) / "starts.json"))
    p = Path(RUNS_DIR or out) / arm / name / "optimized_morphology" / "optimized_morphology.design.json"
    t, par, _ = design_from_dict(json.loads(p.read_text()))
    return t, par, dict(source=str(p))


def suite_d18(topo, params, task, sp, workdir, seed_cfgs=SEED_CONFIGS, budget_2=BUDGET_2, budget_3=BUDGET_3, official=True, log=print,
              per_seed_detail: bool = False, envelope=None):
    """The common three-contact physical suite (fixed physical task of SPEC §2) at the Phase 9.4
    budgets under several seed-search configurations. Exact Full counts use the exact SLSQP force check
    at every sub-step (``MC.evaluate_fixed_contacts_exact``, Phase 8.6). Wrench capacities and the
    physical original KaRMA (all-finger self-collision gate) once per design.

    ``per_seed_detail`` (Phase 10.3): also record, under every seed configuration, each three-contact seed's
    joint configuration and sphere position and its wrench capacities in the 14 standard directions, so the
    task-feasible subset ``S_task`` and the subset scores come from the same seeds as the suite's counts.

    ``envelope`` (Phase 11, D19): the admissibility record includes the servo envelope; call with ``official=False``
    (the original KaRMA's kernels apply one radius to every capsule)."""
    hm = build_hand_model(topo, params)
    act = AK.actuation_for(hm, task)
    fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle")) if topo.n_fingers >= 3 else None
    cfg_mc = replace(MC_KARMA, commands=CMDS); cfg_full = replace(FULL_KARMA, commands=CMDS); cfg_2 = replace(DIFF_KARMA, commands=CMDS)
    sm = cfg_full.smooth
    out = dict(admissibility=admissibility(params, topo, envelope=envelope), budgets=dict(two_finger=budget_2, three_contact=budget_3), per_seed_config={})
    for cname, ckw in seed_cfgs.items():
        rec = {}
        t0 = time.perf_counter()
        if fingers is not None:
            s3 = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=budget_3, **ckw))
            rec["n_seeds_3"] = int(s3.valid.sum()); rec["seed_stats_3"] = {k: v for k, v in s3.stats.items() if isinstance(v, (int, float))}
            if rec["n_seeds_3"]:
                ex_full = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, s3, sp, task, act, cfg_full)
                ex_mc = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, s3, sp, task, act, cfg_mc)
                rec.update(full_exact_K_T=ex_full.K_T, full_exact_K_SO3=ex_full.K_SO3, mc_exact_K_T=ex_mc.K_T, mc_exact_K_SO3=ex_mc.K_SO3,
                           full_cells=[len(r["cells"]) for r in ex_full.per_seed], full_bins=[len(r["bins"]) for r in ex_full.per_seed],
                           full_seed_force_ok=[bool(r.get("seed_force_ok", True)) for r in ex_full.per_seed],
                           mc_cells=[len(r["cells"]) for r in ex_mc.per_seed], mc_bins=[len(r["bins"]) for r in ex_mc.per_seed])
                rec["full_exact_score"] = sm.lambda_T * ex_full.K_T / sm.ref_T + sm.lambda_R * ex_full.K_SO3 / sm.ref_R
                if per_seed_detail:
                    asg3 = assignment(topo, 3, fingers)
                    rec["seed_q_3"] = np.asarray(s3.q)[: rec["n_seeds_3"]].tolist(); rec["seed_p_3"] = np.asarray(s3.p)[: rec["n_seeds_3"]].tolist()
                    rec["wrench_alpha_3"] = [np.asarray(AK.wrench_capacity_at(hm, GraspState(s3.q[s], s3.p[s], jnp.eye(3)), asg3, sp, act, n_dir=14)[0]).tolist()
                                             for s in range(rec["n_seeds_3"])]
                if cname == "default":
                    asg3 = assignment(topo, 3, fingers)
                    caps = []
                    for s in range(rec["n_seeds_3"]):
                        a, _ = AK.wrench_capacity_at(hm, GraspState(s3.q[s], s3.p[s], jnp.eye(3)), asg3, sp, act, n_dir=14)
                        caps.append(dict(mean=float(a.mean()), min=float(a.min()), frac_ge_1=float(np.mean(a >= 1.0))))
                    out["wrench_capacity_3"] = caps
            else:
                rec.update(full_exact_K_T=0.0, full_exact_K_SO3=0.0, mc_exact_K_T=0.0, mc_exact_K_SO3=0.0, full_exact_score=0.0)
        s2 = find_seeds(hm, topo, sp, SeedConfig(budget=budget_2, **ckw))
        rec["n_seeds_2"] = int(s2.valid.sum())
        if envelope is not None:
            rec["seed_stats_2"] = {k: v for k, v in s2.stats.items() if isinstance(v, (int, float))}
        if rec["n_seeds_2"]:
            ex2 = evaluate_exact(hm, topo, s2, sp, task, cfg_2)
            rec.update(two_exact_K_T=ex2.K_T, two_exact_K_R=ex2.K_R, two_exact_score=sm.lambda_T * ex2.K_T / sm.ref_T + sm.lambda_R * ex2.K_R / sm.ref_R)
        else:
            rec.update(two_exact_K_T=0.0, two_exact_K_R=0.0, two_exact_score=0.0)
        rec["seed_config"] = SeedConfig(budget=budget_3, **ckw).__dict__ | {"ik": "lm/100"}
        rec["time_s"] = time.perf_counter() - t0
        out["per_seed_config"][cname] = rec
        log(f"    [{cname}] seeds3 {rec.get('n_seeds_3')} full K_T {rec.get('full_exact_K_T', 0):.5f} K_SO3 {rec.get('full_exact_K_SO3', 0):.4f} score {rec.get('full_exact_score', 0):.4f} | "
            f"seeds2 {rec['n_seeds_2']} two K_T {rec['two_exact_K_T']:.4f} K_R {rec['two_exact_K_R']:.4f} ({rec['time_s']:.0f}s)")
    if official:
        from dkarma.metrics.exact_library import official_karma
        t0 = time.perf_counter()
        out["official_physical_allfinger"] = official_karma_physical(topo, params, task, workdir / "official_allfinger", name="suite", self_collision_all=True)
        out["official_physical_pair"] = official_karma_physical(topo, params, task, workdir / "official_pair", name="suite", self_collision_all=False)
        # the objective arm A' optimizes: the published (L_ref-scaled) original KaRMA, with the D18 all-finger gate
        out["official_published_allfinger"] = official_karma(topo, params, workdir / "official_published", name="suite", self_collision_all=True)
        out["official_time_s"] = time.perf_counter() - t0
        log(f"    official physical KaRMA: cube fraction all-finger {out['official_physical_allfinger']['cube_fraction']:.5f} pair {out['official_physical_pair']['cube_fraction']:.5f}")
    return out


def run_suite(key: str, out: Path, log, starts_single_config=True, per_seed_detail: bool = False):
    task = load_task_spec(); sp = task.step_params()
    topo, params, src = design_by_key(key, out)
    sdir = out / "suite" / key.replace("/", "__"); sdir.mkdir(parents=True, exist_ok=True)
    if (sdir / "suite.json").exists():
        log(f"[suite/{key}] exists, skipping"); return
    if BOX["envelope"] is not None and key.startswith("start/"):
        starts_single_config = False          # Phase 11: N11's own S_task under three configurations is criterion v2's R3 reference
    cfgs = SEED_CONFIGS if not (key.startswith("start/") and starts_single_config) else {"default": SEED_CONFIGS["default"]}
    log(f"[suite/{key}] from {src['source']}; seed configs {list(cfgs)}")
    t0 = time.perf_counter()
    res = suite_d18(topo, params, task, sp, sdir, seed_cfgs=cfgs, log=log, per_seed_detail=per_seed_detail, official=BOX["official"],
                    envelope=BOX["envelope"])
    res.update(design=key, source=src["source"], wall_s=time.perf_counter() - t0, task="fixed physical (SPEC §2)", platform=jax.default_backend(),
               jax_version=jax.__version__)
    if BOX["envelope"] is not None:
        res.update(box="d19", cap_radius_m=np.asarray(params.cap_radius).tolist())
    (sdir / "suite.json").write_text(json.dumps(res, indent=1, default=float))
    log(f"[suite/{key}] done in {res['wall_s']:.0f}s")


# ── robustness sweep (9.5 criterion item 3) ──────────────────────────────────

def run_sweep(key: str, out: Path, log, taus=(0.215, 0.108, 0.043), masses=(0.035, 0.070), budget=BUDGET_3):
    """Exact Full-KaRMA counts of one design under the torque ceilings of SPEC §2.1 (stall, primary, continuous)
    and under a doubled object mass at the primary ceiling, on one fixed set of fresh three-contact seeds (default
    seed configuration, 12 seeds; the seed gate and every sub-step's exact SLSQP check use the swept load);
    fixed physical task otherwise. Writes ``<out>/sweeps/<key>.json``."""
    task = load_task_spec(); sp = task.step_params()
    topo, params, src = design_by_key(key, out)
    sdir = out / "sweeps"; sdir.mkdir(parents=True, exist_ok=True)
    fn = sdir / f"{key.replace('/', '__')}.json"
    if fn.exists():
        log(f"[sweep/{key}] exists, skipping"); return
    hm = build_hand_model(topo, params)
    act0 = AK.actuation_for(hm, task)
    fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
    cfg_full = replace(FULL_KARMA, commands=CMDS); sm = cfg_full.smooth
    t0 = time.perf_counter()
    s3 = find_multi_seeds(hm, topo, fingers, sp, act0, SeedConfig(budget=budget))
    n3 = int(s3.valid.sum())
    rows = []
    settings = [("torque", tau, task.obj.mass_kg) for tau in taus] + [("mass", float(act0.tau_bar[0]), m) for m in masses if abs(m - task.obj.mass_kg) > 1e-12]
    for kind, tau, m in settings:
        act = act0._replace(tau_bar=jnp.full_like(act0.tau_bar, tau), w_ext=act0.w_ext * (m / task.obj.mass_kg))
        if n3:
            ex = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, s3, sp, task, act, cfg_full)
            score = sm.lambda_T * ex.K_T / sm.ref_T + sm.lambda_R * ex.K_SO3 / sm.ref_R
            seed_ok = [bool(r.get("seed_force_ok", True)) for r in ex.per_seed]
            rows.append(dict(kind=kind, tau_bar=tau, mass_kg=m, full_exact_K_T=ex.K_T, full_exact_K_SO3=ex.K_SO3, full_exact_score=score,
                             seeds_hold=int(sum(seed_ok)), cells=[len(r["cells"]) for r in ex.per_seed]))
        else:
            rows.append(dict(kind=kind, tau_bar=tau, mass_kg=m, full_exact_K_T=0.0, full_exact_K_SO3=0.0, full_exact_score=0.0, seeds_hold=0, cells=[]))
        log(f"  [sweep/{key}] {kind} tau {tau:.3f} mass {m:.3f}: score {rows[-1]['full_exact_score']:.4f} seeds hold {rows[-1]['seeds_hold']}/{n3}")
    rec = dict(design=key, source=src["source"], n_seeds_3=n3, seed_config=SeedConfig(budget=budget).__dict__ | {"ik": "lm/100"}, rows=rows,
               task="fixed physical (SPEC §2), torque ceiling and object mass swept", wall_s=time.perf_counter() - t0, platform=jax.default_backend())
    fn.write_text(json.dumps(rec, indent=1, default=float))
    log(f"[sweep/{key}] done in {rec['wall_s']:.0f}s")


# ── summary ──────────────────────────────────────────────────────────────────

def _paired_stats(a, b, n_boot=10000, seed=0):
    """Paired comparison of ``b - a`` over starts: mean, median, sign counts, Wilcoxon signed-rank p, sign-test p
    and a percentile bootstrap 95 % CI of the mean difference (resampling starts)."""
    from scipy.stats import wilcoxon, binomtest
    a = np.asarray(a, float); b = np.asarray(b, float); d = b - a
    w = int(np.sum(d > 1e-12)); l_ = int(np.sum(d < -1e-12)); t = len(d) - w - l_
    try:
        pw = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
    except ValueError:
        pw = float("nan")
    ps = float(binomtest(w, w + l_, 0.5).pvalue) if w + l_ else 1.0
    rng = np.random.RandomState(seed)
    boots = np.array([d[rng.randint(0, len(d), len(d))].mean() for _ in range(n_boot)]) if len(d) else np.zeros(1)
    return dict(n=len(d), a=a.tolist(), b=b.tolist(), diff=d.tolist(), mean_diff=float(d.mean()), median_diff=float(np.median(d)),
                wins_b=w, ties=t, losses_b=l_, wilcoxon_p=pw, sign_test_p=ps,
                boot_ci95=[float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))])


def _describe(v):
    v = np.asarray(v, float)
    return dict(mean=float(v.mean()), median=float(np.median(v)), best=float(v.max()), worst=float(v.min()),
                iqr=[float(np.percentile(v, 25)), float(np.percentile(v, 75))]) if v.size else {}


def summarize(out: Path, log=print):
    """Tables of the 9.2 comparison (fixed physical task) with the 9.4 seed-configuration spread."""
    from scipy.stats import spearmanr
    suites = {p.parent.name.replace("__", "/"): json.loads(p.read_text()) for p in sorted((out / "suite").glob("*/suite.json"))}
    runs = {}
    for arm in ("armAprime", "full_d18"):
        for p in sorted((out / arm).glob("*/run.json")):
            runs[f"{arm}/{p.parent.name}"] = json.loads(p.read_text())
    starts = sorted({k.split("/")[1] for k in suites if k.startswith(("armAprime/", "full_d18/"))})
    cfg_names = list(SEED_CONFIGS)
    metrics = [("full_exact_score", "exact Full-KaRMA score (cells/8000/0.1 + SO(3) bins/4608/0.3)"),
               ("full_exact_K_T", "exact actuated three-contact cell fraction"), ("full_exact_K_SO3", "exact SO(3) bin fraction"),
               ("mc_exact_K_T", "exact MC (unactuated) three-contact cell fraction"), ("two_exact_score", "exact two-finger library score (K_T/0.1 + K_R/0.3)"),
               ("n_seeds_3", "three-contact seeds found (of 12)")]
    rec = dict(task="fixed physical (SPEC §2)", d18=dict(min_base_separation_m=MIN_BASE_SEPARATION_M, reach_max_m=REACH_MAX_M, self_collision_all=True),
               starts=starts, designs=sorted(suites), per_design={}, paired={}, paired_per_config={}, ranking_stability={}, budgets={})

    def val(key, metric, cfg=None):
        """Mean over the seed configurations run for the design (or one configuration)."""
        s = suites.get(key)
        if s is None:
            return None
        pcs = s["per_seed_config"]
        if cfg is not None:
            return pcs.get(cfg, {}).get(metric)
        vs = [c.get(metric) for c in pcs.values() if c.get(metric) is not None]
        return float(np.mean(vs)) if vs else None

    def spread(key, metric):
        s = suites.get(key)
        vs = [c.get(metric) for c in s["per_seed_config"].values() if c.get(metric) is not None] if s else []
        return (float(np.mean(vs)), float(np.std(vs)), vs) if vs else (None, None, [])

    f4 = lambda x: "–" if x is None else f"{x:.4f}"   # noqa: E731
    lines = ["# Phase 9.2 / 9.4 summary", "",
             "Task: **fixed physical** (SPEC §2: 15 mm / 35 g sphere, ±100 mm cube at 10 mm, XL330 primary ceiling 0.108 N·m) unless a column says *published*. "
             "Design box (D18): finger bases ≥ 24.1 mm apart, reach ≤ 250 mm per finger, all-finger clearance in every self-collision gate. "
             "Seeds: 12 three-contact and 16 two-finger fresh seeds per design under three seed-search configurations "
             f"({', '.join(cfg_names)}); values are the mean over the configurations ± their standard deviation (start designs: default configuration only).", ""]
    # budgets
    for arm in ("armAprime", "full_d18"):
        rs = [r for k, r in runs.items() if k.startswith(arm)]
        if rs:
            walls = [r.get("wall_s", 0.0) for r in rs]
            opt = [r.get("optimizer_time_s") for r in rs if r.get("optimizer_time_s") is not None]
            rec["budgets"][arm] = dict(n_runs=len(rs), wall_s=_describe(walls), optimizer_time_s=(_describe(opt) if opt else None),
                                       n_evaluations=[r.get("n_evaluations") for r in rs],
                                       n_inadmissible_candidates=[r.get("n_inadmissible_candidates") for r in rs if r.get("n_inadmissible_candidates") is not None],
                                       n_nonfinite=[r.get("n_nonfinite") for r in rs if r.get("n_nonfinite") is not None])
    lines += ["## Optimizer budgets", "", "| arm | runs | evaluations per run | wall clock per run (s, mean [min, max]) | notes |", "| --- | --- | --- | --- | --- |"]
    for arm, b in rec["budgets"].items():
        w = b["wall_s"]
        note = (f"proxy evaluations of the published original KaRMA; inadmissible candidates scored 0 without evaluation: {b['n_inadmissible_candidates']}" if arm == "armAprime"
                else f"Adam gradient evaluations of the smooth Full objective; optimizer time {b['optimizer_time_s']['mean']:.0f} s mean (validations excluded); non-finite gradients per run {b['n_nonfinite']}" if b.get("optimizer_time_s") else "")
        lines.append(f"| {arm} | {b['n_runs']} | {sorted(set(b['n_evaluations']))} | {w['mean']:.0f} [{w['worst']:.0f}, {w['best']:.0f}] | {note} |")
    lines.append("")
    # per-design table
    lines += ["## Every design", "",
              "| design | admissible | min base sep (mm) | max reach (mm) | three-contact seeds (per config) | Full exact score | Full K_T | Full K_SO3 | two-finger exact score | min wrench capacity (×weight) | physical original KaRMA cube fraction, all-finger / pair gate | *published* original KaRMA-T, all-finger gate |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for key in sorted(suites):
        s = suites[key]; a = s["admissibility"]
        fs = spread(key, "full_exact_score"); kt = spread(key, "full_exact_K_T"); ks = spread(key, "full_exact_K_SO3"); tw = spread(key, "two_exact_score")
        off = s.get("official_physical_allfinger", {}).get("cube_fraction"); offp = s.get("official_physical_pair", {}).get("cube_fraction")
        pub = s.get("official_published_allfinger", {}).get("KaRMA_T")
        caps = [c["min"] for c in s.get("wrench_capacity_3", [])]
        fmt = lambda m: (f"{m[0]:.4f} ± {m[1]:.4f}" if m[0] is not None and len(m[2]) > 1 else f4(m[0]))  # noqa: E731
        n3 = [c.get("n_seeds_3") for c in s["per_seed_config"].values()]
        lines.append(f"| {key} | {a['admissible']} | {a['min_base_separation_m'] * 1e3:.1f} | {a['max_reach_m'] * 1e3:.1f} | {n3} | {fmt(fs)} | {fmt(kt)} | {fmt(ks)} | {fmt(tw)} | "
                     f"{(f'{min(caps):.2f}' if caps else '–')} | {('–' if off is None else f'{off:.5f}')} / {('–' if offp is None else f'{offp:.5f}')} | {('–' if pub is None else f'{pub:.4f}')} |")
        rec["per_design"][key] = dict(admissibility=a, full_exact_score=fs, full_exact_K_T=kt, full_exact_K_SO3=ks, two_exact_score=tw,
                                      official_physical_allfinger=off, official_physical_pair=offp, official_published_allfinger_KaRMA_T=pub,
                                      min_wrench_capacity=(min(caps) if caps else None), n_seeds_3=n3,
                                      run={k: v for k, v in runs.get(key, {}).items() if k in ("wall_s", "n_evaluations", "n_inadmissible_candidates", "best_proxy_karma_t",
                                                                                                 "n_nonfinite", "best_validation", "optimizer_time_s")})
    lines.append("")
    # paired comparisons: Full-D18 vs arm A' (mean over seed configurations, and per configuration)
    for metric, desc in metrics:
        pairs = [(st, val(f"armAprime/{st}", metric), val(f"full_d18/{st}", metric), val(f"start/{st}", metric)) for st in starts]
        pairs = [p for p in pairs if p[1] is not None and p[2] is not None]
        if len(pairs) < 2:
            continue
        ps = _paired_stats([p[1] for p in pairs], [p[2] for p in pairs])
        rec["paired"][metric] = dict(ps, starts=[p[0] for p in pairs], start_values=[p[3] for p in pairs],
                                     armAprime=_describe([p[1] for p in pairs]), full_d18=_describe([p[2] for p in pairs]))
        da, db = rec["paired"][metric]["armAprime"], rec["paired"][metric]["full_d18"]
        lines += [f"## Paired per start: Full KaRMA (D18) − arm A′ on the {desc}", "",
                  "| start | start design (default config) | arm A′ | Full-D18 | Full − A′ |", "| --- | --- | --- | --- | --- |"]
        for st, av, bv, sv in pairs:
            lines.append(f"| {st} | {f4(sv)} | {av:.4f} | {bv:.4f} | {bv - av:+.4f} |")
        lines += ["", f"| arm | mean | median | best | IQR |", "| --- | --- | --- | --- | --- |",
                  f"| arm A′ | {da['mean']:.4f} | {da['median']:.4f} | {da['best']:.4f} | [{da['iqr'][0]:.4f}, {da['iqr'][1]:.4f}] |",
                  f"| Full-D18 | {db['mean']:.4f} | {db['median']:.4f} | {db['best']:.4f} | [{db['iqr'][0]:.4f}, {db['iqr'][1]:.4f}] |", "",
                  f"Full-D18 wins / ties / losses {ps['wins_b']} / {ps['ties']} / {ps['losses_b']}; mean difference {ps['mean_diff']:+.4f} "
                  f"(bootstrap 95 % CI [{ps['boot_ci95'][0]:+.4f}, {ps['boot_ci95'][1]:+.4f}]); median {ps['median_diff']:+.4f}; "
                  f"Wilcoxon signed-rank p = {ps['wilcoxon_p']:.3f}; sign test p = {ps['sign_test_p']:.3f} (n = {ps['n']} starts)", ""]
        per_cfg = {}
        for c in cfg_names:
            pc = [(val(f"armAprime/{st}", metric, c), val(f"full_d18/{st}", metric, c)) for st in starts]
            pc = [x for x in pc if x[0] is not None and x[1] is not None]
            if len(pc) >= 2:
                per_cfg[c] = _paired_stats([x[0] for x in pc], [x[1] for x in pc])
        if per_cfg:
            rec["paired_per_config"][metric] = per_cfg
            lines += ["| seed configuration | Full wins / ties / losses | mean diff | Wilcoxon p |", "| --- | --- | --- | --- |"]
            lines += [f"| {c} | {v['wins_b']} / {v['ties']} / {v['losses_b']} | {v['mean_diff']:+.4f} | {v['wilcoxon_p']:.3f} |" for c, v in per_cfg.items()]
            lines.append("")
    # MuJoCo leg (9.3 controllers): three-contact library of every A' and Full-D18 design, 12 fresh seeds, default seed configuration
    simdir = out / "sim"
    if simdir.exists():
        lines += ["## MuJoCo, three-contact library (12 fresh seeds, default seed configuration; 9.3 scene and controllers)", "",
                  "Per design: three-contact seeds found, seeds that hold in simulation, exactly valid primitives, and how many of them "
                  "also succeed in simulation (strict: pose within 4 mm / 0.2 rad and every assigned contact at the primitive end; pose only). "
                  "The paired measure is strict successes per budgeted seed (successes / 12): the simulated three-contact capability of the design.", ""]
        for ctrl in ("position", "force"):
            per = {}
            for st in starts:
                k = st.split("_")[0]
                for arm in ("armAprime", "full_d18"):
                    pth = simdir / f"simulation_{ctrl}__{arm}_{k}.json"
                    if not pth.exists():
                        continue
                    r = json.loads(pth.read_text())[f"{arm}_{k}"]
                    rows3 = [x for x in r["rows"] if x["library"] == "three_contact"]
                    sep = r.get("separated", {}).get("three_contact", {})
                    per.setdefault(st, {})[arm] = dict(n_seeds=len({x["seed"] for x in rows3}), hold=sep.get("seeds_hold", 0),
                                                       valid=sum(x["n_exact"] for x in rows3), strict=sum(min(x["n_sim_rot"], x["n_exact"]) for x in rows3),
                                                       pose=sum(min(x["n_pose"], x["n_exact"]) for x in rows3),
                                                       rate_strict=sep.get("frac", {}).get("strict_rot"), rate_pose=sep.get("frac", {}).get("pose"))
            both = [st for st in starts if st in per and len(per[st]) == 2]
            if not both:
                continue
            lines += [f"### controller: {ctrl}", "",
                      "| start | arm A′: seeds / hold / valid / strict / pose | Full-D18: seeds / hold / valid / strict / pose | strict per budgeted seed: A′ → Full |",
                      "| --- | --- | --- | --- |"]
            for st in both:
                a, b = per[st]["armAprime"], per[st]["full_d18"]
                lines.append(f"| {st} | {a['n_seeds']} / {a['hold']} / {a['valid']} / {a['strict']} / {a['pose']} | {b['n_seeds']} / {b['hold']} / {b['valid']} / {b['strict']} / {b['pose']} | "
                             f"{a['strict'] / 12:.2f} → {b['strict'] / 12:.2f} |")
            ps = _paired_stats([per[st]["armAprime"]["strict"] / 12 for st in both], [per[st]["full_d18"]["strict"] / 12 for st in both])
            pp = _paired_stats([per[st]["armAprime"]["pose"] / 12 for st in both], [per[st]["full_d18"]["pose"] / 12 for st in both])
            rec.setdefault("mujoco", {})[ctrl] = dict(per_start=per, strict_per_seed=ps, pose_per_seed=pp)
            lines += ["", f"strict successes per budgeted seed, Full − A′: wins / ties / losses {ps['wins_b']} / {ps['ties']} / {ps['losses_b']}, mean {ps['mean_diff']:+.2f} "
                      f"(bootstrap 95 % CI [{ps['boot_ci95'][0]:+.2f}, {ps['boot_ci95'][1]:+.2f}]), Wilcoxon p = {ps['wilcoxon_p']:.3f}; "
                      f"pose successes per budgeted seed: {pp['wins_b']} / {pp['ties']} / {pp['losses_b']}, mean {pp['mean_diff']:+.2f} "
                      f"(CI [{pp['boot_ci95'][0]:+.2f}, {pp['boot_ci95'][1]:+.2f}]), Wilcoxon p = {pp['wilcoxon_p']:.3f} (n = {ps['n']} starts)", ""]
    # arm B and arm A against the A' and Full-D18 distributions (single designs, start0 analogue)
    for key in ("armB", "armA"):
        if key in suites:
            v = val(key, "full_exact_score"); t2 = val(key, "two_exact_score")
            lines.append(f"{key}: Full exact score {f4(v)} (spread {spread(key, 'full_exact_score')[1] or 0:.4f}), two-finger exact score {f4(t2)}; "
                         f"arm A′ start0 {f4(val('armAprime/start0_nominal', 'full_exact_score'))}, Full-D18 start0 {f4(val('full_d18/start0_nominal', 'full_exact_score'))}")
    lines.append("")
    # ranking stability over the seed configurations (all candidate designs scored under every configuration)
    cands = [k for k in suites if not k.startswith("start/") and len(suites[k]["per_seed_config"]) == len(cfg_names)]
    for metric, desc in metrics[:3] + metrics[4:5]:
        mat = np.array([[suites[k]["per_seed_config"].get(c, {}).get(metric, np.nan) for c in cfg_names] for k in cands], float)
        if mat.shape[0] >= 3 and not np.isnan(mat).any():
            rhos = {f"{cfg_names[i]} vs {cfg_names[j]}": float(spearmanr(mat[:, i], mat[:, j]).correlation) for i in range(len(cfg_names)) for j in range(i + 1, len(cfg_names))}
            top = [cands[int(np.argmax(mat[:, i]))] for i in range(len(cfg_names))]
            order = [[cands[j] for j in np.argsort(-mat[:, i])[:5]] for i in range(len(cfg_names))]
            rec["ranking_stability"][metric] = dict(candidates=cands, configs=cfg_names, matrix=mat.tolist(), spearman=rhos, best_per_config=top, top5_per_config=order,
                                                    sd_over_configs=np.std(mat, axis=1).tolist(), mean_over_configs=np.mean(mat, axis=1).tolist())
            lines += [f"## Ranking stability over the three seed configurations: {desc} ({len(cands)} candidate designs)", "",
                      "| pair of seed configurations | Spearman rho |", "| --- | --- |"]
            lines += [f"| {k} | {v:.3f} |" for k, v in rhos.items()]
            lines += ["", "| seed configuration | best design | top 5 |", "| --- | --- | --- |"]
            lines += [f"| {c} | {t} | {', '.join(o)} |" for c, t, o in zip(cfg_names, top, order)]
            lines.append("")
    (out / "summary.json").write_text(json.dumps(rec, indent=1, default=float))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["nominal", "starts", "armAprime", "full", "suite", "sweep", "summary"])
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--design", default=None, help="suite: design key (armAprime/<start>, full_d18/<start>, start/<start>, armA, armB)")
    ap.add_argument("--kkt-method", default="exact", help="full: hard-QP derivative (Phase 9.1 choice)")
    ap.add_argument("--out", default="workspace/phase9/9.2")
    ap.add_argument("--starts-dir", default=str(PHASE84_FULL3), help="directory of starts.json (Phase 10.2: workspace/phase10/10.2)")
    ap.add_argument("--n-starts", type=int, default=8, help="number of starts of the 8.4 procedure (Phase 10.2: 33)")
    ap.add_argument("--per-seed-detail", action="store_true", help="suite: record seed states and wrench capacities under every seed configuration (Phase 10.3)")
    ap.add_argument("--box", choices=sorted(BOXES), default="d18", help="design box: d18 (Phases 9-10) or d19 (Phase 11 servo envelope)")
    ap.add_argument("--runs-dir", default=None, help="suite/sweep: directory of the optimizer runs (<arm>/<start>/), default --out")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    global STARTS_DIR, N_STARTS, BOX, RUNS_DIR
    STARTS_DIR, N_STARTS, BOX, RUNS_DIR = Path(args.starts_dir), args.n_starts, BOXES[args.box], args.runs_dir
    log_path = out / "log.txt"

    def log(s):
        print(s, flush=True)
        with open(log_path, "a") as fh:
            fh.write(str(s) + "\n")

    if args.stage == "nominal":
        run_nominal(log)
    elif args.stage == "starts":
        run_starts(log)
    elif args.stage == "armAprime":
        run_armAprime(args.only, out, log)
    elif args.stage == "full":
        run_full(args.only, out, log, kkt_method=args.kkt_method)
    elif args.stage == "suite":
        run_suite(args.design, out, log, per_seed_detail=args.per_seed_detail)
    elif args.stage == "sweep":
        run_sweep(args.design, out, log)
    else:
        summarize(out, log)
    log("done")


if __name__ == "__main__":
    main()
