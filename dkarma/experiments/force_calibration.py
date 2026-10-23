"""Phase 8.1 (SPEC.md §11.1): calibrate the smooth force credit against the exact solver.

Held-out states: every exactly valid primitive-end state of Experiment II's two-finger
libraries (baseline and pert, 6 seeds, the Phase 7 settings) plus the seed states, and the
three-contact seed and primitive-end states of the 3x4 anthropomorphic start. Each state is
labelled by ``contact_forces.solve_exact`` (feasible / infeasible) under

* the Experiment II settings (torque ceiling 0.215 / 0.108 / 0.043 / 0.02 / 0.01 N·m at 35 g;
  70 g and 140 g at the primary ceiling), and
* states "around the margin": for each state the critical load scale, torque scale and
  friction scale are bisected with the exact solver, then the load / ceiling / friction are
  set to fixed multiples of the critical value on both sides of it.

For every candidate hinge scale the smooth energy ``E_F*`` is solved in one batched GPU call,
and for every credit mapping (weight, hard knee, two-term) the per-state credit
``exp(-penalty(E_F*))`` is compared with the exact label: AUC (rank statistic, ties count
half), accuracy at the credit-0.5 threshold, the credit distribution of each label. AUC is
invariant to the weight and to any strictly monotone mapping; only the hinge scale (the
energy's own ranking) moves it. The weight and the mapping decide *where* the credit falls,
which is the second acceptance number (actuation-aware coverage <= 10 % of the kinematic
value where the exact evaluator rejects every state), measured by re-evaluating Act-KaRMA
per candidate at the continuous ceiling on the Experiment II designs.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.force_calibration --out workspace/phase8/8.1
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import build_hand_model, load_task_spec, Topology, anthropomorphic  # noqa: E402
from dkarma.core.contacts import contact_pass  # noqa: E402
from dkarma.solvers.seed_projection import find_seeds, find_multi_seeds, SeedConfig, assignment  # noqa: E402
from dkarma.solvers.rollout import GraspState, rollout, exact_gates, EnergyWeights  # noqa: E402
from dkarma.solvers.exact import validate_translation  # noqa: E402
from dkarma.solvers.contact_forces import solve_exact, solve_soft, ForceWeights, ForceProblem  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, ACT_KARMA, FULL_KARMA  # noqa: E402
from dkarma.metrics.commands import build_library, CommandConfig  # noqa: E402
from dkarma.metrics import diff_karma as DK, act_karma as AK, multicontact as MC  # noqa: E402
from dkarma.experiments.actuation_awareness import designs  # noqa: E402
from karma_jax.kinematics import forward_kinematics  # noqa: E402

TAUS = [0.215, 0.108, 0.043, 0.02, 0.01]
MASSES = [0.035, 0.07, 0.14]
MARGIN_FACTORS = [0.5, 0.7, 0.85, 0.95, 1.05, 1.15, 1.3, 1.6, 2.5]


# ── held-out states ──────────────────────────────────────────────────────────

def two_finger_states(name, hm, topo, seeds, sp, task, cfg, act):
    """Seed states and exactly valid primitive-end states of the two-finger library (as
    Experiment II's ``exact_force_fraction``), each as a host ``ForceProblem``."""
    asg = assignment(topo)
    out = []
    for s in range(int(np.asarray(seeds.valid).sum())):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        n0 = contact_pass(hm, forward_kinematics(hm.dm, q), p, sp.sphere_radius, asg).normal[0]
        lib = build_library(frame, -n0, cfg.commands)
        st0 = GraspState(q, p, jnp.eye(3))
        out.append(dict(design=name, seed=s, command=-1, primitive=-1, pb=AK.force_problem_at(hm, st0, asg, sp, act)))
        exT = jax.vmap(lambda a: validate_translation(hm, st0, a, asg, sp, cfg.commands.n_sub))(lib.dp_T)
        okT = np.asarray(exT.ok)
        for u in range(okT.shape[0]):
            for k in range(okT.shape[1]):
                if not okT[u, k]:
                    break
                st = GraspState(exT.q[u, k + 1], exT.p[u, k + 1], exT.R[u, k + 1])
                pb = AK.force_problem_at(hm, st, asg._replace(caps=jnp.asarray(exT.pair[u, k + 1])), sp, act)
                out.append(dict(design=name, seed=s, command=u, primitive=k, pb=pb))
    return out


def three_contact_states(name, hm, topo, fingers, seeds, sp, task, act, cfg):
    """Seed states and primitive-end states of the three-contact library that pass every
    kinematic exact gate and the unactuated exact equilibrium (the MC-KaRMA exact path)."""
    asg = assignment(topo, 3, fingers)
    n_sub = cfg.commands.n_sub
    out = []
    for s in range(int(np.asarray(seeds.valid).sum())):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        st0 = GraspState(q, p, jnp.eye(3))
        out.append(dict(design=name, seed=s, command=-1, primitive=-1, pb=AK.force_problem_at(hm, st0, asg, sp, act)))
        dp_T, dth_T, _, _ = MC.multi_library(frame, cfg.commands)
        roll_T = cfg.roll._replace(kind="translation", mode="hard", grasp_mode="equilibrium", actuation=False)
        sT, dT = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_T, None, act))(dp_T, dth_T)
        gT = jax.vmap(lambda d: exact_gates(d, sp))(dT)
        ok_step = np.array(gT["qp"] & gT["slip"] & gT["gap"] & gT["collision"] & gT["self_collision"] & gT["limits"])
        for u in range(6):
            for k in range(dp_T.shape[1] // n_sub):
                sl = slice(k * n_sub, (k + 1) * n_sub)
                if not np.all(ok_step[u, sl]):
                    break
                qk, pk = sT.q[u, (k + 1) * n_sub], sT.p[u, (k + 1) * n_sub]
                if not MC._exact_equilibrium(hm, qk, pk, asg, sp, act, actuated=False):
                    break
                out.append(dict(design=name, seed=s, command=u, primitive=k, pb=AK.force_problem_at(hm, GraspState(qk, pk, sT.R[u, (k + 1) * n_sub]), asg, sp, act)))
    return out


# ── labelling ────────────────────────────────────────────────────────────────

def label(pb: ForceProblem, w, tau_scale=1.0, mu_scale=1.0):
    pb2 = pb._replace(tau_bar=pb.tau_bar * tau_scale, mu=pb.mu * mu_scale, gamma=pb.gamma * mu_scale)
    ex = solve_exact(pb2, w)
    return bool(ex["feasible"]), float(ex["viol"])


def bisect(feasible_at, lo, hi, n=14):
    """Log-bisection of a scalar ``x`` between ``lo`` (feasible) and ``hi`` (infeasible)."""
    for _ in range(n):
        mid = np.sqrt(lo * hi)
        if feasible_at(mid):
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def instances_for(state, w_ext, tau_primary):
    """Labelled instances of one state: Experiment II settings and the three margin families."""
    pb = state["pb"]
    w0 = np.asarray(w_ext)
    base_tau = float(np.asarray(pb.tau_bar)[0])
    rows = []

    def add(fam, tau_scale, load_scale, mu_scale):
        feas, viol = label(pb, load_scale * w0, tau_scale, mu_scale)
        rows.append(dict(family=fam, tau_scale=tau_scale, load_scale=load_scale, mu_scale=mu_scale, feasible=feas, viol=viol))

    for tau in TAUS:
        add("expII_torque", tau / base_tau, 1.0, 1.0)
    for m in MASSES[1:]:
        add("expII_mass", tau_primary / base_tau, m / MASSES[0], 1.0)
    crit = {}
    # load margin at the primary ceiling
    ts = tau_primary / base_tau
    if label(pb, 0.05 * w0, ts)[0] and not label(pb, 40.0 * w0, ts)[0]:
        lam = bisect(lambda x: label(pb, x * w0, ts)[0], 0.05, 40.0)
        crit["load"] = lam
        for f in MARGIN_FACTORS:
            add("margin_load", ts, lam * f, 1.0)
    # torque margin at the nominal load (feasibility increases with the ceiling: bisect on "infeasible")
    if label(pb, w0, 10.0)[0] and not label(pb, w0, 0.01)[0]:
        t_c = bisect(lambda x: not label(pb, w0, x)[0], 0.01, 10.0)
        crit["tau"] = t_c * base_tau
        for f in MARGIN_FACTORS:
            add("margin_torque", t_c * f, 1.0, 1.0)
    # friction margin at the nominal load and primary ceiling (feasibility increases with mu)
    if label(pb, w0, ts, 5.0)[0] and not label(pb, w0, ts, 0.05)[0]:
        m_c = bisect(lambda x: not label(pb, w0, ts, x)[0], 0.05, 5.0)
        crit["mu"] = m_c * float(pb.mu)
        for f in MARGIN_FACTORS:
            add("margin_friction", ts, 1.0, m_c * f)
    return rows, crit


# ── smooth energies (batched) ────────────────────────────────────────────────

def batch_energies(states, rows_per_state, w_ext, wts: ForceWeights):
    """``E_F*`` of every (state, instance) for one ``ForceWeights`` in one vmapped call."""
    pbs, ws = [], []
    for st, rows in zip(states, rows_per_state):
        for r in rows:
            pb = st["pb"]
            pbs.append(pb._replace(tau_bar=pb.tau_bar * r["tau_scale"], mu=pb.mu * r["mu_scale"], gamma=pb.gamma * r["mu_scale"]))
            ws.append(r["load_scale"] * w_ext)
    stacked = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *pbs)
    W = jnp.stack(ws)
    f = jax.jit(jax.vmap(lambda pb, w: solve_soft(pb, w, wts).E))
    return np.asarray(f(stacked, W))


# ── statistics ───────────────────────────────────────────────────────────────

def auc(score, y):
    from scipy.stats import rankdata
    y = np.asarray(y, bool); score = np.asarray(score, float)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def penalty(E, w, knee, steep, knee_eps):
    """The step penalty of a credit mapping (``rollout.step_energy``): ``w E + steep * hinge((E - knee) / knee)``
    with ``hinge(x) = knee_eps * softplus(x / knee_eps)``."""
    E = np.asarray(E, float)
    out = w * E
    if steep > 0:
        x = (E - knee) / knee / knee_eps
        out = out + steep * knee_eps * np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))
    return out


def mapping_stats(E, y, w, knee, steep, knee_eps):
    y = np.asarray(y, bool)
    c = np.exp(-penalty(E, w, knee, steep, knee_eps))
    return dict(auc=auc(c, y), acc_at_half=float(np.mean((c >= 0.5) == y)),
                feasible_credit_median=float(np.median(c[y])) if y.any() else None,
                feasible_credit_p05=float(np.percentile(c[y], 5)) if y.any() else None,
                infeasible_credit_median=float(np.median(c[~y])) if (~y).any() else None,
                infeasible_credit_p95=float(np.percentile(c[~y], 95)) if (~y).any() else None,
                infeasible_credit_gt_half=float(np.mean(c[~y] >= 0.5)) if (~y).any() else None,
                feasible_credit_lt_half=float(np.mean(c[y] < 0.5)) if y.any() else None)


def youden_threshold(E, y):
    """Energy threshold maximizing sensitivity + specificity (feasible <=> E <= thr)."""
    E = np.asarray(E, float); y = np.asarray(y, bool)
    cand = np.unique(E)
    best = (-1.0, float(np.median(E)))
    for t in cand:
        pred = E <= t
        tpr = np.mean(pred[y]) if y.any() else 0.0
        tnr = np.mean(~pred[~y]) if (~y).any() else 0.0
        j = tpr + tnr - 1.0
        if j > best[0]:
            best = (j, float(t))
    return best[1], best[0]


# ── Experiment II leg: coverage ratio at the continuous ceiling ──────────────

def coverage_ratio(name, hm, topo, seeds, sp, task, wts: ForceWeights, en: EnergyWeights, taus):
    cfg_act = replace(ACT_KARMA, roll=ACT_KARMA.roll._replace(force_weights=wts), energy=en)
    act0 = AK.actuation_for(hm, task)
    d_kin = jax.jit(lambda h: DK.evaluate(h, topo, seeds, sp, task, DIFF_KARMA))(hm)
    v = np.asarray(seeds.valid)
    kin_T = float(np.asarray(d_kin.K_T)[v].mean()); kin_R = float(np.asarray(d_kin.K_R)[v].mean())
    ev = jax.jit(lambda h, a: DK.evaluate(h, topo, seeds, sp, task, cfg_act, None, a))
    out = dict(kinematic_K_T=kin_T, kinematic_K_R=kin_R, by_tau={})
    for tau in taus:
        d = ev(hm, act0._replace(tau_bar=jnp.full_like(act0.tau_bar, tau)))
        K_T = float(np.asarray(d.K_T)[v].mean()); K_R = float(np.asarray(d.K_R)[v].mean())
        out["by_tau"][str(tau)] = dict(K_T=K_T, K_R=K_R, ratio_T=K_T / kin_T if kin_T else None, ratio_R=K_R / kin_R if kin_R else None,
                                       score=float(d.score))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase8/8.1")
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--no-three", action="store_true")
    ap.add_argument("--hinges", nargs="*", type=float, default=[0.02, 0.01, 0.005, 0.002])
    ap.add_argument("--iters", nargs="*", type=int, default=[25, 50])
    ap.add_argument("--weights", nargs="*", type=float, default=[1.0, 5.0, 20.0, 50.0])
    ap.add_argument("--coverage-taus", nargs="*", type=float, default=[0.108, 0.043, 0.02])
    ap.add_argument("--skip-coverage", action="store_true")
    ap.add_argument("--coverage-only", default=None, help="an existing calibration.json: run only the coverage leg for its candidates")
    ap.add_argument("--coverage-out", default="coverage.json")
    ap.add_argument("--mappings", nargs="*", default=None, help="mapping keys to evaluate in the coverage leg (default: linear w1/20/50 and every knee)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    task = load_task_spec(); sp = task.step_params()
    t_start = time.perf_counter()
    if args.coverage_only:
        results = json.loads(Path(args.coverage_only).read_text())
        seeds_of = {}
        for name, topo, params in designs(["baseline", "pert"]):
            hm = build_hand_model(topo, params)
            seeds_of[name] = (hm, topo, find_seeds(hm, topo, sp, SeedConfig(budget=args.budget)))
        cov = {}
        for cand in results["candidates"]:
            wts = ForceWeights(eps_hinge=cand["eps_hinge"], iters=cand["iters"])
            for mk, m in cand["mappings"].items():
                if args.mappings is not None and mk not in args.mappings:
                    continue
                if args.mappings is None and mk.startswith("linear") and m["w"] not in (1.0, 20.0, 50.0):
                    continue
                en = EnergyWeights(force=m["w"], force_knee=m["knee"] if m["steep"] > 0 else 1.0, force_steep=m["steep"],
                                   force_knee_eps=m.get("knee_eps", 0.25))
                key = f"{cand['key']}/{mk}"
                cov[key] = dict(energy=en._asdict(), force_weights=wts._asdict())
                for name, (hm, topo, seeds) in seeds_of.items():
                    cov[key][name] = coverage_ratio(name, hm, topo, seeds, sp, task, wts, en, args.coverage_taus)
                    r = cov[key][name]["by_tau"]
                    print(f"[coverage {key}] {name}: " + " ".join(f"tau {t}: T {v['ratio_T']:.3f} R {v['ratio_R']:.3f}" for t, v in r.items())
                          + f" ({time.perf_counter() - t_start:.0f}s)", flush=True)
                (out / args.coverage_out).write_text(json.dumps(cov, indent=1, default=float))
        print("done")
        return

    # 1. states
    groups = {}
    two = designs(["baseline", "pert"])
    seeds_of = {}
    for name, topo, params in two:
        hm = build_hand_model(topo, params)
        seeds = find_seeds(hm, topo, sp, SeedConfig(budget=args.budget))
        act = AK.actuation_for(hm, task)
        seeds_of[name] = (hm, topo, seeds)
        sts = two_finger_states(name, hm, topo, seeds, sp, task, ACT_KARMA, act)
        groups.setdefault("two_finger", []).extend(sts)
        print(f"[{name}] {len(sts)} two-finger states ({time.perf_counter() - t_start:.0f}s)", flush=True)
    if not args.no_three:
        topo3 = Topology.three_finger(4); p3 = anthropomorphic(topo3)
        hm3 = build_hand_model(topo3, p3); act3 = AK.actuation_for(hm3, task)
        fingers = (topo3.thumb_finger, topo3.index_finger, topo3.finger_index("middle"))
        s3 = find_multi_seeds(hm3, topo3, fingers, sp, act3, SeedConfig(budget=args.budget))
        cfg3 = replace(FULL_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10))
        sts = three_contact_states("start3x4", hm3, topo3, fingers, s3, sp, task, act3, cfg3)
        groups["three_contact"] = sts
        print(f"[start3x4] {len(sts)} three-contact states ({time.perf_counter() - t_start:.0f}s)", flush=True)

    w_ext = jnp.asarray(AK.actuation_for(seeds_of['baseline'][0], task).w_ext)
    tau_primary = float(task_spec_torque())

    # 2. labels
    rows = {}
    crits = {}
    for g, sts in groups.items():
        rows[g] = []; crits[g] = []
        for i, st in enumerate(sts):
            r, c = instances_for(st, w_ext, tau_primary)
            rows[g].append(r); crits[g].append(c)
        n = sum(len(r) for r in rows[g]); nf = sum(rr["feasible"] for r in rows[g] for rr in r)
        print(f"[{g}] {n} labelled instances, {nf} feasible ({time.perf_counter() - t_start:.0f}s)", flush=True)

    # 3. smooth energies per hinge candidate, statistics per mapping
    results = dict(groups={g: dict(n_states=len(sts), n_instances=sum(len(r) for r in rows[g]),
                                   n_feasible=int(sum(rr["feasible"] for r in rows[g] for rr in r),),
                                   states=[{k: v for k, v in st.items() if k != "pb"} | dict(critical=c) for st, c in zip(sts, crits[g])],
                                   instances=[dict(state=i, **rr) for i, r in enumerate(rows[g]) for rr in r])
                           for g, sts in groups.items()},
                   candidates=[])
    energies = {}
    for eps in args.hinges:
        for it in args.iters:
            wts = ForceWeights(eps_hinge=eps, iters=it)
            key = f"h{eps:g}_i{it}"
            E_all, y_all, fam_all, grp_all = [], [], [], []
            per_group = {}
            for g, sts in groups.items():
                E = batch_energies(sts, rows[g], w_ext, wts)
                y = np.array([rr["feasible"] for r in rows[g] for rr in r])
                fam = np.array([rr["family"] for r in rows[g] for rr in r])
                energies[(key, g)] = E
                per_group[g] = dict(auc=auc(-E, y), by_family={f: auc(-E[fam == f], y[fam == f]) for f in np.unique(fam)},
                                    E_feasible_p50=float(np.median(E[y])) if y.any() else None,
                                    E_feasible_p95=float(np.percentile(E[y], 95)) if y.any() else None,
                                    E_feasible_max=float(E[y].max()) if y.any() else None,
                                    E_infeasible_p05=float(np.percentile(E[~y], 5)) if (~y).any() else None,
                                    E_infeasible_p50=float(np.median(E[~y])) if (~y).any() else None)
                E_all.append(E); y_all.append(y); fam_all.append(fam); grp_all.append(np.full(E.shape[0], g))
            E_all = np.concatenate(E_all); y_all = np.concatenate(y_all); fam_all = np.concatenate(fam_all)
            thr, j = youden_threshold(E_all, y_all)
            cand = dict(key=key, eps_hinge=eps, iters=it, auc_pooled=auc(-E_all, y_all), per_group=per_group,
                        youden_threshold_E=thr, youden_J=j, mappings={})
            for w in args.weights:
                cand["mappings"][f"linear_w{w:g}"] = dict(w=w, knee=0.0, steep=0.0) | mapping_stats(E_all, y_all, w, 0.0, 0.0, 1.0)
            knee = thr
            ke = 0.125        # softplus width as a fraction of the knee: the floor steep * ke * softplus(-8) is ~4e-5 steep per step
            for steep in (1.0, 3.0, 10.0):
                cand["mappings"][f"knee_s{steep:g}"] = dict(w=0.0, knee=knee, steep=steep, knee_eps=ke) | mapping_stats(E_all, y_all, 0.0, knee, steep, ke)
            cand["mappings"]["twoterm_w1_s3"] = dict(w=1.0, knee=knee, steep=3.0, knee_eps=ke) | mapping_stats(E_all, y_all, 1.0, knee, 3.0, ke)
            results["candidates"].append(cand)
            print(f"[{key}] AUC pooled {cand['auc_pooled']:.4f} " + " ".join(f"{g}:{v['auc']:.4f}" for g, v in per_group.items())
                  + f" youden thr {thr:.2e} J {j:.3f} ({time.perf_counter() - t_start:.0f}s)", flush=True)
            (out / "calibration.json").write_text(json.dumps(results, indent=1, default=float))
    np.savez(out / "energies.npz", **{f"{k[0]}__{k[1]}": v for k, v in energies.items()})

    # 4. coverage at the continuous ceiling per (hinge, mapping) candidate
    if not args.skip_coverage:
        cov = {}
        for cand in results["candidates"]:
            if cand["iters"] != args.iters[0]:
                continue
            wts = ForceWeights(eps_hinge=cand["eps_hinge"], iters=cand["iters"])
            for mk, m in cand["mappings"].items():
                if mk.startswith("linear") and m["w"] not in (1.0, 20.0, 50.0):
                    continue
                en = EnergyWeights(force=m["w"], force_knee=m["knee"] if m["steep"] > 0 else 1.0, force_steep=m["steep"],
                                   force_knee_eps=m.get("knee_eps", 0.25))
                key = f"{cand['key']}/{mk}"
                cov[key] = {}
                for name, (hm, topo, seeds) in seeds_of.items():
                    cov[key][name] = coverage_ratio(name, hm, topo, seeds, sp, task, wts, en, args.coverage_taus)
                    r = cov[key][name]["by_tau"]
                    print(f"[coverage {key}] {name}: " + " ".join(f"tau {t}: T {v['ratio_T']:.3f} R {v['ratio_R']:.3f}" for t, v in r.items())
                          + f" ({time.perf_counter() - t_start:.0f}s)", flush=True)
                results["coverage"] = cov
                (out / "calibration.json").write_text(json.dumps(results, indent=1, default=float))
    print("done")


def task_spec_torque():
    from dkarma.core.actuation import load_actuation_spec
    return load_actuation_spec().torque_limit_nm


if __name__ == "__main__":
    main()
