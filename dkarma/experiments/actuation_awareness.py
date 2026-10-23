"""Experiment II (SPEC.md §12): does actuation awareness change the prediction?

Hold the kinematics fixed and lower the torque ceiling progressively (stall → primary →
continuous → below); also raise the object mass. For each setting report the kinematic
(Diff-KaRMA) coverage, which must not move, the actuation-feasible (Act-KaRMA) smooth
coverage, the exact force-feasible fraction of the library's exactly valid states (SLSQP
at every primitive end: the independent quasi-static check), and the number of states the
kinematic evaluator accepts that the actuation-aware one rejects.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.actuation_awareness --out workspace/phase7/exp2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import build_hand_model, load_task_spec, Topology, anthropomorphic  # noqa: E402
from dkarma.solvers.seed_projection import find_seeds, SeedConfig, assignment  # noqa: E402
from dkarma.solvers.rollout import GraspState  # noqa: E402
from dkarma.solvers.contact_forces import solve_exact  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, ACT_KARMA  # noqa: E402
from dkarma.metrics import diff_karma as DK, act_karma as AK  # noqa: E402
from dkarma.metrics.exact_library import evaluate_exact  # noqa: E402
from dkarma.tests.conftest import load_design  # noqa: E402


def pert_design():
    """The perturbed two-finger design of Experiment II: the baseline's unconstrained coordinates
    plus ``0.25 * RandomState(1).randn`` (Phase 7, reproduced exactly)."""
    from dkarma.core import Bounds, encode, decode, fixed_from_params
    t, p, _ = load_design("baseline")
    B = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
    rng = np.random.RandomState(1)
    phi = jax.tree_util.tree_map(lambda a: a + 0.25 * rng.randn(*a.shape), encode(p, t, B))
    return t, decode(phi, t, B, fixed_from_params(p))


def designs(names):
    out = []
    for n in names:
        if n == "anthro3":
            t = Topology.three_finger(4); out.append((n, t, anthropomorphic(t)))
        elif n == "pert":
            t, p = pert_design(); out.append((n, t, p))
        elif n.endswith(".json"):                         # Phase 9.5: a dkarma design JSON (fabrication candidate)
            from dkarma.export import design_from_dict
            t, p, nm = design_from_dict(json.loads(Path(n).read_text())); out.append((Path(n).stem.replace(".design", ""), t, p))
        else:
            t, p, _ = load_design(n); out.append((n, t, p))
    return out


def exact_force_fraction(hm, topo, seeds, sp, task, cfg, act):
    """Among exactly valid primitive-end states of the library, the fraction whose exact force
    allocation is feasible under ``act`` (and the count rejected)."""
    ex = evaluate_exact(hm, topo, seeds, sp, task, cfg)
    asg = assignment(topo)
    n_ok = n_states = 0
    n_seed_ok = 0
    rejected = []
    for s, r in enumerate(ex.per_seed):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        # the seed state itself (Phase 8.1: every state, the seed included, must hold the object)
        seed_feas = solve_exact(AK.force_problem_at(hm, GraspState(q, p, jnp.eye(3)), asg, sp, act), act.w_ext)["feasible"]
        n_states += 1; n_ok += seed_feas; n_seed_ok += seed_feas
        if not seed_feas:
            rejected.append(dict(seed=s, command=-1, primitive=-1))
        from dkarma.metrics.commands import build_library
        from karma_jax.kinematics import forward_kinematics
        from dkarma.core.contacts import contact_pass
        n0 = contact_pass(hm, forward_kinematics(hm.dm, q), p, sp.sphere_radius, asg).normal[0]
        lib = build_library(frame, -n0, cfg.commands)
        from dkarma.solvers.exact import validate_translation
        exT = jax.vmap(lambda a: validate_translation(hm, GraspState(q, p, jnp.eye(3)), a, asg, sp, cfg.commands.n_sub))(lib.dp_T)
        okT = np.asarray(exT.ok)
        for u in range(okT.shape[0]):
            for k in range(okT.shape[1]):
                if not okT[u, k]:
                    break
                st = GraspState(exT.q[u, k + 1], exT.p[u, k + 1], exT.R[u, k + 1])
                pb = AK.force_problem_at(hm, st, asg._replace(caps=jnp.asarray(exT.pair[u, k + 1])), sp, act)
                feas = solve_exact(pb, act.w_ext)["feasible"]
                n_states += 1; n_ok += feas
                if not feas:
                    rejected.append(dict(seed=s, command=u, primitive=k))
    return dict(n_states=n_states, n_force_feasible=n_ok, fraction=(n_ok / n_states if n_states else None), rejected=rejected[:20],
                n_seeds=len(ex.per_seed), n_seed_force_feasible=n_seed_ok, exact_K_T=ex.K_T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase7/exp2.json")
    ap.add_argument("--designs", nargs="*", default=["baseline", "pert"])
    ap.add_argument("--torques", nargs="*", type=float, default=[0.215, 0.108, 0.043, 0.02, 0.01])
    ap.add_argument("--masses", nargs="*", type=float, default=[0.035, 0.07, 0.14])
    ap.add_argument("--budget", type=int, default=6)
    args = ap.parse_args()
    task = load_task_spec(); sp = task.step_params()
    cfg_kin = DIFF_KARMA; cfg_act = ACT_KARMA
    ds = designs(args.designs)
    results = []
    for name, topo, params in ds:
        hm = build_hand_model(topo, params)
        seeds = find_seeds(hm, topo, sp, SeedConfig(budget=args.budget))
        if int(seeds.valid.sum()) == 0:
            results.append(dict(design=name, skipped="no seeds")); continue
        act0 = AK.actuation_for(hm, task)
        d_kin = jax.jit(lambda h: DK.evaluate(h, topo, seeds, sp, task, cfg_kin))(hm)
        rec = dict(design=name, n_seeds=int(seeds.valid.sum()), kinematic=dict(K_T=float(np.asarray(d_kin.K_T)[np.asarray(seeds.valid)].mean()),
                                                                            K_R=float(np.asarray(d_kin.K_R)[np.asarray(seeds.valid)].mean()), score=float(d_kin.score)), sweeps=[])
        ev = jax.jit(lambda h, a: DK.evaluate(h, topo, seeds, sp, task, cfg_act, None, a))
        for tau in args.torques:
            act = act0._replace(tau_bar=jnp.full_like(act0.tau_bar, tau))
            d = ev(hm, act)
            s = AK.summarize(d)
            ff = exact_force_fraction(hm, topo, seeds, sp, task, cfg_act, act)
            rec["sweeps"].append(dict(kind="torque", tau_bar=tau, mass=task.obj.mass_kg, smooth=dict(K_T=float(np.asarray(d.K_T)[np.asarray(seeds.valid)].mean()),
                                 K_R=float(np.asarray(d.K_R)[np.asarray(seeds.valid)].mean()), score=float(d.score),
                                 torque_usage_max=s["torque_usage_max"], force_feasible_step_frac=s["force_feasible_step_frac"]), exact=ff))
            print(f"[{name}] tau {tau:.3f}: smooth K_T {rec['sweeps'][-1]['smooth']['K_T']:.5f} (kin {rec['kinematic']['K_T']:.5f}) "
                  f"exact force-feasible {ff['n_force_feasible']}/{ff['n_states']}", flush=True)
        for m in args.masses:
            act = act0._replace(w_ext=act0.w_ext * (m / task.obj.mass_kg))
            d = ev(hm, act); s = AK.summarize(d)
            ff = exact_force_fraction(hm, topo, seeds, sp, task, cfg_act, act)
            rec["sweeps"].append(dict(kind="mass", tau_bar=float(act0.tau_bar[0]), mass=m, smooth=dict(K_T=float(np.asarray(d.K_T)[np.asarray(seeds.valid)].mean()),
                                 K_R=float(np.asarray(d.K_R)[np.asarray(seeds.valid)].mean()), score=float(d.score),
                                 torque_usage_max=s["torque_usage_max"], force_feasible_step_frac=s["force_feasible_step_frac"]), exact=ff))
            print(f"[{name}] mass {m:.3f}: smooth K_T {rec['sweeps'][-1]['smooth']['K_T']:.5f} exact force-feasible {ff['n_force_feasible']}/{ff['n_states']}", flush=True)
        results.append(rec)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
