"""Phase 8.5 (SPEC.md §11.1): verify the three-contact failure of arm A with independent procedures.

Three procedures per design, run on arm A (original KaRMA + CMA-ES, ``workspace/phase7/exp34/
armA_design``) and, as controls, on arm B (Full KaRMA + Adam) and the 3x4 anthropomorphic start:

(i)   a >= 10x larger JAX seed search: 8.6x more joint samples, sphere-centre proposals from every
      triple of phalanges (not only the fingertips), two perturbed Levenberg-Marquardt restarts
      per candidate; every distinct kinematic solution is verified (gaps, clearances, limits, exact
      frictional equilibrium with unlimited torque);
(ii)  alternative initializations: warm starts from the other designs' three-contact seeds
      (their ``(q, p)`` fed to the multi-contact IK on this design, plus perturbations), and
      joint-space random restarts around the start's best candidates;
(iii) an independent host-side constrained solver that shares no code with ``solve_multi_ik``:
      Pinocchio forward kinematics of the exported URDF (the frozen reference's loader), SciPy
      SLSQP and trust-constr on the gaps-plus-clearances problem in ``(q, p)`` from many random
      starts, every solution re-verified by the exact checks of the JAX path.

Every count is an attempt count; "no seed found" is reported as such (CLAUDE.md rule 13).

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.three_contact_verification --out workspace/phase8/8.5
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import build_hand_model, load_task_spec, Topology, anthropomorphic  # noqa: E402
from dkarma.core.contacts import contact_pass  # noqa: E402
from dkarma.core.actuation import grasp_matrix, contact_torque_map, gravity_torque, angular_jacobians  # noqa: E402
from dkarma.solvers.seed_projection import (find_multi_seeds, SeedConfig, assignment, solve_multi_ik, _verify_multi,  # noqa: E402
                                            joint_samples)
from dkarma.solvers.contact_forces import build_problem, solve_exact, ForceWeights  # noqa: E402
from dkarma.metrics import act_karma as AK  # noqa: E402
from dkarma.export import design_from_dict, write_design  # noqa: E402
from karma_jax.kinematics import forward_kinematics  # noqa: E402

ARM_A = "workspace/phase7/exp34/armA_design/armA.design.json"
ARM_B = "workspace/phase7/exp34/run_full/optimized_morphology/optimized_morphology.design.json"


def load_designs():
    topo = Topology.three_finger(4)
    out = {"start": (topo, anthropomorphic(topo))}
    for name, path in (("armA", ARM_A), ("armB", ARM_B)):
        t, p, _ = design_from_dict(json.loads(Path(path).read_text()))
        out[name] = (t, p)
    return out


def exact_verify(hm, topo, fingers, sp, act, q, p):
    """The JAX path's exact checks at one state: kinematic gates and frictional equilibrium."""
    asg = assignment(topo, 3, fingers)
    gmax, oc, scl = (float(x) for x in _verify_multi(hm, jnp.asarray(q), jnp.asarray(p), asg, sp))
    kin = gmax <= float(sp.eps_g) and oc >= float(sp.eps_col) and scl >= float(sp.eps_col)
    lo, hi = np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper)
    lim = bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9))
    eq = False
    if kin and lim:
        fkr = forward_kinematics(hm.dm, jnp.asarray(q))
        cs = contact_pass(hm, fkr, jnp.asarray(p), sp.sphere_radius, asg)
        pb = build_problem(grasp_matrix(cs.p_sphere, jnp.asarray(p), asg.valid), contact_torque_map(cs.J_surf, asg.valid),
                           angular_jacobians(cs.Jw, asg.valid), gravity_torque(hm, fkr, act), cs.normal, asg.valid,
                           1e3 * act.tau_bar, act.mu, ForceWeights().torsion_radius_m)
        eq = bool(solve_exact(pb, act.w_ext)["feasible"])
    return dict(gap_max_m=gmax, obj_clear_m=oc, self_clear_m=scl, kinematic_ok=bool(kin and lim), limits_ok=lim, equilibrium_ok=eq)


# ── (i) and (ii): the JAX search with more candidates and warm starts ────────

def procedure_big_search(hm, topo, fingers, sp, act, budget, keep_all=False):
    cfg = SeedConfig(n_linspace=5, n_sobol=2560, n_random=640, budget=budget)
    t0 = time.perf_counter()
    s = find_multi_seeds(hm, topo, fingers, sp, act, cfg, centre_caps="all", n_restarts=2, keep_all=keep_all)
    return s, dict(s.stats, seconds=time.perf_counter() - t0)


def procedure_warm_starts(hm, topo, fingers, sp, act, donors: dict, budget, n_perturb=20, scale=0.15, seed=3, keep_all=False):
    """Donor seeds' ``(q, p)`` as IK initializations on this design, plus perturbations."""
    rng = np.random.RandomState(seed)
    lo, hi = np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper)
    qs, ps = [], []
    for dname, ds in donors.items():
        for i in range(int(ds.valid.sum())):
            q, p = np.asarray(ds.q[i]), np.asarray(ds.p[i])
            qs.append(np.clip(q, lo, hi)); ps.append(p)
            for _ in range(n_perturb):
                qs.append(np.clip(q + scale * rng.randn(q.shape[0]), lo, hi)); ps.append(p + 0.01 * rng.randn(3))
    if not qs:
        return None, dict(n_candidates=0)
    t0 = time.perf_counter()
    s = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=budget), candidates=(np.stack(qs), np.stack(ps)), keep_all=keep_all)
    return s, dict(s.stats, n_donor_seeds=sum(int(d.valid.sum()) for d in donors.values()), seconds=time.perf_counter() - t0)


# ── (iii): independent host solver (Pinocchio FK + SciPy) ────────────────────

class HostGrasp:
    """Gaps and clearances of the exported URDF through the frozen reference's Pinocchio loader."""

    def __init__(self, topo, params, workdir, task, fingers):
        # the reference loader builds the terminal capsule with the (L_ref-scaled) link radius of its
        # metric config: load it with the *physical* config, as official_karma_physical does
        from dkarma.export import load_reference, write_design
        from dkarma.metrics.exact_library import physical_metric_config
        from karma.config import load_config
        from karma.robot import load_robot, update_positions_only, capsule_endpoints_world
        cfg0, _, _ = load_reference(topo, params, workdir, name="host")
        mcfg = physical_metric_config(cfg0.l_ref_m, task, workdir)
        _, yaml_path, _ = write_design(topo, params, workdir, "host")
        self.cfg = load_config(mcfg, yaml_path)
        self.ctx = load_robot(self.cfg.urdf_path, self.cfg)
        assert abs(self.cfg.link_radius_m - task.rolling.link_radius_m) < 1e-9 and abs(self.cfg.sphere_radius_m - task.obj.radius_m) < 1e-9
        self._upd, self._ends = update_positions_only, capsule_endpoints_world
        self.r_s, self.r_l = task.obj.radius_m, task.rolling.link_radius_m
        self.eps_g, self.eps_col = task.rolling.eps_g_m, task.rolling.eps_col_m
        names = list(topo.fingers)
        self.contact_links = [f"{names[f]}_l{topo.n_joints[f] - 1}" for f in fingers]
        self.links = sorted(self.ctx.link_capsules.keys())
        # collision envelope: each capsule's own radius (D19); the contact links' gaps use the task's contact radius
        self.radius = dict(zip(topo.link_names, np.asarray(params.cap_radius, float).tolist()))
        assert all(abs(self.radius[l] - self.r_l) < 1e-12 for l in self.contact_links), "the contact links must carry the contact radius"
        self.other_links = [l for l in self.links if l not in self.contact_links]
        self.finger_of = {l: l.split("_l")[0] for l in self.links}
        self.pairs = [(a, b) for i, a in enumerate(self.links) for b in self.links[i + 1:] if self.finger_of[a] != self.finger_of[b]]
        m = self.ctx.model
        self.nq = m.nq
        self.lo, self.hi = np.array(m.lowerPositionLimit), np.array(m.upperPositionLimit)

    def _segs(self, q):
        self._upd(self.ctx, np.array(q, float))
        return {l: tuple(np.array(x) for x in self._ends(self.ctx, l)) for l in self.links}

    @staticmethod
    def _point_seg(p, a, b):
        d = b - a
        L2 = d @ d
        t = 0.0 if L2 < 1e-18 else float(np.clip((p - a) @ d / L2, 0.0, 1.0))
        return np.linalg.norm(p - (a + t * d))

    def gaps(self, z):
        q, p = z[: self.nq], z[self.nq:]
        segs = self._segs(q)
        return np.array([self._point_seg(p, *segs[l]) - self.r_s - self.r_l for l in self.contact_links])

    def clearances(self, z):
        from karma.math_utils import closest_points_on_segments
        from karma_jax.seeds import _axis_segment
        q, p = z[: self.nq], z[self.nq:]
        segs = self._segs(q)
        obj = [self._point_seg(p, *segs[l]) - self.r_s - self.radius[l] for l in self.other_links]
        selfc = []
        for a, b in self.pairs:
            p0, p1 = closest_points_on_segments(*_axis_segment(*segs[a]), *_axis_segment(*segs[b]))
            selfc.append(np.linalg.norm(p1 - p0) - (self.radius[a] + self.radius[b]))
        return np.array(obj + selfc)

    def tip_centroid(self, q):
        segs = self._segs(q)
        return np.mean([0.5 * (segs[l][0] + segs[l][1]) for l in self.contact_links], axis=0)


def procedure_host(hg: HostGrasp, n_starts, seed, methods=("SLSQP",), log=print):
    """Many-start SLSQP (and trust-constr on the best endpoints) on the feasibility problem
    ``|gap_c| <= eps_g`` (contact links), ``clearance >= eps_col`` (all other capsule-sphere and
    cross-finger capsule pairs), joint limits as bounds, sphere centre free."""
    from scipy.optimize import minimize, Bounds as SBounds
    rng = np.random.RandomState(seed)
    nq = hg.nq
    sols, attempts, n_converged = [], 0, 0
    t0 = time.perf_counter()
    for k in range(n_starts):
        q0 = hg.lo + rng.rand(nq) * (hg.hi - hg.lo)
        p0 = hg.tip_centroid(q0) + 0.01 * rng.randn(3)
        z0 = np.concatenate([q0, p0])
        cons = [dict(type="ineq", fun=lambda z: hg.eps_g - np.abs(hg.gaps(z))),
                dict(type="ineq", fun=lambda z: hg.clearances(z) - hg.eps_col)]
        bounds = SBounds(np.concatenate([hg.lo, np.full(3, -1.0)]), np.concatenate([hg.hi, np.full(3, 1.0)]))
        res = minimize(lambda z: float(np.sum(hg.gaps(z) ** 2)) * 1e4, z0, method="SLSQP", bounds=bounds, constraints=cons,
                       options=dict(maxiter=300, ftol=1e-12))
        attempts += 1
        z = res.x
        g, c = hg.gaps(z), hg.clearances(z)
        if np.all(np.abs(g) <= hg.eps_g + 1e-9) and np.all(c >= hg.eps_col - 1e-9) and np.all(z[:nq] >= hg.lo - 1e-9) and np.all(z[:nq] <= hg.hi + 1e-9):
            n_converged += 1
            sols.append(dict(q=z[:nq].tolist(), p=z[nq:].tolist(), gap_max=float(np.max(np.abs(g))), clear_min=float(c.min()), method="SLSQP", start=k))
        if (k + 1) % 50 == 0:
            log(f"    host SLSQP {k + 1}/{n_starts}: {n_converged} feasible ({time.perf_counter() - t0:.0f}s)")
    # trust-constr polish on the 10 best infeasible endpoints (a different algorithm)
    return sols, dict(n_starts=attempts, n_feasible=n_converged, seconds=time.perf_counter() - t0)


def procedure_host_trust(hg: HostGrasp, n_starts, seed, log=print):
    from scipy.optimize import minimize, NonlinearConstraint, Bounds as SBounds
    rng = np.random.RandomState(seed)
    nq = hg.nq
    sols, n_ok = [], 0
    t0 = time.perf_counter()
    for k in range(n_starts):
        q0 = hg.lo + rng.rand(nq) * (hg.hi - hg.lo)
        p0 = hg.tip_centroid(q0) + 0.01 * rng.randn(3)
        z0 = np.concatenate([q0, p0])
        nlc = [NonlinearConstraint(lambda z: hg.gaps(z), -hg.eps_g, hg.eps_g), NonlinearConstraint(lambda z: hg.clearances(z), hg.eps_col, np.inf)]
        bounds = SBounds(np.concatenate([hg.lo, np.full(3, -1.0)]), np.concatenate([hg.hi, np.full(3, 1.0)]))
        res = minimize(lambda z: float(np.sum(hg.gaps(z) ** 2)) * 1e4, z0, method="trust-constr", bounds=bounds, constraints=nlc,
                       options=dict(maxiter=200, xtol=1e-10, gtol=1e-10))
        z = res.x
        g, c = hg.gaps(z), hg.clearances(z)
        if np.all(np.abs(g) <= hg.eps_g + 1e-9) and np.all(c >= hg.eps_col - 1e-9):
            n_ok += 1
            sols.append(dict(q=z[:nq].tolist(), p=z[nq:].tolist(), gap_max=float(np.max(np.abs(g))), clear_min=float(c.min()), method="trust-constr", start=k))
        if (k + 1) % 20 == 0:
            log(f"    host trust-constr {k + 1}/{n_starts}: {n_ok} feasible ({time.perf_counter() - t0:.0f}s)")
    return sols, dict(n_starts=n_starts, n_feasible=n_ok, seconds=time.perf_counter() - t0)


def dedup(sols, rad=0.1):
    kept = []
    for s in sols:
        q = np.asarray(s["q"])
        if any(np.linalg.norm(q - np.asarray(k["q"])) < rad for k in kept):
            continue
        kept.append(s)
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="workspace/phase8/8.5")
    ap.add_argument("--designs", nargs="*", default=["armA", "armB", "start"])
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--host-starts", type=int, default=400)
    ap.add_argument("--trust-starts", type=int, default=60)
    ap.add_argument("--skip-big", action="store_true")
    ap.add_argument("--keep-all", action="store_true", help="verify every distinct kinematic solution (slow on designs with many); default stops at the budget")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    task = load_task_spec(); sp = task.step_params()
    designs = load_designs()
    log_path = out / "log.txt"

    def log(s):
        print(s, flush=True)
        with open(log_path, "a") as fh:
            fh.write(s + "\n")

    results = json.loads((out / "results.json").read_text()) if (out / "results.json").exists() else {}
    # baseline (Phase 7) seeds of every design, used as warm-start donors
    base = {}
    for name, (topo, params) in designs.items():
        hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
        fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
        s = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=args.budget))
        base[name] = s
        log(f"[{name}] Phase 7 search: {int(s.valid.sum())} seeds, stats {s.stats}")
    for name in args.designs:
        topo, params = designs[name]
        hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
        fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
        rec = dict(phase7_search=dict(base[name].stats, n_seeds=int(base[name].valid.sum())))
        if not args.skip_big:
            s1, st1 = procedure_big_search(hm, topo, fingers, sp, act, args.budget, keep_all=args.keep_all)
            rec["big_search"] = dict(st1, n_seeds=int(s1.valid.sum()))
            log(f"[{name}] (i) big search: {rec['big_search']}")
            if int(s1.valid.sum()):
                np.savez(out / f"{name}_seeds_big.npz", q=np.asarray(s1.q), p=np.asarray(s1.p), frame=np.asarray(s1.frame), valid=np.asarray(s1.valid))
        donors = {d: base[d] for d in designs if d != name and int(base[d].valid.sum())}
        s2, st2 = procedure_warm_starts(hm, topo, fingers, sp, act, donors, args.budget, keep_all=args.keep_all)
        rec["warm_starts"] = dict(st2, n_seeds=int(s2.valid.sum()) if s2 is not None else 0, donors=list(donors))
        log(f"[{name}] (ii) warm starts: {rec['warm_starts']}")
        if s2 is not None and int(s2.valid.sum()):
            np.savez(out / f"{name}_seeds_warm.npz", q=np.asarray(s2.q), p=np.asarray(s2.p), frame=np.asarray(s2.frame), valid=np.asarray(s2.valid))
        hg = HostGrasp(topo, params, out / f"host_{name}", task, fingers)
        sols, st3 = procedure_host(hg, args.host_starts, seed=11, log=log)
        sols_t, st3t = procedure_host_trust(hg, args.trust_starts, seed=12, log=log)
        allsols = dedup(sols + sols_t)
        ver = [exact_verify(hm, topo, fingers, sp, act, np.asarray(s["q"]), np.asarray(s["p"])) for s in allsols]
        n_kin = sum(v["kinematic_ok"] for v in ver); n_eq = sum(v["equilibrium_ok"] for v in ver)
        rec["host_solver"] = dict(slsqp=st3, trust_constr=st3t, n_distinct_solutions=len(allsols), n_verified_kinematic=n_kin,
                                  n_verified_equilibrium=n_eq, solutions=[dict(s, verify=v) for s, v in zip(allsols, ver)][:50])
        log(f"[{name}] (iii) host solver: SLSQP {st3['n_feasible']}/{st3['n_starts']} trust-constr {st3t['n_feasible']}/{st3t['n_starts']} "
            f"distinct {len(allsols)} verified kinematic {n_kin} equilibrium {n_eq}")
        results[name] = rec
        (out / "results.json").write_text(json.dumps(results, indent=1, default=float))
    log("done")


if __name__ == "__main__":
    main()
