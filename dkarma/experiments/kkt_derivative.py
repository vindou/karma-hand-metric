"""Phase 9.1 (SPEC.md §11.2): the derivative of the hard rolling QP at degenerate active sets.

Three stages, each a sub-command:

``trace``: re-run one Phase 8.2 Adam run (objective, variant, start) without validations and
record every iterate, its raw gradient, finiteness and the seed set in use
(``optimizers.adam(iterate_log=...)``); the recorded gradient norms are compared with the
Phase 8 ``run.json`` to show that the trace reproduces the run.

``corpus``: at chosen iterations of the traces (every non-finite iteration, the largest finite
gradients, and regular well-conditioned iterations) rebuild the design and the seed set, replay
every command of the library with the hard solver and record every rolling QP: ``(P, c, A, l, u,
row_mask, x_qp, qp_ok)``, the identified active set, its rank deficiency, the condition number
of the KKT matrix and the norm of the current (``exact``) derivative's vector-Jacobian product
for a fixed cotangent. States whose exact derivative is non-finite or whose VJP norm exceeds a
threshold are the *degenerate* corpus; a random sample of the others is the *well-conditioned*
corpus. The corpus is stored as ``corpus_T.npz`` / ``corpus_R.npz`` (translation and rotation QPs
have different shapes) with a JSON index.

``compare``: for every corpus state and every derivative method of
``dkarma.solvers.rolling_qp.kkt_resolve`` (exact, regularized at several levels, pseudoinverse,
dependent rows removed), the directional derivative of the KKT solution along random
perturbations of the problem data against central finite differences of the tightly solved QP
where the active set is stable on both sides, and against both one-sided differences where it
changes. Writes ``compare.json`` and ``compare.md``.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.kkt_derivative trace --objective diff2 --variant phase7 --start start1_sobol0_s0.25 --out workspace/phase9/9.1
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.kkt_derivative corpus --out workspace/phase9/9.1
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.kkt_derivative compare --out workspace/phase9/9.1
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

from karma_jax.kinematics import forward_kinematics  # noqa: E402
from karma_jax.qp import QPSettings, default_settings, solve_box_qp  # noqa: E402
from dkarma.core import load_task_spec, decode, fixed_from_params, build_hand_model  # noqa: E402
from dkarma.core.contacts import contact_pass  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig, SeedSet, assignment, reproject  # noqa: E402
from dkarma.solvers.rolling_qp import assemble, solve_hard, kkt_resolve, KKT_METHODS  # noqa: E402
from dkarma.solvers.rollout import GraspState, step  # noqa: E402
from dkarma.metrics.commands import build_library  # noqa: E402
from dkarma.metrics import multicontact as MC  # noqa: E402
from dkarma.metrics.continuation import schedule  # noqa: E402
from dkarma.optimization import make_objective, adam  # noqa: E402
from dkarma.experiments.continuation_study import objective_spec, make_starts, VARIANTS, BOUNDS  # noqa: E402

TIGHT = QPSettings(iters=80, tol=1e-13, infeas_tol=1e-6, reg=1e-13)   # FD reference solve (dkarma/tests/test_rolling_qp.py)
METHODS = [("exact", dict(method="exact", reg=1e-10)),
           ("reg1e-8", dict(method="reg", reg=1e-8)), ("reg1e-6", dict(method="reg", reg=1e-6)), ("reg1e-4", dict(method="reg", reg=1e-4)),
           ("pinv", dict(method="pinv", rcond=1e-10)),
           ("reduced", dict(method="reduced", dep_tol=1e-6)), ("reduced1e-8", dict(method="reduced", dep_tol=1e-8)),
           ("reduced1e-4", dict(method="reduced", dep_tol=1e-4))]


# ── trace ────────────────────────────────────────────────────────────────────

def trace(objective: str, variant: str, start: str, out: Path, phase8_root: Path, log=print, stage_select=None):
    task = load_task_spec()
    spec = objective_spec(objective)
    starts, _ = make_starts(spec, task, 5, 0.25, phase8_root / objective, log=log)
    phi = dict(starts)[start]
    kw = dict(VARIANTS[variant])
    policy = kw.pop("seed_policy", "reproject")
    if stage_select is not None:
        kw["stage_select"] = stage_select
    topo, cfg = spec["topo"], spec["cfg"]
    params = decode(phi, topo, BOUNDS, fixed_from_params(spec["params0"]))
    obj = make_objective(topo, params, BOUNDS, task, cfg, SeedConfig(budget=spec["budget"]), seed_policy=policy, reseed_every=10)
    log_ = []
    t0 = time.perf_counter()
    res = adam(obj, phi, n_iters=spec["n_iters"], stages=spec["stages"], validate_every=10**9, validation_fn=None, log=log, seed=0,
               instrument=False, iterate_log=log_, **kw)
    wall = time.perf_counter() - t0
    its = [r for r in log_ if "iter" in r]
    seeds = {r["seed_id"]: r["seeds"] for r in log_ if "seeds" in r}
    tdir = out / f"trace_{objective}_{variant}_{start}"; tdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(tdir / "iterates.npz", x=np.stack([r["x"] for r in its]), grad=np.stack([r["grad"] for r in its]),
                        iter=np.array([r["iter"] for r in its]), stage=np.array([r["stage"] for r in its]),
                        finite=np.array([r["finite"] for r in its]), seed_id=np.array([r["seed_id"] for r in its]),
                        smooth=np.array([r["smooth"] for r in its]),
                        **{f"seeds{k}_{f}": np.asarray(getattr(s, f)) for k, s in seeds.items() for f in ("q", "p", "frame", "valid")})
    # reproduction check against the Phase 8 run
    rj = phase8_root / objective / variant / start / "run.json"
    rep = {}
    if rj.exists():
        th = json.loads(rj.read_text())["training_history"]
        g8 = np.array([h["grad_norm"] for h in th]); f8 = np.array([h["grad_finite"] for h in th])
        g9 = np.array([np.linalg.norm(np.where(np.isfinite(r["grad"]), r["grad"], 0.0)) for r in its]); f9 = np.array([r["finite"] for r in its])
        n = min(len(g8), len(g9))
        both = f8[:n] & f9[:n]
        rel = np.abs(g9[:n] - g8[:n]) / np.maximum(g8[:n], 1e-12)
        rep = dict(n_iters=n, nonfinite_phase8=[int(h["iter"]) for h in th if not h["grad_finite"]],
                   nonfinite_trace=[int(r["iter"]) for r in its if not r["finite"]],
                   same_finiteness=int(np.sum(f8[:n] == f9[:n])), max_rel_grad_norm_dev=float(rel[both].max()) if both.any() else None,
                   median_rel_grad_norm_dev=float(np.median(rel[both])) if both.any() else None,
                   smooth_phase8_first_last=[th[0]["smooth"], th[-1]["smooth"]], smooth_trace_first_last=[its[0]["smooth"], its[-1]["smooth"]])
    meta = dict(objective=objective, variant=variant, start=start, options=res.meta["options"], schedule=res.meta["schedule"], wall_s=wall,
                n_iters=len(its), reproduction=rep, stage_select_override=stage_select)
    (tdir / "trace.json").write_text(json.dumps(meta, indent=1, default=float))
    log(f"trace {objective}/{variant}/{start}: {len(its)} iterates, non-finite {sum(not r['finite'] for r in its)}, wall {wall:.0f}s, reproduction {rep}")
    return tdir


# ── corpus ───────────────────────────────────────────────────────────────────

def recording_rollout(hm, st0, dps, dths, asg, sp, roll_cfg, act, qp_settings):
    """Hard-mode rollout that also returns every step's rolling problem and QP solution."""
    free_theta = roll_cfg.kind == "translation"

    def body(st, cmd):
        dp, dth = cmd
        fkr = forward_kinematics(hm.dm, st.q)
        cs = contact_pass(hm, fkr, st.p, sp.sphere_radius, asg)
        prob = assemble(hm, cs, asg, dp, dth, st.q, sp, free_theta=free_theta, lambda_theta=roll_cfg.lambda_theta,
                        theta_bound=roll_cfg.theta_bound)
        res = solve_box_qp(prob.P, prob.c, prob.A, prob.l, prob.u, qp_settings, row_mask=prob.row_mask)
        st1, dg = step(hm, st, dp, dth, asg, sp, roll_cfg, qp_settings, act)
        rec = dict(P=prob.P, c=prob.c, A=prob.A, l=prob.l, u=prob.u, row_mask=prob.row_mask, x_qp=res.x, qp_ok=res.success,
                   q=st.q, p=st.p)
        return st1, rec

    _, recs = jax.lax.scan(body, st0, (dps, dths))
    return recs


_REC_FNS = {}


def _rec_fn(key, roll_cfg, settings):
    """One compiled recording rollout per (topology, kind, stage config); the model, state, library,
    assignment and actuation are traced arguments so iterations do not recompile."""
    if key not in _REC_FNS:
        _REC_FNS[key] = jax.jit(lambda hm, st0, dps, dths, asg, sp, act: jax.vmap(
            lambda a, b: recording_rollout(hm, st0, a, b, asg, sp, roll_cfg, act, settings))(dps, dths))
    return _REC_FNS[key]


def record_design(topo, params, seeds: SeedSet, cfg, task, fingers=None):
    """Every rolling QP of every command of every valid seed, as flat arrays per step kind."""
    sp = task.step_params()
    hm = build_hand_model(topo, params)
    act = None
    if cfg.contacts >= 3:
        from dkarma.metrics import act_karma as AK
        act = AK.actuation_for(hm, task)
        asg = assignment(topo, 3, fingers)
    else:
        asg = assignment(topo)
    roll = cfg.roll._replace(mode="hard")
    settings = default_settings(jnp.float64)
    out = {"T": [], "R": []}
    index = []
    for s in range(seeds.q.shape[0]):
        if not bool(seeds.valid[s]):
            continue
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        q0, ok, _ = reproject(hm, q, p, asg, sp)
        st0 = GraspState(q0, p, jnp.eye(3))
        if cfg.contacts >= 3:
            dp_T, dth_T, dp_R, dth_R = MC.multi_library(frame, cfg.commands)
            roll_T = roll._replace(kind="translation", grasp_mode="equilibrium"); roll_R = roll._replace(kind="rotation", grasp_mode="equilibrium")
        else:
            n0 = contact_pass(hm, forward_kinematics(hm.dm, q0), p, sp.sphere_radius, asg).normal[0]
            lib = build_library(frame, -n0, cfg.commands)
            dp_T, dth_T, dp_R, dth_R = lib.dp_T, lib.dth_T, lib.dp_R, lib.dth_R
            roll_T = roll._replace(kind="translation"); roll_R = roll._replace(kind="rotation")
        for kind, dps, dths, rc in (("T", dp_T, dth_T, roll_T), ("R", dp_R, dth_R, roll_R)):
            fn = _rec_fn((topo, kind, rc, cfg.commands), rc, settings)
            recs = fn(hm, st0, dps, dths, asg, sp, act)
            recs = {k: np.asarray(v) for k, v in recs.items()}
            U, H = recs["x_qp"].shape[:2]
            flat = {k: v.reshape((U * H,) + v.shape[2:]) for k, v in recs.items()}
            out[kind].append(flat)
            index += [dict(kind=kind, seed=s, command=u, step=h, reproj_ok=bool(ok)) for u in range(U) for h in range(H)]
    cat = {k: ({f: np.concatenate([d[f] for d in v]) for f in v[0]} if v else None) for k, v in out.items()}
    return cat, index


def exact_vjp_norms(rec, kind_n_gap, kw, cotangent):
    """Norm of the VJP of ``x_kkt`` w.r.t. ``(P, c, A, l, u)`` for a fixed unit cotangent, per state
    (``vmap``); also the condition number of the KKT matrix and the active-row rank deficiency."""
    act_tol = 1e-7

    def one(P, c, A, l, u, x):
        def f(P, c, A, l, u):
            return kkt_resolve(P, c, A, l, u, x, act_tol, n_gap=kind_n_gap, **kw)[0]
        xk, vjp = jax.vjp(f, P, c, A, l, u)
        g = vjp(cotangent)
        gn = jnp.sqrt(sum(jnp.sum(jnp.where(jnp.isfinite(gi), gi, 0.0) ** 2) for gi in g))
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(gi)) for gi in g])) & jnp.all(jnp.isfinite(xk))
        return xk, gn, finite

    return jax.jit(jax.vmap(one))(rec["P"], rec["c"], rec["A"], rec["l"], rec["u"], rec["x_qp"])


def active_set_stats(rec, act_tol=1e-7):
    """Host: active mask, number active, rank of the active rows, deficiency, cond(K) per state."""
    P, A, l, u, x = rec["P"], rec["A"], rec["l"], rec["u"], rec["x_qp"]
    N, n = x.shape
    G = np.concatenate([A, -A], axis=1); h = np.concatenate([u, -l], axis=1)
    slack = h - np.einsum("nij,nj->ni", G, x)
    m = slack <= act_tol * (1.0 + np.abs(h))
    n_act = m.sum(1); rank = np.zeros(N, int); cond = np.zeros(N); smin = np.zeros(N)
    for i in range(N):
        Ga = G[i][m[i]]
        if Ga.shape[0]:
            sv = np.linalg.svd(Ga, compute_uv=False)
            rank[i] = int(np.sum(sv > 1e-9 * sv.max())); smin[i] = sv.min() / sv.max()
        mi = m[i].astype(float)
        K = np.block([[P[i], G[i].T * mi[None, :]], [G[i] * mi[:, None], -1e-10 * np.diag(mi) + np.diag(1.0 - mi)]])
        cond[i] = np.linalg.cond(K)
    return dict(active=m, n_active=n_act, rank=rank, deficiency=n_act - rank, cond_K=cond, sigma_ratio_active=smin)


def build_corpus(out: Path, traces, n_well=300, vjp_threshold=1e3, seed=0, log=print):
    task = load_task_spec()
    rng = np.random.RandomState(seed)
    corpus = {}            # (objective, kind) -> list of states; translation/rotation QPs and the two objectives differ in shape
    index = {}
    summary = []
    for tdir in traces:
        meta = json.loads((tdir / "trace.json").read_text())
        spec = objective_spec(meta["objective"])
        z = np.load(tdir / "iterates.npz")
        x_all, it_all, st_all, fin_all, sid_all = z["x"], z["iter"], z["stage"], z["finite"], z["seed_id"]
        gnorm = np.array([np.linalg.norm(np.where(np.isfinite(g), g, 0.0)) for g in z["grad"]])
        cfgs = schedule(spec["cfg"], n_stages=spec["stages"], **(meta["options"].get("schedule_kw") or {}))
        hard = np.array([cfgs[s].roll.mode == "hard" for s in st_all])
        # iterations to record: all non-finite, the 3 largest finite hard-stage gradients, 3 regular hard-stage iterations
        sel = set(np.nonzero(~fin_all)[0].tolist())
        fin_hard = np.nonzero(fin_all & hard)[0]
        if fin_hard.size:
            sel |= set(fin_hard[np.argsort(-gnorm[fin_hard])[:3]].tolist())
            sel |= set(fin_hard[np.linspace(0, fin_hard.size - 1, 3).astype(int)].tolist())
        # consecutive non-finite iterations under skip_nonfinite repeat the same x: dedup by x
        seen = set(); chosen = []
        for i in sorted(sel):
            key = x_all[i].round(12).tobytes()
            if key in seen:
                continue
            seen.add(key); chosen.append(i)
        topo = spec["topo"]
        fixed = fixed_from_params(spec["params0"])
        from dkarma.core import encode
        _, unravel = jax.flatten_util.ravel_pytree(encode(spec["params0"], topo, BOUNDS))
        fingers = None
        if spec["cfg"].contacts >= 3:
            fingers = (topo.thumb_finger, topo.index_finger, [f for f in range(topo.n_fingers) if f not in (topo.thumb_finger, topo.index_finger)][0])
        for i in chosen:
            params = decode(unravel(jnp.asarray(x_all[i])), topo, BOUNDS, fixed)
            k = int(sid_all[i])
            seeds = SeedSet(jnp.asarray(z[f"seeds{k}_q"]), jnp.asarray(z[f"seeds{k}_p"]), jnp.asarray(z[f"seeds{k}_frame"]), jnp.asarray(z[f"seeds{k}_valid"]), 0, {})
            cfg_i = cfgs[int(st_all[i])]
            recs, idx = record_design(topo, params, seeds, cfg_i, task, fingers)
            for kind in ("T", "R"):
                rec = recs[kind]
                if rec is None:
                    continue
                n_gap = rec["A"].shape[1] - rec["P"].shape[1]
                cot = jnp.asarray(rng.randn(rec["P"].shape[1])); cot = cot / jnp.linalg.norm(cot)
                xk, gn, finite = exact_vjp_norms({k_: jnp.asarray(v) for k_, v in rec.items() if k_ in ("P", "c", "A", "l", "u", "x_qp")}, n_gap, dict(method="exact", reg=1e-10), cot)
                gn, finite, xk = np.asarray(gn), np.asarray(finite), np.asarray(xk)
                stats = active_set_stats(rec)
                kkt_gap = np.max(np.abs(np.where(np.isfinite(xk), xk, 0.0) - rec["x_qp"]), axis=1)
                degenerate = (~finite) | (gn > vjp_threshold) | (~np.isfinite(gn))
                kind_idx = [d for d in idx if d["kind"] == kind]
                N = rec["x_qp"].shape[0]
                summary.append(dict(trace=tdir.name, iter=int(it_all[i]), stage=int(st_all[i]), kind=kind, grad_finite=bool(fin_all[i]), grad_norm=float(gnorm[i]),
                                    n_qp=N, n_qp_fail=int((~rec["qp_ok"]).sum()), n_nonfinite_vjp=int((~finite).sum()), n_large_vjp=int((finite & (gn > vjp_threshold)).sum()),
                                    n_rank_deficient=int((stats["deficiency"] > 0).sum()), median_vjp=float(np.median(gn[finite])) if finite.any() else None,
                                    max_vjp=float(gn[finite].max()) if finite.any() else None, n_degenerate=int(degenerate.sum())))
                log(f"  {tdir.name} it {it_all[i]} {kind}: {summary[-1]}")
                deg_ids = np.nonzero(degenerate)[0]
                well_ids = np.nonzero(~degenerate & rec["qp_ok"])[0]
                take = list(deg_ids) + list(rng.choice(well_ids, size=min(len(well_ids), max(n_well // (len(chosen) * len(traces)), 5)), replace=False)) if well_ids.size else list(deg_ids)
                ck = f"{meta['objective']}_{kind}"
                corpus.setdefault(ck, []); index.setdefault(ck, [])
                for j in take:
                    corpus[ck].append({f: rec[f][j] for f in ("P", "c", "A", "l", "u", "row_mask", "x_qp", "qp_ok", "q", "p")})
                    index[ck].append(dict(trace=tdir.name, iter=int(it_all[i]), stage=int(st_all[i]), grad_finite=bool(fin_all[i]), label=("degenerate" if degenerate[j] else "well"),
                                            exact_vjp_finite=bool(finite[j]), exact_vjp_norm=float(gn[j]) if np.isfinite(gn[j]) else None, kkt_gap=float(kkt_gap[j]),
                                            n_active=int(stats["n_active"][j]), rank=int(stats["rank"][j]), deficiency=int(stats["deficiency"][j]),
                                            cond_K=float(stats["cond_K"][j]), sigma_ratio_active=float(stats["sigma_ratio_active"][j]), qp_ok=bool(rec["qp_ok"][j]), **kind_idx[j]))
    for ck, states in corpus.items():
        if states:
            np.savez_compressed(out / f"corpus_{ck}.npz", **{f: np.stack([c[f] for c in states]) for f in states[0]})
    (out / "corpus_index.json").write_text(json.dumps(dict(index=index, per_iteration=summary, vjp_threshold=vjp_threshold), indent=1, default=float))
    for ck, rows in index.items():
        labels = [d["label"] for d in rows]
        log(f"corpus {ck}: {len(labels)} states, degenerate {labels.count('degenerate')}, well {labels.count('well')}")
    return summary


# ── compare ──────────────────────────────────────────────────────────────────

def _directions(rng, P, c, A, l, u, n_dirs):
    """Random perturbation directions of the problem data, each block scaled to its own size."""
    dirs = []
    for _ in range(n_dirs):
        dP = rng.randn(*P.shape); dP = (dP + dP.T) / 2 * (np.linalg.norm(P) / P.shape[0])
        dc = rng.randn(*c.shape) * max(np.linalg.norm(c) / np.sqrt(c.size), 1e-3)
        dA = rng.randn(*A.shape) * (np.linalg.norm(A) / np.sqrt(A.size))
        dA[-P.shape[0]:] = 0.0                                   # the identity rows of the box are structural
        dl = rng.randn(*l.shape) * 1e-2; du = rng.randn(*u.shape) * 1e-2
        dirs.append((dP, dc, dA, dl, du))
    return dirs


def compare(out: Path, n_dirs=3, h=1e-6, log=print, max_states=None):
    rng = np.random.RandomState(1)
    idx = json.loads((out / "corpus_index.json").read_text())["index"]
    results = []
    act_tol = 1e-7
    for kind in sorted(idx):
        if not (out / f"corpus_{kind}.npz").exists():
            continue
        z = np.load(out / f"corpus_{kind}.npz")
        N = z["x_qp"].shape[0]
        n = z["P"].shape[1]; n_gap = z["A"].shape[1] - n
        rows = idx[kind]
        ids = range(N) if max_states is None else range(min(N, max_states))
        methods_fn = {}
        for name, kw in METHODS:
            def make(kw):
                def f(P, c, A, l, u, x_qp, d):
                    def g(t):
                        return kkt_resolve(P + t * d[0], c + t * d[1], A + t * d[2], l + t * d[3], u + t * d[4], x_qp, act_tol, n_gap=n_gap, **kw)[0]
                    x0, dx = jax.jvp(g, (0.0,), (1.0,))
                    return x0, dx
                return jax.jit(f)
            methods_fn[name] = make(kw)
        solve_tight = jax.jit(lambda P, c, A, l, u, rm: solve_box_qp(P, c, A, l, u, TIGHT, row_mask=rm))

        def active(P, c, A, l, u, x):
            G = np.concatenate([A, -A]); hh = np.concatenate([u, -l])
            return (hh - G @ x) <= act_tol * (1.0 + np.abs(hh))

        # finiteness at every state (failed QPs included): forward KKT solution and VJP for a unit cotangent
        cot = jnp.asarray(rng.randn(n)); cot = cot / jnp.linalg.norm(cot)
        fin_rows = []
        for name, kw in METHODS:
            xk, gn, fin = exact_vjp_norms({k: jnp.asarray(z[k]) for k in ("P", "c", "A", "l", "u", "x_qp")}, n_gap, kw, cot)
            fin_rows.append((name, np.asarray(fin), np.asarray(gn), np.asarray(xk)))
        for i in range(N):
            results.append(dict(kind=kind, state=i, label=rows[i]["label"], qp_ok=bool(z["qp_ok"][i]), deficiency=rows[i]["deficiency"], record="finiteness",
                                methods={name: dict(finite=bool(fin[i]), vjp_norm=float(gn[i]) if np.isfinite(gn[i]) else None,
                                                    forward_finite=bool(np.all(np.isfinite(xk[i])))) for name, fin, gn, xk in fin_rows}))
        t0 = time.perf_counter()
        for i in ids:
            P, c, A, l, u, rm, x_qp = (np.asarray(z[k][i]) for k in ("P", "c", "A", "l", "u", "row_mask", "x_qp"))
            if not bool(z["qp_ok"][i]):
                continue
            base = solve_tight(P, c, A, l, u, rm)
            if not bool(base.success):
                continue
            xb = np.asarray(base.x)
            m0 = active(P, c, A, l, u, xb)
            for d in _directions(rng, P, c, A, l, u, n_dirs):
                xs = {}
                stable = True
                masks = {}
                for s in (+1, -1):
                    r = solve_tight(P + s * h * d[0], c + s * h * d[1], A + s * h * d[2], l + s * h * d[3], u + s * h * d[4], rm)
                    if not bool(r.success):
                        xs = None; break
                    xs[s] = np.asarray(r.x); masks[s] = active(P + s * h * d[0], c + s * h * d[1], A + s * h * d[2], l + s * h * d[3], u + s * h * d[4], xs[s])
                    stable &= bool(np.all(masks[s] == m0))
                if xs is None:
                    continue
                fd_c = (xs[1] - xs[-1]) / (2 * h); fd_p = (xs[1] - xb) / h; fd_m = (xb - xs[-1]) / h
                one_sided_gap = float(np.linalg.norm(fd_p - fd_m) / (np.linalg.norm(fd_c) + 1e-12))
                rec = dict(kind=kind, state=i, label=rows[i]["label"], deficiency=rows[i]["deficiency"], cond_K=rows[i]["cond_K"], record="fd",
                           active_set_stable=stable, fd_norm=float(np.linalg.norm(fd_c)), one_sided_rel_gap=one_sided_gap, methods={})
                for name, f in methods_fn.items():
                    x0, dx = f(P, c, A, l, u, xb, tuple(jnp.asarray(v) for v in d))
                    x0, dx = np.asarray(x0), np.asarray(dx)
                    fin = bool(np.all(np.isfinite(dx)) and np.all(np.isfinite(x0)))
                    den = np.linalg.norm(fd_c) + 1e-12
                    rec["methods"][name] = dict(finite=fin, value_gap=float(np.max(np.abs(x0 - xb))) if fin else None, dd_norm=float(np.linalg.norm(dx)) if fin else None,
                                                rel_err_central=float(np.linalg.norm(dx - fd_c) / den) if fin else None,
                                                rel_err_plus=float(np.linalg.norm(dx - fd_p) / (np.linalg.norm(fd_p) + 1e-12)) if fin else None,
                                                rel_err_minus=float(np.linalg.norm(dx - fd_m) / (np.linalg.norm(fd_m) + 1e-12)) if fin else None)
                results.append(rec)
            if (i + 1) % 25 == 0:
                log(f"  {kind}: {i + 1}/{N} states ({time.perf_counter() - t0:.0f}s)")
    (out / "compare.json").write_text(json.dumps(dict(results=results, methods=[m for m, _ in METHODS], n_dirs=n_dirs, h=h), indent=1, default=float))
    md = summarize_compare(results)
    (out / "compare.md").write_text(md)
    log(md)
    return results


def summarize_compare(results):
    names = list(results[0]["methods"]) if results else []
    lines = []
    fin = [r for r in results if r.get("record") == "finiteness"]
    if fin:
        lines += ["### Finiteness on the whole corpus (forward KKT solution and VJP for a unit cotangent)", "",
                  "| method | QP solved, full-rank active set | QP solved, rank-deficient active set | QP failed (value masked, derivative must only be finite) |",
                  "| --- | --- | --- | --- |"]
        groups = [("ok_full", lambda r: r["qp_ok"] and r["deficiency"] == 0), ("ok_def", lambda r: r["qp_ok"] and r["deficiency"] > 0), ("fail", lambda r: not r["qp_ok"])]
        for nm in names:
            cells = []
            for _, sel in groups:
                rs = [r for r in fin if sel(r)]
                nf = sum(1 for r in rs if r["methods"][nm]["finite"] and r["methods"][nm]["forward_finite"])
                big = sum(1 for r in rs if r["methods"][nm]["vjp_norm"] is not None and r["methods"][nm]["vjp_norm"] > 1e3)
                cells.append(f"{nf}/{len(rs)} finite, {big} with VJP > 1e3")
            lines.append(f"| {nm} | " + " | ".join(cells) + " |")
        lines.append("")
    results = [r for r in results if r.get("record", "fd") == "fd"]
    for label in ("well", "degenerate"):
        for stable in (True, False):
            rs = [r for r in results if r["label"] == label and r["active_set_stable"] == stable]
            if not rs:
                continue
            lines += [f"### {label} states, active set {'stable' if stable else 'changes'} under the FD perturbation ({len(rs)} directions, {len({(r['kind'], r['state']) for r in rs})} states)", "",
                      "| method | finite | median rel. err. vs central FD | 90 % | max | median rel. err. vs nearer one-sided FD | share within 1e-3 (central) | share within 1e-2 (one-sided) | median value gap |x_kkt - x_qp| |",
                      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
            for nm in names:
                ms = [r["methods"][nm] for r in rs]
                fin = [m for m in ms if m["finite"]]
                ec = np.array([m["rel_err_central"] for m in fin]); eo = np.array([min(m["rel_err_plus"], m["rel_err_minus"]) for m in fin])
                vg = np.array([m["value_gap"] for m in fin])
                if ec.size:
                    lines.append(f"| {nm} | {len(fin)}/{len(ms)} | {np.median(ec):.2e} | {np.percentile(ec, 90):.2e} | {ec.max():.2e} | {np.median(eo):.2e} | "
                                 f"{np.mean(ec <= 1e-3):.2f} | {np.mean(eo <= 1e-2):.2f} | {np.median(vg):.1e} |")
                else:
                    lines.append(f"| {nm} | 0/{len(ms)} | – | – | – | – | – | – | – |")
            lines.append("")
    return "\n".join(lines) + "\n"


def effect(rerun_root: Path, phase8_root: Path, log=print):
    """The optimization effect of the chosen derivative: every re-run (``<rerun_root>/<objective>/<variant>/<start>/run.json``)
    paired with the Phase 8 run of the same objective, variant and start. Per run: non-finite iterations, clipped
    iterations, finite gradient norms above 100, the first iteration at which the gradient history departs from Phase 8
    (relative gradient-norm difference > 1e-9 or a change of finiteness), and the start / final selected / returned exact
    scores (8 fresh validation seeds, the Phase 8 protocol)."""
    from scipy.stats import wilcoxon
    out = {}
    lines = ["# Phase 9.1: re-runs with the chosen derivative, paired with the Phase 8 runs", "",
             "Exact-library validation with fresh seeds (8 two-finger or 6 three-contact), Phase 8 protocol; fixed physical task.", ""]
    for obj_dir in sorted(p for p in rerun_root.iterdir() if p.is_dir()):
        for var_dir in sorted(p for p in obj_dir.iterdir() if p.is_dir()):
            rows = []
            for rj in sorted(var_dir.glob("*/run.json")):
                p8 = phase8_root / obj_dir.name / var_dir.name / rj.parent.name / "run.json"
                if not p8.exists() and var_dir.name.startswith("policy_"):
                    p8 = phase8_root / obj_dir.name / var_dir.name / rj.parent.name / "run.json"
                if not p8.exists():
                    continue
                a, b = json.loads(p8.read_text()), json.loads(rj.read_text())
                ta, tb = a["training_history"], b["training_history"]
                first_div = None
                for x, y in zip(ta, tb):
                    if x["grad_finite"] != y["grad_finite"] or abs(x["grad_norm"] - y["grad_norm"]) > 1e-9 * max(abs(x["grad_norm"]), 1e-12):
                        first_div = x["iter"]; break
                cnt = lambda th: dict(nonfinite=sum(1 for h in th if not h["grad_finite"]), clipped=sum(1 for h in th if h.get("clipped")),   # noqa: E731
                                      big=sum(1 for h in th if h["grad_finite"] and h["grad_norm"] > 100.0))
                sa, sb = a["meta"]["stages_log"], b["meta"]["stages_log"]
                rows.append(dict(start=rj.parent.name, phase8=dict(cnt(ta), start=sa[0].get("exact_start"), final_selected=a["meta"]["final_iterate_exact_selected"],
                                                                     returned=a["best_validation"].get("exact_score")),
                                 rerun=dict(cnt(tb), start=sb[0].get("exact_start"), final_selected=b["meta"]["final_iterate_exact_selected"],
                                            returned=b["best_validation"].get("exact_score")),
                                 first_divergence_iter=first_div, phase8_first_nonfinite=next((h["iter"] for h in ta if not h["grad_finite"]), None)))
            if not rows:
                continue
            ra = np.array([r["phase8"]["returned"] for r in rows]); rb = np.array([r["rerun"]["returned"] for r in rows]); d = rb - ra
            fa = np.array([r["phase8"]["final_selected"] for r in rows]); fb = np.array([r["rerun"]["final_selected"] for r in rows])
            w = int(np.sum(d > 1e-12)); l_ = int(np.sum(d < -1e-12))
            try:
                pv = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
            except ValueError:
                pv = float("nan")
            key = f"{obj_dir.name}/{var_dir.name}"
            out[key] = dict(rows=rows, returned_diff=d.tolist(), wins=w, losses=l_, ties=len(d) - w - l_, wilcoxon_p=pv,
                            totals=dict(phase8={k: sum(r["phase8"][k] for r in rows) for k in ("nonfinite", "clipped", "big")},
                                        rerun={k: sum(r["rerun"][k] for r in rows) for k in ("nonfinite", "clipped", "big")}))
            lines += [f"## {key} ({len(rows)} starts)", "",
                      "| start | Phase 8 non-finite / clipped / norm > 100 | re-run non-finite / clipped / norm > 100 | first iteration that differs (Phase 8 first non-finite) | start exact | final selected: Phase 8 → re-run | returned: Phase 8 → re-run |",
                      "| --- | --- | --- | --- | --- | --- | --- |"]
            for r in rows:
                A, B = r["phase8"], r["rerun"]
                lines.append(f"| {r['start']} | {A['nonfinite']} / {A['clipped']} / {A['big']} | {B['nonfinite']} / {B['clipped']} / {B['big']} | "
                             f"{r['first_divergence_iter'] if r['first_divergence_iter'] is not None else 'none (identical)'} ({r['phase8_first_nonfinite']}) | {A['start']:.4f} | "
                             f"{A['final_selected']:.4f} → {B['final_selected']:.4f} | {A['returned']:.4f} → {B['returned']:.4f} |")
            T = out[key]["totals"]
            lines += ["", f"totals non-finite / clipped / norm > 100: Phase 8 {T['phase8']['nonfinite']} / {T['phase8']['clipped']} / {T['phase8']['big']}, "
                      f"re-run {T['rerun']['nonfinite']} / {T['rerun']['clipped']} / {T['rerun']['big']}; returned score: mean {ra.mean():.4f} → {rb.mean():.4f} "
                      f"(paired diff {d.mean():+.4f}, wins / ties / losses {w} / {len(d) - w - l_} / {l_}, Wilcoxon p {pv:.3f}); "
                      f"final selected: mean {fa.mean():.4f} → {fb.mean():.4f}", ""]
    (rerun_root / "effect.json").write_text(json.dumps(out, indent=1, default=float))
    (rerun_root / "effect.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return out


def rescore(rerun_root: Path, phase8_root: Path, objective: str = "full3", log=print, only_variant: str | None = None):
    """Phase 8.2 validated the Full-objective runs with the pre-8.6 exact path (smooth force gate); the re-runs use the
    8.6 path (exact SLSQP check at every sub-step). For a paired comparison under one validator, the returned designs of
    both runs are re-scored here with the current exact path and the runs' own validation protocol (fresh seeds,
    ``objective_spec(objective)``'s budget and commands, default seed configuration)."""
    from dkarma.export import design_from_dict
    from dkarma.core import build_hand_model
    from dkarma.solvers.seed_projection import find_multi_seeds
    from dkarma.metrics import act_karma as AK
    from scipy.stats import wilcoxon
    task = load_task_spec(); sp = task.step_params()
    spec = objective_spec(objective); cfg = spec["cfg"]; sm = cfg.smooth

    def score(design_json):
        topo, params, _ = design_from_dict(json.loads(Path(design_json).read_text()))
        hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
        fingers = (topo.thumb_finger, topo.index_finger, [f for f in range(topo.n_fingers) if f not in (topo.thumb_finger, topo.index_finger)][0])
        seeds = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=spec["budget"]))
        ex = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, seeds, sp, task, act, cfg)
        return float(sm.lambda_T * ex.K_T / sm.ref_T + sm.lambda_R * ex.K_SO3 / sm.ref_R), int(seeds.valid.sum())

    out = {}
    lines = [f"# Phase 9.1: {objective} returned designs re-scored with the current exact path (Phase 8.6), Phase 8 vs re-run", "",
             f"Fresh seeds (budget {spec['budget']}, default seed configuration); exact Full-KaRMA score; fixed physical task.", ""]
    prev = json.loads((rerun_root / f"rescore_{objective}.json").read_text()) if (rerun_root / f"rescore_{objective}.json").exists() else {}
    for var_dir in sorted(p for p in (rerun_root / objective).iterdir() if p.is_dir()):
        if only_variant is not None and var_dir.name != only_variant:
            if var_dir.name in prev:                     # keep the earlier re-score of the other variants
                out[var_dir.name] = prev[var_dir.name]
            continue
        rows = []
        for rd in sorted(p for p in var_dir.iterdir() if (p / "run.json").exists()):
            p8 = phase8_root / objective / var_dir.name / rd.name / "optimized_morphology" / "optimized_morphology.design.json"
            p9 = rd / "optimized_morphology" / "optimized_morphology.design.json"
            if not p8.exists() or not p9.exists():
                continue
            (a, na), (b, nb) = score(p8), score(p9)
            rows.append(dict(start=rd.name, phase8=a, rerun=b, seeds_phase8=na, seeds_rerun=nb))
            log(f"  {var_dir.name}/{rd.name}: Phase 8 returned {a:.4f} ({na} seeds), re-run returned {b:.4f} ({nb} seeds)")
        if not rows:
            continue
        d = np.array([r["rerun"] - r["phase8"] for r in rows])
        w = int(np.sum(d > 1e-12)); l_ = int(np.sum(d < -1e-12))
        try:
            pv = float(wilcoxon(d).pvalue) if np.any(np.abs(d) > 1e-12) else 1.0
        except ValueError:
            pv = float("nan")
        out[var_dir.name] = dict(rows=rows, wins=w, losses=l_, ties=len(d) - w - l_, mean_diff=float(d.mean()), wilcoxon_p=pv)
        lines += [f"## {objective}/{var_dir.name}", "", "| start | Phase 8 returned design | re-run returned design | diff |", "| --- | --- | --- | --- |"]
        lines += [f"| {r['start']} | {r['phase8']:.4f} | {r['rerun']:.4f} | {r['rerun'] - r['phase8']:+.4f} |" for r in rows]
        lines += ["", f"mean {np.mean([r['phase8'] for r in rows]):.4f} → {np.mean([r['rerun'] for r in rows]):.4f}; wins / ties / losses {w} / {len(d) - w - l_} / {l_}; "
                  f"mean diff {d.mean():+.4f}; Wilcoxon p {pv:.3f}", ""]
    (rerun_root / f"rescore_{objective}.json").write_text(json.dumps(out, indent=1, default=float))
    (rerun_root / f"rescore_{objective}.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["trace", "corpus", "compare", "effect", "rescore"])
    ap.add_argument("--rerun-root", default="workspace/phase9/9.1/rerun_reduced")
    ap.add_argument("--only-variant", default=None, help="rescore: only this variant (the others are kept from the previous rescore JSON)")
    ap.add_argument("--objective", default="diff2")
    ap.add_argument("--variant", default="phase7")
    ap.add_argument("--start", default="start1_sobol0_s0.25")
    ap.add_argument("--no-stage-select", action="store_true", help="trace: run the variant without stage selection (no validations available)")
    ap.add_argument("--out", default="workspace/phase9/9.1")
    ap.add_argument("--phase8-root", default="workspace/phase8/8.2")
    ap.add_argument("--traces", nargs="*", default=None, help="corpus: trace directories (default: every trace_* under --out)")
    ap.add_argument("--n-well", type=int, default=300)
    ap.add_argument("--n-dirs", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=None)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"{args.stage}.log"

    def log(s):
        print(s, flush=True)
        with open(log_path, "a") as fh:
            fh.write(str(s) + "\n")

    if args.stage == "trace":
        trace(args.objective, args.variant, args.start, out, Path(args.phase8_root), log=log, stage_select=(False if args.no_stage_select else None))
    elif args.stage == "corpus":
        traces = [Path(t) for t in args.traces] if args.traces else sorted(out.glob("trace_*"))
        build_corpus(out, traces, n_well=args.n_well, log=log)
    elif args.stage == "effect":
        effect(Path(args.rerun_root), Path(args.phase8_root), log=log)
    elif args.stage == "rescore":
        rescore(Path(args.rerun_root), Path(args.phase8_root), objective=args.objective, log=log, only_variant=args.only_variant)
    else:
        compare(out, n_dirs=args.n_dirs, log=log, max_states=args.max_states)
    log("done")


if __name__ == "__main__":
    main()
