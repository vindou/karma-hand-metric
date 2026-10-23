"""Morphology optimizers (SPEC.md §8.3-§8.5): Adam on the smooth objective with a continuation
schedule and exact validation every ``k`` iterations; CMA-ES on the same unconstrained
coordinates (the gradient-free baseline, reusing the ``cma`` package of ``optimize/``); and
the hybrid (CMA-ES or Sobol starts, gradient refinement, exact selection).

Every optimizer logs the parameters, the smooth and validation scores and the PRNG seeds
so a run regenerates exactly (``checkpoint.save_run``). The reported design is the best by
the *validation* criterion, never by the smooth score (SPEC §8.5).

Phase 8.2/8.3 additions to :func:`adam` (``docs/phase8_report.md``): per-stage instrumentation
(exact validation and smooth diagnostics at every stage boundary), gradient hygiene
(``grad_clip``, ``skip_nonfinite``, ``reset_moments``), within-stage selection with warm
starts (``stage_select``), the seed policies of :mod:`objectives` (``seed_policy``), and the
schedule options of :mod:`dkarma.metrics.continuation` (``schedule_kw``).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

from ..core.morphology import Phi, HandParams
from ..metrics.continuation import schedule, describe
from .objectives import Objective, ValidationFn


@dataclass
class OptResult:
    best_params: HandParams
    best_phi: Phi
    best_validation: dict
    training_history: list = field(default_factory=list)      # dicts: iter, smooth, stage, step, time
    validation_history: list = field(default_factory=list)    # dicts: iter, validation dict, smooth
    constraint_history: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def _flat(phi: Phi):
    return jax.flatten_util.ravel_pytree(phi)


# Phase 8.2 defaults, adopted from the continuation study (docs/phase8_report.md §8.2, workspace/phase8/8.2/diff2):
# gradient hygiene (clip the 2-norm at 100, no update on a non-finite gradient, Adam moments reset at stage
# boundaries) and stage selection (each stage warm-starts from its predecessor's best validated iterate).
# The Phase 7 behaviour is grad_clip=None, skip_nonfinite=False, reset_moments=False, stage_select=False.
ADAM_DEFAULTS = dict(grad_clip=100.0, grad_spike=None, skip_nonfinite=True, reset_moments=True, stage_select=True, schedule_kw=None)


def adam(obj: Objective, phi0: Phi, *, n_iters: int = 60, lr: float = 0.02, betas=(0.9, 0.999), eps: float = 1e-8,
         stages: int = 3, validate_every: int = 10, validation_fn: ValidationFn | None = None,
         log: Callable[[str], None] = print, seed: int = 0, schedule_kw: dict | None = None,
         grad_clip: float | None = None, skip_nonfinite: bool | None = None, reset_moments: bool | None = None,
         stage_select: bool | None = None, seed_policy: str | None = None, reseed_every: int | None = None,
         instrument: bool = True, warmup: bool = False, grad_spike: float | None = None,
         iterate_log: list | None = None) -> OptResult:
    """Adam ascent on the smooth objective over a continuation schedule (soft solver first,
    kernels tightening). Validation every ``validate_every`` iterations and at every stage
    boundary selects the result.

    ``grad_clip``: clip the gradient 2-norm (``None`` = off). ``grad_spike``: an iteration whose
    gradient norm exceeds ``grad_spike`` times the median of the last ten accepted norms makes no
    update (the hard-QP derivative blows up at degenerate active sets; ``None`` = off). ``skip_nonfinite``: an iteration
    whose gradient has a non-finite entry makes no update (otherwise non-finite entries are
    zeroed and Adam steps along its momentum). ``reset_moments``: zero the Adam moments at
    stage boundaries. ``stage_select``: warm-start each stage from the best validated design
    *within* the previous stage. ``seed_policy`` / ``reseed_every``: see
    :mod:`dkarma.optimization.objectives` (defaults: the objective's). ``None`` options take
    ``ADAM_DEFAULTS``. ``iterate_log`` (Phase 9.1 tracing): a list that receives, per iteration,
    ``dict(iter, stage, x, grad, finite, seed_id)`` with the pre-step coordinates and the raw
    gradient, and ``dict(seed_id, seeds)`` whenever the seed set changes."""
    d = ADAM_DEFAULTS
    grad_clip = d["grad_clip"] if grad_clip is None else grad_clip
    grad_spike = d["grad_spike"] if grad_spike is None else grad_spike
    skip_nonfinite = d["skip_nonfinite"] if skip_nonfinite is None else skip_nonfinite
    reset_moments = d["reset_moments"] if reset_moments is None else reset_moments
    stage_select = d["stage_select"] if stage_select is None else stage_select
    schedule_kw = dict(d["schedule_kw"] or {}) if schedule_kw is None else dict(schedule_kw)
    policy = obj.seed_policy if seed_policy is None else seed_policy
    if reseed_every is not None:
        obj.reseed_every = reseed_every
    cfgs = schedule(obj.cfg, n_stages=stages, **schedule_kw)
    x, unravel = jax.flatten_util.ravel_pytree(phi0)
    m = jnp.zeros_like(x); v = jnp.zeros_like(x)
    hist, vhist, stage_recs, seed_events = [], [], [], []
    best = None
    t0 = time.perf_counter()
    per_stage = max(n_iters // stages, 1)
    it = 0
    t_adam = 0
    seeds = obj.seeds
    seed_id = 0
    if iterate_log is not None:
        iterate_log.append(dict(seed_id=seed_id, seeds=seeds))
    objs = [obj.with_config(c) for c in cfgs]
    fns = [o.value_and_grad() for o in objs]
    if warmup:                      # compile every stage's program before the clock starts (matched budgets, 8.4)
        for f in fns:
            jax.block_until_ready(f(unravel(x), seeds))
        t0 = time.perf_counter()
    val_cache = {}      # iteration -> validation record (a validation of the iterate at that iteration)
    t_val = 0.0
    accepted_norms = []

    def validate(x_, it_, stage_, tag):
        nonlocal best, t_val
        if it_ in val_cache:
            return val_cache[it_]
        params = obj.params_of(unravel(x_))
        tv = time.perf_counter()
        vd = validation_fn(params) if validation_fn else {}
        t_val += time.perf_counter() - tv
        crit = vd.get("exact_score", float("nan"))
        rec = dict(iter=it_, stage=stage_, tag=tag, validation=vd, crit=crit, x=np.asarray(x_))
        val_cache[it_] = rec
        vhist.append(dict(iter=it_, stage=stage_, tag=tag, validation=vd))
        log(f"  it {it_:4d} stage {stage_} validation {crit:.4f} [{tag}] ({time.perf_counter() - t0:.0f}s)")
        if validation_fn and (best is None or crit > best[0]):
            best = (crit, unravel(x_), params, vd, it_)
        return rec

    for si, f in enumerate(fns):
        o = objs[si]
        if reset_moments:
            m = jnp.zeros_like(x); v = jnp.zeros_like(x); t_adam = 0
        if si > 0:
            seeds, info = obj.seeds_for_policy(obj.params_of(unravel(x)), policy, seeds, at_stage_boundary=True, it=it)
            if info:
                seed_events.append(dict(iter=it, stage=si, **info)); log(f"  reseed at stage {si}: {info.get('n_seeds')} seeds")
                seed_id += 1
                if iterate_log is not None:
                    iterate_log.append(dict(seed_id=seed_id, seeds=seeds))
        rec = dict(stage=si, iter_start=it, config=describe(cfgs[si]), n_nonfinite=0, n_clipped=0, n_spikes=0, n_seeds=int(seeds.valid.sum()))
        if instrument:
            vs = validate(x, it, si, "stage_start")
            rec["exact_start"] = vs["crit"]; rec["validation_start"] = vs["validation"]
            rec["diag_start"] = o.diagnostics(unravel(x), seeds)
        best_in_stage = None
        smooth_first = None
        for _ in range(per_stage):
            if policy == "regenerate_every_k":
                seeds, info = obj.seeds_for_policy(obj.params_of(unravel(x)), policy, seeds, at_stage_boundary=False, it=it)
                if info:
                    seed_events.append(dict(iter=it, stage=si, **info)); log(f"  reseed at it {it}: {info.get('n_seeds')} seeds")
                    seed_id += 1
                    if iterate_log is not None:
                        iterate_log.append(dict(seed_id=seed_id, seeds=seeds))
            val, g = f(unravel(x), seeds)
            gflat, _ = jax.flatten_util.ravel_pytree(g)
            finite = bool(jnp.all(jnp.isfinite(gflat)))
            if iterate_log is not None:
                iterate_log.append(dict(iter=it + 1, stage=si, x=np.asarray(x), grad=np.asarray(gflat), finite=finite, seed_id=seed_id,
                                        smooth=float(val)))
            gnorm_raw = float(jnp.linalg.norm(jnp.where(jnp.isfinite(gflat), gflat, 0.0)))
            if smooth_first is None:
                smooth_first = float(val)
            it += 1
            clipped = False
            spike = False
            if not finite:
                rec["n_nonfinite"] += 1
            if finite and grad_spike is not None and len(accepted_norms) >= 3 and gnorm_raw > grad_spike * float(np.median(accepted_norms[-10:])):
                spike = True; rec["n_spikes"] += 1
            if (finite or not skip_nonfinite) and not spike:
                gflat = jnp.where(jnp.isfinite(gflat), gflat, 0.0)
                gn = jnp.linalg.norm(gflat)
                if grad_clip is not None and float(gn) > grad_clip:
                    gflat = gflat * (grad_clip / gn); clipped = True; rec["n_clipped"] += 1
                if finite:
                    accepted_norms.append(gnorm_raw)
                t_adam += 1
                m = betas[0] * m + (1 - betas[0]) * gflat
                v = betas[1] * v + (1 - betas[1]) * gflat ** 2
                mh = m / (1 - betas[0] ** t_adam); vh = v / (1 - betas[1] ** t_adam)
                step = lr * mh / (jnp.sqrt(vh) + eps)
                x = x + step          # ascent
                step_norm = float(jnp.linalg.norm(step))
            else:
                step_norm = 0.0
            hist.append(dict(iter=it, smooth=float(val), stage=si, grad_norm=gnorm_raw, grad_finite=finite, clipped=clipped, spike=spike,
                             step_norm=step_norm, n_seeds=int(seeds.valid.sum()), time=time.perf_counter() - t0))
            at_stage_end = (_ == per_stage - 1) or it >= n_iters
            if (it % max(validate_every, 1) == 0 or it == n_iters) and not at_stage_end:
                vr = validate(x, it, si, "periodic")
                if validation_fn and (best_in_stage is None or vr["crit"] > best_in_stage["crit"]):
                    best_in_stage = vr
            if it >= n_iters:
                break
        # stage end
        rec["iter_end"] = it
        rec["smooth_first"] = smooth_first; rec["smooth_last"] = hist[-1]["smooth"] if hist else None
        if instrument or stage_select or validation_fn is not None:
            ve = validate(x, it, si, "stage_end")
            rec["exact_end_raw"] = ve["crit"]; rec["validation_end_raw"] = ve["validation"]
            if validation_fn and (best_in_stage is None or ve["crit"] > best_in_stage["crit"]):
                best_in_stage = ve
        if instrument:
            rec["diag_end"] = o.diagnostics(unravel(x), seeds)
        if stage_select and best_in_stage is not None and best_in_stage["iter"] != it:
            x = jnp.asarray(best_in_stage["x"])
            rec["selected_iter"] = best_in_stage["iter"]; rec["exact_end_selected"] = best_in_stage["crit"]
            log(f"  stage {si}: warm start from it {best_in_stage['iter']} (validation {best_in_stage['crit']:.4f})")
        else:
            rec["selected_iter"] = it; rec["exact_end_selected"] = rec.get("exact_end_raw")
        stage_recs.append(rec)
        if it >= n_iters:
            break
    if best is None:
        params = obj.params_of(unravel(x))
        best = (float("nan"), unravel(x), params, validation_fn(params) if validation_fn else {}, it)
    crits = [(r["iter"], r["validation"].get("exact_score", float("nan"))) for r in vhist]
    final_raw = stage_recs[-1].get("exact_end_raw") if stage_recs else float("nan")
    final_sel = stage_recs[-1].get("exact_end_selected") if stage_recs else float("nan")
    intermediate = [c for i, c in crits if i < (stage_recs[-1]["iter_end"] if stage_recs else n_iters) and np.isfinite(c)]
    meta = dict(method="adam", lr=lr, n_iters=n_iters, stages=stages, seed=seed, schedule=[describe(c) for c in cfgs],
                options=dict(grad_clip=grad_clip, grad_spike=grad_spike, skip_nonfinite=skip_nonfinite, reset_moments=reset_moments, stage_select=stage_select,
                             seed_policy=policy, reseed_every=obj.reseed_every, schedule_kw=schedule_kw),
                stages_log=stage_recs, seed_events=seed_events, best_iter=best[4],
                final_iterate_exact_raw=final_raw, final_iterate_exact_selected=final_sel,
                best_intermediate_exact=(max(intermediate) if intermediate else float("nan")),
                n_validations=len(vhist), validation_time_s=t_val, wall_clock_s=time.perf_counter() - t0,
                optimizer_time_s=time.perf_counter() - t0 - t_val)
    return OptResult(best[2], best[1], best[3], hist, vhist, [], meta)


def cmaes(obj: Objective, phi0: Phi, *, popsize: int = 12, n_gens: int = 30, sigma0: float = 0.3,
          validate_every_gens: int = 5, validation_fn: ValidationFn | None = None, log=print, seed: int = 1,
          time_budget_s: float | None = None, warmup: bool = False) -> OptResult:
    """CMA-ES on the unconstrained coordinates with the smooth objective as fitness (batched
    on the GPU through ``vmap``) and exact validation of the incumbent every few generations.
    ``time_budget_s`` stops after the generation that exhausts it (validation time excluded)."""
    import cma
    x0, unravel = jax.flatten_util.ravel_pytree(phi0)
    f_batch = jax.jit(jax.vmap(lambda xx: obj.value(unravel(xx))))
    es = cma.CMAEvolutionStrategy(np.asarray(x0), sigma0, dict(popsize=popsize, seed=seed, verbose=-9))
    hist, vhist = [], []
    best = None
    if warmup:
        jax.block_until_ready(f_batch(jnp.broadcast_to(x0, (popsize, x0.shape[0]))))
    t0 = time.perf_counter()
    t_val = 0.0
    last_xb, last_vd = None, None        # an unchanged incumbent has an unchanged (deterministic) validation
    for g in range(n_gens):
        X = es.ask()
        vals = np.asarray(f_batch(jnp.asarray(np.stack(X))))
        vals = np.where(np.isfinite(vals), vals, -1e6)
        es.tell(X, (-vals).tolist())
        i = int(np.argmax(vals))
        t_opt = time.perf_counter() - t0 - t_val
        hist.append(dict(gen=g, best=float(vals[i]), mean=float(vals.mean()), time=t_opt))
        done = (g == n_gens - 1) or (time_budget_s is not None and t_opt >= time_budget_s)
        if (g + 1) % validate_every_gens == 0 or done:
            tv = time.perf_counter()
            xb = jnp.asarray(es.result.xbest)
            params = obj.params_of(unravel(xb))
            if last_xb is not None and np.array_equal(np.asarray(xb), last_xb):
                vd = dict(last_vd)
            else:
                vd = validation_fn(params) if validation_fn else {}
                last_xb, last_vd = np.asarray(xb).copy(), vd
            sm = float(obj.value(unravel(xb)))
            vhist.append(dict(gen=g, smooth=sm, validation=vd))
            crit = vd.get("exact_score", sm)
            log(f"  gen {g:3d} best smooth {sm:.4f} validation {crit:.4f} ({time.perf_counter() - t0:.0f}s)")
            if best is None or crit > best[0]:
                best = (crit, unravel(xb), params, vd)
            t_val += time.perf_counter() - tv
        if done:
            break
    return OptResult(best[2], best[1], best[3], hist, vhist, [],
                     dict(method="cmaes", popsize=popsize, n_gens=len(hist), sigma0=sigma0, seed=seed, validation_time_s=t_val,
                          wall_clock_s=time.perf_counter() - t0, optimizer_time_s=time.perf_counter() - t0 - t_val))


def hybrid(obj: Objective, phi0: Phi, *, n_starts: int = 4, sobol_scale: float = 0.5, adam_iters: int = 30,
           validation_fn: ValidationFn | None = None, log=print, seed: int = 2, time_budget_s: float | None = None,
           warmup: bool = False, **adam_kw) -> OptResult:
    """Sobol-perturbed starts around ``phi0`` -> short Adam refinement each -> exact validation
    -> the best physically validated candidate (SPEC §8.3 hybrid). ``time_budget_s``: no new start
    begins once the optimizer time (validations excluded) is exhausted."""
    from scipy.stats.qmc import Sobol
    x0, unravel = jax.flatten_util.ravel_pytree(phi0)
    n = x0.shape[0]
    pts = Sobol(d=n, scramble=True, seed=seed).random(n_starts)
    starts = [x0] + [x0 + sobol_scale * jnp.asarray(2.0 * p - 1.0) for p in pts[: n_starts - 1]]
    results = []
    t0 = time.perf_counter()
    t_opt = 0.0
    for k, xs in enumerate(starts):
        if time_budget_s is not None and k > 0 and t_opt >= time_budget_s:
            log(f"hybrid: budget exhausted after {k} starts"); break
        log(f"start {k}/{len(starts)}")
        r = adam(obj, unravel(xs), n_iters=adam_iters, validate_every=adam_iters, validation_fn=validation_fn, log=log,
                 seed=seed + k, instrument=False, warmup=(warmup and k == 0), **adam_kw)
        results.append(r)
        t_opt += r.meta["optimizer_time_s"]
    crit = [r.best_validation.get("exact_score", r.training_history[-1]["smooth"] if r.training_history else -np.inf) for r in results]
    i = int(np.argmax(crit))
    out = results[i]
    t_val = sum(r.meta.get("validation_time_s", 0.0) for r in results)
    out.meta = dict(method="hybrid", n_starts=len(results), adam_iters=adam_iters, seed=seed, start_criteria=[float(c) for c in crit],
                    validation_time_s=t_val, wall_clock_s=time.perf_counter() - t0, optimizer_time_s=t_opt,
                    per_start=[r.meta for r in results])
    out.constraint_history = [r.best_validation for r in results]
    return out


__all__ = ["OptResult", "adam", "cmaes", "hybrid", "ADAM_DEFAULTS"]
