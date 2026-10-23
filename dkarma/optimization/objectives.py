"""Objectives for the morphology optimizer (SPEC.md §7, §8.1): the smooth metric variants as
functions of the unconstrained ``Phi``, with the design-space constraints as penalties, plus
the exact validation callbacks the optimizer uses to pick the reported design.

An :class:`Objective` holds everything a run needs: the topology, bounds, fixed parameters,
the seeds (found once per run on the start design and re-projected inside the smooth
objective), the task and metric configuration. ``value_and_grad`` is jitted with the seed
set as a traced argument, so a run can change its seeds (Phase 8.3 seed policies) without
recompiling; ``validate`` re-scores a design exactly (exact library, and the original KaRMA
on the physical task when asked).

Seed policies (SPEC §11.1 item 8.3; the gradient is conditional on the seed set in use):

* ``"reproject"``: the start design's seeds, re-projected onto every iterate (Phase 7);
* ``"regenerate_stage"``: a fresh seed search on the current design at every stage boundary;
* ``"regenerate_every_k"``: a fresh search every ``reseed_every`` iterations;
* ``"union_stage"``: at every stage boundary, the union of the re-projected current seeds
  and a fresh search, ranked by each seed's *exact* library score, cut to the fixed budget.

Phase 8.3 (docs/phase8_report.md §8.3): ``"reproject"`` stays the default (no non-finite
gradients on either objective); ``"union_stage"`` is the recommended setting for two-finger
objectives (better returned design on 4 of 5 starts, 1.5x wall clock); every reseeding policy
exposes non-finite hard-QP derivatives on the 3x4 Full objective.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

from ..core.morphology import Topology, HandParams, Phi, Bounds, FixedParams, decode, build_hand_model
from ..core.object_model import TaskSpec
from ..core.actuation import ActuationParams
from ..solvers.seed_projection import SeedSet, SeedConfig, find_seeds, find_multi_seeds, assignment, reproject
from ..metrics.config import MetricConfig
from ..metrics import diff_karma as DK
from ..metrics import act_karma as AK
from ..metrics import multicontact as MC
from ..metrics.exact_library import evaluate_exact, official_karma_physical

SEED_POLICIES = ("reproject", "regenerate_stage", "regenerate_every_k", "union_stage")
GATE_NAMES_2 = ("qp", "slip", "gap", "collision", "self_collision", "grasp", "limits")
GATE_NAMES_3 = GATE_NAMES_2 + ("force",)


@dataclass(frozen=True)
class Constraints:
    """Global design-space constraints (SPEC §8.1) as smooth penalties on the physical design.
    ``base_sep_min_m`` (Phase 9.2, D18): minimum distance between any two finger bases
    (``None`` = no penalty; 0.0241 m for the admissible design space)."""
    reach_max_m: float | None = 0.25       # sum of a finger's link lengths incl. tip
    reach_weight: float = 10.0
    base_sep_min_m: float | None = None
    base_sep_weight: float = 10.0


def reach_penalty(params: HandParams, topo: Topology, cons: Constraints):
    pen = jnp.zeros(())
    if cons.reach_max_m is not None:
        wire = jnp.sum(jnp.where(jnp.asarray(topo.joint_mask), params.link_len, 0.0), axis=1)   # [F]
        pen = pen + cons.reach_weight * jnp.sum(jax.nn.softplus((wire - cons.reach_max_m) / 0.005) * 0.005) ** 2
    if cons.base_sep_min_m is not None and topo.n_fingers >= 2:
        b = params.base_xyz
        F = b.shape[0]
        i, j = np.triu_indices(F, 1)
        d = jnp.sqrt(jnp.sum((b[i] - b[j]) ** 2, axis=-1) + 1e-12)
        pen = pen + cons.base_sep_weight * jnp.sum(jax.nn.softplus((cons.base_sep_min_m - d) / 0.002) * 0.002) ** 2
    return pen


D18_CONSTRAINTS = Constraints(reach_max_m=0.25, base_sep_min_m=0.0241)


ValidationFn = Callable[[HandParams], dict]


def _arrays_only(seeds: SeedSet) -> SeedSet:
    """The array part of a seed set (what the jitted objective traces)."""
    return SeedSet(seeds.q, seeds.p, seeds.frame, seeds.valid, 0, {})


def _pair_topology(topo: Topology, hm, a, b):
    t2 = Topology(fingers=topo.fingers, n_joints=topo.n_joints, pair=(a, b))
    hm2 = hm._replace(dm=hm.dm._replace(thumb_caps=hm.finger_caps[t2.thumb_finger], index_caps=hm.finger_caps[t2.index_finger]))
    return t2, hm2


@dataclass
class Objective:
    topo: Topology
    bounds: Bounds
    fixed: FixedParams
    seeds: SeedSet
    task: TaskSpec
    sp: object
    cfg: MetricConfig
    act: ActuationParams | None = None
    fingers: tuple[int, ...] | None = None        # multi-contact variants
    constraints: Constraints = field(default_factory=Constraints)
    pair_seeds: dict = field(default_factory=dict)  # pairwise objective: {(finger_a, finger_b): SeedSet}
    seed_policy: str = "reproject"                  # Phase 8.3 default policy (see module docstring)
    reseed_every: int = 10                          # for "regenerate_every_k"
    seed_cfg: SeedConfig = field(default_factory=SeedConfig)

    # ── smooth value ────────────────────────────────────────────────────────

    def _diag(self, hm, seeds, pair_seeds):
        """The smooth evaluator's diagnostics object for a built model (pairwise: the pair of
        ``self.topo``, whose seeds are those of the start pair)."""
        if self.cfg.contacts >= 3:
            return MC.evaluate_fixed_contacts(hm, self.topo, self.fingers, seeds, self.sp, self.task, self.act, self.cfg)
        if self.cfg.roll.actuation:
            return DK.evaluate(hm, self.topo, seeds, self.sp, self.task, self.cfg, None, self.act)
        return DK.evaluate(hm, self.topo, seeds, self.sp, self.task, self.cfg)

    def _score(self, hm, seeds, pair_seeds):
        if self.cfg.name == "pairwise":
            scores = []
            for (a, b), sd in pair_seeds.items():
                t2, hm2 = _pair_topology(self.topo, hm, a, b)
                scores.append(DK.evaluate(hm2, t2, sd, self.sp, self.task, self.cfg).score)
            return jnp.mean(jnp.stack(scores))
        return self._diag(hm, seeds, pair_seeds).score

    def value(self, phi: Phi, seeds: SeedSet | None = None, pair_seeds: dict | None = None):
        seeds = self.seeds if seeds is None else seeds
        pair_seeds = self.pair_seeds if pair_seeds is None else pair_seeds
        params = decode(phi, self.topo, self.bounds, self.fixed)
        return self._score(build_hand_model(self.topo, params), seeds, pair_seeds) - reach_penalty(params, self.topo, self.constraints)

    def value_and_grad(self):
        """``f(phi, seeds=None, pair_seeds=None) -> (value, grad)``; one compiled program per
        seed-set *shape*, so changing the seeds does not recompile."""
        vg = jax.jit(jax.value_and_grad(lambda phi, seeds, pair_seeds: self.value(phi, seeds, pair_seeds)))

        def f(phi, seeds=None, pair_seeds=None):
            seeds = _arrays_only(self.seeds if seeds is None else seeds)
            ps = {k: _arrays_only(v) for k, v in (self.pair_seeds if pair_seeds is None else pair_seeds).items()}
            return vg(phi, seeds, ps)
        return f

    def params_of(self, phi: Phi) -> HandParams:
        return decode(phi, self.topo, self.bounds, self.fixed)

    def with_config(self, cfg: MetricConfig) -> "Objective":
        return Objective(self.topo, self.bounds, self.fixed, self.seeds, self.task, self.sp, cfg, self.act, self.fingers,
                         self.constraints, self.pair_seeds, self.seed_policy, self.reseed_every, self.seed_cfg)

    # ── diagnostics (Phase 8.2 instrumentation) ─────────────────────────────

    def diagnostics(self, phi: Phi, seeds: SeedSet | None = None) -> dict:
        """Host summary of the smooth evaluation at ``phi``: smooth score and coverages, the
        share of smooth steps failing each exact gate, the mean end-of-command credit, and the
        number of seeds whose re-projection failed. Pairwise objectives report the start pair."""
        seeds = _arrays_only(self.seeds if seeds is None else seeds)
        if not hasattr(self, "_diag_fn"):
            self._diag_fn = jax.jit(lambda phi, s: self._diag(build_hand_model(self.topo, decode(phi, self.topo, self.bounds, self.fixed)), s, {}))
        d = self._diag_fn(phi, seeds)
        v = np.asarray(d.valid)
        ps = d.per_seed
        names = GATE_NAMES_3 if self.cfg.contacts >= 3 else GATE_NAMES_2
        gT = np.asarray(ps.gates_T)[v]; gR = np.asarray(ps.gates_R)[v]
        g = np.concatenate([gT.reshape(-1, gT.shape[-1]), gR.reshape(-1, gR.shape[-1])]) if v.any() else np.zeros((0, len(names)), bool)
        K_R = ps.K_SO3 if self.cfg.contacts >= 3 else ps.K_R
        out = dict(smooth=float(d.score), n_valid=int(v.sum()),
                   smooth_K_T=float(np.asarray(ps.K_T)[v].mean()) if v.any() else 0.0,
                   smooth_K_R=float(np.asarray(K_R)[v].mean()) if v.any() else 0.0,
                   gate_fail_share={n: float(1.0 - g[:, i].mean()) if g.size else None for i, n in enumerate(names)},
                   all_gates_pass_share=float(g.all(axis=1).mean()) if g.size else None,
                   end_credit_T_mean=float(np.exp(np.asarray(ps.end_logw_T)[v]).mean()) if v.any() else 0.0,
                   end_credit_R_mean=float(np.exp(np.asarray(ps.end_logw_R)[v]).mean()) if v.any() else 0.0,
                   reproj_failed=int((~np.asarray(ps.reproj_ok)[v]).sum()) if v.any() else 0)
        return out

    # ── seed policies (Phase 8.3) ───────────────────────────────────────────

    @property
    def budget(self) -> int:
        return int(self.seeds.q.shape[0])

    def _assignment(self):
        return assignment(self.topo, 3, self.fingers) if self.cfg.contacts >= 3 else assignment(self.topo)

    def regenerate_seeds(self, params: HandParams) -> SeedSet:
        """A fresh seed search on ``params`` with the run's seed configuration and budget."""
        hm = build_hand_model(self.topo, params)
        cfg = SeedConfig(**{**self.seed_cfg.__dict__, "budget": self.budget})
        if self.cfg.contacts >= 3:
            return find_multi_seeds(hm, self.topo, self.fingers, self.sp, self.act, cfg)
        return find_seeds(hm, self.topo, self.sp, cfg)

    def reproject_seeds(self, params: HandParams, seeds: SeedSet) -> SeedSet:
        """``seeds`` re-projected onto ``params`` on the host (joints move, centres fixed); lanes
        whose projection fails become invalid."""
        hm = build_hand_model(self.topo, params)
        asg = self._assignment()
        q, ok, _ = jax.vmap(lambda q, p: reproject(hm, q, p, asg, self.sp))(seeds.q, seeds.p)
        valid = seeds.valid & ok
        return SeedSet(q, seeds.p, seeds.frame, valid, int(valid.sum()), dict(seeds.stats, reprojected=True))

    def exact_seed_scores(self, params: HandParams, seeds: SeedSet) -> np.ndarray:
        """Per-seed exact library score ``K_T / ref_T + K_R / ref_R`` (zero for invalid lanes)."""
        hm = build_hand_model(self.topo, params)
        sm = self.cfg.smooth
        valid = np.asarray(seeds.valid)
        idx = np.nonzero(valid)[0]
        packed = _pack(seeds, idx, self.budget if idx.size <= self.budget else int(idx.size))
        out = np.zeros(valid.shape[0])
        if idx.size == 0:
            return out
        if self.cfg.contacts >= 3:
            ex = MC.evaluate_fixed_contacts_exact(hm, self.topo, self.fingers, packed, self.sp, self.task, self.act, self.cfg)
            n_bins = self.task.orientation.so3_bins_exact
            per = [(len(r["cells"]), len(r["bins"])) for r in ex.per_seed]
        else:
            ex = evaluate_exact(hm, self.topo, packed, self.sp, self.task, self.cfg)
            n_bins = 228
            per = [(len(r.cells), len(r.bins)) for r in ex.per_seed]
        for j, (nc, nb) in zip(idx, per):
            out[j] = sm.lambda_T * (nc / self.task.task.n_cells) / sm.ref_T + sm.lambda_R * (nb / n_bins) / sm.ref_R
        return out

    def union_seeds(self, params: HandParams, current: SeedSet) -> tuple[SeedSet, dict]:
        """Re-projected ``current`` seeds plus a fresh search, deduplicated, ranked by the exact
        per-seed score, cut to the budget. Returns ``(seeds, info)``."""
        rep = self.reproject_seeds(params, current)
        fresh = self.regenerate_seeds(params)
        q_r, q_f = np.asarray(rep.q), np.asarray(fresh.q)
        keep_r = [i for i in range(q_r.shape[0]) if bool(rep.valid[i])]
        keep_f = [i for i in range(q_f.shape[0]) if bool(fresh.valid[i])
                  and not any(np.linalg.norm(q_f[i] - q_r[j]) < self.seed_cfg.dedup_rad for j in keep_r)]
        ir, if_ = np.asarray(keep_r, int), np.asarray(keep_f, int)
        cat = lambda a, b: jnp.asarray(np.concatenate([np.asarray(a)[ir], np.asarray(b)[if_]]), rep.q.dtype)   # noqa: E731
        cand = SeedSet(cat(rep.q, fresh.q), cat(rep.p, fresh.p), cat(rep.frame, fresh.frame),
                       jnp.ones(len(keep_r) + len(keep_f), bool), len(keep_r) + len(keep_f), {})
        scores = self.exact_seed_scores(params, cand) if cand.q.shape[0] else np.zeros(0)
        order = np.argsort(-scores, kind="stable")[: self.budget]
        from_rep = int(np.sum(order < len(keep_r)))
        out = _pack(cand, order, self.budget)
        info = dict(n_reprojected_valid=len(keep_r), n_fresh_valid=len(keep_f), n_fresh_dedup=int(fresh.valid.sum()) - len(keep_f),
                    kept_from_reprojected=from_rep, kept_from_fresh=int(order.size) - from_rep,
                    exact_scores_kept=scores[order].tolist(), exact_scores_all=scores.tolist())
        return out, info

    def seeds_for_policy(self, params: HandParams, policy: str, current: SeedSet, at_stage_boundary: bool,
                         it: int) -> tuple[SeedSet, dict | None]:
        """The seed set to use next under ``policy`` (``None`` info when nothing changed)."""
        if policy not in SEED_POLICIES:
            raise ValueError(f"unknown seed policy {policy}; choose from {SEED_POLICIES}")
        if policy == "reproject":
            return current, None
        if policy == "regenerate_stage" and at_stage_boundary:
            s = self.regenerate_seeds(params)
            return s, dict(policy=policy, n_seeds=int(s.valid.sum()), stats=s.stats)
        if policy == "regenerate_every_k" and it > 0 and it % max(self.reseed_every, 1) == 0:
            s = self.regenerate_seeds(params)
            return s, dict(policy=policy, n_seeds=int(s.valid.sum()), stats=s.stats)
        if policy == "union_stage" and at_stage_boundary:
            s, info = self.union_seeds(params, current)
            return s, dict(policy=policy, n_seeds=int(s.valid.sum()), **info)
        return current, None


def _pack(seeds: SeedSet, idx, budget: int) -> SeedSet:
    """Seed lanes ``idx`` first, padded to ``budget`` with invalid lanes."""
    idx = np.asarray(idx, int)
    S, J = budget, seeds.q.shape[1]
    q = np.zeros((S, J)); p = np.zeros((S, 3)); fr = np.tile(np.eye(3), (S, 1, 1)); valid = np.zeros(S, bool)
    n = min(idx.size, S)
    if n:
        q[:n] = np.asarray(seeds.q)[idx[:n]]; p[:n] = np.asarray(seeds.p)[idx[:n]]; fr[:n] = np.asarray(seeds.frame)[idx[:n]]; valid[:n] = True
    dt = seeds.q.dtype
    return SeedSet(jnp.asarray(q, dt), jnp.asarray(p, dt), jnp.asarray(fr, dt), jnp.asarray(valid), n, {})


def make_objective(topo: Topology, params0: HandParams, bounds: Bounds, task: TaskSpec, cfg: MetricConfig,
                   seed_cfg: SeedConfig = SeedConfig(), pin_fingers: tuple[int, ...] = (), fingers=None,
                   constraints: Constraints = Constraints(), dtype=jnp.float64, seed_policy: str = "reproject",
                   reseed_every: int = 10) -> Objective:
    """Find the seeds on the start design and bundle everything the optimizer needs.
    ``seed_policy`` (SPEC §11.1 item 8.3) is the default policy of ``adam`` for this objective."""
    from ..core.morphology import fixed_from_params
    sp = task.step_params(dtype)
    hm0 = build_hand_model(topo, params0)
    act = AK.actuation_for(hm0, task, dtype=dtype)
    pair_seeds = {}
    if cfg.name == "pairwise":
        names = topo.fingers
        pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
        for a, b in pairs:
            t2, hm2 = _pair_topology(topo, hm0, a, b)
            pair_seeds[(a, b)] = find_seeds(hm2, t2, sp, seed_cfg, dtype)
        seeds = pair_seeds[(topo.pair[0], topo.pair[1])] if (topo.pair[0], topo.pair[1]) in pair_seeds else pair_seeds[(topo.pair[1], topo.pair[0])]
    elif cfg.contacts >= 3:
        fingers = fingers or (topo.thumb_finger, topo.index_finger, [f for f in range(topo.n_fingers) if f not in (topo.thumb_finger, topo.index_finger)][0])
        seeds = find_multi_seeds(hm0, topo, fingers, sp, act, seed_cfg, dtype)
    else:
        seeds = find_seeds(hm0, topo, sp, seed_cfg, dtype)
    return Objective(topo, bounds, fixed_from_params(params0, pin_fingers), seeds, task, sp, cfg, act, fingers, constraints,
                     pair_seeds, seed_policy, reseed_every, seed_cfg)


def exact_validation(obj: Objective, workdir, official: bool = True, name: str = "cand", enforce_d18: bool = False,
                     self_collision_all: bool = False) -> ValidationFn:
    """Validation callback: exact library coverage on fresh seeds of the candidate, and
    optionally the original KaRMA on the physical task (SPEC §8.5: the saved candidate is the
    best by this criterion, never by the smooth score). ``enforce_d18``: an inadmissible candidate
    (base separation, reach) scores zero. ``self_collision_all``: the original KaRMA's
    self-collision gate tests all fingers (D18)."""
    from pathlib import Path
    counter = {"n": 0}

    def validate(params: HandParams) -> dict:
        counter["n"] += 1
        hm = build_hand_model(obj.topo, params)
        out = {}
        if enforce_d18:
            from ..core.admissibility import admissibility
            adm = admissibility(params, obj.topo)
            out.update(admissible=adm["admissible"], min_base_separation_m=adm["min_base_separation_m"], max_reach_m=adm["max_reach_m"])
            if not adm["admissible"]:      # an inadmissible candidate scores zero and is never returned (CLAUDE.md rule 11 b)
                out.update(n_seeds=0, exact_K_T=0.0, exact_K_R=0.0, exact_score=0.0, d18_violations=adm["violations"])
                return out
        if obj.cfg.contacts >= 3:
            seeds = find_multi_seeds(hm, obj.topo, obj.fingers, obj.sp, obj.act, SeedConfig(budget=obj.seeds.q.shape[0]))
            ex = MC.evaluate_fixed_contacts_exact(hm, obj.topo, obj.fingers, seeds, obj.sp, obj.task, obj.act, obj.cfg)
            out.update(n_seeds=int(seeds.valid.sum()), exact_K_T=ex.K_T, exact_K_R=ex.K_SO3)
        else:
            seeds = find_seeds(hm, obj.topo, obj.sp, SeedConfig(budget=obj.seeds.q.shape[0]))
            ex = evaluate_exact(hm, obj.topo, seeds, obj.sp, obj.task, obj.cfg)
            out.update(n_seeds=int(seeds.valid.sum()), exact_K_T=ex.K_T, exact_K_R=ex.K_R)
        sm = obj.cfg.smooth
        out["exact_score"] = sm.lambda_T * out["exact_K_T"] / sm.ref_T + sm.lambda_R * out["exact_K_R"] / sm.ref_R
        if official:
            wd = Path(workdir) / f"{name}_{counter['n']:04d}"
            row = official_karma_physical(obj.topo, params, obj.task, wd, name=name, self_collision_all=self_collision_all)
            out.update(official_cube_fraction=row["cube_fraction"], official_KaRMA_R=row["KaRMA_R"],
                       official_voxels=row["n_voxels"], official_runtime_s=row.get("runtime_s"))
        return out
    return validate


__all__ = ["Constraints", "D18_CONSTRAINTS", "reach_penalty", "Objective", "make_objective", "exact_validation", "ValidationFn", "SEED_POLICIES"]
