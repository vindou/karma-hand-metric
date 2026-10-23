# Phase 9 report: admissible baselines and controller-independent validation

Scope: SPEC.md §11.2 (added 2026-10-13 from the owner's Phase 8 review,
`docs/phase8_review_2026-10-13.md`), items 9.1 to 9.5 in priority order. No metric
formulation, kernel or contact model changed; the frozen task of SPEC §2 is untouched. Every
table states which task definition it reports: **fixed physical** (15 mm / 35 g sphere, ±100 mm
cube at 10 mm, XL330 ceilings; SPEC §2) or **published `L_ref`-scaled** (the original KaRMA's
sphere `0.05 L_ref` and scaled voxels). Design-box values (SPEC §16 item 10): finger bases
≥ 24.1 mm apart (`2 r_l + eps_col`), reach ≤ 250 mm per finger (the global hand envelope),
all-finger clearance in every self-collision gate.

Raw outputs: `workspace/phase9/{9.1,9.2,9.3}/`; chain scripts `workspace/phase9/chain_*.sh`;
logs beside them. Everything below names its script, JSON, design files, seed configuration and
optimizer seed.

## 9.1 Repair of the KKT derivative at degenerate active sets

### Files changed

| File | Change |
| --- | --- |
| `dkarma/solvers/rolling_qp.py` | `kkt_resolve(method=...)`: the Phase 2 derivative (`"exact"`, LU of the active-set KKT system with `-1e-10` on the active dual block) plus three alternatives: `"reg"` (the same system with a larger dual regularization), `"pinv"` (minimum-norm least-squares solution, `jnp.linalg.pinv` with a relative cutoff, constant-rank derivative), `"reduced"` (sequential Gram-Schmidt over the active rows of `G = [A; -A]`, box rows first, drops rows linearly dependent on the rows kept so far at relative tolerance `dep_tol`; the KKT matrix of the kept rows is nonsingular because `P ≻ 0`). `solve_hard(kkt_method, kkt_reg, kkt_dep_tol, kkt_rcond)`, `kkt_row_order`, `KKT_METHODS` |
| `dkarma/solvers/rollout.py` | `RollConfig.kkt_method / kkt_reg / kkt_dep_tol`, passed to `solve_hard` by `step` (default `"reduced"`, chosen below); the two rotation updates use `exp_so3_safe` |
| `dkarma/core/geometry.py` | `exp_so3_safe`: the reference's `exp_so3` with the small-angle guard decided on `stop_gradient(omega)` and the norm taken of `where(small, 1, omega)`; forward values and derivatives bit-identical to `karma_jax.steps.exp_so3` wherever `|omega| ≥ 1e-10` (checked on 12,000 random rotation vectors on CPU and GPU), zero derivative instead of NaN at `omega = 0` |
| `dkarma/optimization/optimizers.py` | `adam(iterate_log=...)`: per-iteration record of the pre-step coordinates, the raw gradient, its finiteness and the seed set in use (the tracing hook of this item; no change to the update) |
| `dkarma/experiments/kkt_derivative.py` (new) | `trace`: re-run a Phase 8.2 Adam run without validations and compare its gradient history with the Phase 8 `run.json`; `corpus`: record every rolling QP of chosen iterations (problem data, QP solution, active set, rank deficiency, condition number, norm of the exact derivative's VJP); `compare`: every method's directional derivative against central and one-sided finite differences of the tightly solved QP along random perturbations of the problem data |
| `dkarma/experiments/continuation_study.py` | `--kkt-method`, `--starts-from` (same five starts as Phase 8.2), `study.md` reports the number of finite gradients above norm 100 per run |
| `dkarma/tests/test_rolling_qp.py` | the Phase 2 finite-difference test parametrized over the four methods (unchanged tolerance 1e-3 on stable active sets, both reference hands); a synthetic QP with duplicated active rows on which `reg`, `pinv` and `reduced` must be finite and equal to the true Jacobian; the reduced mask drops the dependent gap row and keeps the box row |
| `dkarma/tests/test_rollout.py` | `test_rollout_gradients_wrt_morphology` parametrized over `("hard", method)` for the four methods; its finite-difference functions are jitted once (the eager scan recompiled at every call and the module reached > 18 GB of host memory) and compiled programs are dropped after every test |
| `dkarma/tests/test_gradients.py` | `exp_so3_safe` equals the reference in value and derivative away from zero and is finite at zero (where the reference's derivative is NaN) |
| `dkarma/tests/conftest.py` | JAX caches cleared after every test module (host-memory hygiene; values unaffected) |
| `dkarma/tests/test_optimization.py` | the iterate log is checked in the hygiene test |

### Procedure

1. **Where the Phase 8 gradient events are.** The Phase 8.2 `run.json` histories
   (`workspace/phase8/8.2/{diff2,full3}/*/*/run.json`) record per iteration the gradient norm,
   its finiteness and the stage. Non-finite gradients occur only in hard-solver stages
   (`diff2`: `phase7/start1` iterations 35–60, `hygiene_reseed` starts 1–3, `hygiene_select_reseed`
   starts 1–2; `full3`: every reseeding variant, iterations 22–40). Gradient norms above 100 (the
   "spikes" that the Phase 8.2 clipping handled) occur in **both** solver stages: the largest
   (`diff2/hygiene/start2`, norms 4e4 to 5e6 at iterations 9–14) are in stage 0, whose solver is the
   **soft** closed-form solve. Those spikes are therefore not KKT events; they are reported
   separately below.
2. **Re-tracing the runs.** The saved artifacts hold only the best iterate, so the runs were
   re-traced (`kkt_derivative trace`, same start, seeds, variant and optimizer seed, no
   validations). Reproduction is exact for the reseeding variants (identical non-finite iterations,
   gradient norms equal to 3e-16 relative on every iteration) and for the `full3` reseed runs
   (identical non-finite iterations; their Phase 8 stage selection had chosen the stage end, so the
   validation-free re-trace follows the same path). The `phase7` variant does not reproduce beyond
   iteration 9: the gradient norms agree through iteration 9 and differ from iteration 10 on, without a
   non-finite event, and its re-trace has no non-finite gradient. Those runs date from 2026-10-11 20:30,
   in the first multi-variant study process that was later OOM-killed; every run that reproduces dates
   from 2026-10-12 or later. The cause could not be determined from the artifacts. The `phase7` traces
   contribute well-conditioned and spike states only.
3. **Corpus.** At every non-finite iteration (deduplicated: under `skip_nonfinite` the iterate does
   not move, so consecutive non-finite iterations share one design), at the three largest finite
   hard-stage gradients and at three regular hard-stage iterations of each trace, every rolling QP of
   the stage's library (all seeds × commands × steps, translation and rotation QPs separately) was
   recorded by a hard-mode rollout that also returns the problem data (`recording_rollout`). For
   each QP: the active set at the Phase 2 tolerance (`1e-7 (1 + |h_i|)`), the number of active rows,
   their rank (SVD, 1e-9 relative), the condition number of the exact method's KKT matrix, and the
   norm of the exact derivative's vector–Jacobian product for one fixed unit cotangent. A state is
   *degenerate* when that VJP is non-finite or exceeds 1e3; a random sample of the remaining
   QP-successful states is the *well-conditioned* corpus.
4. **Comparison.** For every corpus state and every method, the directional derivative of the KKT
   solution along three random perturbation directions of `(P, c, A_gap, l, u)` (each block scaled to
   its own size; the identity rows of the box are structural and are not perturbed) is compared with
   central finite differences of the QP solved to tolerance 1e-13 at `± h = 1e-6`, and with both
   one-sided differences. The active set of the perturbed solutions is recorded: where it equals
   the base active set on both sides the solution map is differentiable along that direction and the
   central difference is the reference; where it changes, the solution is at best directionally
   differentiable and a method is judged against the nearer one-sided difference. The value gap
   `|x_kkt − x_qp|` is recorded too (a method whose base value departs from the QP solution would
   change the forward pass through the `x_qp + (x_kkt − stop_grad(x_kkt))` construction).
5. **Optimization effect.** The 8.2 `hygiene_select` variant was re-run on the same five `diff2` starts
   (`workspace/phase8/8.2/diff2/starts.json`) with the chosen derivative. So was the variant that exposed
   the defect (`hygiene_select_reseed`), and on the Full objective the 8.3 `hygiene_select` and
   per-stage reseeding (`policy_regenerate_stage`) runs. Non-finite and spike counts, final and returned
   exact scores are paired against the Phase 8 runs.

### Results

**Corpus** (`workspace/phase9/9.1/corpus_index.json`, `corpus_{diff2,full3}_{T,R}.npz`, `corpus.log`;
traces `trace_*/iterates.npz`): 53 recorded iterations of 10 traces, 9 of them with a non-finite
gradient; 102,312 rolling QPs recorded. 17,309 are kept: the 16,385 with a failed interior-point solve and
924 solved ones. Of the solved ones, 612 were sampled as well-conditioned and 312 have an exact-derivative
VJP above 1e3.

| | translation QPs | rotation QPs |
| --- | --- | --- |
| recorded (`diff2` / `full3`) | 53,760 / 28,800 | 10,752 / 9,000 |
| interior-point solve failed | 9,234 (17.2 %) / 6,848 (23.8 %) | 121 (1.1 %) / 226 (2.5 %) |
| rank-deficient active set | 9,233 / 6,848 | 121 / 226 |
| exact-derivative VJP norm > 1e3 | 9,294 / 7,025 | 125 / 253 |
| exact-derivative VJP non-finite (GPU, fp64) | 0 / 0 | 0 / 0 |

Three facts the corpus establishes:

1. **Every rank-deficient active set belongs to a QP whose interior-point solve failed**
   (16,385 of 16,385 failed QPs have more active rows than independent ones; 0 of the 924 solved
   QPs do). A failed solve returns a point that is not a KKT point; the slack test then marks more
   rows active than the problem has variables, the KKT matrix has condition ~1e15 (median) and the
   exact derivative's VJP is ~6e9 (median). The value of such a step is already masked to zero by
   `solve_hard` (`x = where(success, x, 0)`), so this derivative never enters the objective's
   gradient except through the backward pass of the masked branch.
2. **Solved QPs are full-rank, and some have large but legitimate sensitivities**: cond(K) median
   2e7, VJP median 111, 90th percentile 3.7e3, maximum 5.8e4; 312 of 924 solved QPs have a VJP above
   1e3 under *every* method (table below). `P`'s spectrum does not explain them: its smallest
   eigenvalue is the regularization `qp_regularization / r_s² = 4.4e-3` on every solved state, its
   condition number 2.3e7 on every translation QP (the rotation prior 1e5 against that floor), and both
   are identical for the large-VJP states and the others. What separates them is the active
   constraints' multipliers: the large-VJP states have an active gap-band row with a large
   multiplier (median largest |λ| 22 / 30 / 75 on `diff2` translation / `full3` translation / `full3`
   rotation QPs, against 0.008 / 0.014 / 0.004 for the other solved states; Spearman ρ between VJP
   norm and largest |λ| 0.71 and 0.83 on the translation QPs). A rolling step whose ±`eps_g` gap
   band binds hard is close to infeasible, and the derivative of its solution with respect to the
   constraint data scales with that multiplier. These are the finite spikes the Phase 8.2 clipping at
   norm 100 handles. The stage-0 spikes have no KKT system at all: stage 0 uses the soft closed-form
   solver, and its Phase 8.2 `hygiene/start2` norms of up to 5e6 are sensitivities of that solve.
3. **The non-finite gradients are not produced in `kkt_resolve` at all.** On the recorded
   non-finite iterates the objective gradient is NaN (reproduced on the CPU, all 36 entries, every
   seed lane), while every QP's forward KKT solution and VJP are finite under every method. The
   iteration-level signature is a failed **rotation** QP: 9 of the 9 non-finite iterations have
   failed rotation QPs (3 to 60 of 384/360), 43 of the 44 finite iterations have none (the one
   exception has one). `jax_debug_nans` places the NaN in the rotation rollout's scan. The
   mechanism: a failed QP zeroes the joint step; in a rotation step the achieved rotation
   `dtheta_achieved(dq = 0)` is then exactly zero, and `karma_jax.steps.exp_so3(0)` has a NaN
   derivative (the norm's gradient at zero, `d exp_so3 / d omega |_0 = [nan nan nan]`, verified
   directly). The `where(success, x, 0)` guard quarantines this in translation steps (the rotation
   comes from the masked QP solution, whose cotangent the `where` zeros by selection), but in rotation
   steps the zero rotation is a function of the contact Jacobians and the NaN cotangent flows into the
   morphology gradient around the guard.

**Derivative comparison** (`compare.md`, `compare.json`; 3 directions per state, FD at `h = 1e-6`
on the QP solved to 1e-13; relative error = `|J d − FD| / |FD|`):

| method | well-conditioned, active set stable (941 directions, 321 states): median / 90 % rel. error vs central FD, share ≤ 1e-3 | degenerate-labelled, solved QP, stable (104 dir., 39 states) | active set changes under ±h (1,079 dir.): share within 1e-2 of the nearer one-sided FD | failed QPs (16,385): finite / VJP > 1e3 | median value gap `|x_kkt − x_qp|` (well) |
| --- | --- | --- | --- | --- | --- |
| exact (Phase 2) | 2.2e-6 / 1.0e-5, 0.99 | 4.8e-6 / 2.5e-5, 0.94 | 0.03 / 0.01 | 16,385 / 16,385 | 1.6e-11 |
| reg 1e-8 | 2.3e-6 / 1.0e-5, 0.99 | 2.6e-4 / 6.2e-4, 0.94 | 0.03 / 0.01 | 16,385 / 15,092 | 2.4e-10 |
| reg 1e-6 | 4.3e-6 / 3.3e-5, 0.99 | 2.2e-2 / 5.8e-2, 0.03 | 0.03 / 0.00 | 16,385 / 14,781 | 7.3e-9 |
| reg 1e-4 | 1.0e-5 / 2.8e-3, 0.82 | 6.1e-1 / 8.8e-1, 0.00 | 0.02 / 0.00 | 16,385 / 11,337 | 5.9e-7 |
| pinv (rcond 1e-10) | 2.2e-6 / 1.0e-5, 0.99 | 4.8e-6 / 2.5e-5, 0.94 | 0.03 / 0.01 | 16,385 / 5,983 | 1.6e-11 |
| **reduced** (dep_tol 1e-6; 1e-8 and 1e-4 identical) | 2.2e-6 / 1.0e-5, 0.99 | 4.8e-6 / 2.5e-5, 0.94 | 0.03 / 0.01 | 16,385 / 3,575 | 1.6e-11 |

Reading: where the QP is solved, `exact`, `pinv` and `reduced` are the same derivative to 1e-12 (no
solved QP has a dependent active row, so the reduction never removes anything and the pseudoinverse
never truncates); regularizing the dual block at 1e-6 or more buys nothing in finiteness and costs
accuracy (the "degenerate-labelled, solved" states are the ill-conditioned full-rank ones, where a
1e-6 dual regularization already moves the derivative by 2 %). Where the active set changes under the
perturbation (49 % of the sampled directions, i.e. the solution is at a kink), no method is within
1e-2 of either one-sided difference in more than 3 % of the directions: at a kink the KKT derivative
of the identified active set is one side's derivative at best, and the random problem-data
perturbations move several constraints at once; this is the expected behaviour of a piecewise-smooth
solution map and is not changed by any of the four methods (SPEC §4.2). At failed QPs the derivative
is masked; the reduction keeps the forward KKT solution and its VJP bounded in 78 % of them (3,575
against 16,385 with VJP > 1e3), which is the only place the methods differ.

**Decision.** Two changes, neither of which is damping:

1. `RollConfig.kkt_method = "reduced"` (dependent active rows removed, `dep_tol = 1e-6`) is the
   default hard-mode derivative. On every solved QP of the corpus it *is* the Phase 2 derivative;
   its only effect is a nonsingular KKT system at failed QPs (whose value is masked anyway), so
   the masked branch's backward pass multiplies zero cotangents by bounded numbers instead of
   ~1e10 ones (an overflow risk in fp32 that the fp64 corpus did not realize). The Phase 2 gradient
   tests pass unchanged for all four methods (`test_rolling_qp.py`, `test_rollout.py`).
2. `dkarma.core.geometry.exp_so3_safe` replaces `karma_jax.steps.exp_so3` in the rollout's two
   rotation updates. Its forward value and derivative are bit-identical to the reference wherever
   `|omega| ≥ 1e-10`. Below that it returns the identity with a zero derivative (the reference's
   guard), finite at exactly zero. This removes the actual NaN source. On the two recorded non-finite
   iterates (`diff2/hygiene_reseed/start1` iteration 54, `full3/policy_regenerate_stage/start2`
   iteration 22) the objective value is unchanged (0.113848, 0.041097) and the gradient is finite.

   *Reproduction check.* Unchanged code reproduces the Phase 8 runs to 2e-16 in every gradient norm
   (`trace_*/trace.json`, `reproduction`). A first version of `exp_so3_safe` computed the angle as
   `sqrt(sum(omega²))`, which differs from the reference's norm by one ulp. That was enough to change
   the stage-0 gradient by 4 % by iteration 4 and every later iterate, because the soft-stage
   dynamics amplify rounding by about 1e4 per iteration through the gradient spikes. The final version is
   bit-identical. A validation-free re-trace of the Phase 8 `hygiene_select` start 0 run with the final
   code reproduces all 60 iterations to 2.2e-16, including both hard stages under `reduced`
   (`workspace/phase9/9.1/bitcheck/`). A run without degenerate events is therefore unchanged bit for
   bit by the repair. On the two-finger objective any difference in the re-runs below is caused by the
   repair itself. On the Full objective the exact validator also changed between Phase 8.2 and now (8.6),
   which is separated out below.

What is **not** changed: the clipping at norm 100 (Phase 8.2) stays, because the large finite
gradients are legitimate sensitivities of nearly infeasible rolling steps (large gap-band
multipliers) that no derivative formula removes;
the QP failure rate itself (17–24 % of translation sub-steps in these hard-stage states) is a
property of the reference interior-point solver on infeasible or badly scaled steps and is reported
as the dominant source of masked steps; changing the solver would be a formulation change (rule 11).

### How to reproduce (9.1)

```bash
# re-trace a Phase 8.2 run (GPU; one process per trace): iterates, gradients, seeds, reproduction check
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.kkt_derivative trace --objective diff2 --variant hygiene_reseed --start start2_sobol1_s0.25 --out workspace/phase9/9.1
# the corpus of rolling QPs at the non-finite, spike and regular iterations of the traces (GPU)
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.kkt_derivative corpus --out workspace/phase9/9.1 --traces workspace/phase9/9.1/trace_*
# the derivative comparison (CPU, fp64)
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.kkt_derivative compare --out workspace/phase9/9.1
# the optimization effect: the 8.2 variants on the Phase 8 starts with the chosen derivative (GPU)
workspace/phase9/chain_91_rerun.sh reduced
```

### Optimization effect (8.2 `hygiene_select` re-run with the chosen derivative)

Procedure: `continuation_study --kkt-method reduced --starts-from workspace/phase8/8.2/<objective>` on the
Phase 8.2 starts (same `starts.json`, same start validations, optimizer seed 0, the Phase 8.2 variant
options), one process per variant (`workspace/phase9/chain_91_rerun*.sh`); analysis
`kkt_derivative effect --rerun-root workspace/phase9/9.1/rerun_reduced`
(`workspace/phase9/9.1/rerun_reduced/effect.{md,json}`). Exact-library validation with fresh seeds as in
Phase 8.2 (8 two-finger or 6 three-contact seeds, default seed configuration; fixed physical task). The
"first iteration that differs" is the first iteration whose gradient norm differs from the Phase 8 run by
more than 1e-9 relative, or whose finiteness differs.

| objective / variant | start | non-finite / clipped (Phase 8 → re-run) | first differing iteration (Phase 8's first non-finite) | returned exact score, Phase 8 → re-run | final selected, Phase 8 → re-run |
| --- | --- | --- | --- | --- | --- |
| diff2 / hygiene_select | start0 … start4 | 0 / 63 → 0 / 63 (all five) | none: identical | 0.2904, 0.3101, 0.2370, 0.2596, 0.3049 → identical | identical |
| diff2 / hygiene_select_reseed | start0 | 0 / 19 → 0 / 19 | none | 0.3046 → 0.3046 | 0.3046 → 0.3046 |
| | start1 | 17 / 14 → 0 / 15 | 44 (44) | 0.2516 → **0.3195** | 0.2002 → 0.3195 |
| | start2 | 37 / 13 → 0 / 19 | 23 (23) | 0.2268 → **0.3312** | 0.2268 → 0.3312 |
| | start3 | 0 / 13 → 0 / 13 | none | 0.2437 → 0.2437 | 0.1582 → 0.1582 |
| | start4 | 0 / 10 → 0 / 10 | none | 0.3366 → 0.3366 | 0.3054 → 0.3054 |

*Two-finger objective.* The repair leaves every run without a degenerate event bit-for-bit unchanged:
`hygiene_select` 5 of 5 and the reseeding variant 3 of 5. It changes the two runs that had non-finite
gradients exactly from their first non-finite iteration. Non-finite iterations fall from 54 to 0. Both
changed runs return better designs (+0.068 and +0.104 exact; mean returned over the five starts 0.2727 →
0.3071; final selected 0.2390 → 0.2838). With 2 non-tied starts of 5 the paired test cannot reach
significance (Wilcoxon p = 0.5). Each gain is 2 to 3 times the 0.03 standard deviation of one 8-seed
validation step (8.2), but two runs remain two runs.

*Full (3×4) objective.* The Phase 8.2/8.3 Full-objective runs were validated with the exact path *before*
8.6, whose smooth force pre-filter rejected some exactly feasible sub-steps. The re-runs are validated with
the 8.6 path, the exact SLSQP check at every sub-step (rule 11c). Identical iterates therefore validate
differently: start 0 at iteration 20 reads 0.019 in Phase 8 and 0.020 now. Where the Phase 8 stage
selection chose a different warm start than the corrected validator does (iteration 20 against 10 on
starts 1 and 2 of both variants), the runs diverge at iteration 21 without any non-finite event. The
gradient histories and validations show this (`effect.md`, `run.json`). Their returned scores mix the
validator correction with the repair. For a paired comparison under one validator, the returned designs of
both runs are re-scored with the current exact path and the runs' own protocol (6 fresh three-contact
seeds; `kkt_derivative rescore`, `rescore_full3.{md,json}`).

| objective / variant | start | non-finite (Phase 8 → re-run) | first differing iteration (Phase 8's first non-finite) | cause of the first difference | returned design re-scored with the current exact path: Phase 8 → re-run |
| --- | --- | --- | --- | --- | --- |
| full3 / hygiene_select | start0 | 0 → 0 | none | — | 0.0200 → 0.0200 (same design) |
| | start1 | 0 → 0 | 21 | warm start chosen by the corrected validator | 0.0163 → 0.0159 |
| | start2 | 0 → 0 | 21 | warm start chosen by the corrected validator | 0.0226 → 0.0146 |
| | start3 | 0 → 0 | none | — | 0.0248 → 0.0248 (same design) |
| | start4 | 3 → 0 | 38 (38) | the repair | 0.0183 → 0.0183 (same design: both runs return iteration 20) |
| full3 / policy_regenerate_stage | start0 | 17 → 0 | 24 (24) | the repair | 0.0200 → 0.0200 (same design: both runs return iteration 20, before the divergence at 24) |
| | start1 | 14 → 0 | 21 (27) | warm start chosen by the corrected validator | 0.0152 → 0.0151 |
| | start2 | 19 → 0 | 21 (22) | warm start chosen by the corrected validator | 0.0146 → 0.0146 (0.01460 → 0.01463) |
| | start3 | 0 → 0 | none | — | 0.0248 → 0.0248 (same design) |
| | start4 | 0 → 0 | none | — | 0.0174 → 0.0183 (same trajectory; the corrected validator returns iteration 20 instead of 30) |

On the Full objective the repair removes every non-finite iteration: 3 → 0 under `hygiene_select` and
50 → 0 under per-stage reseeding, the variant 8.3 found to expose the defect most. The repair changes a
run only from its first non-finite iteration. Under `hygiene_select` that changed no returned design: the
one affected run, start 4, returns iteration 20, before its divergence at 38. The two lower re-scores
(starts 1 and 2) come from runs that diverged at the stage boundary through the corrected validator, not
through the derivative. Iterations 1–20 are identical to Phase 8, and the warm start is chosen by
validating those identical iterates. Under per-stage reseeding the same holds: the one run the repair changed (start 0, from iteration 24) returns its iteration-20 design in both versions, and the remaining differences are validator-driven (mean re-scored returned score 0.0184 → 0.0185, paired difference +0.0001, Wilcoxon p = 0.75; under `hygiene_select` 0.0204 → 0.0187, −0.0017, p = 0.5, entirely from starts 1 and 2). On this objective, at these budgets, the repaired derivative removes the 53 non-finite iterations without a measurable effect on the returned designs; the Phase 8.2 hygiene (skip the non-finite iteration) had already contained them.

### Acceptance (9.1)

* *The chosen derivative passes the existing gradient tests unchanged on well-conditioned states.* All four
  methods pass the Phase 2 finite-difference tests at the Phase 2 tolerances, on both reference hands and
  over 4-step rollouts (`test_rolling_qp.py`, `test_rollout.py`). The one exception is `pinv`, which needs
  an absolute floor of 1e-5 on near-zero components and is not the chosen method. On the corpus,
  `reduced` equals the Phase 2 derivative on every solved QP: no solved QP has a dependent active row, so
  the KKT system is the same. **Met.**
* *No non-finite gradient on the corpus.* Every method's KKT solution and VJP are finite on all 17,309
  corpus QPs. The objective's gradient at the recorded non-finite iterates is finite after the
  `exp_so3_safe` change. The 20 re-runs have 0 non-finite iterations, against 107 in the same 20 Phase 8
  runs. **Met.**
* *The re-run matches or improves the Phase 8 returned scores (paired, 5 starts).* The variant the item
  names, 8.2's `hygiene_select` on the two-finger objective, matches bit for bit on 5 of 5 starts. Under
  per-stage reseeding it improves the 2 runs it changes (+0.068, +0.104) and leaves 3 identical. On the
  Full objective the repair changes no returned design. The differences there come from 8.6's corrected
  validator choosing different warm starts and returned iterates, measured above. The usual caveat applies:
  one 8-seed validation step moves the score by about 10 % (8.2), and two improved runs are two runs.
  **Met** for the named variant; no degradation attributable to the repair elsewhere.

### Limitations (9.1)

* The derivative comparison perturbs the QP data (`P, c, A_gap, l, u`) along random directions, not along
  the morphology-gradient direction of a particular objective. The corpus comes from 10 traces of two
  objectives and is dominated by failed-QP states (16,385 of 17,309).
* At active-set changes (49 % of the sampled directions) no method reproduces either one-sided derivative
  in more than 3 % of directions. This is a property of the piecewise-smooth hard QP, which the smooth
  objective's prefix weights absorb, not something a KKT formula repairs.
* The interior-point solve fails on 17 to 24 % of translation sub-steps in these hard-stage states. Those
  steps are masked (zero joint step) and remain the main loss of credit in the hard stages; changing the
  solver would be a formulation change (rule 11).
* Large finite gradients (VJP > 1e3 on 312 of 924 solved QPs) are legitimate sensitivities of nearly
  infeasible rolling steps (large gap-band multipliers). The norm-100 clipping of 8.2 stays.

## 9.2 Collision-valid original-KaRMA baseline (arm A′) and the key comparison

### Files changed

| File | Change |
| --- | --- |
| `karma_jax/model.py` | `FlatModel.cap_group` / `DeviceModel.cap_group` (default `None`): the finger group of every capsule, i.e. the first joint of its chain below the shared palm joint, or the chain's own root when no joint is shared. `build_flat_model(..., self_collision_all=True)` fills it. The default model is unchanged bit for bit, and `karma_jax/tests` pass on CPU and GPU: 101 passed, 1 skipped |
| `karma_jax/geometry.py` | `check_self_collisions` dispatches to `check_self_collisions_all` (every pair of valid capsules on different chains, clearance `dist − 2 r_l`) when `cap_group` is set, else to the reference's thumb-vs-index check (`check_self_collisions_pair`). Because the gate lives in the model, the seed filter, the nullspace polish and every step kernel of the proxy and of the official search apply it without further plumbing |
| `dkarma/core/admissibility.py` (new) | `admissibility(params, topo)`: the hard D18 check (minimum base separation 24.1 mm = `2 r_l + eps_col`, reach ≤ 250 mm per finger) with the pairwise separations and reaches |
| `dkarma/optimization/objectives.py` | `Constraints.base_sep_min_m` (smooth softplus penalty on every base pair, width 2 mm) and `D18_CONSTRAINTS`; `exact_validation(enforce_d18=True)` scores an inadmissible iterate zero so it is never returned; `self_collision_all` forwarded to the physical original KaRMA |
| `dkarma/metrics/exact_library.py` | `official_karma_physical(self_collision_all=...)` |
| `dkarma/experiments/admissible_baseline.py` (new) | `armAprime`: the Phase 7 `cmaes_original_karma` with D18 enforced on every candidate. `full`: Adam Full KaRMA under D18 from the same starts. `suite`: the common suite at the 9.4 budgets under three seed configurations, with the platform recorded. `sweep`: the torque and mass sweep of 9.5. `summary`: paired statistics (sign counts, Wilcoxon, sign test, bootstrap CI), ranking stability and the MuJoCo leg |
| `dkarma/experiments/simulation_replay.py` | MuJoCo leg of 9.2: three-contact libraries at 12 seeds per design (`export --libraries three_contact --budget 12`) replayed under both 9.3 controllers |
| `dkarma/tests/test_admissibility.py` (new) | the hard check on the start, arm A (inadmissible, 17.8 mm) and arm B; the penalty (zero when admissible, active and differentiable when not); the all-finger gate (three groups on 3×4 hands, equals dkarma's `self_clearance` up to the radii, blind pair check vs failing all-finger check on arm A, identical to the pair check on two-finger hands) |

### Procedure

Starts: the eight Phase 8.4 `full3` starts (`workspace/phase8/8.4/full3/starts.json`: the
anthropomorphic start, four Sobol perturbations at scale 0.25 and three at 0.5, Sobol seed 321); all
eight satisfy D18 (minimum base separation 29.7 to 49.9 mm, reach ≤ 250 mm).

*Arm A′* (`admissible_baseline armAprime --only k`, `workspace/phase9/9.2/armAprime/<start>/`): CMA-ES
(`cma` 4.0, population 8, 12 generations = 96 evaluations, σ₀ 0.3, CMA seed = start index) against
the original KaRMA via the `karma_jax` fast proxy with the Phase 7 settings (5 linspace samples per
joint, 256 Sobol + 64 random joint samples, 5 offsets, dedup 0.1, 48 seeds, stage-1 budget 15, 3
finalists at 10,000 states; **published `L_ref`-scaled task**, as arm A). D18 throughout: a candidate
whose base separation is below 24.1 mm or whose reach exceeds 250 mm is scored zero without
evaluation; the proxy runs on `build_flat_model(self_collision_all=True)`, so its seed filter, polish
and BFS gates reject any configuration in which any two fingers' capsules come within `eps_col`
(at the proxy's own `L_ref`-scaled link radius). The returned design is the best admissible candidate
by the proxy (as arm A: `es.result.xbest`). Same start, bounds and evaluation budget as arm A; arm A
itself is the `start0` run of this procedure without D18 (Phase 7, CMA seed 0).

*Full KaRMA under D18* (`admissible_baseline full --only k --kkt-method reduced`,
`workspace/phase9/9.2/full_d18/<start>/`): the Phase 8.4 Adam settings (40 iterations, 2 stages, 6
three-contact seeds, `ADAM_DEFAULTS`, learning rate 0.02, validation every 10 iterations and at stage
ends, optimizer seed 0) with the D18 base-separation penalty added to the reach penalty and the exact
validation gated by admissibility (an inadmissible iterate scores zero and cannot be returned); the
9.1 derivative. Arm B (Phase 7 `run_full`) is kept as a design and re-scored on the same suite.

*Suite* (`admissible_baseline suite --design <key>`, `workspace/phase9/9.2/suite/<key>/suite.json`;
**fixed physical task**): for every start, every arm A′ and Full-D18 design, arm A and arm B: the
exact Full-KaRMA counts (actuated three-contact cells of 8,000 and SO(3) bins of 4,608, exact SLSQP
force check at every sub-step and the 8.1 seed gate, `MC.evaluate_fixed_contacts_exact`), the exact
MC counts, the exact two-finger library coverage (`evaluate_exact`), with **12 three-contact and 16
two-finger fresh seeds** under three seed-search configurations (`default`: the Phase 7/8 recipe,
Sobol 256 + random 64, `RandomState(42)`; `sobol512_rs7`: 512 + 96, seed 7; `lin7_rs13`: 7 linspace
samples, 192 + 64, seed 13, offsets ±7.5/±12.5 mm); wrench capacities at the three-contact seeds (14
directions) and the physical original KaRMA (top-3 mean cube fraction, both with the all-finger gate
and with the reference's pair gate) once per design. The start designs are scored under the default
configuration only.

### Results

*Admissibility without the constraint.* All 16 Full-KaRMA designs of Phase 8.4 (8 Adam, 8 CMA-ES; same
eight starts, no D18 in the objective) satisfy D18. Their minimum base separation is 35.8 to 74.5 mm and
their reach at most 131 mm (`workspace/phase8/8.4/full3/start*/{adam,cmaes}/optimized_morphology/`). The
three-contact objective keeps all three fingers in play, so the D18 penalty is inactive at those designs.
The original-KaRMA objective scores the thumb–index pair only. Unconstrained, it produced arm A (index and
middle bases 17.8 mm apart). Under D18 it proposes inadmissible candidates in every run, 2 to 18 of the
96 (`armAprime/*/run.json`); they are scored zero without evaluation.

*Exact comparison* (`admissible_baseline summary`, `workspace/phase9/9.2/summary.{md,json}`, suites under
`workspace/phase9/9.2/suite/<design>/suite.json`; **fixed physical task**; D18 design box: base separation
≥ 24.1 mm, reach ≤ 250 mm, all-finger clearance; 12 three-contact and 16 two-finger fresh seeds per
configuration, mean over three seed configurations). Starts: the 8.4 `full3` set. Arm A′: CMA seed =
start index. Full-D18: optimizer seed 0, `kkt_method reduced`. Design files:
`workspace/phase9/9.2/{armAprime,full_d18}/<start>/optimized_morphology/optimized_morphology.design.json`.

| start | start design (default config) | arm A′ | Full-D18 | Full − A′ |
| --- | --- | --- | --- | --- |
| start0 (anthropomorphic) | 0.0182 | 0.0079 | 0.0215 | +0.0136 |
| start1 | 0.0172 | 0.0156 | 0.0191 | +0.0036 |
| start2 | 0.0152 | 0.0000 (no three-contact seed found: 12 / 45 / 24 IK attempts from 4 / 15 / 8 reachable joint samples, none kinematically valid) | 0.0176 | +0.0176 |
| start3 | 0.0129 | 0.0108 | 0.0123 | +0.0015 |
| start4 | 0.0062 | 0.0123 | 0.0105 | −0.0017 |
| start5 | 0.0083 | 0.0175 | 0.0000 (no three-contact seed found: none of 381 / 669 / 341 joint samples brings the three fingertips around a sphere, so no IK attempt) | −0.0175 |
| start6 | 0.0119 | 0.0093 | 0.0154 | +0.0061 |
| start7 | 0.0067 | 0.0117 | 0.0109 | −0.0008 |

Exact Full-KaRMA score (cells / 8000 / 0.1 + SO(3) bins / 4608 / 0.3), fixed physical task.

| measure (fixed physical task unless stated) | arm A′: mean / median / best / IQR | Full-D18: mean / median / best / IQR | Full − A′: wins / ties / losses | mean difference, bootstrap 95 % CI | Wilcoxon p |
| --- | --- | --- | --- | --- | --- |
| **exact Full-KaRMA score** (primary) | 0.0106 / 0.0112 / 0.0175 / [0.0089, 0.0131] | 0.0134 / 0.0138 / 0.0215 / [0.0108, 0.0180] | 5 / 0 / 3 | +0.0028, [−0.0044, +0.0093] | 0.383 |
| exact actuated three-contact cell fraction | 0.0003 / 0.0003 / 0.0005 | 0.0004 / 0.0004 / 0.0007 | 6 / 0 / 2 | +0.0001, [−0.0001, +0.0003] | 0.195 |
| exact SO(3) bin fraction | 0.0024 / 0.0024 / 0.0039 | 0.0028 / 0.0030 / 0.0044 | 5 / 0 / 3 | +0.0004, [−0.0011, +0.0018] | 0.641 |
| exact MC (unactuated) cell fraction | 0.0003 / 0.0003 / 0.0005 | 0.0004 / 0.0004 / 0.0007 | 5 / 0 / 3 | +0.0001, [−0.0001, +0.0003] | 0.250 |
| exact two-finger library score | 0.2047 / 0.2088 / 0.2374 / [0.1802, 0.2294] | 0.1435 / 0.1468 / 0.1764 / [0.1299, 0.1748] | 1 / 0 / 7 | −0.0612, [−0.0877, −0.0337] | 0.016 |
| original KaRMA, physical task, all-finger gate (cube fraction) | mean 0.0145 | mean 0.0093 | 3 / 0 / 5 | | |
| original KaRMA, *published* `L_ref`-scaled task, all-finger gate (KaRMA-T) | mean 0.0255 | mean 0.0112 | 1 / 0 / 7 | | |

*Each arm optimizes what it is built to.* Arm A′ raises its own objective, the published KaRMA-T with
the all-finger gate, over the start on 7 of 8 starts (mean 0.0088 → 0.0255), and its two-finger exact
coverage beats Full-D18 on 7 of 8 (p = 0.016, under every seed configuration). Full-D18 raises the exact
Full score over the start on 6 of 8 starts. Arm A′ does so on 3 of 8.

*The key comparison.* On the primary measure, Full KaRMA under D18 is ahead of the collision-valid
original-KaRMA optimum on 5 of 8 starts, with a mean paired difference of +0.0028 (+26 % of A′'s mean).
The bootstrap 95 % interval is [−0.0044, +0.0093], Wilcoxon p = 0.38, and under the default seed
configuration the split is 4 / 4. **The paired data do not support the claim that Full KaRMA produces
better simultaneous-contact manipulation than a collision-valid original-KaRMA optimum** at eight
starts and these budgets. They do not refute it either: the interval includes effects as large as the
start-to-start spread. Two starts decide most of the mean, and each is a total loss of the three-contact
grasp: start 2 for A′ (the CMA-ES returned design has no three-contact seed) and start 5 for Full-D18 (Adam
loses the grasp by iteration 10, as in Phase 8.4). Without them, Full-D18 wins 4 of 6, mean +0.0037.

*Budgets.* The arms are compared at their Phase 7 settings, not at matched compute. Arm A′ spends 96
proxy evaluations, 2,300 s per run on average (1,827–3,799 s). Full-D18 spends 40 Adam iterations, 69 s of
optimizer time per run (816 s with validations). The compute difference favours arm A′ by more than 30×,
which would make a Full-KaRMA win conservative. The data show no significant win, so no budget-matched
control was run. The comparison mixes objective and optimizer (CMA-ES on a proxy against Adam on the
smooth Full objective), as the Experiment IV arms do by design.

*Arm B and arm A.* Arm B (Phase 7, Full KaRMA + Adam from start 0, admissible) scores 0.0146 ± 0.0014,
between arm A′ start 0 (0.0079) and Full-D18 start 0 (0.0215). Arm A (no D18): the seed search found no three-contact seed in 792 / 1,485 / 726 IK attempts under the
three configurations, consistent with 8.5. Its physical original KaRMA collapses under the all-finger gate
from 0.0140 to 0.0001: its index and middle fingers overlap in every configuration.

*MuJoCo leg.* Every arm A′ and Full-D18 design was replayed in the 9.3 scene under both 9.3
controllers. Each replay used the three-contact library with 12 fresh seeds under the default seed
configuration, exported on the CPU (`workspace/phase9/9.2/sim/`, `chain_92_sim_cpu.sh`). The paired measure
is the number of exactly valid three-contact primitives that also succeed in simulation (pose within
4 mm / 0.2 rad and every assigned contact at the primitive end), per budgeted seed. It combines how many
motions the exact model credits with how many of them the simulated hand executes. Fixed physical task.

| start | position servo: arm A′ → Full-D18 | force controller: arm A′ → Full-D18 |
| --- | --- | --- |
| start0 | 0.50 → 5.33 | 0.67 → 8.08 |
| start1 | 3.92 → 7.92 | 4.67 → 9.83 |
| start2 | 0.00 → 8.00 (arm A′: no seed) | 0.00 → 10.25 |
| start3 | 0.17 → 0.67 | 0.33 → 0.67 |
| start4 | 3.67 → 5.25 | 4.25 → 6.75 |
| start5 | 0.50 → 0.00 (Full-D18: no seed) | 2.58 → 0.00 |
| start6 | 4.08 → 10.75 | 5.08 → 12.75 |
| start7 | 2.17 → 1.17 | 4.25 → 1.25 |

| controller | Full − A′ wins / losses | mean difference, bootstrap 95 % CI | Wilcoxon p | same with pose-only success: wins / losses, mean, p |
| --- | --- | --- | --- | --- |
| position servo | 6 / 2 | +3.01 [+0.90, +5.20] | 0.062 | 5 / 3, +3.26, 0.461 |
| force controller | 6 / 2 | +3.47 [+0.24, +6.60] | 0.148 | 5 / 3, +3.11, 0.547 |

In simulation the Full-D18 designs execute more simultaneous-contact primitives than the arm A′ designs
from the same start on 6 of 8 starts, under both controllers. The bootstrap interval of the mean excludes
zero, but the rank test does not reach 5 % (p = 0.062 and 0.148), and the pose-only reading is not
separated. The advantage comes from the exact model crediting more motions (more exactly valid
primitives per design), not from a higher simulation success rate. Full-D18 start0's strict rate on
holding seeds is 0.22 / 0.33 (position / force), against arm A′ start6's 0.37 / 0.46 and arm B's
0.37 / 0.50. This leg uses one seed-search configuration and is secondary to the exact comparison.

**Answer to the question of SPEC §11.2.** On a mechanically admissible design space, with collision-valid
baselines, eight starts and these budgets: Full KaRMA produces three-finger hands whose exact
simultaneous-contact capability is higher than that of the original-KaRMA optimum on 5 of 8 starts
(+26 % on average) and whose simulated three-contact execution is higher on 6 of 8. Neither difference
is significant at 5 % by the paired rank test; the exact one's confidence interval includes zero. The
original-KaRMA optimum is the better two-finger hand (7 of 8, p = 0.016). The paired data therefore do
**not** support the claim that Full KaRMA produces better simultaneous-contact manipulation than a
collision-valid original-KaRMA optimum. They point in that direction and leave it to more starts, or to
hardware, to decide.

### Acceptance (9.2)

The comparison is reported with its uncertainty: eight starts per arm, fresh seeds, three seed
configurations, paired statistics, and the MuJoCo leg under both controllers. The superiority claim is
**not made**, because the paired data do not support it. Arm A′ satisfies D18 throughout (every returned
design admissible, inadmissible proposals scored zero, all-finger gate in the proxy) and achieves its own
objective. The arms use their Phase 7 budgets, which favour arm A′ in compute by more than 30×. **Met**,
as the acceptance is worded: "reported with its uncertainty, whichever way it goes".

### Limitations (9.2)

* Eight starts give a paired test with little power: an effect of +26 % on the exact Full score is not
  separable from the start-to-start spread here. Two starts carry most of the variance, one per arm,
  each a total loss of the three-contact grasp.
* The arms differ in optimizer as well as objective (CMA-ES on the proxy against Adam on the smooth Full
  objective), as Experiment IV's arms do. The original KaRMA has no gradient, and Full KaRMA with CMA-ES
  was behind Adam in 8.4, so a single-optimizer comparison was not run.
* Arm A′'s proxy uses the published `L_ref`-scaled task, as arm A did. All comparisons use the fixed
  physical task, on which arm A′ was not optimized; the published-task numbers are reported for that
  reason.
* The MuJoCo leg uses one seed-search configuration and the 9.3 contact model and controllers.

### How to reproduce (9.2)

```bash
workspace/phase9/chain_92_armAprime.sh                 # arm A' from the eight starts (GPU, ~40-60 min per start)
workspace/phase9/chain_92_full.sh reduced              # Full KaRMA under D18 from the same starts (GPU, ~10 min per start)
workspace/phase9/chain_92_suites.sh                    # suites of every design at the 9.4 budgets, summary, 9.3 reference exports
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline summary --out workspace/phase9/9.2
```

## 9.3 Force-controlled MuJoCo replay with separated criteria

### Files changed

| File | Change |
| --- | --- |
| `dkarma/experiments/simulation_replay.py` | export: `exact_force_path(return_fn=True)` also returns the exact allocation's contact normal forces per state (`fn_exact [H+1, Cn]`); `lane_offsets` gives per-lane joint directions that push one fingertip along its contact normal (`dq_lane [H+1, Cn, J]`); both stored in the trajectory archives. simulate: `controller="force"` (admittance regulation of each lane's squeeze on the measured normal force, set-point = `force_scale ×` the exact allocation), three separated outcomes per primitive (object pose, contact retention, drop / gross instability), `separated_outcomes` summary, `--controller/--force-scale/--k-i/--k-p/--continue-after-failure/--design-dir`. The position servo of 8.6 is unchanged and reproduces the 8.6 baseline numbers (112 / 115 strict, 100 with orientation) |

### Controller and criteria

Scene as 8.6 (capsule links, 15 mm / 35 g sphere, μ 0.6 with torsional 0.0018 m, gravity −z,
`implicitfast` 0.5 ms, position actuators `kp` 20 N·m/rad with `forcerange ±0.108 N·m`). Both
controllers stream the exported joint path (linear ramp over 50 ms per 1.25 mm sub-step) after a
0.3 s settle.

* **Position servo (8.6):** every assigned fingertip commanded 0.5 mm into the sphere along its
  contact normal at every sub-step (fixed geometric squeeze).
* **Force controller (9.3, `controller="force"`):** per lane `c` the squeeze `delta_c` (m) is an
  admittance state driven by the measured fingertip normal force, with the exact allocation as a
  **lower bound**: `d delta_c/dt = k_i max(f*_c − f_c, 0) − leak·k_i max(f_c − f*_c, 0)`,
  `delta_c ∈ [0.5, 3] mm` (never below the 8.6 squeeze), started at 0.5 mm, `k_i = 0.2 m/(N·s)`,
  `leak = 0.1`; `f*_c` is the exact minimum-norm SLSQP allocation's normal force of that lane at the
  current sub-step (interpolated, `force_scale` 1.0) and `f_c` the normal force MuJoCo measures between
  the sphere and the lane's capsule (`mj_contactForce`). The command is `q_path + Σ_c delta_c
  dq_lane[:, c]`: a lane whose force falls below its allocation (a contact slipping away) advances
  along its contact normal at up to `k_i f*_c` (0.1 m/s per newton of deficit) until the force is
  restored; a lane pressing harder leaks back toward the baseline squeeze. The joint servos and the
  torque ceiling are unchanged, so this is an admittance outer loop on the fingertip normal force,
  realizable on a position-controlled hand with fingertip force sensing.

  *Tuning record* (two-finger baseline only, settle phase and the first four primitives of five
  translation trajectories, `workspace/phase9/9.3_smoke/`, scratch script `force_tune.py`): the
  symmetric regulator tried first (`d delta_c/dt = k_i (f*_c − f_c)`, `delta_c ≥ −0.5 mm`,
  `controller="force_sym"`) **does not hold the object**: the minimum-norm allocation (0.44 N and 0.14 N
  at the first seed) is weaker than the grip the 0.5 mm squeeze produces (0.79 and 0.49 N), and a
  two-finger pinch cannot realize two independent normal-force targets, so the two integrators wind up
  against each other and the sphere is lost in the settle phase (0 of 5 seeds hold at `force_scale` 1,
  4 of 5 at 2 or 3 with 1 to 2 first primitives succeeding; a proportional term of 2 mm/N destabilizes
  every setting). The force-floor law holds 5 of 5 seeds and matches the position servo on the same
  trajectories (primitives 1 to 4 succeeding on 5/5/4/2 trajectories against 5/4/4/1; `k_i` 0.05 to 0.2,
  `leak` 0 or 0.1 and `force_scale` 1 to 2 are indistinguishable there). On the full baseline library
  the force-floor controller reproduces the position servo's two-finger numbers (112 of 115 strict,
  101 with orientation, no drop) — as expected, since the two-finger failures of 8.6 were orientation
  losses, not contact losses.

Outcomes per primitive, at the primitive end: **object-pose success** = centre within 4 mm and
orientation within 0.2 rad of the command; **contact retention** = every assigned fingertip in
contact; **drop / gross instability** = centre more than 20 mm from the command or no assigned
fingertip in contact. The force-controlled trajectories continue after a pose or retention failure
and stop only at a drop; the 8.6 strict count (position + contacts, first failure) is the same
quantity under both stopping rules. "Seed does not hold" (settle drift > 4 mm) is reported
separately.

**Decision rule**, fixed on 2026-10-14 before any force-controlled three-contact replay was run
(`simulation_replay.CONTROLLER_INDUCED_GAIN`, `MODEL_INDUCED_GAIN`). The quantity is the strict
three-contact success rate: exactly valid primitives of seeds that hold under that controller, with
pose and every assigned contact. If force control minus the position servo is at least +0.30 on
every three-contact design, the 8.6 discrepancy is called controller-induced. If it is below +0.10
on every design, it is called model-induced. Anything else is called mixed and reported per design.

### Results

Script `dkarma/experiments/simulation_replay.py` (`augment`, `export`, `simulate`, `compare`); outputs
`workspace/phase9/9.3/simulation_<controller>__<design>.json`, `controller_comparison.{md,json}`. The
trajectories are the 8.6 archives (`workspace/phase8/8.6/<design>_trajectories.npz`: 6 seeds per library,
default seed configuration, budget 6), augmented with the exact allocation's normal forces and the per-lane
directions. Recomputing the exact force verdicts reproduced every stored verdict. Arm A′ is the design from
the anthropomorphic start (`workspace/phase9/9.2/armAprime/start0_nominal/`, CMA seed 0), exported with the
same recipe. Both controllers run every trajectory to its end and stop only at a drop. The position servo
reproduces every 8.6 count (three-contact 21/135 and 39/103; two-finger 100/115, 82/111, 87/133). Fixed
physical task throughout.

**Separated outcomes**, over exactly valid primitives of the seeds that hold under that controller
(settle drift ≤ 4 mm):

| design | library | controller | seeds hold | exactly valid primitives | strict: pose + all contacts | pose | contact retention | no drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two-finger baseline | two-finger | position | 5/6 | 115 | 0.87 | 0.87 | 0.97 | 0.98 |
| two-finger baseline | two-finger | force | 5/6 | 115 | 0.88 | 0.88 | 0.97 | 1.00 |
| 3×4 start | two-finger | position | 6/6 | 111 | 0.74 | 0.75 | 0.88 | 0.94 |
| 3×4 start | two-finger | force | 6/6 | 111 | 0.77 | 0.80 | 0.97 | 1.00 |
| 3×4 start | three-contact | position | 4/6 | 135 | 0.16 | 0.56 | 0.16 | 0.94 |
| 3×4 start | three-contact | force | 4/6 | 135 | 0.28 | 0.69 | 0.29 | 0.99 |
| arm B | two-finger | position | 4/6 | 133 | 0.65 | 0.65 | 0.98 | 1.00 |
| arm B | two-finger | force | 4/6 | 133 | 0.65 | 0.65 | 0.98 | 0.99 |
| arm B | three-contact | position | 4/6 | 103 | 0.38 | 0.78 | 0.38 | 0.97 |
| arm B | three-contact | force | 4/6 | 103 | 0.52 | 0.81 | 0.52 | 0.98 |
| arm A′ (start 0) | two-finger | position | 5/6 | 159 | 0.53 | 0.59 | 0.67 | 0.75 |
| arm A′ (start 0) | two-finger | force | 5/6 | 159 | 0.67 | 0.72 | 0.89 | 0.97 |
| arm A′ (start 0) | three-contact | position | 2/6 | 25 | 0.24 | 0.60 | 0.24 | 0.80 |
| arm A′ (start 0) | three-contact | force | 3/6 | 43 | 0.14 | 0.53 | 0.14 | 0.81 |
| arm A, middle finger removed | two-finger | either | 6/6 | 122 | 0.79 | 0.79 | 1.00 | 1.00 |

**Confusion matrices** (8.6 convention: all seeds, including those that do not hold, per primitive up to
the first exactly rejected one, criterion pose + all contacts; succeed / fail):

| design | library | position: exactly valid | position: exactly rejected | force: exactly valid | force: exactly rejected |
| --- | --- | --- | --- | --- | --- |
| baseline | two-finger | 100 / 16 | 16 / 39 | 101 / 15 | 16 / 39 |
| 3×4 start | two-finger | 82 / 29 | 13 / 43 | 86 / 25 | 13 / 43 |
| 3×4 start | three-contact | 21 / 158 | 7 / 57 | 38 / 141 | 10 / 54 |
| arm B | two-finger | 87 / 76 | 8 / 48 | 87 / 76 | 10 / 46 |
| arm B | three-contact | 39 / 92 | 12 / 55 | 54 / 77 | 11 / 56 |
| arm A′ (start 0) | two-finger | 85 / 92 | 6 / 44 | 106 / 71 | 8 / 42 |
| arm A′ (start 0) | three-contact | 6 / 74 | 4 / 68 | 6 / 74 | 5 / 67 |

**Verdict by the pre-registered rule: mixed.** The force controller's gain in strict three-contact success
is +0.13 on the start, +0.15 on arm B and −0.10 on arm A′. The arm A′ change compares different holding
seeds (2 against 3 of 6) over 25 and 43 primitives. The three-contact discrepancy of 8.6 is therefore
neither removed by force control nor unaffected by it.

**What fails, and how** (post hoc, `controller_comparison.md` "Post hoc"; not part of the rule):

* *Drops are rare* under both controllers. On the start and arm B, 94 to 99 % of the exactly valid
  three-contact primitives keep the sphere within 20 mm with at least one contact. *The object pose
  follows the command* in 56 to 81 % of them. The strict criterion fails mostly on contact retention: on
  the start under the position servo, 47 of the 114 failing primitives keep the pose but lose a contact.
* *Under the position servo the lost contact is absent for the whole primitive.* Among lanes failing
  retention, the median contact duty cycle is 0.00, and 70 % (start) and 60 % (arm B) never touched the
  sphere during the primitive. The lost lane is the lightest loaded: on the start it is the middle finger
  in 44 of 51 losses with the pose kept, on arm B the index finger in 22 of 31. Its minimum-norm
  allocation is often ~0 N, and the fixed 0.5 mm geometric squeeze does not keep that fingertip on
  the sphere.
* *Under the force controller the losses are intermittent.* The failing lanes touched the sphere for a
  median of 61 % (start) and 52 % (arm B) of the primitive, at a mean normal force of 0.21 and 0.25 N.
  Only 2 % and 0 % never touched it, and the lane's squeeze is far from its 3 mm limit (median 0.65 and
  0.82 mm). The remaining strict failures are contact chatter at the sub-millimetre scale, not a
  fingertip that cannot reach the sphere. Reading retention as "in contact for at least half of the
  primitive", force-controlled three-contact success is 0.53 (start) and 0.75 (arm B), against 0.80
  and 0.65 for the same designs' two-finger libraries. The position servo gives 0.21 and 0.54.
* *Arm A′'s three-contact grasps hold poorly in simulation*: 2 or 3 of 6 seeds, against 4 of 6 for the
  start and arm B. On the seeds that hold, the failing lanes touch for only a median 23 % of the primitive
  under force control. Its two-finger library gains most from force control (strict 0.53 → 0.67,
  retention 0.67 → 0.89).

**Interpretation, with the numbers.** On the start and arm B, the exact quasi-static model predicts the
object motion (pose success 0.69 and 0.81 under force control). It does not predict whether a lightly
loaded third fingertip stays on the sphere at every instant. That second quantity depends on the
controller: the position servo loses the contact outright, and the force-floor controller turns most of
those losses into intermittent contact. By the strict, instantaneous criterion that 8.6 used, roughly a
fifth (start: 0.13 of a 0.58 deficit to its two-finger library) to a half (arm B: 0.15 of 0.27) of the
three-contact deficit is controller-induced. The remainder sits in contact maintenance at the
sub-millimetre scale under this contact model (`solref` 0.002, rigid capsules), not in pose tracking.
Arm A′ is the exception: its three-contact seeds rarely hold in simulation under either controller.

### Limitations (9.3)

* One force controller, an admittance outer loop on the measured normal force, tuned on the two-finger
  baseline only. The minimum contact force (0.05 N) was set from the allocation statistics before any
  three-contact force replay; no gain was tuned on three-contact data. A controller with tangential force
  control, or compliant fingertips, could change the remaining chatter.
* The contact model is unchanged from 8.6 (`solref 0.002 1`, `solimp 0.95 0.99 0.001`, rigid capsules). The
  fabricated hand will have compliant fingertips (SPEC §12 IV), which this scene does not model.
* Six seeds per library (the 8.6 archives), so the per-design rates rest on 25 to 135 primitives of 2 to
  6 holding seeds. Arm A′'s three-contact rates rest on 25 and 43 primitives and are not comparable
  between controllers because the holding seeds differ.
* The duty-cycle reading of retention is post hoc and descriptive; the verdict uses the pre-registered
  strict criterion.

### How to reproduce (9.3)

```bash
# the 8.6 archives with the exact allocation's normal forces and per-lane directions added (CPU)
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay augment --src workspace/phase8/8.6 --out workspace/phase9/9.3 --designs baseline start armB armA
# arm A' (start 0): export with the 8.6 recipe (6 seeds per library)
cp workspace/phase9/9.2/armAprime/start0_nominal/optimized_morphology/optimized_morphology.design.json workspace/phase9/9.3/armAprime_start0.design.json
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay export --out workspace/phase9/9.3 --design-dir workspace/phase9/9.3 --designs armAprime_start0
# replays: one process per design and controller (add --drop-fingers middle for arm A, --design-dir for arm A')
for C in position force; do for D in baseline start armB; do
  JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay simulate --out workspace/phase9/9.3 --designs $D --controller $C --continue-after-failure --tag _${C}__$D
done; done
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay compare --out workspace/phase9/9.3
```

## 9.4 Final validation with more seeds

### Files changed

| File | Change |
| --- | --- |
| `dkarma/metrics/multicontact.py` | `evaluate_fixed_contacts_exact` runs each seed's command rollouts through one jitted program per roll configuration (`_command_rollouts`, model, state and commands as arguments) instead of an eager `vmap` of a fresh closure per seed. The eager form recompiled the scan for every seed, about 3 s each on the CPU, and accumulated compiled programs (about 15 GB of host memory over one suite). `CACHED_ROLLOUTS = False` restores the Phase 8 path |
| `dkarma/metrics/exact_library.py` | the same for `evaluate_exact`'s translation and rotation validators (`_validate_commands`, `CACHED_VALIDATORS`); `official_karma(self_collision_all=...)` |
| `dkarma/tests/test_multicontact.py`, `dkarma/tests/test_diff_karma.py` | the cached and eager paths give identical cells, bins, verdicts and gate codes (three-contact with and without the torque gates; two-finger) |

*Identity check* (scratch script, CPU, fp64). Arm B and arm A′ start 0 were each evaluated twice, with 12
three-contact and 16 two-finger fresh seeds: Full, MC and two-finger exact evaluators, eager against
cached. Every seed's cell set, bin set, per-primitive verdict and gate code is identical. The cached
path takes 94 / 64 / 25 s against 149 / 117 / 112 s for arm B (Full / MC / two-finger) and 45 / 31 / 21 s
against 102 / 81 / 117 s for arm A′; the SLSQP force checks are unchanged and dominate the remainder.
All suites therefore run on the CPU (`jax.default_backend()` is recorded in every `suite.json`), three
at a time (`workspace/phase9/chain_92_suites_cpu.sh`).

### Procedure

Every number that ranks candidates in 9.2 and 9.5 comes from `admissible_baseline suite`
(`workspace/phase9/9.2/suite/<design>/suite.json`, fixed physical task), which re-scores each design
with **fresh seeds**: 12 three-contact seeds (`find_multi_seeds`, budget 12) and 16 two-finger seeds
(`find_seeds`, budget 16), under **three seed-search configurations**:

| configuration | joint samples | random draws | offsets along the capsule axes |
| --- | --- | --- | --- |
| `default` (the Phase 7/8 recipe) | 5 linspace per joint, 256 Sobol | 64, `RandomState(42)` | 0, ±5, ±10 mm |
| `sobol512_rs7` | 5 linspace per joint, 512 Sobol | 96, `RandomState(7)` | 0, ±5, ±10 mm |
| `lin7_rs13` | 7 linspace per joint, 192 Sobol | 64, `RandomState(13)` | 0, ±7.5, ±12.5 mm |

IK (projected Levenberg–Marquardt, 100 iterations), the exact verification and the budget cut are
the same in all three. The reported value of a design is the mean over the three configurations,
with the standard deviation over configurations as its seed-set spread; per-seed counts are in the
JSON. Start designs, which rank nothing, are scored under `default` only. Ranking stability is
measured over the candidate designs: arm A, arm B, the eight arm A′ and the eight Full-D18 designs.
For each pair of configurations it is the Spearman rank correlation of the per-design scores and
whether the best design is the same; the paired 9.2 comparison is repeated under each configuration
separately.

*Acceptance as applied:* the 9.2 ranking is stable if, under every seed configuration, the paired
comparison has the same direction (the sign of the mean Full − A′ difference and the larger side of
the win/loss count), and if the Spearman correlation between every pair of configurations is at least
0.8 over the candidate designs. This threshold was fixed before the suites ran.

### Results

`workspace/phase9/9.2/summary.md` (fixed physical task; 18 candidate designs: arm A, arm B, eight arm A′,
eight Full-D18; 12 three-contact and 16 two-finger fresh seeds per configuration).

| measure | Spearman ρ: default vs sobol512_rs7 / default vs lin7_rs13 / sobol512_rs7 vs lin7_rs13 | best design under each configuration |
| --- | --- | --- |
| exact Full-KaRMA score | 0.903 / 0.905 / 0.946 | Full-D18 start0, Full-D18 start0, Full-D18 start0 |
| exact actuated three-contact cell fraction | 0.953 / 0.954 / 0.957 | Full-D18 start0 (all three) |
| exact SO(3) bin fraction | 0.888 / 0.885 / 0.921 | Full-D18 start0 (all three) |
| exact two-finger library score | 0.893 / 0.922 / 0.878 | arm A′ start6, arm A′ start3, arm A′ start6 |

The paired 9.2 comparison under each seed configuration (Full-D18 − arm A′, exact Full score):

| seed configuration | Full wins / ties / losses | mean difference | Wilcoxon p |
| --- | --- | --- | --- |
| default | 4 / 0 / 4 | +0.0021 | 0.641 |
| sobol512_rs7 | 5 / 0 / 3 | +0.0025 | 0.461 |
| lin7_rs13 | 5 / 0 / 3 | +0.0038 | 0.250 |

The seed-set spread of a single design is small next to the design-to-design spread: standard deviation
over configurations 0.0001 to 0.0020 on the Full score, against arm means of 0.0106 and 0.0134. The
largest spreads come from designs with few three-contact seeds (arm A′ start3: 3, 3 and 8 seeds).

### Acceptance (9.4)

* *Every ranking number uses at least 16 fresh two-finger and 12 three-contact seeds with the spread over
  three seed-search configurations:* **met** for every number in 9.2 and 9.5 that ranks designs. The 9.3
  simulations use the 8.6 archives (6 seeds) and rank nothing. The 9.2 MuJoCo leg uses 12 seeds under the
  default configuration only; it is a secondary measure and does not enter the 9.5 ranking.
* *The 9.2 ranking is stable across the three seed sets.* The design ranking is stable: Spearman
  ρ ≥ 0.878 between every pair of configurations on every measure (≥ 0.903 on the primary one; threshold 0.8), with the
  same best design on the primary measure. The sign of the mean arm difference is stable too: Full-D18
  ahead under all three. The win/loss majority is not: 4 / 4 under the default configuration, 5 / 3 under
  the other two. The pre-registered criterion requires both, so **the arm-level ranking is not stable**,
  and that part of the acceptance is **not met**. This is consistent with the arm difference in 9.2 being
  within noise. It is reported as measured, not loosened.

## 9.5 Fabrication-candidate selection

### Files changed

| File | Change |
| --- | --- |
| `dkarma/experiments/candidate_selection.py` (new) | the criterion below as code. It covers the pre-registered candidates, eligibility, the primary measure with its seed-configuration spread, robustness from the suites and the sweeps, and simulation agreement from the 9.3 and 9.2 replays (12-seed libraries take precedence per library). It writes `selection.{md,json}`. `--export` copies a selected design's specification and re-scores the copy (`--rescore`); `--preserve` copies the candidates that were not selected, with `candidate_numbers.json` |
| `dkarma/experiments/admissible_baseline.py` | `sweep`: exact Full counts at the stall, primary and continuous ceilings and at a doubled object mass, on one set of 12 fresh three-contact seeds |
| `dkarma/experiments/actuation_awareness.py` | accepts a design JSON path (`--designs <file>.design.json`) |
| `results/fabrication_candidates/README.md` | the Phase 9 outcome; `full_d18_start0/`, `armAprime_start6/` preserved as candidates that were not selected |

### Criterion as applied (fixed 2026-10-14, before the 9.2 suites and the sweeps ran)

The criterion of `docs/phase8_report.md` §8.7(b), amended for D18, made operational as follows
(`dkarma/experiments/candidate_selection.py`). Every number is on the **fixed physical task**.

*Candidates.* There are three. **Arm B** (Phase 7) is one. The second is **the arm A′ design that
scores highest on its own objective**: the published original KaRMA-T with the all-finger gate,
which is the final scoring an original-KaRMA practitioner would apply. The third is **the Full-KaRMA
D18 design with the highest primary measure** (below) among the eight Phase 9 runs. Arm A is
ineligible and is listed as the failure case. For the like-for-like 9.3 comparison, the arm A′ from
the anthropomorphic start is used: start 0, the start of arms A and B.

1. *Eligibility.* The design is admissible under D18. Its exact two-finger library score is above
   zero under all three seed configurations. Fresh three-contact seeds are found under all three
   configurations. A seed found and verified exactly (gaps, clearances, limits, frictional
   equilibrium) establishes existence. The three procedures of 8.5 were needed only to support a
   claim of non-existence. The exported design files must reproduce the suite's default-configuration
   numbers when re-scored from the exported copy.
2. *Primary measure.* The exact Full-KaRMA score, actuated three-contact cells / 8,000 / 0.1 plus
   SO(3) bins / 4,608 / 0.3, as the mean over the three seed configurations. It is reported with its
   standard deviation over configurations and over seeds. The exact two-finger library score is
   reported alongside.
3. *Robustness (requirement).* At every three-contact seed of the default configuration, the minimum
   wrench capacity is at least 1 × the object's weight in all 14 directions. In the torque/mass sweep
   at the primary ceiling (`admissible_baseline sweep`), at least one seed holds a 70 g object. The
   Full score at 70 g must retain at least 50 % of its 35 g value, which is the meaning of "degrades
   gracefully". The scores at the stall and continuous ceilings are reported.
4. *Simulation agreement (requirement).* This is applied on the candidate's three-contact library,
   because it carries the primary measure. The strict false-positive rate counts exactly valid
   primitives of holding seeds that fail pose or contact retention in MuJoCo. It must be below the
   two-finger baseline's two-finger false-positive rate under the same controller, for at least one of
   the two 9.3 controllers. 8.6 had a single controller. Requiring the condition under one realizable
   controller follows the criterion's purpose, "the exact counts are trusted for that design".
5. *Decision.* Among candidates meeting 1, 3 and 4, the higher primary measure wins. A difference
   smaller than the larger seed-configuration standard deviation of the two counts as a tie, and the
   tie is broken by the higher minimum wrench capacity. The two-finger cost is stated in two ways:
   the selected design's two-finger exact score against the best two-finger score among all scored
   designs, and against the best arm A′. If no candidate meets 1, 3 and 4, nothing is selected and
   the failing measurements are reported. The criterion is not loosened.

### Results (9.5)

Script `dkarma/experiments/candidate_selection.py` (`workspace/phase9/9.5/selection.{md,json}`); suites and
sweeps of 9.2/9.4 (`workspace/phase9/9.2/suite/`, `workspace/phase9/9.2/sweeps/`); simulations of
`workspace/phase9/9.2/sim/` (12-seed three-contact libraries, 16-seed baseline two-finger library) and
`workspace/phase9/9.3/`. Fixed physical task throughout.

*Candidates* (by the pre-registered rules): arm B; arm A′ **start6**, which has the highest published
original KaRMA-T with the all-finger gate among the eight arm A′ designs (0.0408; next start3 0.0358);
Full-D18 **start0**, which has the highest primary measure among the eight Full-D18 designs (0.0215; best under
all three seed configurations). Arm A: ineligible (17.8 mm base separation), kept as the failure case.

| candidate | 1 eligible | primary: Full exact score, mean ± sd over 3 configs | two-finger exact score | 3: weakest seed's minimum wrench capacity (default config, 12 seeds) | 3: Full score at stall / primary / continuous ceiling; at 70 g (seeds holding) | 3: 70 g retention | 4: three-contact strict FP rate, position / force | 4: baseline two-finger FP, position / force |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Full-D18 start0 | yes | **0.0215 ± 0.0017** | 0.1747 | 0.00 (10 of 12 seeds ≥ 1; 1 seed does not hold) | 0.0208 / 0.0205 / 0.0111; 0.0190 (10/12) | 0.93 | 0.78 / 0.67 | 0.19 / 0.18 |
| arm B | yes | 0.0146 ± 0.0014 | 0.2055 | 0.00 (9 of 12 ≥ 1; 2 seeds do not hold) | 0.0142 / 0.0140 / 0.0010; 0.0122 (9/12) | 0.87 | 0.63 / 0.50 | 0.19 / 0.18 |
| arm A′ start6 | yes | 0.0093 ± 0.0010 | 0.2374 | 0.16 (11 of 12 ≥ 1) | 0.0080 / 0.0080 / 0.0071; 0.0074 (11/12) | 0.93 | 0.63 / 0.54 | 0.19 / 0.18 |
| arm A (failure case) | no | 0.0000 | 0.2098 | – (no three-contact seed) | – | – | – | – |

**Nothing is selected.** Every candidate fails requirement 3. Each has at least one three-contact seed
whose wrench capacity is below the object's weight in some direction: 0.00, 0.00 and 0.16 at the
weakest seed. The median seed of each design is at 2.3 to 2.5 × weight. Reading the requirement over
seeds that hold at the primary ceiling would not change the outcome: the weakest holding seeds are at
0.06, 0.66 and 0.16. Every candidate also fails requirement 4 under both controllers. Their three-contact
false-positive rates (0.50 to 0.78) are far above the baseline's two-finger rate (0.18 to 0.19; 16 seeds, 15 and 16 holding). The
criterion's decision step is not reached and the criterion is not loosened. The 9.5 acceptance ("a
selection with the criterion's numbers") is therefore **not met**, with the measurements above.

*What the numbers say for the owner's decision* (not a selection). On the primary measure Full-D18 start0 is
the best design of Phase 9: 0.0215, against arm B's 0.0146 (+47 %) and the best arm A′'s 0.0175 (start5).
It is the best under all three seed configurations and is robust to the doubled mass (93 %). It keeps more
of its score at the continuous ceiling than arm B (0.0111 against 0.0010). Its two-finger cost against the
best arm A′ two-finger design (0.2374) is −0.063 exact (−26 %). Its three-contact exact counts transfer to
MuJoCo at a lower rate than the other candidates' (strict 0.22 / 0.33 against 0.37 to 0.50), though it
executes as many three-contact primitives per seed as arm B (5.33 / 8.08 against 6.00 / 8.08). Requirements 3 and 4 as written would reject
any three-finger design whose weakest of 12 grasps is weak, and any three-contact library, because
three-contact retention is a sub-millimetre controller and contact-model question (9.3). Whether to amend
them, for example to a quantile of the per-seed capacities or to agreement under the force controller with
a duty-cycle retention, is the owner's decision. It is not taken here.

*Specifications preserved.* Full-D18 start0 and arm A′ start6 are copied to
`results/fabrication_candidates/{full_d18_start0,armAprime_start6}/`, labelled not selected: design JSON,
URDF, robot YAML, actuator specification and the criterion's numbers. Arms A and B stay as before.

### How to reproduce (9.5)

```bash
# robustness sweeps of the candidates (CPU)
for K in armB armAprime/start6_sobol9_s0.5 full_d18/start0_nominal; do workspace/phase9/run_sweep.sh $K; done
# MuJoCo references and candidate libraries at the 9.4 seed budgets: the export pool (CPU)
workspace/phase9/chain_92_sim_cpu.sh
# the selection by the written criterion; preserve the candidates that were not selected
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.candidate_selection --out workspace/phase9/9.5 --export results/fabrication_candidates --rescore --preserve results/fabrication_candidates
```

## Phase 9 in one page: established, refuted, remaining

Fixed physical task (SPEC §2) throughout, unless *published* is stated. D18 design box: finger bases
≥ 24.1 mm apart, reach ≤ 250 mm, all-finger clearance. Tests at the end of Phase 9: `dkarma/tests`
193 passed, 6 skipped (CPU, 28 min); `karma_jax/tests` 101 passed, 1 skipped (GPU and CPU).

### Established

1. **The non-finite gradients of Phases 7–8 were not a KKT defect** (9.1). They came from
   `karma_jax.steps.exp_so3`, whose derivative is NaN at an exactly zero rotation. A failed rotation QP
   produces such a rotation, and the NaN reached the morphology gradient around the step's
   `where(success, …)` guard; 9 of 9 non-finite iterations have failed rotation QPs, 43 of 44 finite ones
   have none. The KKT system is rank-deficient only at failed QPs, whose contribution is masked. The repair
   is a bit-identical `exp_so3_safe` plus the `reduced` KKT derivative, which equals the Phase 2 derivative
   on every solved QP. Non-finite iterations fall from 107 to 0 in 20 re-runs. Runs without degenerate
   events reproduce Phase 8 bit for bit. The two two-finger runs the repair changed return better designs
   (+0.068, +0.104). Large finite gradients are legitimate (large gap-band multipliers), and the norm-100
   clipping stays.
2. **D18 can be enforced in every optimizer without changing a metric** (9.2). The original-KaRMA objective
   proposes inadmissible designs in every CMA-ES run (2 to 18 of 96). All 16 Full-KaRMA designs of 8.4
   were admissible without the constraint. Arm A (17.8 mm) loses its two-finger capability under the
   all-finger gate (0.0140 → 0.0001 physical cube fraction).
3. **A collision-valid original-KaRMA baseline exists and does what it optimizes** (9.2): its published
   KaRMA-T rises over the start on 7 of 8 starts, and it is the better two-finger hand (7 of 8 against
   Full-D18, p = 0.016).
4. **The candidate ranking is stable across seed-search configurations** (9.4): Spearman ρ 0.90 to 0.95 on
   the exact Full score, with the same best design (Full-D18 start0) under all three configurations.
5. **The 8.6 three-contact discrepancy is partly controller-induced** (9.3). The verdict by the
   pre-registered rule is *mixed*. A force-floor admittance controller raises strict three-contact success
   from 0.16 to 0.28 (start) and from 0.38 to 0.52 (arm B). Object poses follow the commands in 0.69 to 0.81
   of exactly valid primitives, and drops are rare. Under the position servo the lost contact is absent
   for the whole primitive; under force control it is intermittent, touching for about half the primitive at
   about 0.2 N.

### Refuted or not supported

- "The KKT derivative at degenerate active sets is the root of the NaNs" (Phase 8 one-page summary,
  remaining item 2): refuted, see 1. The Phase 8 hygiene had contained the symptom correctly.
- "Full KaRMA produces better simultaneous-contact manipulation than a collision-valid original-KaRMA
  optimum": **not supported** at 8 starts. The exact Full score is ahead on 5 of 8 (+26 %, CI
  [−0.0044, +0.0093], p = 0.38). Simulated three-contact execution is ahead on 6 of 8 under both
  controllers (p = 0.062 and 0.148). The direction is consistent; the evidence is not sufficient.
- "Force control makes three-contact replay as reliable as two-finger replay": not supported. Under force
  control, strict three-contact success stays below each design's two-finger success on the start, and the
  candidates' three-contact false-positive rates (0.50 to 0.67) are far above the baseline's two-finger rate
  (0.18).

### Not met, and why

- 9.4: the arm-level comparison is not stable across seed sets. The mean difference keeps its sign, but
  the win/loss majority is 4/4 under the default configuration and 5/3 under the other two. This is
  consistent with the difference being within noise.
- 9.5: **nothing is selected**. Every candidate (arm B, arm A′ start6, Full-D18 start0) fails the
  robustness requirement: its weakest of 12 three-contact seeds has a wrench capacity below 1 × weight
  (0.00 to 0.16) while the median seed is at 2.3 to 2.5 ×. Every candidate also fails the
  simulation-agreement requirement. The written criterion was not loosened.

### Fabrication recommendation, with its uncertainty

Under the criterion as written (§8.7(b) amended for D18), **no design is recommended for fabrication**.
If the owner amends requirements 3 and 4, the leading candidate on the primary measure is **Full-D18
start0** (`results/fabrication_candidates/full_d18_start0/`). For example, requirement 3 could become a
quantile of the per-seed capacities, and requirement 4 three-contact agreement under the force controller
judged against a three-contact rather than a two-finger reference. On the primary measure Full-D18 start0
scores 0.0215 ± 0.0017 exact Full score (seed-configuration sd), the best of all 18 candidate designs
under every seed configuration and +47 % over arm B (0.0146 ± 0.0014). It holds 11 of 12 seeds at the
primary ceiling, keeps 93 % of its score with a 70 g object and 54 % at the continuous ceiling (arm B 7 %),
and executes as many simulated three-contact primitives per seed as arm B. Its two-finger cost is
−0.063 exact against the best arm A′ (−26 %). The uncertainty is twofold. The advantage of Full KaRMA as
an *objective* over the collision-valid original KaRMA is not established at 8 starts. And its
three-contact simulation success rate on holding seeds (0.22 / 0.33) is below the other candidates', so
its exact three-contact counts are the least trusted in simulation. Arm A′ start6 is the
two-finger-oriented alternative: the best two-finger coverage of Phase 9, 0.2374, and the most
torque-robust three-contact grasps, but 0.0093 on the primary measure. Arm A remains the failure case.

### Remaining

1. The owner's decision on requirements 3 and 4 of the selection criterion (above), and with it the
   fabrication candidate.
2. More starts per arm (16 or more) if the Full-KaRMA-versus-original-KaRMA claim is to be settled before
   hardware. At the measured effect size and spread, 8 starts cannot separate it.
3. A compliant-fingertip contact model in the simulation scene (SPEC §12 IV fabricates compliant
   fingertips), to test whether the residual three-contact chatter is a property of rigid capsules.
4. SPEC §14's bullet "the exact-vs-simulation error term of Phase 8.6 is controller-dependent until 9.3"
   can now cite 9.3: partly controller-dependent, with the three-contact residual in contact maintenance.
   The SPEC text is left to the owner.
