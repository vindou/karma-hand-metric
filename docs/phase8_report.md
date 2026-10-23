# Phase 8 report: calibration and validation (2026-10-11)

Phase 8 of `SPEC.md` §11.1. No new metric formulation; every item calibrates or validates the
modules delivered in Phases 0-7. Reported numbers come from the exact evaluators (CLAUDE.md
rule 2); smooth values are labelled as such. Raw outputs are under `workspace/phase8/`; every
number below names its script and JSON. Tests: `JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python
-m pytest dkarma/tests -q` (results per item).

## 8.1 Recalibration of the smooth force credit

### Files changed

| File | Change |
| --- | --- |
| `dkarma/experiments/force_calibration.py` (new) | held-out states, exact labels, batched smooth energies, AUC and credit statistics per candidate, coverage ratio at the continuous ceiling per candidate (`--coverage-only`, `--mappings`) |
| `dkarma/solvers/contact_forces.py` | `solve_soft`: damped **Newton** (full Hessian of the convex cost) with Marquardt-scaled damping in the rescaled variables `y = [f, m / gamma]`; 50 iterations; `ForceWeights.grad_mode = "envelope"` (gradient at the converged minimizer) or `"unrolled"`; `eps_hinge` default 0.02 → **0.005** |
| `dkarma/solvers/rollout.py` | `EnergyWeights.force_knee / force_steep / force_knee_eps` (the credit mapping, defaults **7.0e-7 / 3.0 / 0.125**); `force_penalty`, `seed_force_energy` |
| `dkarma/metrics/diff_karma.py`, `dkarma/metrics/multicontact.py` | with actuation on, the **seed state** is gated by its own force energy (`SeedEval.seed_force_E`, `MultiSeedEval.seed_force_E`); the exact Full path checks the seed state's actuated equilibrium before crediting its cell and bin |
| `dkarma/experiments/actuation_awareness.py` | `pert_design()` factored out; the exact force-feasible count includes the seed states (`n_seed_force_feasible`) |
| `dkarma/tests/test_actuation.py` | `test_force_energy_gradient_wrt_morphology` rewritten for the converged regime (three load scales, envelope gradient vs finite differences at 1e-3, convergence 50 vs 100 iterations); `test_force_credit_mapping_knee` (new) |

### Procedure

Held-out states (`workspace/phase8/8.1/final/calibration.json`, `--hinges 0.02 0.005 --iters 50`):
the seed states and every exactly valid primitive-end state of Experiment II's two-finger
libraries (baseline and pert, 6 seeds each, the Phase 7 settings; 64 states) and the seed and
primitive-end states of the 3x4 anthropomorphic start's three-contact library (29 states).
Each state is labelled by `contact_forces.solve_exact` under the Experiment II settings
(ceilings 0.215 / 0.108 / 0.043 / 0.02 / 0.01 N·m at 35 g; 70 g and 140 g at 0.108 N·m) and
around its own margins: the critical load scale, torque ceiling and friction coefficient are
bisected with the exact solver (14 steps), then sampled at 0.5, 0.7, 0.85, 0.95, 1.05, 1.15,
1.3, 1.6 and 2.5 times the critical value. 3,153 labelled instances (1,557 feasible); median
critical values: load 2.3× the sphere's weight, ceiling 0.064 N·m (two-finger) / 0.067 N·m
(three-contact), friction 0.17 / 0.28.

For each hinge scale the smooth energy `E_F*` is solved for every instance in one batched GPU
call. AUC is the rank statistic of `-E_F*` against the label (ties count half). Because AUC is
invariant to the weight and to any strictly monotone credit mapping, the weight and the mapping
are judged by where the credit falls: accuracy at a credit threshold of 0.5, the share of
exactly infeasible states keeping credit ≥ 0.5, the share of feasible states dropping below
0.5, and the 5th / 95th percentiles of the credit per label.

### What the calibration found before any tuning

1. **The Phase 4 force solver did not converge at infeasible states.** Levenberg-Marquardt is a
   small-residual method; at an infeasible state the residual at the optimum is large and
   Gauss-Newton converged only linearly, and the `lam * I` damping was blind to the torsion
   columns (scale `1/gamma ≈ 555` times the force columns). 200 iterations left three-contact
   feasible states at `E ~ 1e-3` (scratch check, `scaled_i200` vs `current_i200`). The
   derivative the Phase 4 test validated was the unrolled derivative of the unconverged
   25-iteration map, which at a 6× load has the **opposite sign** to the converged energy's
   gradient (finite differences: −0.091 at 25 iterations vs +0.028 converged, coordinate 31
   of the baseline). The fix is numerical, not a formulation change: damped Newton with the
   full Hessian of the convex cost, Marquardt scaling, variables `m / gamma`; the energy at 50
   and 100 iterations agrees to 1e-6 relative and the envelope gradient matches finite
   differences to ≤ 1e-6 relative at feasible, marginal and 6× loads (test rewritten).
2. **The energy ranks states almost perfectly; the mapping was the problem.** Exactly feasible
   states have `E_F* ≤ 4.5e-7` (hinge 0.005, 99th percentile 9e-8); infeasible states start at
   `~1e-6` and reach `1e-3` for a 10 % violation. With `exp(-w E_F*)` and `w = 1` to `50`, a
   10 % violation keeps 95 to 100 % of its credit.
3. **The seed state was never force-gated.** The smooth coverage counts the initial state of all
   ten commands at full credit, which with the 6 mm kernel on the 10 mm grid occupies ≈ 13
   cell-equivalents: 27 % (baseline) and 42 % (pert) of the kinematic `K_T~`, the exact floor
   every mapping hit at the continuous ceiling. SPEC §5 requires the hold at every state, so
   the seed state is now gated like every step (smooth and exact paths).

### AUC of the smooth credit vs the exact label

Corrected labelling (`workspace/phase8/8.1/final/`; the first run, `workspace/phase8/8.1/calibration.json`,
had the torque and friction margins bisected in the wrong direction and gives 0.9987 / 0.9998):

| hinge `eps_hinge` | AUC pooled | two-finger | three-contact | by family (two-finger / three-contact) | Youden threshold `E_F*` (J) |
| --- | --- | --- | --- | --- | --- |
| 0.02 (Phase 4) | 0.9843 | 0.9864 | 0.9864 | Exp II torque 1.00 / 1.00, mass 0.999 / 0.999, load margin 0.995 / 0.984, torque margin 0.998 / 0.995, friction margin 0.987 / 0.994 | 5.6e-5 (0.84) |
| **0.005** | **0.9988** | 0.9998 | 0.9974 | torque 1.00 / 1.00, mass 1.00 / 0.994, load 1.00 / 0.996, torque margin 1.00 / 0.999, friction 0.999 / 0.995 | **6.6e-7 (0.98)** |

Both hinge scales meet AUC ≥ 0.95; 0.005 is adopted (the tighter hinge lowers the energy of
feasible states at active constraints by two orders of magnitude without hurting convergence).

### Credit mappings (hinge 0.005, 3,153 instances)

| mapping | accuracy at credit 0.5 | infeasible with credit ≥ 0.5 | feasible with credit < 0.5 | infeasible credit p95 | feasible credit p5 |
| --- | --- | --- | --- | --- | --- |
| linear `w = 1` (Phase 4) | 0.509 | 97.1 % | 0.0 % | 1.000 | 1.000 |
| linear `w = 20` | 0.618 | 75.4 % | 0.0 % | 1.000 | 1.000 |
| linear `w = 50` | 0.667 | 65.9 % | 0.0 % | 1.000 | 1.000 |
| hard knee at 6.6e-7, slope 1 per knee-unit | 0.984 | 2.3 % | 1.0 % | 0.109 | 0.999 |
| **hard knee, slope 3** | **0.990** | **0.8 %** | **1.3 %** | **0.001** | **0.996** |
| hard knee, slope 10 | 0.990 | 0.4 % | 1.6 % | 0.000 | 0.988 |
| two-term (`w = 1` + knee, slope 3) | 0.990 | 0.8 % | 1.3 % | 0.001 | 0.996 |

The step penalty adopted is `force * E_F* + force_steep * hinge((E_F* - knee) / knee)` with
`hinge(x) = 0.125 softplus(x / 0.125)`, `force = 1`, `knee = 7.0e-7` (the Youden threshold of
the first labelling; the corrected one gives 6.6e-7, inside the knee's width), `force_steep = 3`:
the two-term form (soft inside the margin, steep outside). Its gradient is zero below the knee
and `3 / knee` per unit of energy above it; the knee test checks both.

### Actuation-aware coverage at the continuous ceiling (`workspace/phase8/8.1/coverage_seedgate.json`)

Smooth `K_T~` of Act-KaRMA as a fraction of the kinematic `K_T~` on the Experiment II designs
(6 seeds), with the seed gate; the exact evaluator rejects every state at 0.043 and 0.02 N·m
(Experiment II table below):

| mapping (hinge) | baseline 0.108 | baseline 0.043 | baseline 0.02 | pert 0.108 | pert 0.043 | pert 0.02 |
| --- | --- | --- | --- | --- | --- | --- |
| linear `w = 1` (0.02) | 1.00 | 0.90 | 0.43 | 1.00 | 0.97 | 0.77 |
| linear `w = 20` (0.02) | 1.00 | 0.43 | 0.01 | 1.00 | 0.68 | 0.14 |
| linear `w = 50` (0.005) | 1.00 | 0.29 | 0.00 | 1.00 | 0.54 | 0.02 |
| knee slope 1 (0.005) | 0.99 | 0.02 | 0.00 | 1.00 | 0.00 | 0.00 |
| **knee slope 3 (0.005)** | **0.99** | **0.00** | **0.00** | **1.00** | **0.00** | **0.00** |

Without the seed gate (`workspace/phase8/8.1/calibration.log`, coverage lines) every mapping
floors at 0.27 (baseline) and 0.42 (pert): the seed-state artefact of finding 3.

*Acceptance:* AUC 0.9988 ≥ 0.95; at the continuous ceiling the actuation-aware coverage is
0.0 % of the kinematic value on both designs (≤ 10 %); the Phase 4 gradient test passes in its
stricter form (`dkarma/tests/test_actuation.py`: 10 passed). Both criteria met.

### Experiment II re-run (replaces the Phase 7 table)

`dkarma/experiments/actuation_awareness.py` → `workspace/phase8/exp2/exp2.json` (same designs, 6
seeds, settings and script as Phase 7; the force credit of this section; "exact force-feasible"
now counts the seed states too, so the state counts are 36 and 23 instead of 30 and 17). Smooth
values are labelled smooth; the exact column is the independent SLSQP allocation.

| design | setting | kinematic K_T~ (smooth) | Act K_T~ (smooth) | kinematic K_R~ | Act K_R~ | smooth max torque usage | exact force-feasible states | seed states |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | τ̄ 0.215 N·m (stall) | 0.00603 | 0.00603 | 0.1095 | 0.1095 | 1.01 | 36 / 36 | 6 / 6 |
| baseline | τ̄ 0.108 (primary) | 0.00603 | 0.00599 | 0.1095 | 0.1090 | 1.02 | 35 / 36 | 6 / 6 |
| baseline | τ̄ 0.043 (continuous) | 0.00603 | **0.00000** | 0.1095 | 0.0000 | 1.08 | 0 / 36 | 0 / 6 |
| baseline | τ̄ 0.020 | 0.00603 | 0.00000 | 0.1095 | 0.0000 | 1.29 | 0 / 36 | 0 / 6 |
| baseline | τ̄ 0.010 | 0.00603 | 0.00000 | 0.1095 | 0.0000 | 1.51 | 0 / 36 | 0 / 6 |
| baseline | mass 70 g | 0.00603 | 0.00396 | 0.1095 | 0.0821 | 1.07 | 23 / 36 | 4 / 6 |
| baseline | mass 140 g | 0.00603 | 0.00001 | 0.1095 | 0.0001 | 1.17 | 1 / 36 | 0 / 6 |
| pert | τ̄ 0.215 | 0.00390 | 0.00390 | 0.0728 | 0.0728 | 1.01 | 23 / 23 | 6 / 6 |
| pert | τ̄ 0.108 | 0.00390 | 0.00390 | 0.0728 | 0.0728 | 1.02 | 23 / 23 | 6 / 6 |
| pert | τ̄ 0.043 | 0.00390 | **0.00000** | 0.0728 | 0.0000 | 1.05 | 0 / 23 | 0 / 6 |
| pert | τ̄ 0.020 | 0.00390 | 0.00000 | 0.0728 | 0.0000 | 1.11 | 0 / 23 | 0 / 6 |
| pert | τ̄ 0.010 | 0.00390 | 0.00000 | 0.0728 | 0.0000 | 1.15 | 0 / 23 | 0 / 6 |
| pert | mass 70 g | 0.00390 | 0.00309 | 0.0728 | 0.0578 | 1.07 | 19 / 23 | 5 / 6 |
| pert | mass 140 g | 0.00390 | 0.00000 | 0.0728 | 0.0000 | 1.14 | 1 / 23 | 0 / 6 |

Against Phase 7: at the continuous ceiling the smooth coverage kept 75 % and 90 % of the
kinematic value; it now keeps 0 % where the exact solver rejects every state, and 66 % / 79 %
at 70 g where the exact solver accepts 64 % / 83 % of the states. The smooth "max torque usage"
is now reported at the converged minimizer (1.02 at the primary ceiling against 1.10 in Phase 7,
where the unconverged solver overstated it); it is still the smooth allocation's number, not a
margin. Findings 1, 2 and 4 of the Phase 7 section stand; finding 3 (the calibration gap) is
closed by this item.

### Limitations

- The held-out set is two designs' two-finger libraries plus one three-contact design (93
  states, 3,153 labelled instances); the margin families are generated from those states.
- The torque-margin family of the first labelling was mis-centred (bisection direction); the
  corrected run reports lower AUC for hinge 0.02 (0.984) and the same conclusion.
- The knee is a calibrated constant of the smooth energy at hinge 0.005; changing
  `eps_hinge` requires re-deriving it (the script prints the Youden threshold).
- The damped-Newton solve costs more per step than the Phase 4 LM (Hessian of a 16-variable
  cost, 50 iterations); the Full objective's iteration time is reported in 8.2.
- The envelope gradient is exact at a minimizer; states where 50 iterations do not converge
  (none observed on the held-out set) would carry an error `∇_x E · dx*/dphi`.

### How to reproduce

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.force_calibration --out workspace/phase8/8.1/final --hinges 0.02 0.005 --iters 50 --skip-coverage
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.force_calibration --out workspace/phase8/8.1 --coverage-only workspace/phase8/8.1/calibration.json \
    --coverage-out coverage_seedgate.json --mappings linear_w1 linear_w20 linear_w50 knee_s1 knee_s3 knee_s10 twoterm_w1_s3
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.actuation_awareness --out workspace/phase8/exp2/exp2.json
```

## 8.2 Repair of the continuation schedule

### Files changed

| File | Change |
| --- | --- |
| `dkarma/optimization/optimizers.py` | `adam`: per-stage instrumentation (exact validation and smooth diagnostics at every stage boundary, `meta["stages_log"]`), gradient hygiene (`grad_clip`, `grad_spike`, `skip_nonfinite`, `reset_moments`), `stage_select` (warm start from the best validated iterate of the previous stage), `seed_policy` / `reseed_every`, `warmup`; `cmaes` / `hybrid`: `time_budget_s`, `warmup`, optimizer time separated from validation time; `ADAM_DEFAULTS` (the adopted schedule) |
| `dkarma/optimization/objectives.py` | seeds are a traced argument of the jitted objective (no recompile on reseeding); `Objective.diagnostics` (gate-failure shares, end credits, reprojection failures); seed policies `regenerate_seeds`, `reproject_seeds`, `union_seeds`, `exact_seed_scores`, `seeds_for_policy`; `make_objective(seed_policy=..., reseed_every=...)` |
| `dkarma/metrics/continuation.py` | `schedule(sigma_energy_end, sigma_p_end_m, sigma_r_end_rad, skip_strict)`, `describe` |
| `dkarma/experiments/continuation_study.py` (new) | the study driver: starts, variants, resumable runs, `study.md` |
| `dkarma/experiments/validation_noise.py` (new) | the exact validation's sensitivity to one Adam step |
| `dkarma/tests/test_optimization.py` | stage-boundary validations and instrumentation; hygiene, stage selection and the seed policies (5 tests) |

### Procedure

`workspace/phase8/8.2/diff2/` (`log.txt`, `study.md`, `study.json`, one `run.json` and the §13
artifact layout per run): the two-finger Diff-KaRMA objective of Experiment I (Phase 0 baseline,
8 seeds, 5 cells, 12 rotation steps, 60 Adam iterations, 3 stages, learning rate 0.02) from five
starts: the baseline and four Sobol perturbations of its unconstrained coordinates (scale 0.25,
`starts.json`) that have seeds. Twelve variants, every run instrumented: exact-library
validation with fresh seeds at iteration 0, every 10 iterations and at every stage boundary
(the best of these is "best intermediate"; the start's validation is shared across variants),
the smooth-vs-exact gap and the share of smooth steps failing each exact gate at every stage
boundary, the gradient norm, the step norm and the non-finite / clipped / spike flags at every
iteration. Wall clock per run is 10 to 15 min on the RTX 4090, of which 7 to 9 min are
validations.

| variant | gradient hygiene | stage selection | seeds | schedule |
| --- | --- | --- | --- | --- |
| `phase7` | none (non-finite entries zeroed, Adam steps on) | no | reproject | 3 stages, soft → hard, kernels 15 → 6 mm, 0.25 → 0.09 rad, energy sigma 8 → 1 |
| `hygiene` | clip the gradient 2-norm at 100; no update on a non-finite gradient; Adam moments reset at stage boundaries | no | reproject | same |
| `hygiene2` | as hygiene but no clipping: an iteration whose gradient norm exceeds 10× the median of the last ten accepted norms makes no update | no | reproject | same |
| `*_select` | | yes: each stage warm-starts from the best validated iterate (periodic or end) of the previous stage | | |
| `*_reseed` | | | fresh seed search at every stage boundary (8.3 policy b) | |
| `hygiene_gentle` | hygiene | no | reproject | final energy sigma 2 instead of 1 |
| `*_skipstrict` | | | | the geometric path laid over 4 points with the strict end point dropped |

### What the instrumentation found

1. **Phase 7's drops are gradient events, not the schedule's geometry.** In the hard stages the
   KKT re-solve of the rolling QP (`rolling_qp.kkt_resolve`, regularization 1e-10) returns
   non-finite or exploding derivatives at degenerate active sets: 26 non-finite iterations on
   one start, norms of 1e4 to 4e4 against a typical 1 to 10 on the others. With Phase 7's
   handling (non-finite entries zeroed, momentum kept) Adam steps along stale directions for the
   rest of the stage; on `start1` the exact score fell from 0.249 to 0.172 with no intermediate
   gain. Finite spikes contaminate Adam's second moment for ~20 iterations (steps shrink to the
   spike's direction), and when they are clipped instead (`hygiene`) the full-size step is taken
   along the spike's direction: on `start2` the smooth value fell within stage 1 from 1.41 to 0.75
   under ascent, 10 of 20 iterations clipped.
2. **The strict stage is not the culprit.** With hygiene, the exact score of the stage-2 end
   exceeded its start on 3 of 5 starts, and the gentle and skip-strict variants do not separate
   from the full schedule (paired differences of the returned score within ±0.013).
3. **Reseeding per stage reintroduces the failure.** A fresh seed set exposes new degenerate
   active sets: 34 to 54 non-finite iterations per five runs for the `*_reseed` variants against
   0 for every reprojecting variant.
4. **The exact validation is noisier than the acceptance.** `validation_noise.py` on the
   `phase7/start0` returned design: eight random perturbations of one median Adam step
   (0.039 in the unconstrained coordinates) move the exact score from 0.312 to 0.224..0.317
   (sd 0.032, 10 %); at 3 steps sd 0.021. With 16 instead of 8 validation seeds the sd is 0.013
   (4.3 %) and the base reads 0.302 (the 8-seed 0.312 is the best-of-seven selection bias).
   `workspace/phase8/8.2/validation_noise.json`, `validation_noise_b16.json`. A run has six
   intermediate validations; their maximum sits several per cent above the typical reading, so
   "final ≥ best intermediate − 2 %" fails for a stationary optimizer most of the time at this
   precision.

### Results (5 starts each; exact-library score with fresh seeds, 8 seeds)

| variant | raw final ≥ best − 2 % | selected final ≥ best − 2 % | returned > start | mean exact: start → returned → raw final → selected final | best returned | non-finite / clipped / spikes (5 runs) |
| --- | --- | --- | --- | --- | --- | --- |
| phase7 | 2/5 | 2/5 | 4/5 | 0.210 → 0.302 → 0.272 → 0.272 | 0.371 | 26 / 0 / 0 |
| hygiene | 3/5 | 3/5 | 5/5 | 0.210 → 0.287 → 0.261 → 0.261 | 0.323 | 0 / 64 / 0 |
| hygiene2 (spike skip) | 2/5 | 2/5 | 4/5 | 0.210 → 0.278 → 0.250 → 0.250 | 0.326 | 0 / 0 / 187 |
| **hygiene + select** | 3/5 | **5/5** | 5/5 | 0.210 → 0.280 → 0.267 → 0.280 | 0.310 | 0 / 63 / 0 |
| hygiene2 + select | 2/5 | 3/5 | 4/5 | 0.210 → 0.278 → 0.250 → 0.258 | 0.326 | 0 / 0 / 187 |
| hygiene + reseed | 1/5 | 1/5 | 5/5 | 0.210 → 0.273 → 0.243 → 0.243 | 0.337 | 49 / 67 / 0 |
| hygiene2 + select + reseed | 3/5 | 3/5 | 4/5 | 0.210 → 0.265 → 0.252 → 0.252 | 0.332 | 34 / 0 / 131 |
| hygiene + select + reseed | 1/5 | 2/5 | 5/5 | 0.210 → 0.273 → 0.229 → 0.239 | 0.337 | 54 / 69 / 0 |
| hygiene + gentle final sigma | 2/5 | 2/5 | 5/5 | 0.210 → 0.300 → 0.272 → 0.272 | 0.348 | 0 / 52 / 0 |
| hygiene + skip strict | 2/5 | 2/5 | 5/5 | 0.210 → 0.292 → 0.266 → 0.266 | 0.361 | 0 / 64 / 0 |
| hygiene + select + skip strict | 2/5 | 4/5 | 5/5 | 0.210 → 0.292 → 0.257 → 0.278 | 0.361 | 0 / 78 / 0 |
| hygiene2 + select + skip strict | 2/5 | 3/5 | 5/5 | 0.210 → 0.262 → 0.234 → 0.241 | 0.319 | 0 / 0 / 177 |

"Raw final" is the last Adam iterate; "selected final" is the design the schedule hands over
when stage selection is on (the best validated iterate of the last stage), equal to the raw
final otherwise. "Returned" is the best validated design of the run (SPEC §8.5). Paired
differences of the returned score against `phase7` on the same five starts are −0.003 to
−0.040 (1 to 2 wins of 5) for every variant: at a one-step validation sd of 0.03 none of them is
distinguishable from zero, in either direction. Per-start values are in `study.json`.

### Acceptance

The criterion (final iterate ≥ best intermediate − 2 % in ≥ 4 of 5 starts) is **met by
`hygiene + select` (5/5) and `hygiene + select + skip strict` (4/5) on the stage-selected
final iterate, and by no variant on the raw last iterate** (best 3/5). The raw reading is the
literal one; it is not reached, and finding 4 says why: the validation moves by 10 % per Adam
step at 8 seeds, so a 2 % criterion against the maximum of six readings is below the
measurement's resolution. The item is reported as measured and the criterion is not loosened.
The 3×4 start is covered by the 8.3 runs (`workspace/phase8/8.2/full3/`, next section).

### Adopted default (`ADAM_DEFAULTS`)

`hygiene + select`: gradient 2-norm clipped at 100, no update on a non-finite gradient, Adam
moments reset at stage boundaries, each stage warm-started from the best validated iterate of
its predecessor; seeds re-projected (8.3); the Phase 7 three-stage geometric schedule
unchanged (gentle sigma and skip-strict bring nothing the data can see). Reasons: it is the
only variant with no non-finite gradient and 5/5 improvement of the start together with 5/5
preservation of the final design, and its returned scores are within noise of every other
variant's. The gradient's conditionality on the start's seed set is a property of this default
(8.3).

### Limitations

- Five starts per variant on one objective; the per-variant pass counts differ by one or two
  starts, which the validation noise alone produces. The adopted default is chosen for its
  stability properties, not for a demonstrated score gain.
- The spike events are symptoms of `kkt_resolve` at degenerate active sets; the optimizer-level
  remedies treat the symptom. A regularized or least-squares KKT derivative would treat the
  cause and is a Phase 9 numerics item (it changes a validated derivative).
- Validation at 8 seeds is the Phase 7 setting; 16 seeds halve the noise at twice the cost
  (8.4 uses fresh seeds and the physical original KaRMA for the returned designs only).
- The first study process was OOM-killed (24 GB RSS after 20 runs: jitted programs accumulate
  per run); the matrix was finished as one process per variant (`workspace/phase8/chain_82c.sh`).

### How to reproduce

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.continuation_study --objective diff2 --starts 5 --variants phase7 --out workspace/phase8/8.2/diff2   # one variant per process
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.continuation_study --summary workspace/phase8/8.2/diff2
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.validation_noise --run workspace/phase8/8.2/diff2/phase7/start0_nominal --n 8 [--budget 16]
```

## 8.3 Seed-policy study

### Files changed

`dkarma/optimization/objectives.py` (the four policies, `Objective.seed_policy` / `reseed_every`,
`make_objective(seed_policy=..., reseed_every=...)`, `adam(seed_policy=..., reseed_every=...)`,
listed under 8.2) and `dkarma/experiments/continuation_study.py` (`policy_*` variants,
`--base-variant`). The driver, starts and instrumentation are those of 8.2.

### Procedure

Same starts and budgets as 8.2, on the adopted schedule (`hygiene_select`, 8.2): the two-finger
Diff-KaRMA objective (5 starts, 60 iterations, 3 stages, 8 seeds) and the 3×4 Full-KaRMA
objective from the anthropomorphic start (5 starts: the start plus four Sobol perturbations
with ≥ 3 three-contact seeds; 40 iterations, 2 stages, 6 seeds, Experiment III's setting;
`workspace/phase8/8.2/full3/`). Policies: (a) re-project the start's seeds onto every iterate
(Phase 7); (b) a fresh seed search on the current design at every stage boundary; (c) a fresh
search every 10 iterations; (d) at every stage boundary the union of the re-projected current
seeds and a fresh search, deduplicated (0.1 rad), ranked by each seed's exact library score and
cut to the budget. The Phase 7 schedule is included on the Full objective as the reference. Every
run reports the exact score of the start, of every validated iterate, of the raw and selected
final iterate, the number of feasible seeds at every iteration, the seed events (for (d): how
many kept seeds came from the re-projected set and how many from the fresh search), the
non-finite gradient count and the wall clock.

### Results

Two-finger Diff-KaRMA (`workspace/phase8/8.2/diff2/{hygiene_select,hygiene_select_reseed,policy_regenerate_every_k,policy_union_stage}`):

| policy | raw final ≥ best − 2 % | selected | returned > start | mean exact: start → returned → raw final | best returned | feasible seeds along the run | non-finite gradients (5 runs) | wall s | paired vs (a), returned score: mean, wins/ties/losses |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (a) re-project | 3/5 | 5/5 | 5/5 | 0.210 → 0.280 → 0.267 | 0.310 | 8 / 8 | 0 | 607 | – |
| (b) regenerate per stage | 1/5 | 2/5 | 5/5 | 0.210 → 0.273 → 0.229 | 0.337 | 8 / 8 | 54 | 608 | −0.008, 2/0/3 |
| (c) regenerate every 10 | 2/5 | 3/5 | 4/5 | 0.210 → 0.261 → 0.230 | 0.291 | 8 / 8 | 0 | 638 | −0.019, 1/0/4 |
| **(d) union, exact-ranked** | **4/5** | 5/5 | 5/5 | 0.210 → **0.293** → 0.286 | **0.357** | 8 / 8 | 0 | 928 | **+0.012, 4/0/1** |

3×4 Full KaRMA (`workspace/phase8/8.2/full3/`):

| policy | raw final ≥ best − 2 % | selected | returned > start | mean exact: start → returned → raw final | best returned | non-finite gradients (5 runs) | wall s | paired vs (a) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Phase 7 schedule, re-project | 1/5 | 1/5 | 4/5 | 0.0141 → 0.0178 → 0.0146 | 0.0228 | 0 | 661 | −0.0006, 2/1/2 |
| (a) re-project (adopted schedule) | 1/5 | 2/5 | 5/5 | 0.0141 → 0.0184 → 0.0169 | 0.0220 | 3 | 760 | – |
| (b) regenerate per stage | 1/5 | 2/5 | 4/5 | 0.0141 → 0.0172 → 0.0138 | 0.0220 | 50 | 531 | −0.0012, 1/2/2 |
| (c) regenerate every 10 | 4/5 | 4/5 | 5/5 | 0.0141 → 0.0182 → 0.0179 | 0.0220 | 29 | 565 | −0.0003, 2/1/2 |
| (d) union, exact-ranked | 2/5 | 2/5 | 4/5 | 0.0141 → 0.0188 → 0.0166 | **0.0314** | 28 | 703 | +0.0004, 1/2/2 |

The union keeps 2 to 7 re-projected seeds and adds 1 to 6 fresh ones per boundary on both
objectives; the budget is always filled (8 or 6 feasible seeds at every iteration under every
policy, no run lost seeds).

### What the data say

1. On the two-finger objective the exact-ranked union is the best policy by every column:
   it preserves the final iterate on 4 of 5 starts (raw) and 5 of 5 (selected), improves every
   start, returns the best design of the study (0.357) and beats re-projection on 4 of 5 starts
   (mean +0.012, about one third of the one-step validation sd of 8.2), at 1.5× the wall clock
   (one fresh search and one exact ranking per stage boundary).
2. On the Full objective the three reseeding policies all expose the hard-QP derivative
   failure of 8.2 finding 1: 28 to 50 non-finite gradient iterations per five runs (skipped by
   the adopted hygiene, so 14 to 25 % of the iterations make no update) against 3 for
   re-projection; the returned scores of all policies are tied within noise (paired means
   ±0.001, 1 to 2 wins of 5). Policy (c) preserves the final iterate on 4 of 5 starts; its
   non-finite rate says this is not a stability property of the policy but of which
   iterations happened to be skipped. The union found the single best three-contact design of
   the study (0.0314, start 3) and no gain elsewhere.
3. The 8.2 acceptance on the 3×4 start (final ≥ best intermediate − 2 % on ≥ 4 of 5 starts)
   is met on the Full objective only by policy (c) (4/5 raw and selected); the adopted schedule
   with re-projection reaches 1/5 raw and 2/5 selected. The Full objective's exact validation
   (6 three-contact seeds, cells and 4608-bin counts of a few units) is the discrete count
   problem of 8.2 finding 4 in a harsher form: its scores are 0.010 to 0.031 and move by one
   cell or bin between consecutive validations. As in 8.2 the counts are reported as measured.

### Adopted default

`seed_policy = "reproject"` (unchanged) is the default of `make_objective` and `adam`: it is the
only policy without the non-finite-gradient failure on both objectives and its returned scores
are within noise of every other policy's on the Full objective. `seed_policy = "union_stage"` is
the recommended setting for two-finger objectives, where its gain is consistent across starts
(4 of 5) and costs 50 % more wall clock; it is used as such in the 8.4 two-finger arm's note but
not in its runs, so that 8.4 compares the default pipeline. Every report that uses policy (a)
states that the gradient is conditional on the start design's seed set, re-projected: the
Experiment I and III results of Phase 7, the 8.2 study and the 8.4 comparison all are.

### Limitations

- Five starts per policy and objective; differences between policies on the Full objective are
  below the measurement's resolution.
- The non-finite gradients that fresh seeds expose are the same KKT-derivative defect as in
  8.2; the policies are compared under the adopted hygiene (skipped iterations), which penalizes
  reseeding by lost iterations rather than by corrupted steps. A regularized KKT derivative
  would change this comparison and is left for Phase 9.
- The union's exact ranking uses the objective's own exact library on the current design (one
  evaluation per candidate seed); its cost scales with the number of distinct seeds found.

### How to reproduce

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.continuation_study --objective diff2 --starts 5 --variants policy_union_stage --base-variant hygiene_select --out workspace/phase8/8.2/diff2
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.continuation_study --objective full3 --starts 5 --variants hygiene_select phase7 policy_regenerate_stage policy_regenerate_every_k policy_union_stage --base-variant hygiene_select --out workspace/phase8/8.2/full3
```

## 8.4 Optimizer comparison with statistics

### Files changed

| File | Change |
| --- | --- |
| `dkarma/experiments/optimizer_comparison.py` (new) | ≥ 8 starts, the three optimizers at matched optimizer time, final evaluation of every returned design (fresh-seed exact library and the original KaRMA on the physical task), `summary.md` / `summary.json` with mean, median, best, IQR, paired differences, sign counts and Wilcoxon signed-rank; one process per start (`--only`) |
| `dkarma/optimization/optimizers.py` | `adam(warmup=...)` and `cmaes(warmup=..., time_budget_s=...)` compile before the clock starts; `hybrid(time_budget_s=...)`; `cmaes` validates an unchanged incumbent once; optimizer time and validation time reported separately |

### Procedure

`workspace/phase8/8.4/{diff2,full3}/` (`log.txt`, `summary.md`, `summary.json`, per start `run.json` and
the §13 artifacts of each method). Objectives as in 8.2/8.3: the two-finger Diff-KaRMA objective
from the Phase 0 baseline (8 seeds, 5 cells, 12 rotation steps) and the 3×4 Full-KaRMA objective
from the anthropomorphic start (6 three-contact seeds, 4 cells, 10 rotation steps). Eight starts
per objective (`starts.json`, Sobol seed 321): the nominal start, four Sobol perturbations of its
unconstrained coordinates at scale 0.25 and three at scale 0.5, each required to have seeds
(≥ 4 two-finger, ≥ 3 three-contact). Optimizers, in this order per start:

1. **Adam** with the 8.2 default schedule and the 8.3 default seed policy (re-projection; the
   gradient is conditional on the start's seed set): 60 iterations / 3 stages (two-finger) or
   40 / 2 (Full), learning rate 0.02, validation every 10 iterations and at stage ends. Its
   optimizer time per start, measured on the RTX 4090 after a warm-up compile and with every
   validation excluded, is the budget `B` of that start: **65 to 68 s** on every start of both
   objectives.
2. **CMA-ES** (`cma` 4.0, population 8, σ₀ 0.3) on the same unconstrained coordinates with the
   smooth objective batched on the GPU, run until its optimizer time reaches `B`: 118 to 120
   generations (944 to 960 evaluations) on the two-finger objective, 50 generations (400
   evaluations) on the Full objective. The incumbent is validated every 16 generations (every
   2 on `diff2/start0`, the run before the cadence was set) and at the end.
3. **Hybrid**: Sobol-perturbed starts around the start design (scale 0.25), each refined by
   Adam for 15 (two-finger) or 10 (Full) iterations over 2 or 1 stages and validated at the end,
   no new refinement once the optimizer time reaches `B`.

The design each optimizer returns is its best validated design (SPEC §8.5); every returned
design and every start is then evaluated with fresh seeds by the exact library (the validation
criterion, `exact_score = K_T/0.1 + K_R/0.3`) and by the original KaRMA BFS on the physical task
(`official_karma_physical`: top-3 mean voxel count over the 8,000 cells of the task cube and
KaRMA-R). Statistics per method: mean, median, best, interquartile range, the fraction of starts
whose returned design beats the start; per pair: the per-start differences, sign counts and the
Wilcoxon signed-rank p-value (8 pairs).

**Budget caveat for the hybrid.** Its optimizer time includes the re-tracing of the jitted
objective for every refinement after the first (the warm-up covers only the first), so it
overran the budget (102 s mean on `diff2`, 107 s on `full3`, against `B` ≈ 66 s) while
performing fewer gradient evaluations than Adam (2 to 3 refinements of 15 iterations, 28 to 45,
against 60). The hybrid rows are therefore not a matched comparison in either direction; they
are reported, not ranked.

### Results: two-finger Diff-KaRMA (8 starts; `workspace/phase8/8.4/diff2/summary.md`)

Exact-library score of the returned design (fresh seeds):

| method | mean | median | best | IQR | beats start | mean gain vs start |
| --- | --- | --- | --- | --- | --- | --- |
| start | 0.156 | 0.145 | 0.238 | | | |
| Adam | 0.291 | 0.306 | 0.387 | [0.272, 0.329] | 7/8 | +0.135 |
| **CMA-ES** | **0.374** | **0.376** | **0.488** | [0.334, 0.385] | **8/8** | **+0.218** |
| hybrid (over budget, fewer gradient evaluations) | 0.239 | 0.258 | 0.294 | [0.203, 0.276] | 6/8 | +0.084 |

| pair | mean diff | median diff | wins / ties / losses | Wilcoxon p |
| --- | --- | --- | --- | --- |
| Adam − CMA-ES | −0.083 | −0.057 | 1 / 0 / 7 | 0.078 |
| Adam − hybrid | +0.052 | +0.075 | 7 / 0 / 1 | 0.195 |
| CMA-ES − hybrid | +0.134 | +0.131 | 8 / 0 / 0 | 0.008 |

Original KaRMA on the physical task, cube fraction of the returned design:

| method | mean | median | best | IQR | beats start |
| --- | --- | --- | --- | --- | --- |
| start | 0.0085 | 0.0101 | 0.0118 | | |
| Adam | 0.0264 | 0.0242 | 0.0516 | [0.0168, 0.0311] | 7/8 |
| **CMA-ES** | **0.0640** | **0.0383** | **0.2124** | [0.0255, 0.0693] | **8/8** |
| hybrid | 0.0161 | 0.0161 | 0.0254 | [0.0115, 0.0196] | 8/8 |

| pair | mean diff | median diff | wins / ties / losses | Wilcoxon p |
| --- | --- | --- | --- | --- |
| Adam − CMA-ES | −0.038 | −0.014 | 2 / 0 / 6 | 0.078 |
| Adam − hybrid | +0.010 | +0.006 | 7 / 0 / 1 | 0.078 |
| CMA-ES − hybrid | +0.048 | +0.018 | 8 / 0 / 0 | 0.008 |

What the returned designs look like (`*/optimized_morphology/*.design.json`): Adam's designs
keep the start's proportions (longest finger reach 113 to 135 mm against the starts' 99 to
117 mm; 60 steps of 0.02 in the unconstrained coordinates move the morphology little); CMA-ES's
designs lengthen both fingers to 132 to 195 mm reach, inside the design box (links ≤ 90 mm,
reach bound 250 mm, no reach penalty active) and rewarded by the fixed ±100 mm task cube. The
best CMA-ES design (`start1`, exact 0.488, cube fraction 0.212, reach 195 mm) is a long-fingered
two-finger hand; whether such a hand is wanted is a design-box question (SPEC §8.1), not a metric
one, and every returned design satisfies the box.

### Results: 3×4 Full KaRMA (8 starts; `workspace/phase8/8.4/full3/summary.md`)

Exact score of the returned design (fresh three-contact seeds, actuated three-contact cells / 8,000 + 4608-bin SO(3) fraction / 0.3),
re-scored after 8.6 with the exact SLSQP force check at every sub-step (`workspace/phase8/8.4/full3/rescore_exact.json`;
the runs themselves selected on the Phase 5 path, whose smooth force pre-filter rejected some exactly feasible sub-steps and
gave means 0.0107 / 0.0142 / 0.0096 / 0.0117 for start / Adam / CMA-ES / hybrid, `summary.md`):

| method | mean | median | best | IQR | beats start | mean gain vs start |
| --- | --- | --- | --- | --- | --- | --- |
| start | 0.0115 | 0.0119 | 0.0195 | | | |
| **Adam** | **0.0147** | **0.0152** | **0.0214** | [0.0138, 0.0198] | **7/8** | **+0.0032** |
| CMA-ES | 0.0097 | 0.0091 | 0.0203 | [0.0056, 0.0137] | 2/8 | −0.0018 |
| hybrid (over budget, fewer gradient evaluations) | 0.0120 | 0.0134 | 0.0154 | [0.0095, 0.0145] | 5/8 | +0.0005 |

| pair | mean diff | median diff | wins / ties / losses | Wilcoxon p |
| --- | --- | --- | --- | --- |
| Adam − CMA-ES | +0.0050 | +0.0033 | 6 / 1 / 1 | 0.031 |
| Adam − hybrid | +0.0027 | +0.0031 | 7 / 0 / 1 | 0.078 |
| CMA-ES − hybrid | −0.0024 | −0.0027 | 2 / 0 / 6 | 0.312 |

Seeds of the returned designs that cannot hold the sphere at the primary ceiling (the 8.1 seed gate): Adam 0 of 48,
CMA-ES 1, hybrid 4, the starts 3.

Original KaRMA on the physical task (two-finger BFS on the thumb–index pair), cube fraction:

| method | mean | median | best | IQR | beats start |
| --- | --- | --- | --- | --- | --- |
| start | 0.0075 | 0.0082 | 0.0105 | | |
| Adam | 0.0100 | 0.0131 | 0.0165 | [0.0055, 0.0141] | 6/8 |
| CMA-ES | 0.0094 | 0.0066 | 0.0222 | [0.0024, 0.0181] | 5/8 |
| hybrid | 0.0089 | 0.0092 | 0.0130 | [0.0064, 0.0111] | 7/8 |

| pair | mean diff | median diff | wins / ties / losses | Wilcoxon p |
| --- | --- | --- | --- | --- |
| Adam − CMA-ES | +0.0005 | −0.0004 | 4 / 0 / 4 | 1.000 |
| Adam − hybrid | +0.0011 | +0.0014 | 6 / 0 / 2 | 0.547 |
| CMA-ES − hybrid | +0.0005 | −0.0007 | 3 / 0 / 5 | 1.000 |

Per start (`run.json`): on `start5` (a scale-0.5 perturbation with 5 seeds) both Adam and
CMA-ES returned designs whose fresh seed search finds **no** three-contact grasp (exact 0; the
best validated design of those runs was the first validated iterate, which had already lost
the grasps); CMA-ES also lost the grasps on `start4` (2 seeds left) and the two-finger
physical score on `start6` (0.0070 → 0.0007). The Full objective's 50 generations (400
evaluations, σ₀ 0.3) wander out of the narrow region where three fingertips hold the sphere,
and the exact selection then returns whatever validated best, which can be below the start
(4 of 8 for CMA-ES). Adam's 40 steps stay in that region on 7 of 8 starts.

### Conclusion, with its uncertainty

On the two-finger Diff-KaRMA objective at matched optimizer time (≈ 66 s on the RTX 4090,
compile and validations excluded), **CMA-ES returned a better design than Adam on 7 of 8 starts**
by the exact library (mean +0.083, Wilcoxon p = 0.078) and on 6 of 8 by the original KaRMA on
the physical task (p = 0.078), and beat the hybrid on 8 of 8 (p = 0.008). The paired data do not
support the statement "differentiable optimization is superior to CMA-ES" for this objective and
budget; they support its opposite at the 10 % level and not at the 5 % level with 8 starts. The
mechanism is the cost ratio: one batched evaluation of 8 designs costs about as much as one
gradient of one design, so CMA-ES spends ~950 evaluations where Adam spends 60 gradients, and
with σ₀ = 0.3 it explores the design box (longer fingers) that Adam's 60 steps of 0.02 do not
reach. Adam does improve the start on 7 of 8 starts (mean +0.135 exact, 2.5× the physical cube
fraction), and the 8.2 study shows what its runs do between validations.

On the 3×4 Full-KaRMA objective at the same matched budget the order reverses:
**Adam returned a better design than CMA-ES on 6 of 8 starts** (one tie, one loss; mean +0.0050,
Wilcoxon p = 0.031) by the exact actuated three-contact library, while the two-finger
physical KaRMA of the same designs is a tie (4 / 0 / 4). Here CMA-ES's 400 evaluations are not
enough to search a box in which most designs have no three-contact grasp, and its best
smooth designs are often exactly infeasible; the gradient's locality is an asset. The honest
statement is therefore conditional: differentiable optimization is not superior to CMA-ES on
the two-finger objective at this budget (the paired data lean the other way); it is superior
on the three-contact actuated objective at this budget (6 of 8 with a tie, p = 0.031, 8 starts),
where the feasible region is narrow. Neither claim extends beyond the measured budget (≈ 66 s), the
design box of SPEC §8.1, the Phase 7 hyper-parameters, or eight starts.

### Limitations

- Eight starts per objective: differences with p = 0.078 are suggestive, not conclusive; the
  physical-task scores of the CMA-ES designs have a wide IQR (one design at 0.21).
- The hybrid's budget is not matched (see the caveat) and its results are not ranked.
- The CMA-ES incumbent on `diff2/start0` was validated every 2 generations (65 validations),
  the others every 16 (8); the selection opportunity differs for that one run.
- Matched *optimizer* time excludes compile (one-off, cached) and validations (identical for all
  methods per checkpoint); matched *wall clock including validations* would favour the method
  with fewer checkpoints.
- Adam's learning rate (0.02) and iteration count are Phase 7's; a larger rate or more
  iterations at the same budget were not tried (the budget is set by Adam's own run).

### How to reproduce

```bash
for K in 0 1 2 3 4 5 6 7; do ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.optimizer_comparison --objective diff2 --starts 8 --only $K --out workspace/phase8/8.4/diff2; done
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.optimizer_comparison --summary workspace/phase8/8.4/diff2
```

## 8.5 Verification of the three-contact failure of arm A

### Files changed

| File | Change |
| --- | --- |
| `dkarma/experiments/three_contact_verification.py` (new) | the three procedures below, controls, attempt counts, `workspace/phase8/8.5/results.json` |
| `dkarma/solvers/seed_projection.py` | `multi_candidates(centre_caps="all", n_restarts, restart_scale)`: sphere-centre proposals from every triple of phalanges and perturbed LM initializations; `find_multi_seeds(candidates=..., keep_all=...)`: warm-start candidates and full verification counts (`n_distinct_kinematic`, `n_equilibrium_ok/rejected`); vectorized dedup. Defaults unchanged (Phase 7 behaviour; `dkarma/tests/test_multicontact.py` passes) |

### Procedure

Three independent procedures per design, on arm A (`workspace/phase7/exp34/armA_design/armA.design.json`)
and as controls on arm B (`run_full/optimized_morphology/optimized_morphology.design.json`) and the
3x4 anthropomorphic start. A three-contact seed is a configuration in which the three fingertip
capsules touch the 15 mm sphere within 0.35 mm, every other capsule clears it by 0.5 mm, fingers
clear each other by 0.5 mm, joints are within their limits, and the three contacts hold the
35 g sphere in a frictional equilibrium (mu 0.6, 3 mm torsional patch) with unlimited torque.

| procedure | what is independent | attempts per design |
| --- | --- | --- |
| (i) 10x JAX seed search | 3,261 joint samples instead of 381 (Sobol 2,560 + random 640 + per-joint linspace), centre proposals from all 64 phalanx triples instead of the fingertip triple, two perturbed restarts (0.2 rad) per candidate, projected LM `solve_multi_ik` as Phase 7 | arm A 1,633,284 IK solves; arm B 934,668; start 1,390,986 |
| (ii) alternative initializations | the other designs' three-contact seeds (`q`, sphere centre) as IK starts on this design, each with 20 perturbations (0.15 rad, 10 mm) | arm A 252 (12 donor seeds from start and arm B); arm B 126 (6 donors from the start) |
| (iii) independent host solver | Pinocchio FK of the exported URDF through the frozen reference's loader (physical metric config), SciPy SLSQP (400 random starts in the joint box, sphere centre at the fingertip centroid + 10 mm noise) and trust-constr (60 starts) on the feasibility problem `|gap_c| <= eps_g`, clearances `>= eps_col`, limits as bounds; no code shared with `solve_multi_ik`; every solution re-verified by the JAX path's exact checks and the exact equilibrium | 460 starts per design |

The host geometry was checked against dkarma's: capsule endpoints agree to 1e-11 once the
reference loader uses the physical config (its default config shortens the tip capsule by the
`L_ref`-scaled radius, 1.5 mm here). On the start, the smoke run's host solutions verified 4 of 4
kinematically.

### Results (`workspace/phase8/8.5/results.json`, `run.log`, `run_controls.log`)

| design | Phase 7 search (381 samples, fingertip centres): seeds | (i) 10x search: IK attempts → kinematically valid → distinct → equilibrium-ok | (ii) warm starts: attempts → kinematic → equilibrium | (iii) host SLSQP / trust-constr feasible → distinct → verified kinematic → equilibrium |
| --- | --- | --- | --- | --- |
| **arm A** | 0 (0 of 792 kinematic) | **1,633,284 → 0 → 0 → 0** | **252 → 0 → 0** | **0 / 400, 0 / 60 → 0 → 0 → 0** |
| arm B (control) | 6 (335 of 486 kinematic) | 934,668 → 495,313 → 7 checked (budget 6 reached) → 6 | 126 → 126 → 6 (budget) | 329 / 400, 46 / 60 → 373 → 367 → 268 |
| start (control) | 6 (389 of 819 kinematic) | 1,390,986 → 379,751 → 7 checked (budget 6 reached) → 6 | 126 → 45 → 6 (budget) | 240 / 400, 19 / 60 → 259 → 257 → 166 |

Arm A's two-finger grasps are unaffected (6 seeds, Phase 7 table); the search is for three
simultaneous fingertip contacts, which requires the middle fingertip to reach the 30 mm sphere
held between thumb and index. Arm A's finger bases lie 17.8 mm (index-middle), 28.8 mm
(thumb-index) and 40.6 mm (thumb-middle) apart, the first less than two capsule diameters
(23.6 mm), so index and middle run nearly parallel and the thumb opposes the index alone; arm B's
bases are 37.1, 54.2 and 85.6 mm apart (`armA.design.json`, `optimized_morphology.design.json`;
finger reaches 97 to 108 mm on both).

### Verdict

The Phase 7 claim is **upgraded**, in the form SPEC §11.1 prescribes: *no feasible three-contact
grasp of arm A was found by three independent procedures with 1,633,284 + 252 IK attempts of the
JAX search (64 phalanx triples, two restarts per candidate, 12 donor seeds) and 460 starts of an
independent Pinocchio + SciPy constrained solver; the same procedures find hundreds of thousands
of kinematically valid and hundreds of equilibrium-feasible configurations on arm B and the
start.* The searches alone are not a proof of nonexistence (CLAUDE.md rule 13).

**Addendum from 8.6 (a proof, not a search).** Building arm A's MuJoCo scene exposed what the
searches could not say: at every two-finger seed of arm A the index and middle capsules
interpenetrate by 7 to 20 mm, and dkarma's own cross-finger clearance table agrees
(`capsule_pair_clearance`: −20.2 mm for `index_c1`–`middle_c1` at the first seed). The cause is
configuration-independent: the index and middle finger bases are **17.8 mm apart** while two
capsules of radius 11.8 mm need 23.6 mm, so the proximal capsules, whose axes start at the bases,
have a clearance of at most **−5.8 mm for every joint configuration** (start: +6.4 mm, arm B:
+13.5 mm). The three-contact seed definition requires ≥ 0.5 mm clearance between every pair of
capsules of different fingers, so **no three-contact grasp of arm A exists under the model**; the
three procedures' zero counts are the consequence. The two-finger evaluators did not see this
because the reference metric checks self-collision only among the links of the fingers in its
robot config (thumb and index for arm A, `robot_armA.yaml`), which dkarma reproduces (CLAUDE.md
rule 1); the CMA-ES against the original KaRMA was therefore free to park the middle finger
inside the index. The Experiment IV conclusion is sharpened: the two-finger objective did not
merely move the third finger out of reach, it produced a hand that cannot be built as three
separate fingers. The design box of SPEC §8.1 has no minimum base separation; adding one
(≥ 2 r_l + eps_col between any two finger bases) is a design-box decision for the owner.

### Limitations

- The host solver samples joint space uniformly and starts the sphere at the fingertip
  centroid; a grasp reachable only from a narrow basin could be missed by all procedures (the
  JAX search covers 1.6 M starts, which bounds how narrow).
- The equilibrium check uses the unlimited-torque frictional equilibrium (the seed definition of
  SPEC §6.3), not the actuated one; arm B's 268 of 367 ratio shows the kinematic and the
  frictional conditions are distinct.
- The controls were run with the budget cut (6 seeds) for procedures (i) and (ii), so their
  "distinct" counts stop at the budget; the host procedure (iii) is counted in full.

### How to reproduce

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.three_contact_verification --out workspace/phase8/8.5 --designs armA
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.three_contact_verification --out workspace/phase8/8.5 --designs armB start
```

## 8.6 Independent simulation (MuJoCo)

MuJoCo 3.15 was installed into `~/mamba/envs/karma-jax` on 2026-10-13 after the owner's confirmation.

### Files changed

| File | Change |
| --- | --- |
| `dkarma/experiments/simulation_replay.py` (new) | `export`: every command of the exact trajectory libraries as the hard-mode fixed-assignment joint path at sub-step resolution, the exact verdict per primitive (reference kernels for two contacts; kinematic gates plus the exact SLSQP force check of every sub-step and the 8.1 seed gate for three), the exact allocation's joint torques and the geometric squeeze offsets per state; `simulate`: MJCF scene from the design, position-servo replay, success criteria, confusion matrices (`--libraries`, `--drop-fingers`, `--kp`, `--squeeze-mm`, `--tag`) |
| `dkarma/metrics/multicontact.py` | `evaluate_fixed_contacts_exact`: the actuated force check of every sub-step is the **exact** SLSQP allocation (lazily, up to the first failed sub-step) instead of the smooth allocation's gate, whose 1e-6 N equilibrium tolerance rejected exactly feasible states near the torque margin once the solver converged (14 of the first 40 such states on the start were exactly feasible); the 8.4 Full designs were re-scored with it (§8.4) |

### Scene and controller

Built from the design parameters, not the URDF: hinge joints with the design's axes and limits
(radians), one capsule collision geometry per link with the design's radius (terminal capsule
axis `tip − r_l` as in dkarma), same-finger capsule pairs excluded (dkarma has no same-finger
self-collision), link masses 28 g / 10 g, the sphere as a free body (15 mm, 35 g), Coulomb
friction 0.6 with torsional coefficient `mu·a = 0.0018 m` for the 3 mm patch and rolling 1e-4,
gravity −z, `implicitfast` at 0.5 ms, elliptic cones, position actuators (`kp` 20 N·m/rad,
`forcerange ±0.108 N·m`). The scene's capsule endpoints agree with dkarma's to 1e-17 at the seed
configurations.

The controller streams the exported joint path (a linear ramp between consecutive sub-step
commands, 50 ms per 1.25 mm sub-step) and commands every assigned fingertip **0.5 mm into the
sphere along its contact normal** (joint offsets from the surface Jacobians, computed per state),
after a 0.3 s settle at the seed. Three controller variants were rejected by measurement before
this one: a pure position replay (zero normal force, the sphere drops in the settle phase), a
torque-offset replay `q + tau_exact / kp` (holds the sphere at rest at twice the minimum-norm
allocation, loses it along the path whose contact gaps open by up to the 0.35 mm tolerance),
and a stepped instead of streamed command (the saturated servo slams the fingertip 2.4 mm within
milliseconds and ejects the sphere). A 1 mm squeeze displaces the sphere by 2 to 4 mm in the
settle phase (the two normals are not antipodal); 0.5 mm holds it to 0.2 to 0.7 mm.

Success per primitive: at the primitive end the sphere centre is within 4 mm of the commanded
centre, every assigned fingertip is in contact, and (secondary criterion) the sphere's orientation
is within 0.2 rad of the commanded rotation. A trajectory stops at its first failed primitive.
A seed whose sphere drifts more than 4 mm in the settle phase ("seed does not hold") fails every
primitive; those are reported separately because they measure the hold, not the path.

### Results (`workspace/phase8/8.6/simulation.json`, `simulation_armA_two_finger.json`; 6 seeds per library)

| design | library | seeds that hold | exactly valid primitives | succeed (position + contacts) | succeed (+ orientation) | exactly rejected first primitives → succeed (+ orientation) | holding seeds only: valid → succeed (+ orient.); rejected → succeed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| two-finger baseline | two-finger | 5/6 | 116 | 112 (97 %) | 100 (86 %) | 55 → 16 (29 %) | 115 → 100 (87 %); 16/45 |
| 3×4 start | two-finger | 6/6 | 111 | 98 (88 %) | 82 (74 %) | 56 → 13 (23 %) | 111 → 82 (74 %); 13/56 |
| 3×4 start | three-contact | 4/6 | 179 | 21 (12 %) | 21 (12 %) | 64 → 7 (11 %) | 135 → 21 (16 %); 7/42 |
| arm A (three-finger body) | two-finger | 0/6 | 122 | 0 | 0 | 56 → 0 | – |
| arm A (middle finger removed) | two-finger | 6/6 | 122 | 122 (100 %) | 96 (79 %) | 56 → 22 (39 %) | 122 → 96 (79 %); 22/56 |
| arm B | two-finger | 4/6 | 163 | 131 (80 %) | 87 (53 %) | 56 → 8 (14 %) | 133 → 87 (65 %); 8/36 |
| arm B | three-contact | 4/6 | 131 | 39 (30 %) | 39 (30 %) | 67 → 12 (18 %) | 103 → 39 (38 %); 12/43 |

Arm A has no three-contact library (8.5). Its three-finger scene cannot hold any seed: the index
and middle capsules interpenetrate (8.5 addendum) and the contact solver flings the fingers
apart in the first 10 ms; with the middle finger removed it behaves like the other two-finger
hands.

### Disagreements examined

*Exactly valid, fails in simulation* (two-finger libraries, holding seeds: 13 to 35 %):
mostly rotation commands that lose the orientation criterion while the centre stays within
tolerance (baseline: 112 of 115 primitives keep position and contacts, 100 keep orientation),
i.e. the sphere slips about the pinch axis instead of rolling with the fingertips; the
quasi-static model credits the rotation through the torsional patch and the rolling constraint,
the simulation's torsional friction and the 0.5 mm squeeze do not always transmit it. For arm B
the two-finger loss is larger (65 %): its longer fingers (reach 100 to 108 mm) give the servo
less tip stiffness per joint.

*Three-contact libraries* (16 % and 38 % of holding seeds' valid primitives succeed): the failures
are at the **first** primitive in 21 of 22 (start) and 15 of 17 (arm B) trajectories, with the
sphere within 0.8 mm of its commanded centre and 0.04 to 0.08 rad of its rotation, and **one of
the three contacts lost** (start: the middle fingertip in 20 of 22; arm B: the index in 9, the
middle in 5). The quasi-static model keeps all three gaps at zero by construction; in the
simulation the geometric squeeze and the rolling leave one fingertip a fraction of a millimetre
off the sphere, which the strict "all assigned contacts" criterion counts as failure although
the object is held and placed. Translations fare better than rotations (start 40 % vs 11 %, arm B
33 % vs 38 %). The three-contact error term is therefore dominated by contact maintenance at
the sub-millimetre scale, not by drops or large pose errors.

*Exactly rejected, succeeds in simulation* (11 to 39 %): rejections by the rolling QP (`qp_fail`),
by the rotation efficiency / slip gates (`rot_slip`) and by the 0.5 mm clearance gates
(`collision`, `rot_collision`) can still be executed by a compliant simulated hand: the gates
are conservative where the simulation tolerates a sliding contact or a sub-millimetre clearance.
The `force` and `seed_force` rejections (three-contact, torque ceiling) never succeed
(0 of 17): the actuation gate is not conservative.

*Seeds that do not hold* (baseline 1 of 6, start 2 of 6 three-contact, arm B 2 of 6): these
seed grasps satisfy the exact equilibrium with the minimum-norm allocation but the 0.5 mm
geometric squeeze pushes the sphere out (non-antipodal normals); they are a controller
limitation, counted separately.

### Acceptance

The four designs are evaluated, the confusion matrices are in the table and the disagreements
are attributed to gates and assumptions: **met**, with the caveats that the three-contact
error term is large and criterion-driven (contact maintenance), and that the controller is a
simple position servo with a fixed geometric squeeze, not a force controller.

### Limitations

- One controller and one set of contact parameters (`solref 0.002`, `solimp 0.95 0.99 0.001`); a
  force-controlled or compliant-fingertip replay would move both error fractions.
- The position replay assumes the quasi-static path is executable at 25 mm/s; dynamics are
  minimal at this speed but not absent (the first-sub-step kick of a stepped command showed how
  sensitive the pinch is).
- The three-contact "all contacts" criterion is strict; a criterion on the held pose alone would
  count most of those failures as successes (position error 0.8 mm).
- Two-finger exact verdicts come from the reference kernels, which ignore fingers outside the
  pair (arm A).

### How to reproduce

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay export --out workspace/phase8/8.6
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay simulate --out workspace/phase8/8.6
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay simulate --out workspace/phase8/8.6 --designs armA --drop-fingers middle --tag _armA_two_finger
PYTHONPATH=. ~/mamba/envs/karma-jax/bin/python workspace/phase8/8.4/rescore_full3.py
```

## 8.7 Write-ups and fabrication candidates

### (a) Why the physical-task scores and the published KaRMA-T are negatively correlated

Experiment I (`workspace/phase3/surrogate.json`, `docs/phase3_report.md` §3.2) found Spearman
rho = −0.33 between the smooth `K_T~` and the published KaRMA-T over nine two-finger designs,
and rho = −0.26 between the original KaRMA on the physical task and the published number. The
two numbers answer different questions.

**Two task definitions.** The published metric (`karma/`, `karma_config.yaml`) scales every
length of its task by `L_ref / 0.2`: the sphere radius is 10 mm × `L_ref / 200 mm`, the voxel 10 mm
× the same factor, and so are the contact tolerance, the collision clearance, the sub-step and
the slip gate. `L_ref` is the hand's own characteristic length (reach from the palm). A hand
half the size is scored on a sphere half the size in a cube half the size, and its score is a
dimensionless fraction of *its own* workspace. That is the design intent of the paper: compare
hands of different scales on a task proportional to each. The physical task of SPEC §2 fixes the
object (15 mm radius, 35 g, mu 0.6), the task cube (±100 mm at 10 mm) and the tolerances for
every design, because the hand to be fabricated must manipulate that object.

**What each rewards.** Under the published definition a design is rewarded for rolling a sphere
of radius `0.05 L_ref` through voxels of edge `0.05 L_ref`; shrinking the hand shrinks the
object and the voxels together, so a short-fingered hand that rolls a small object over many
small voxels scores high. Under the fixed task a design is rewarded for rolling the 15 mm sphere
through 10 mm cells; a short finger simply cannot wrap a sphere that is large relative to its
links, and the capsule radius (11.8 mm, fixed in both definitions) leaves little room between the
fingertips.

**Why the Phase 0 exploit designs separate the two.** The CMA-ES winners T and A1 of the
`optimize/` study pushed `L_ref` down to 118 and 131 mm (SPEC §1; the family's lower link bound
bites). Their published task uses a 5.9 to 6.5 mm sphere: 2.3 to 2.5 times smaller than the
15 mm physical object. Their published KaRMA-T (0.56, 0.43) is 5 to 6 times LEAP's, and on the
physical task they reach exactly one cell (the seed) under every evaluator. The baseline
(`L_ref` 155 mm) scores 0.03 published and is the most capable physical hand of the set. The
correlation is negative because the set is dominated by this trade: the designs that exploited
the scaling are the ones that lost the physical object.

Per-design table (nine two-finger designs of Experiment I; 3x4 designs of Experiments III/IV
below; `L_ref` from the exported robot YAML; "published sphere" is the radius the published task
scores the hand on, `0.05 L_ref`; the physical object has radius 15 mm, so the ratio column is
hand reach over object diameter):

| design | L_ref (mm) | L_ref / 30 mm | published-task sphere radius (mm) | smooth K_T~ | exact-library K_T | physical cube fraction (orig. KaRMA, physical task) | physical KaRMA-R | published KaRMA-T | published KaRMA-R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 154.7 | 5.2 | 7.7 | 0.00588 | 0.00067 | 0.01071 | 0.109 | 0.0297 | 0.215 |
| T | 117.5 | 3.9 | 5.9 | 0.00163 | 0.00013 | 0.00013 | 0.004 | **0.5579** | 0.270 |
| A1 | 130.7 | 4.4 | 6.5 | 0.00163 | 0.00013 | 0.00013 | 0.004 | **0.4295** | 0.274 |
| pert00 | 203.2 | 6.8 | 10.2 | 0.00205 | 0.00014 | 0.00025 | 0.024 | 0.0002 | 0.045 |
| pert01 | 185.3 | 6.2 | 9.3 | 0.00213 | 0.00025 | 0.00108 | 0.023 | 0.0001 | 0.004 |
| pert02 | 157.4 | 5.2 | 7.9 | 0.00446 | 0.00047 | 0.00762 | 0.094 | 0.0200 | 0.190 |
| pert03 | 179.9 | 6.0 | 9.0 | 0.00433 | 0.00047 | 0.00258 | 0.063 | 0.0028 | 0.103 |
| pert04 | 126.4 | 4.2 | 6.3 | 0.00163 | 0.00014 | 0.00079 | 0.017 | 0.0213 | 0.153 |
| pert05 | 173.1 | 5.8 | 8.7 | 0.00392 | 0.00028 | 0.00229 | 0.032 | 0.0008 | 0.048 |

| 3x4 design (`workspace/phase7/exp34/results.json`) | L_ref (mm) | L_ref / 30 mm | published-task sphere radius (mm) | three-contact seeds (Phase 7 search) | 2-finger exact K_T | physical cube fraction | physical KaRMA-R | published-style KaRMA-T |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| start | 174.3 | 5.8 | 8.7 | 6 | 0.00063 | 0.01050 | 0.112 | – |
| (1) thumb–index Diff | 188.8 | 6.3 | 9.4 | 6 | 0.00067 | 0.01162 | 0.091 | – |
| (2) pairwise | 186.6 | 6.2 | 9.3 | 6 | 0.00071 | 0.00971 | 0.098 | – |
| (3) MC | 188.0 | 6.3 | 9.4 | 6 | 0.00081 | 0.01562 | 0.093 | – |
| (4) Full = arm B | 188.1 | 6.3 | 9.4 | 6 | 0.00073 | 0.01129 | 0.106 | – |
| arm A (original KaRMA + CMA-ES) | 139.8 | 4.7 | 7.0 | 0 | 0.00050 | 0.01421 | 0.094 | 0.0416 |

Two remarks the table supports. First, the exploit is not a property of CMA-ES but of the
objective: arm A, optimized against the original KaRMA with the proxy of `optimize/`, again
shrank `L_ref` (174 → 140 mm) and raised the published-style number, and it is the best
two-finger hand of its set on the physical task as well, because its scale stayed above 4.5
object diameters; T and A1 went below 4.5 and lost the object. Second, the smooth `K_T~`, the
exact library and the original KaRMA on the physical task agree with each other (rho 0.98 to
0.99) because they share the task; they disagree with the published number because they do not.
The published KaRMA-T remains the right number for the paper's question (how dexterous is a
hand relative to its own size); the physical scores are the right numbers for this project's
question (which hand manipulates this object). Neither is wrong; they must not be compared.

### (b) Fabrication candidates and the selection criterion

Arm A (original KaRMA + CMA-ES) and arm B (Full KaRMA + Adam) are preserved with their exported
specifications under `results/fabrication_candidates/` (`README.md` there lists the files and
their provenance). **Nothing is selected.** Note from 8.5/8.6: arm A's index and middle finger
bases are 17.8 mm apart, less than two capsule radii, so its index and middle links overlap in
every configuration; as exported it is buildable only as a two-finger hand (thumb and index), and
its three-finger specification is void. Arm B's finger bases are separable (≥ 13.5 mm clearance
bound). The criterion, written here before any selection and
derived from SPEC §12 IV, is:

1. *Eligibility.* A candidate must have, on the frozen physical task, exact two-finger coverage
   and at least one exactly valid three-contact grasp verified by the three independent
   procedures of 8.5 (or, if the design is intended as a two-finger hand, the claim "no
   three-contact grasp" stated with its attempt counts); its exported URDF and torque
   specification must reproduce its Phase 7/8 scores when re-scored (`official_karma_physical`,
   `codesign_comparison.suite`).
2. *Primary measure.* The exact Full-KaRMA counts (actuated three-contact cells and SO(3) bins,
   fresh seeds, `MC.evaluate_fixed_contacts_exact` with the 8.1 seed gate) and the exact
   two-finger library coverage, each with its fresh-seed uncertainty (standard deviation over
   seeds, and over at least three seed-search budgets).
3. *Robustness.* Minimum wrench capacity at the three-contact seeds ≥ 1 × the object's weight
   in every one of the 14 standard directions, and the torque and mass sweeps of Experiment II
   on the candidate (the design must hold the object at the primary ceiling with margin, and the
   score must degrade gracefully at 70 g).
4. *Simulation agreement* (8.6): the exact-vs-simulation confusion matrix of the candidate must
   show a false-positive rate (exactly valid, fails in simulation) below the rate of the
   two-finger baseline, so that the exact counts are trusted for that design.
5. *Decision.* Between eligible candidates, prefer the higher primary measure; break ties by
   robustness; report the two-finger cost of the choice (SPEC §12 IV's "same resources,
   underutilization made explicit").

The protocol after selection is SPEC §12 IV: export skeleton and torque spec → mechanical design
preserving joint kinematics → compliant fingertips → as-built measurements → the same pipeline on
designed and as-built → physical trials with a tracked sphere.

## Phase 8 in one page: established, refuted, remaining

### Established

1. **The smooth force credit now tracks the exact solver** (8.1). The Phase 4 force solver
   had never converged at infeasible states (Gauss-Newton on a large-residual problem, damping
   blind to the torsion variables), its validated gradient was that of an unconverged map, and
   the seed state was never force-gated. With damped Newton in scaled variables, the envelope
   gradient (finite-difference agreement ≤ 1e-6 at feasible, marginal and 6× loads), hinge 0.005
   and the knee credit mapping, the energy separates exact labels with AUC 0.9988 on 3,153
   held-out instances, and the actuation-aware coverage is 0 % of the kinematic value wherever
   the exact solver rejects every state (Experiment II re-run; both acceptance numbers met).
2. **Why the Phase 7 continuation stages lost exact score** (8.2): non-finite and exploding
   derivatives of the hard rolling QP at degenerate active sets, carried by Adam's momentum;
   not the kernel widths. Gradient hygiene (clip, skip non-finite, reset moments) and
   stage selection remove the failure (0 non-finite iterations, 5 of 5 starts improved, final
   design preserved on 5 of 5 under the selected reading) and are the new default.
3. **The exact validation's resolution** (8.2): one Adam step moves the 8-seed exact score by
   10 % (sd), 4.3 % at 16 seeds; the best-of-run reading is biased upward. Criteria at the
   2 % level are below this resolution; comparative claims need paired, multi-start data.
4. **Seed policy** (8.3): re-projection stays the default (no non-finite gradients on either
   objective); the exact-ranked union is better on two-finger objectives (4 of 5 starts, +0.012)
   at 1.5× wall clock; every reseeding policy exposes the KKT-derivative defect on the Full
   objective.
5. **Optimizer comparison with statistics** (8.4, 8 starts × 3 methods × 2 objectives, matched
   optimizer time ≈ 66 s, fresh-seed exact library and the physical original KaRMA on every
   returned design): CMA-ES beats Adam on the two-finger objective (7 of 8, p = 0.078; it
   spends ~950 batched evaluations against 60 gradients and lengthens the fingers within the
   design box), Adam beats CMA-ES on the three-contact actuated objective (6 of 8 with a tie, p = 0.031;
   CMA-ES leaves the narrow feasible region). The hybrid's budget was not matched and it is
   not ranked.
6. **Arm A has no three-contact grasp that three independent procedures can find** (8.5):
   1,633,284 + 252 IK attempts (64 phalanx triples, restarts, 12 donor seeds) and 460 starts of a
   Pinocchio + SciPy solver, zero kinematically valid configurations; the same procedures find
   hundreds of thousands on arm B and the start. The Experiment IV claim is upgraded to that
   statement, with its attempt counts, not to nonexistence.
7. **The exact-vs-simulation error term is measured** (8.6, MuJoCo): for seeds the controller
   holds, 65 to 87 % of exactly valid two-finger primitives succeed (position, contacts and
   orientation) and 14 to 39 % of exactly rejected ones succeed anyway; three-contact
   primitives succeed in 16 to 38 %, failing almost always at the first primitive by losing one
   of three contacts with the sphere within 0.8 mm of its commanded pose. Actuation rejections
   never succeed in simulation (0 of 17). Arm A cannot be simulated as a three-finger body: its
   index and middle fingers overlap (item 8 below).
8. **Arm A's third finger is unbuildable, not merely unreachable** (8.5/8.6): the index and
   middle bases are 17.8 mm apart against a 23.6 mm capsule diameter, so the proximal capsules
   overlap by ≥ 5.8 mm in every configuration; the design box has no base-separation constraint.
9. **The negative correlation with the published KaRMA-T is a task-definition effect** (8.7):
   the published task scales its sphere and voxels with the hand (`0.05 L_ref`), the physical
   task fixes them; designs below ~4.5 object diameters of reach score high on the former and
   lose the 15 mm object on the latter.

### Refuted or withdrawn

- "The strict continuation stage is too tight for these budgets" (Phase 7): the strict stage
  raised the exact score once the gradient events were handled; gentle sigma and skipping the
  strict stage bring nothing measurable.
- "A larger force weight (20 to 50) would calibrate the actuation penalty" (Phase 7, finding 3):
  linear weights up to 50 leave 24 to 54 % of exactly infeasible states above half credit and
  29 to 54 % of the kinematic coverage at the continuous ceiling; the fix was the solver, the
  mapping's knee and the seed gate.
- Phase 7's Experiment II table and its "max torque usage" values (overstated by the
  unconverged solver): replaced.
- "Differentiable optimization is superior to CMA-ES" is not supported on the two-finger
  objective at the measured budget; it is supported, with 8 starts (p = 0.031), on the three-contact
  actuated objective.

### Not met, and why

- 8.2's acceptance on the raw last iterate (best 3 of 5) and on the 3×4 start (1 of 5 raw,
  2 of 5 selected with the adopted schedule): the validation moves more per step than the
  criterion allows; reported as measured, criterion not loosened.

### Remaining before fabrication

1. A force- or impedance-controlled replay in MuJoCo (8.6 used a position servo with a fixed
   0.5 mm squeeze) and a pose-based three-contact success criterion, to separate the contact
   maintenance error from manipulation failures; the exact-vs-simulation term is now measured
   but controller-dependent.
2. The KKT-derivative defect at degenerate active sets (regularized or least-squares derivative
   with finite-difference tests), which currently costs 14 to 25 % of iterations under
   reseeding and is the root of the Phase 7 failures.
3. A design-box decision on finger reach (SPEC §8.1): the two-finger CMA-ES winners reach 195 mm
   because the fixed cube rewards it; whether that is wanted is a fabrication constraint.
4. Validation at 16 or more seeds for any selection at the few-per-cent level.
5. The selection itself, by the criterion written in 8.7(b), after 1 to 4.
