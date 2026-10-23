# Phase 10 report: compliant contact, a powered admissible comparison, and deployment-relevant grasps

Scope: SPEC.md §11.3 (added 2026-10-14 from the owner's Phase 9 review, `docs/phase9_review_2026-10-14.md`), items
10.1 to 10.4 in order. The framework is fixed: no metric, kernel, contact model of any evaluator, solver, optimizer,
frozen task (`dkarma/configs/task_spec.yaml`) or D18 design box changed. The Phase 9 settings stay
(`kkt_method="reduced"`, `exp_so3_safe`, norm-100 clipping, cached exact evaluators, suites and exports on the CPU).
Every table states its task definition: **fixed physical** (15 mm / 35 g sphere, ±100 mm cube at 10 mm, XL330
primary ceiling 0.108 N·m; SPEC §2) or **published `L_ref`-scaled** (the original KaRMA's task). Every simulation
result names its controller, contact model and success criterion.

Raw outputs: `workspace/phase10/{10.1,10.2,10.3,10.4}/`; chain scripts `workspace/phase10/chain_*.sh`.

---

## Pre-registration (frozen 2026-10-14 15:50 EDT, before any Phase 10 data was produced)

Everything in this section was written before the first compliant replay, before the first new 10.2 start was
drawn or run, and before any `S_task` or criterion-v2 quantity was computed. It is not edited after that point.
Corrections, if any, are appended below it with their date and are applied only to data produced after them.

What was known when it was written: the Phase 9 numbers (`docs/phase9_report.md`), including the 9.5 candidates'
per-seed wrench capacities under the default seed configuration and their rigid-contact simulation rates. Two model-
building measurements that are not outcomes were made first. One is the location and the allocated normal force of
the assigned contacts at the exported states (76 % of three-contact contact states sit on a hemispherical cap of the
terminal capsule; median exact-allocation normal force 0.26 N, 95th percentile 1.04 N). The other is MuJoCo's soft-
contact law in a calibration scene (a fixed capsule and the free sphere under known forces, no hand, no trajectory).

### P10.1 Compliant fingertips: model, replays, outcomes, decision rule

**Model** (`dkarma/experiments/compliant_pad.py`; the fitted parameter sets frozen in
`dkarma/configs/compliant_pads.json`). Simulation model only: the pad is the outer `t` of the 11.8 mm terminal-capsule
radius. In MuJoCo the capsule keeps its radius and the pad exists only in the contact law of the (terminal capsule,
sphere) pairs, so geometry, morphologies, trajectories and exact verdicts are unchanged. Assumptions:

| quantity | central | grid | basis |
| --- | --- | --- | --- |
| silicone hardness | Shore 20A | 10A, 20A, 40A | platinum-cure fingertip silicones; SPEC §11.3 range |
| Young's modulus | 0.732 MPa | 0.413, 0.732, 1.690 MPa | Gent (1958) Shore-A relation, ν = 0.5 |
| pad thickness `t` | 2 mm | 1, 2, 3 mm | SPEC §11.3 range |
| normal law | Hertz + bonded-layer correction | same | `F = 16/9 E R*^½ δ^{3/2} (1 + 1.133χ + 1.283χ² + 0.769χ³ + 0.0975χ⁴)`, `χ = √(R*δ)/t` (Dimitriadis et al. 2002), `R* = 6.60 mm` (sphere on the capsule cap) |
| indentation at 0.26 N | 0.121 mm | 0.058–0.191 mm | from the law |
| patch radius at 0.26 N | 0.90 mm | 0.62–1.12 mm | `a = √(R*δ)` |
| sliding friction | 0.6 | 0.6 | frozen task value |
| torsional friction | 0.317 mm | 0.219–0.397 mm | Hertzian patch, `(3π/16) μ a` |
| rolling friction | 33.6 µm | 23–42 µm | hysteresis, `(3/16) α a`, loss fraction α = 0.2 |
| normal damping ratio | 1.0 | 1.0 | MuJoCo-normalized, as the rigid 9.3 contact |

*MuJoCo realization:* one explicit `<pair>` per terminal capsule and the sphere, `condim 6`, elliptic cone, direct
`solref = (−s, −2ζ√s)` with `s = s_u A` (`A`: the pair's `diagApprox` from `body_invweight0`, which makes the static
law independent of the link's inertia), `solimp` fitted to the normal law over 0.02–2 N with impedance ≥ 0.9, and
`solreffriction = (−1, −1/dt)`. The impedance bound is needed because MuJoCo regularizes the friction rows with the
same impedance as the normal row. A soft normal row (impedance ≈ 0.1) lets the sphere creep at 5.3 mm/s under 0.1 N
tangential and 0.3 N normal load inside the friction cone, and MuJoCo 3.15's noslip post-solver does not act on
`condim 4/6` contacts (both measured). Torsional friction cannot be dropped: it is the only resistance to spin about a
two-finger pinch axis. *Calibration-scene checks:* static indentation within 5 % of the law at 0.05 / 0.26 / 1 N for
all nine sets (fit error ≤ 11.6 % over 0.02–2 N); creep 0.045–0.065 mm/s against 0.15 mm/s for the rigid contact; a
0.3 N step settles in 44–78 ms against 13 ms (the impedance inflates MuJoCo's apparent mass along the normal by
1/(1 − imp)). These checks are repeated by `dkarma/tests/test_simulation_replay.py`.

*Diagnostic variants* (reported, not part of the verdict): **D1** the central normal law with the rigid model's
friction (`condim 4`, torsional 1.8 mm, no rolling), to separate normal compliance from patch friction; **D2** the
central set with damping ratio 0.3.

**Rigid bound:** the 8.6/9.3 scene unchanged (`solref 0.002 1`, `solimp 0.95 0.99 0.001`, `condim 4`, torsional
1.8 mm). The rigid replays are re-run with the Phase 10 code and must reproduce the 9.3 / 9.2 numbers exactly.

**Replays:** the 9.3 archives (`workspace/phase9/9.3/`: baseline two-finger; start, arm B, arm A′ start0 two-finger
and three-contact; arm A two-finger with the middle finger removed; 6 seeds per library) and the 9.2 12-seed
libraries (`workspace/phase9/9.2/sim/`: 16 arm A′ and Full-D18 three-contact libraries, arm B three-contact 12 seeds,
baseline two-finger 16 seeds). Controllers with their 9.3 parameters: the 8.6 position servo (`kp` 20 N·m/rad,
0.5 mm squeeze) and the 9.3 force-floor controller (`k_i` 0.2 m/(N·s), leak 0.1, set-point = exact allocation,
floor 0.05 N, squeeze in [0.5, 3] mm). Every trajectory runs to its end and stops only at a drop.

*Controller stability* (the only condition for re-tuning, SPEC §11.3). The two-finger baseline libraries (9.3,
6 seeds; 9.2, 16 seeds) are replayed first at the central parameters. A controller is **unstable under compliance** if,
on either baseline library, fewer seeds hold (settle drift ≤ 4 mm) than under rigid contact, or a holding seed drops the
sphere (centre > 20 mm from the command or no assigned contact) during an exactly valid primitive where it did not under
rigid contact. Only then are that controller's parameters re-tuned, on the two-finger baseline only, by the 9.3 tuning
procedure. A change in two-finger success rate is not instability and triggers nothing. No parameter is ever tuned on
three-contact data.

**Outcomes** per design, library, contact model and controller, over the exactly valid primitives of the seeds that
hold under that model and controller (the 9.3 definitions):

* *strict* (primary): object pose (centre within 4 mm and orientation within 0.2 rad of the command) and every
  assigned fingertip in contact at the primitive end;
* *pose* alone; *retention at the primitive end* alone;
* *retention by duty cycle* (secondary, pre-registered here): every assigned lane in contact during at least 50 % of
  the primitive's simulation steps; and *pose + duty-cycle retention*;
* *drop or gross instability*: centre more than 20 mm from the command, or no assigned fingertip in contact;
* the confusion matrices of the 8.6 convention (all seeds, per primitive up to the first exactly rejected one).

**Decision rule** (SPEC §11.3, made operational). For design `D`, model `m` and controller `c`:
`gap(D, m, c) = strict₂(D, m, c) − strict₃(D, m, c)`, the two-finger minus the three-contact strict rate on holding
seeds. A library without a holding seed has rate 0. The rule applies to the three-contact designs that have both
libraries, the 9.3 archives of the start, arm B and arm A′ start0. The 9.2 libraries are three-contact only and have no
gap. Per controller, at the central parameters:

* **resolved** if `gap ≤ 0.10` on every one of the three designs;
* **not resolved** if, on every design, the gap shrinks by less than half of its rigid value
  (`gap_central > 0.5 · gap_rigid`);
* **partial** otherwise.

The verdict is **robust** if the same verdict holds at all nine grid points. The **headline verdict is the force-floor
controller's**, the hardware controller and the controller of the 10.2 endpoint. The position servo's verdict is
reported beside it.

**Compliant three-contact agreement reference** (the second deliverable). It is the strict false-positive rate
(1 − strict rate) of every three-contact library at the central parameters, per controller: 9.3 libraries and 9.2
12-seed libraries, with the distribution over designs and the pooled rate. **Reference value for criterion v2**: the
median, over the 17 three-contact 12-seed libraries of `workspace/phase9/9.2/sim/` (8 arm A′, 8 Full-D18, arm B) that
have ≥ 10 exactly valid primitives on holding seeds, of the strict FP rate under the force-floor controller at the
central parameters.

### P10.2 Arm A′ vs Full-D18 on new starts

**Starts.** The 8.4 procedure continued exactly as coded (`continuation_study.make_starts`, `objective_spec("full3")`,
scrambled Sobol seed 321, 8 × `n_starts` points, scale 0.25 for Sobol index `j < 4`, 0.5 for `4 ≤ j < 20` and the
default 0.25 for `j ≥ 20`, a point accepted if the start design has ≥ 3 three-contact seeds at budget 6). It is run
with `n_starts = 33` on the GPU, as 8.4. The first eight starts must reproduce `workspace/phase8/8.4/full3/starts.json`.
If they do not, the Phase 9 file is kept for them and the sequence continues past Sobol index 10. **New starts:
start8 to start31 (24)**. N is fixed at 24 in advance. If compute fails, the analysis uses the first N ≥ 16 in Sobol
order with both arms complete, and the shortfall is reported. There is no interim analysis of the primary endpoint.

**Arms**, exactly as Phase 9 (`dkarma/experiments/admissible_baseline.py`), one GPU process per run, run sequentially:
**arm A′** = original KaRMA (`karma_jax` fast proxy, published `L_ref`-scaled task, all-finger gate) + CMA-ES, 96
evaluations, population 8, σ₀ 0.3, CMA seed = start index, D18 throughout. **Full-D18** = Full KaRMA + Adam, 40
iterations, 2 stages, 6 three-contact seeds, `ADAM_DEFAULTS`, learning rate 0.02, optimizer seed 0,
`kkt_method="reduced"`, D18 penalty and gated validation. Compute (wall clock, optimizer time, evaluations) is recorded
per run.

**Primary endpoint** `E(D)`, a common manipulation benchmark measured identically for both arms:

1. *Planner.* The three-contact library of `D` (`simulation_replay export --libraries three_contact --budget 12`, CPU):
   12 fresh three-contact seeds from `find_multi_seeds` under the **default** seed configuration. Per seed, the shared
   fixed-assignment hard-mode rollout of the Full-KaRMA configuration (`CommandConfig(n_cells=4, n_rot_steps=10)`):
   6 translation commands × 4 primitives (8 sub-steps of 1.25 mm each) and 6 rotation commands × 10 primitives. At most
   84 primitives per seed.
2. *Simulation.* The 10.1 central compliant fingertips (Shore 20A, 2 mm) under the **force-floor controller** with its
   9.3 parameters, or as re-tuned by the P10.1 stability rule. Every trajectory runs to its end and stops only at a
   drop.
3. *Count.* Per command, the number of primitives executed successfully from the start of the command up to the
   first failure. Success means object pose (4 mm / 0.2 rad) **and** strict retention (every assigned fingertip in
   contact at the primitive end), which is `n_sim_rot` of `simulate_design`. The count is **regardless of the exact
   verdict**: not capped at the exactly valid prefix. Seeds that fail the exact seed gate or do not hold are counted as
   simulated.
4. `E(D)` = the sum over seeds and commands / 12 (per budgeted seed). A missing seed contributes 0, and a design with
   no three-contact seed has `E = 0`.

**Statistic.** Over the new starts `i`: `Δᵢ = E(Full-D18ᵢ) − E(A′ᵢ)`. The test is the two-sided Wilcoxon signed-rank
test (`scipy.stats.wilcoxon`, defaults) at α = 0.05. Reported with it: the percentile bootstrap 95 % CI of the mean
`Δ` (10,000 resamples of starts, `RandomState(0)`); the sign counts; and mean, median, best and IQR per arm.
**Claim rule:** "Full KaRMA produces better simultaneous-contact manipulation than a collision-valid original-KaRMA
optimum" is claimed **only if p < 0.05 with Full-D18 ahead (more wins than losses)**. If p < 0.05 with arm A′ ahead,
the reverse is reported, and it is attributed to the objective only after a budget-matched control. Otherwise no
claim is made. The **pilot** (Phase 9 starts 0–7, same endpoint computed on the 9.2 libraries) is reported separately.
A pooled 32-start analysis is descriptive only.

**Secondary endpoints** (reported, no separate claims):

* the 9.2 measure: exactly valid primitives that succeed (strict), per budgeted seed, under the same model and
  controller;
* pose-only `E` (`n_pose`), and `E` with duty-cycle retention;
* `E` under rigid contact (force-floor controller);
* `E` under the `sobol512_rs7` seed configuration (pre-registered robustness check; same procedure, same test);
* the 9.4 suite: exact Full score and exact two-finger library score under the three seed configurations (12 / 16
  seeds); the original KaRMA on the physical task (all-finger and pair gates) and on the published task (KaRMA-T,
  all-finger gate).

### P10.3 Robust-grasp subset `S_task`

At each three-contact seed `s` of a seed configuration (`find_multi_seeds`, budget 12), `α_s(d)` is the wrench
capacity at the seed state (`act_karma.wrench_capacity_at`, 14 standard directions, primary ceiling 0.108 N·m, the
35 g object's weight and weight × radius as the force and moment scales). `S_task = {s : min_d α_s(d) ≥ 1}`. Reported per
design and seed configuration (default, `sobol512_rs7`, `lin7_rs13`):

* `|S_task| / |S|` (found seeds) and `|S_task| / 12` (budget), with the per-seed `min_d α_s` distribution;
* the **exact Full score on `S_task`**: the suite's formula (`K_T/0.1 + K_SO3/0.3`, `K` the mean per-seed exact counts)
  with the mean taken over `S_task` only (0 if `S_task` is empty). Per-seed counts come from
  `MC.evaluate_fixed_contacts_exact`, exact SLSQP force check at every sub-step;
* the **10.2 primary endpoint on `S_task`**: the sum over `s ∈ S_task` and commands of the success counts, divided by
  `|S_task|` (per robust seed) and by 12 (per budgeted seed). Default configuration, and `sobol512_rs7` for the
  robustness repeat. Export seeds are matched to `S_task` seeds by their joint configuration.

The all-seeds results are always reported beside the subset results, unchanged. The subset results are never
presented as equivalent to the original criterion. Designs: arm B, the 9.5 candidates, every 10.2 design, the Phase 9
pilot designs and the anthropomorphic start (needed by criterion v2, R3).

### P10.4 Criterion v2 (frozen here; recorded in SPEC §16 item 12 as frozen by the session, subject to the owner's review)

*Deployment requirements it is written from.* The hand holds and rolls the frozen task's object (15 mm / 35 g glass
sphere, μ 0.6) with three fingertips (thumb, index, middle) through the ±100 mm task cube and SO(3). Each joint is an
XL330-M077-T at the primary ceiling of 0.108 N·m. The fingertips are compliant silicone pads (SPEC §12 IV), and the
hardware controller is the 9.3 force-floor admittance controller (position servos plus fingertip normal-force
sensing). On hardware a grasp planner chooses among the hand's available grasps. The hand therefore needs enough
grasps that hold the object against disturbances, from which useful manipulation is possible, and whose exact
predictions transfer to compliant-contact simulation as well as three-contact grasps typically do.

* **R1 Eligibility** (as 9.5 requirement 1): admissible under D18. Exact two-finger library score > 0 and at least one
  three-contact seed under each of the three seed configurations. The exported files reproduce the suite's
  default-configuration numbers.
* **R2 Robust grasps:** `|S_task| ≥ 6` of the 12-seed budget under **every** seed configuration (at least half of the
  sampled grasps resist 1 × weight in all 14 directions at the primary ceiling).
* **R3 Manipulation from robust grasps:** the exact Full score on `S_task`, mean over the three seed configurations, is
  at least that of the anthropomorphic start design (`start0_nominal`) on its own `S_task`, mean over the same three
  configurations. Fabricating an optimized design is justified only if its robust grasps give the planner at least the
  capability of the unoptimized anthropomorphic hand's robust grasps.
* **R4 Load robustness** (as 9.5 requirement 3, second part, unchanged): in the `admissible_baseline sweep` at the
  primary ceiling (default configuration, 12 seeds), at least one seed holds a 70 g object, and the Full score at 70 g
  is ≥ 50 % of the 35 g value.
* **R5 Simulation agreement** against the compliant three-contact reference of P10.1: the candidate's 12-seed
  three-contact library (default configuration) has a strict FP rate, at the central compliant parameters under the
  force-floor controller, that is ≤ the reference value (the median over the 9.2 libraries, P10.1). It must have
  ≥ 10 exactly valid primitives on holding seeds, otherwise R5 fails for insufficient evidence.
* **Decision:** among candidates meeting R1–R5, the highest exact Full score on `S_task` (mean over configurations)
  wins. A difference smaller than the larger seed-configuration standard deviation of the two is a tie, broken by the
  larger mean `|S_task|`. The two-finger cost is stated as in 9.5: against the best two-finger score among all scored
  designs and against the best arm A′. If no candidate meets R1–R5, nothing is selected, the failing measurements are
  reported, and the criterion is not loosened.
* **Candidates:** arm B; Full-D18 start0 and arm A′ start6 (the 9.5 candidates); and the best design of each arm from
  10.2 by the primary endpoint `E`, ties broken by the exact Full score (mean over configurations). Arm A stays the
  failure case.
* **Alongside:** the original §8.7(b) criterion as applied in 9.5 (`candidate_selection.py`, rigid-contact simulation,
  unchanged) on the same candidates. The two verdicts are reported side by side and never presented as equivalent.

### Addenda to the pre-registration (dated; each written before the data it governs)

**A1 (2026-10-14 16:05 EDT; after the P10.1 stability check, before any compliant three-contact replay).** The
two-finger baseline replays of the stability check (`workspace/phase10/10.1/{rigid,20A_2mm}/{9.3,9.2}/`,
`stability.json`) found the **force-floor controller stable** under compliance: the same holding seeds (5/6, 16/16)
and no new drop. Its 9.3 parameters stay. The **position servo is unstable** by the rule. The same seeds hold (5/6,
15/16), but on both libraries a holding seed (seed 5, translation command 2) loses the sphere entirely at its fourth
primitive, where rigid contact lost the pose and one contact but kept the other. The cause is visible in the
two-finger data: the pad absorbs part of the fixed 0.5 mm geometric squeeze, and the median grip force after the
settle falls from 1.28 / 1.00 N (rigid) to 0.98 / 0.79 N. *Re-tuning procedure, two-finger baseline only:* the
position servo's set-point parameter, the commanded squeeze, is chosen from {0.5, 0.75, 1.0, 1.25, 1.5} mm. `kp` is the
servo stiffness and stays at 20 N·m/rad. The chosen value is the smallest for which the stability rule passes on both
baseline libraries at the central parameters. It then applies to every compliant position-servo replay (grid and
diagnostics), while rigid replays keep 0.5 mm. If no value passes, the squeeze stays at 0.5 mm and the compliant
position-servo results are reported as those of an unstable controller. No three-contact data is used.

---

## 10.1 Compliant fingertips in MuJoCo

### Files changed (10.1)

| File | Change |
| --- | --- |
| `dkarma/experiments/compliant_pad.py` (new) | the pad model: Gent's Shore-A modulus, the Hertz + bonded-layer normal law, patch radius, torsional and rolling friction, and the MuJoCo realization (`fit_realization`: `solimp`/direct `solref` fitted to the law with impedance ≥ 0.9, deadbeat `solreffriction`, `PadParams.pair_attrs(A)`), `load_pads`, and `python -m dkarma.experiments.compliant_pad`, which wrote the frozen sets |
| `dkarma/configs/compliant_pads.json` (new) | the frozen parameter sets: the 3 × 3 grid (central 20A / 2 mm) and D2 (damping ratio 0.3), with the assumptions and the MuJoCo / SciPy / NumPy versions |
| `dkarma/experiments/simulation_replay.py` | `mjcf_from_design(pad=, pad_variant=)`: one explicit `<pair>` per terminal capsule and the sphere. Without `pad` the scene is the 8.6/9.3 one string for string, reproducing all 23 Phase 9 scene files. `simulate_design(pad=, pad_variant=, archive=)`; `result_record`; CLI `--pad`, `--pad-file`, `--pad-variant` (`patch`, `rigid_friction` = D1, `patch_friction_only` = post-hoc D3), `--archive`; `export --seed-config` (10.2) |
| `dkarma/experiments/compliant_contact_study.py` (new) | the 10.1 replay driver (`replay`: one process per contact model, archive and controller), the pre-registered stability rule (`stability`), the outcomes, gaps, verdict, agreement reference, rigid-reproduction check and post-hoc tables (`analyze`) |
| `dkarma/tests/test_simulation_replay.py` (new) | 12 tests: rigid scene unchanged (string identity, and against the 9.3 file); a pad changes only the terminal contacts (geometry and inertia identical, 3 pairs, `condim`, patch friction); static law within 6 % for a fixed and a jointed link (the `s = s_u A` scaling); creep below the rigid contact's; the frozen file reproduced by the fit; the outcome, verdict, endpoint and `S_task` functions on synthetic rows |

### Procedure (10.1)

*Model*, as pre-registered (P10.1). The central set is Shore 20A (E = 0.732 MPa) at 2 mm. The grid is
{10A, 20A, 40A} × {1, 2, 3} mm, plus D1 (pad normal law, rigid friction) and D2 (central, damping ratio 0.3).
*Realization checks*, beyond the calibration scene of P10.1: in the hand scenes after the 0.3 s settle, the indentation
of every pad contact carrying more than 0.02 N, divided by the law's indentation at the measured force, has median 0.96 /
0.97 / 0.97 and a 5–95 % range of [0.94, 1.09] (central / 10A 1 mm / 40A 3 mm; 50 contacts each, baseline, start and
Full-D18 start0 seeds; `workspace/phase10/10.1/insitu_static_law.json`).

*Rigid bound.* The rigid replays were re-run with the Phase 10 code. Every first-failure index and every settle drift of
the 42 Phase 9 replays is reproduced exactly; the 9.3 baseline, for instance, matches every measured normal force to
0.0 N.

*Controller stability* (P10.1 rule, two-finger baseline only, addendum A1). The force-floor controller is stable. The
position servo is unstable: one new drop on both baseline libraries. It was re-tuned on the two-finger baseline to a
0.75 mm squeeze, the smallest of {0.5, 0.75, 1.0, 1.25, 1.5} mm that passes on both libraries (`retune.json`). That
squeeze also restores the rigid median grip force (1.28 / 1.00 N on the 16-seed library). All compliant position-servo
replays use 0.75 mm; rigid replays keep 0.5 mm. No three-contact data entered either decision.

*Replays*: 12 contact models (rigid, nine grid points, D1, D2) × the two archive sets × both controllers. That is
48 replay jobs (`workspace/phase10/chain_101.sh`, CPU, six workers, about 20 minutes), plus the post-hoc D3 (4 jobs).
Every trajectory runs to its end and stops only at a drop.

### Results (10.1)

Fixed physical task. Contact model and controller per row. Criteria at each primitive end: *strict* = pose (4 mm /
0.2 rad) and every assigned contact; *duty* = every assigned lane in contact for ≥ 50 % of the primitive; *drop* = centre
> 20 mm from the command or no assigned contact. Rates are over exactly valid primitives of the seeds that hold under that
model and controller (`workspace/phase10/10.1/analysis.{md,json}`).

**The pre-registered rule.** Residual gap = two-finger strict − three-contact strict, same design, model and controller
(9.3 archives, 6 seeds per library):

| contact model | controller | start: 2f / 3c / gap | arm B: 2f / 3c / gap | arm A′ start0: 2f / 3c / gap | verdict |
| --- | --- | --- | --- | --- | --- |
| rigid (bound) | force-floor | 0.77 / 0.28 / +0.49 | 0.65 / 0.52 / +0.13 | 0.67 / 0.14 / +0.53 | – |
| **central 20A / 2 mm** | **force-floor** | 0.73 / 0.47 / **+0.26** | 0.65 / 0.72 / **−0.06** | 0.70 / 0.82 / **−0.12** (1 of 6 seeds holds, 11 primitives) | **partial** |
| rigid (bound) | position | 0.74 / 0.16 / +0.58 | 0.65 / 0.38 / +0.28 | 0.53 / 0.24 / +0.29 | – |
| central 20A / 2 mm | position (0.75 mm) | 0.79 / 0.24 / +0.55 | 0.68 / 0.54 / +0.14 | 0.70 / 0.69 / +0.01 (2 of 6) | partial |

Under the force-floor controller the gap closes on arm B and arm A′ start0. It shrinks on the start from 0.49 to 0.26,
just short of half, so the start alone keeps the verdict from "resolved". **Verdict: partial (headline, force-floor
controller), robust**: partial at all nine grid points under both controllers (`analysis.md`, "Residual gap"). Across the
grid the start's force-controlled gap ranges from +0.16 to +0.37 and arm B's from −0.14 to −0.00; arm A′ start0 rests on
one or two holding seeds at every grid point.

**The 12-seed libraries** (9.2: 8 arm A′, 8 Full-D18, arm B; three-contact; pooled over designs):

| contact model | controller | seeds hold | exactly valid | strict | pose | retention (end) | retention (duty) | pose + duty | no drop |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| rigid | force-floor | 108 / 160 | 2306 | 0.41 | 0.73 | 0.43 | 0.71 | 0.64 | 0.93 |
| central 20A / 2 mm | force-floor | 101 / 160 | 2218 | **0.71** | 0.88 | 0.72 | 0.87 | 0.85 | 0.95 |
| rigid | position | 104 / 160 | 2214 | 0.33 | 0.67 | 0.33 | 0.49 | 0.47 | 0.89 |
| central 20A / 2 mm | position (0.75 mm) | 91 / 160 | 1973 | 0.70 | 0.84 | 0.71 | 0.77 | 0.76 | 0.97 |
| D1: pad law, rigid friction | force-floor | 109 / 160 | 2323 | 0.45 | 0.77 | 0.46 | 0.72 | 0.67 | 0.95 |
| D2: central, damping 0.3 | force-floor | 100 / 160 | 2187 | 0.70 | 0.88 | 0.71 | 0.87 | 0.85 | 0.96 |
| grid range (9 sets) | force-floor | 101 / 160 | 2218 | 0.69–0.72 | 0.87–0.89 | 0.69–0.73 | 0.86–0.90 | – | – |

The two-finger baseline (16 seeds) is unchanged by compliance: strict 0.82 → 0.82 (force-floor) and 0.82 → 0.87 (position
servo, 0.75 mm). Compliance costs holding seeds in three-contact grasps (108 → 101 force-floor, 104 → 91 position servo).
The rates above are conditional on holding, so the seed-independent confusion matrices are reported too:

| design (9.3 / 9.2 archive) | library | controller | rigid: exactly valid succeed / fail | central: exactly valid succeed / fail | rigid: exactly rejected succeed / fail | central: exactly rejected succeed / fail |
| --- | --- | --- | --- | --- | --- | --- |
| baseline (9.3) | two-finger | force-floor | 101 / 15 | 96 / 20 | 16 / 39 | 13 / 42 |
| start (9.3) | three-contact | force-floor | 38 / 141 | 64 / 115 | 10 / 54 | 11 / 53 |
| arm B (9.3) | three-contact | force-floor | 54 / 77 | 74 / 57 | 11 / 56 | 17 / 50 |
| arm A′ start0 (9.3) | three-contact | force-floor | 6 / 74 | 9 / 71 | 5 / 67 | 6 / 66 |
| Full-D18 start0 (9.2) | three-contact | force-floor | 97 / 215 | 236 / 76 | 20 / 108 | 42 / 86 |
| arm A′ start6 (9.2) | three-contact | force-floor | 61 / 84 | 90 / 55 | 23 / 121 | 35 / 109 |
| arm B (9.2) | three-contact | force-floor | 97 / 146 | 150 / 93 | 22 / 113 | 37 / 98 |
| baseline (9.2) | two-finger | force-floor | 274 / 61 | 275 / 60 | 62 / 84 | 50 / 96 |

(8.6 convention: all seeds, per primitive up to the first exactly rejected one; criterion: pose and every contact at the
primitive end.) Compliance turns many exactly valid three-contact primitives from failures into successes (Full-D18
start0: 97 → 236 of 312). It also lets more exactly *rejected* first primitives succeed (20 → 42): with soft contacts the
simulation executes some motions the exact model rejects.

**What the remaining strict failures are.** On the start's three-contact library under the force-floor controller,
compliance raises pose success to 1.00 and duty-cycle retention to 0.99, while end-of-primitive retention stays at 0.47.
Every remaining strict failure there is a fingertip that is out of contact at the instant of the check while it touched
the sphere for most of the primitive: sub-millimetre chatter, not lost manipulation. Pooled over the 12-seed libraries,
pose + duty-cycle success is 0.85 against strict 0.71.

**Post hoc** (not pre-registered; after the results): normal law × friction model. D1 suggested the gain is mostly not
from normal compliance, so the missing cell was run: D3, the rigid normal law with the patch's friction (`condim 6`,
torsion 0.32 mm, rolling 34 µm). Strict three-contact success on the 12-seed libraries, pooled, by primitive kind
(rotations are 87 % of the exactly valid primitives):

| controller | kind | rigid normal, rigid friction | pad normal, rigid friction (D1) | rigid normal, patch friction (D3) | pad normal, patch friction (central) |
| --- | --- | --- | --- | --- | --- |
| force-floor | rotation | 0.39 | 0.42 | 0.64 | 0.70 |
| force-floor | translation | 0.56 | 0.61 | 0.68 | 0.77 |
| position servo | rotation | 0.31 | 0.44 | 0.50 | 0.70 |
| position servo | translation | 0.44 | 0.56 | 0.51 | 0.70 |

Under the force-floor controller most of the three-contact gain comes from the friction model. The rigid scene of 8.6/9.3
assumed a 3 mm patch, torsional friction 1.8 mm, which is larger than either a rigid capsule's or a silicone pad's patch
(0.62–1.12 mm radius at 0.26 N). Normal compliance adds a smaller part. Under the position servo both matter, and they
interact strongly. A plausible reading, not tested here: an over-large torsional friction makes three point-like
fingertips fight each other in spin, which a rolling model with free spin does not ask them to do. Part of the 8.6/9.3
three-contact deficit was therefore an artefact of the rigid scene's friction assumption.

**Compliant three-contact agreement reference** (the second deliverable). Strict FP rate at the central parameters,
force-floor controller, per 12-seed library: Full-D18 0.10–0.62, arm A′ 0.15–0.68, arm B 0.23 (per design in
`analysis.md`). **Reference for criterion v2 (P10.1, frozen rule): the median over the 13 9.2 libraries with ≥ 10 exactly
valid primitives on holding seeds = 0.24.** Over all three-contact libraries: median 0.24, quartiles [0.18, 0.52],
pooled 0.30. For comparison, the compliant two-finger baseline FP is 0.18 (16 seeds). Under the position servo
(0.75 mm) the median is 0.28.

### Acceptance (10.1)

Separated outcomes and confusion matrices under both controllers at the central parameters and over the 3 × 3 grid,
with a verdict by the pre-registered rule: **met**. The verdict is **partial, robust over the grid**, under both
controllers. The compliant three-contact agreement reference is measured: 0.24, force-floor median.

*Tests at the end of 10.1* (2026-10-14): `dkarma/tests` 205 passed, 6 skipped (CPU, 28 min; Phase 9's 193 plus the 12
of `test_simulation_replay.py`; `workspace/phase10/dkarma_tests_10.1.log`). `karma_jax/tests` 101 passed, 1 skipped
(GPU, 3.6 min, between two 10.2 optimizer runs; `workspace/phase10/karma_jax_tests_10.1.log`). `karma/` and
`karma_jax/` are not modified in Phase 10.

### Limitations (10.1)

* The material parameters are assumptions (Gent's relation, ν = 0.5, a bonded layer on a rigid core, hysteresis loss 0.2,
  μ 0.6 for silicone on glass, which is likely low). The grid spans a factor of 4 in modulus and 3 in thickness, and the
  verdict does not change over it. Measuring fabricated fingertips replaces these.
* The MuJoCo realization reproduces the static law (in situ within about ±6 %) but not the pad's dynamics. The impedance
  needed to keep the friction creep at the rigid contact's level inflates MuJoCo's apparent mass along the normal: a
  0.3 N step settles in 44–78 ms, against 13 ms rigid. The damping diagnostic D2 (ζ 0.3) changes nothing (0.70 against
  0.71).
* The rule's three designs rest on 6-seed archives. Arm A′ start0's three-contact library holds on only 1 of 6 seeds
  under compliant force control, so its gap rests on 11 primitives.
* Compliance lowers the number of holding three-contact seeds (101 against 108, force-floor). The holding-seed rates
  are conditional; the confusion matrices and the 10.2 endpoint are not.
* The position servo was re-tuned (0.75 mm squeeze) on two-finger data by the pre-registered procedure. Its compliant
  results use a different set-point than its rigid ones.
* The decomposition into friction and normal compliance (D3) is post hoc.

### How to reproduce (10.1)

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_pad          # refits and rewrites dkarma/configs/compliant_pads.json (identical)
# stability check (two-finger baseline, rigid and central, 9.3 squeeze 0.5 mm; then A1 re-tune candidates)
for M in rigid 20A_2mm; do for S in 9.3 9.2; do for C in position force; do
  JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study replay --model $M --src $S --controller $C --designs baseline --squeeze-mm 0.5; done; done; done
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study stability
for Q in 0.75 1.0 1.25 1.5; do for S in 9.3 9.2; do JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study replay --model 20A_2mm --src $S --controller position --designs baseline --squeeze-mm $Q --root workspace/phase10/10.1/retune/squeeze$Q; done; done
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study retune   # writes retune.json (A1 rule); the 0.5 mm central baseline runs live in retune/squeeze0.5
workspace/phase10/chain_101.sh                                                    # the replay matrix and analyze (CPU, ~20 min with 6 workers)
for S in 9.3 9.2; do for C in force position; do JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study replay --model D3_posthoc_patch_friction --src $S --controller $C --squeeze-mm 0.5; done; done
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.compliant_contact_study analyze
```

---

## 10.2 Arm A′ vs Full-D18 on 24 new starts

### Files changed (10.2)

| File | Change |
| --- | --- |
| `dkarma/experiments/admissible_baseline.py` | `starts` stage, which runs the 8.4 start procedure to `--n-starts` starts in `--starts-dir` and checks that the first eight reproduce the 8.4 file; `--starts-dir` / `--n-starts` for every stage; `suite --per-seed-detail`, which also records each three-contact seed's state and its 14 wrench capacities under every seed configuration (10.3). Phase 9 behaviour is unchanged by default |
| `dkarma/experiments/simulation_replay.py` | `export --seed-config` (the `sobol512_rs7` robustness repeat) |
| `dkarma/experiments/powered_comparison.py` (new) | the pre-registered endpoint `E` (`endpoint`, `per_seed_counts`), the secondaries, the paired statistics and the frozen decision; the pilot; and a post-hoc decomposition (seeds found and holding, subsets) |
| `workspace/phase10/chain_102_gpu.sh`, `chain_102_cpu.sh` | the GPU leg (starts, then arm A′ and Full-D18 per start, one process at a time) and the CPU leg (two exports, six replays and the suite per design; three workers, RSS watchdog and memory gate) |

### Procedure (10.2)

*Starts* (P10.2). `admissible_baseline starts --n-starts 33` drew Sobol points 0–42 (`workspace/phase10/10.2/starts.json`).
The eight Phase 9 starts are reproduced exactly (`starts_check.json`). The pre-registered new starts are start8–start31:
6 at scale 0.5 (Sobol points 11 and 13–17) and 18 at scale 0.25 (points 20–41, the coded default for points ≥ 20).
Points 12, 18, 19, 24, 34, 36 and 38 were rejected for fewer than 3 three-contact seeds. start32 was drawn and not used.

*Arms* exactly as Phase 9: arm A′ with CMA seed = start index; Full-D18 with optimizer seed 0 and `kkt_method="reduced"`.
Both on the GPU, one process at a time. *Planner and endpoint* as frozen: three-contact exports with 12 fresh seeds
(default and `sobol512_rs7` configurations), replays under the central pad and the force-floor controller (secondaries:
rigid contact; position servo), and the 9.4 suite. Every design's export seeds equal its suite's default-configuration
seeds, in the same order (checked on arm A′ start8 under both configurations).

*Provenance of the runs.* GPU leg 2026-10-14 15:58 to 2026-10-15 15:27, then 2026-10-22 16:47–20:07; CPU leg
2026-10-14 16:37 to 2026-10-22 23:23. Two interruptions:

1. Paused (SIGSTOP) on 2026-10-15 at 15:27, then stopped (SIGKILL) at 17:07 on the owner's request
   (`workspace/phase10/STOPPED_2026-10-15.md`). In flight were arm A′ start31 (generation 5 of 12) and three exports. None
   of them had written output; they were rerun from scratch on 2026-10-22.
2. On 2026-10-22 at about 17:22 the shared root filesystem reached 100 % (data outside this project). Full-D18 start31
   crashed in its iteration-30 validation, because XLA could not write a PTX file to `/tmp`. No output was written. The CPU
   jobs were suspended at 17:26. Every file written since the resume was checked to load. On the owner's choice, the
   project's XLA compile cache (`~/.cache/karma_jax_xla`, 15 GB of `jit_*` entries) was deleted; the GPU autotune cache
   was kept. Full-D18 start31 was then rerun from scratch at 19:42.

The GPU was shared at times with other users' jobs: a torch job on 2026-10-14 and six processes on the evening of
2026-10-22. Wall clocks include that contention. The Full-D18 start31 rerun is the outlier: 1,466 s wall and 363 s of
optimizer time against a median of 68 s, because of the contention and the recompilation after the cache deletion.

### Results (10.2)

Fixed physical task; the common benchmark of P10.2. Files: `workspace/phase10/10.2/comparison.{md,json}` (the endpoint,
`powered_comparison.py`) and `summary.{md,json}` (the suites, `admissible_baseline summary`). Designs:
`workspace/phase10/10.2/{armAprime,full_d18}/<start>/optimized_morphology/optimized_morphology.design.json`. Columns:
three-contact seeds found / holding (default configuration, 12-seed budget); the primary endpoint `E` (compliant 20A/2 mm,
force-floor controller, strict success counted to the first failure regardless of the exact verdict, per budgeted seed);
`E` under rigid contact; `E` under `sobol512_rs7`; and the exact Full score (mean over three seed configurations).

| start | A′ seeds found / hold | Full seeds found / hold | A′ E | Full E | Full − A′ | rigid E: A′ / Full | E (`sobol512_rs7`): A′ / Full | exact Full score: A′ / Full |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| start8_sobol11_s0.5 | 10 / 9 | 12 / 10 | 13.58 | 25.25 | +11.67 | 5.92 / 19.50 | 10.75 / 26.58 | 0.0089 / 0.0188 |
| start9_sobol13_s0.5 | 12 / 0 | 12 / 10 | 0.00 | 30.00 | +30.00 | 0.25 / 22.00 | 3.08 / 32.75 | 0.0079 / 0.0171 |
| start10_sobol14_s0.5 | 0 / 0 | 8 / 5 | 0.00 | 9.75 | +9.75 | 0.00 / 6.42 | 0.00 / 10.17 | 0.0000 / 0.0162 |
| start11_sobol15_s0.5 | 0 / 0 | 12 / 7 | 0.00 | 7.83 | +7.83 | 0.00 / 3.67 | 0.00 / 12.33 | 0.0000 / 0.0118 |
| start12_sobol16_s0.5 | 0 / 0 | 12 / 12 | 0.00 | 32.08 | +32.08 | 0.00 / 26.00 | 3.00 / 32.75 | 0.0070 / 0.0177 |
| start13_sobol17_s0.5 | 12 / 7 | 12 / 12 | 12.00 | 33.17 | +21.17 | 8.83 / 20.75 | 22.33 / 39.83 | 0.0110 / 0.0236 |
| start14_sobol20_s0.25 | 0 / 0 | 12 / 12 | 0.00 | 35.58 | +35.58 | 0.00 / 21.33 | 0.00 / 30.00 | 0.0000 / 0.0189 |
| start15_sobol21_s0.25 | 12 / 8 | 12 / 10 | 17.92 | 35.67 | +17.75 | 12.25 / 16.00 | 25.83 / 26.33 | 0.0182 / 0.0251 |
| start16_sobol22_s0.25 | 0 / 0 | 12 / 12 | 0.00 | 40.17 | +40.17 | 0.00 / 26.50 | 0.00 / 20.42 | 0.0000 / 0.0187 |
| start17_sobol23_s0.25 | 12 / 2 | 12 / 9 | 3.75 | 29.08 | +25.33 | 3.25 / 15.00 | 5.75 / 18.50 | 0.0058 / 0.0177 |
| start18_sobol25_s0.25 | 0 / 0 | 12 / 11 | 0.00 | 34.67 | +34.67 | 0.00 / 22.75 | 0.00 / 37.67 | 0.0000 / 0.0239 |
| start19_sobol26_s0.25 | 12 / 0 | 12 / 10 | 0.00 | 31.83 | +31.83 | 0.08 / 18.92 | 0.00 / 21.42 | 0.0073 / 0.0211 |
| start20_sobol27_s0.25 | 0 / 0 | 12 / 7 | 0.00 | 20.08 | +20.08 | 0.00 / 9.83 | 0.00 / 21.50 | 0.0000 / 0.0157 |
| start21_sobol28_s0.25 | 12 / 1 | 12 / 11 | 0.50 | 40.67 | +40.17 | 0.33 / 24.92 | 3.08 / 39.83 | 0.0051 / 0.0193 |
| start22_sobol29_s0.25 | 12 / 9 | 12 / 9 | 14.83 | 15.58 | +0.75 | 8.50 / 8.75 | 28.42 / 22.00 | 0.0119 / 0.0205 |
| start23_sobol30_s0.25 | 8 / 1 | 12 / 6 | 0.17 | 8.83 | +8.67 | 0.17 / 9.00 | 0.00 / 9.58 | 0.0064 / 0.0187 |
| start24_sobol31_s0.25 | 12 / 11 | 12 / 7 | 13.25 | 16.75 | +3.50 | 6.50 / 11.17 | 19.50 / 23.33 | 0.0061 / 0.0129 |
| start25_sobol32_s0.25 | 0 / 0 | 12 / 10 | 0.00 | 32.92 | +32.92 | 0.00 / 17.58 | 0.00 / 31.67 | 0.0000 / 0.0201 |
| start26_sobol33_s0.25 | 12 / 12 | 12 / 10 | 42.08 | 23.58 | -18.50 | 32.42 / 14.25 | 35.50 / 20.58 | 0.0141 / 0.0156 |
| start27_sobol35_s0.25 | 2 / 1 | 12 / 5 | 1.00 | 14.17 | +13.17 | 0.83 / 7.33 | 1.00 / 13.92 | 0.0061 / 0.0161 |
| start28_sobol37_s0.25 | 0 / 0 | 12 / 9 | 0.00 | 19.92 | +19.92 | 0.00 / 13.92 | 0.00 / 30.42 | 0.0000 / 0.0163 |
| start29_sobol39_s0.25 | 12 / 4 | 12 / 9 | 8.42 | 35.17 | +26.75 | 3.42 / 23.67 | 11.33 / 39.33 | 0.0074 / 0.0230 |
| start30_sobol40_s0.25 | 12 / 12 | 12 / 7 | 24.17 | 20.33 | -3.83 | 12.17 / 11.75 | 22.92 / 20.50 | 0.0121 / 0.0156 |
| start31_sobol41_s0.25 | 1 / 1 | 12 / 12 | 1.17 | 45.00 | +43.83 | 0.83 / 26.83 | 1.17 / 45.00 | 0.0041 / 0.0237 |

**Primary endpoint and the frozen decision** (24 new starts, the pre-registered set; no interim analysis was run):

| | arm A′ | Full-D18 |
| --- | --- | --- |
| E: mean / median / best | 6.37 / 0.33 / 42.08 | 26.59 / 29.54 / 45.00 |
| E: IQR | [0.00, 12.31] | [19.12, 34.79] |

Full-D18 − arm A′: **22 wins / 0 ties / 2 losses**, mean difference **+20.22 primitives per budgeted seed (bootstrap 95 %
CI [+14.10, +26.28])**, median +20.62. **Two-sided Wilcoxon p = 9.1 × 10⁻⁵**; sign test p = 3.6 × 10⁻⁵. **By the frozen
rule (p < 0.05, Full-D18 ahead) the claim is made: on a common, mechanically admissible design space, Full KaRMA produces
better simultaneous-contact manipulation than a collision-valid original-KaRMA optimum.** Each arm is the method as
specified: Full KaRMA + Adam against original KaRMA + CMA-ES, at their Phase 9 budgets.

**Secondary endpoints** (pre-registered, no separate claims; Full − A′ over the 24 new starts):

| measure | wins / losses | mean difference [95 % CI] | Wilcoxon p |
| --- | --- | --- | --- |
| 9.2 measure: exactly valid primitives succeeding (strict), same model and controller | 22 / 2 | +14.40 [+10.59, +18.14] | 2.3 × 10⁻⁶ |
| pose-only E | 22 / 2 | +26.70 [+19.13, +34.10] | 6.0 × 10⁻⁵ |
| E with duty-cycle retention | 22 / 2 | +24.28 [+17.27, +31.15] | 7.1 × 10⁻⁵ |
| E under rigid contact (force-floor) | 22 / 2 | +12.59 [+8.30, +16.66] | 1.4 × 10⁻⁴ |
| **E under `sobol512_rs7` (pre-registered robustness check)** | **21 / 3** | **+18.03 [+12.21, +23.75]** | **2.0 × 10⁻⁵** |
| E, compliant, position servo (0.75 mm) [descriptive] | 21 / 3 | +18.03 [+12.01, +24.13] | 1.6 × 10⁻⁴ |
| exact Full score (9.4 suite, mean over 3 seed configurations) | 24 / 0 | +0.0129 [+0.0107, +0.0150] (A′ 0.0058, Full 0.0187) | 1.2 × 10⁻⁷ |
| exact Full score per seed configuration: default / `sobol512_rs7` / `lin7_rs13` | 24/0, 24/0, 24/0 | +0.0131 / +0.0121 / +0.0134 | < 10⁻⁴ each |
| exact two-finger library score | 15 / 9 | +0.0052 [−0.0241, +0.0348] (A′ 0.2057, Full 0.2109) | 0.66 |
| original KaRMA, *published* `L_ref`-scaled task, all-finger gate (KaRMA-T; arm A′'s objective) | 4 / 20 | mean A′ 0.0329, Full 0.0194 | – |
| original KaRMA, physical task, all-finger gate (cube fraction) | 10 / 14 | mean A′ 0.0188, Full 0.0159 | – |

Ranking stability of the 48 designs over the three seed configurations (exact Full score): Spearman ρ 0.946–0.956;
the best design is Full-D18 start15 under two configurations and Full-D18 start31 under the third.

*Pilot* (Phase 9 starts 0–7, same endpoint on the 9.2 libraries; not part of the test): Full − A′ 5 / 3, +6.58
[−1.32, +15.05], p = 0.31. *Pooled 32 starts* (descriptive only): 27 / 5, +16.81 [+11.27, +22.08].

*Compute* (wall clock per run on the shared GPU): arm A′ 2,739 s mean [1,587, 3,713] for 96 proxy evaluations, with 1–14
inadmissible proposals per run scored zero. Full-D18 804 s mean [557, 1,466] for 40 Adam iterations, of which optimizer
time is 68 s median (85 s mean, the start31 rerun included). Every returned design is admissible, and no Full-D18 run
had a non-finite gradient iteration. The compute asymmetry favours arm A′: 3.4× the wall clock and about 40× the
optimizer time.

**Post hoc** (after the frozen decision; not part of it): where the difference comes from.

* On 9 of 24 new starts arm A′ returns a design for which the seed search found **no three-contact seed** under the
  default configuration. These are start10, 11, 12, 14, 16, 18, 20, 25 and 28; all but start12 find none under the other
  two configurations either. Attempt counts, default configuration: 381 joint samples, of which 0 (start10, 14, 16, 18,
  28), 1 (start25), 3 (start11), 7 (start12) or 40 (start20) bring the three fingertips around a sphere; that gave 0–120
  IK attempts with none kinematically valid (`suite.json`, `seed_stats_3`; all three configurations listed there). This
  is what the procedure found, not a proof of non-existence (rule 13). It is the pre-registered `E = 0`. Full-D18 finds 12
  seeds on 23 of 24 starts and 8 on the 24th.
* On two more starts (start9, start19) arm A′'s 12 seeds exist but none holds in simulation.
* Restricted to the 15 new starts where both arms have three-contact seeds, Full-D18 is ahead on 13 of 15 (+16.82
  [+8.33, +25.15], p = 0.0034). Restricted to the 13 starts where both arms have a seed that holds, it is ahead on 11 of
  13 (+14.65 [+5.62, +23.78], p = 0.013). The advantage is therefore not only the total losses of the original-KaRMA
  objective's third finger.
* Arm A′'s three-contact capability on these starts is lower than on the pilot: exact Full 0.0058 against 0.0106. The
  pilot had one no-seed arm A′ design in 8 (start2). The start scale does not explain the difference: the no-seed designs
  are 3 of the 6 scale-0.5 starts and 6 of the 18 scale-0.25 starts. The cause was not investigated further.
* Arm A′'s two-finger advantage of Phase 9 (7 of 8, p = 0.016) does not recur on the new starts (15 / 9 for Full-D18,
  p = 0.66), while arm A′ still wins its own objective (published KaRMA-T, 20 of 24).

### Acceptance (10.2)

The paired comparison on 24 new starts (≥ 16), with the pre-registered primary endpoint, its CI and the frozen decision
applied: **met**. The decision is **claim made**: p = 9.1 × 10⁻⁵, Full-D18 ahead on 22 of 24. The secondary endpoints and
the `sobol512_rs7` robustness check agree in direction and significance. The pilot is reported separately.

*Tests at the end of 10.2* (2026-10-23): `dkarma/tests` 205 passed, 6 skipped (CPU, 40 min under contention;
`workspace/phase10/dkarma_tests_10.2.log`). `karma_jax/tests` 101 passed, 1 skipped (GPU shared with other users' jobs,
so run with `XLA_PYTHON_CLIENT_PREALLOCATE=false`; `karma_jax_tests_10.2.log`).

### Limitations (10.2)

* The arms differ in objective and optimizer (Experiment IV design), so the claim is about the two methods as
  specified, not about the objective alone. The compute asymmetry favours arm A′, which makes the Full-KaRMA win
  conservative.
* The endpoint is simulated (MuJoCo, the assumed compliant pad of 10.1, one controller). It is not hardware.
* Nine of arm A′'s zeros come from designs without a found three-contact seed. That is the outcome of its objective,
  but it makes `E` partly a test of whether the third finger stays in play. The subset analyses address this post hoc.
* The start set's composition (18 of 24 at scale 0.25) follows the coded 8.4 procedure. The pilot's effect was smaller,
  and arm A′ lost the third finger more often on the new starts, for reasons not established. The claim applies to starts
  drawn this way.
* The runs were interrupted twice (above). The interrupted runs were rerun from scratch, and no completed output was
  touched. The wall clocks include GPU sharing with other users.

### How to reproduce (10.2)

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline starts --out workspace/phase10/10.2 --starts-dir workspace/phase10/10.2 --n-starts 33   # GPU
workspace/phase10/chain_102_gpu.sh 8 31        # arm A' and Full-D18 per new start (GPU, ~1 h per start)
workspace/phase10/chain_102_cpu.sh 8 31        # exports, replays, suites (CPU, ~1.5 h per design, 3 workers)
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.powered_comparison --out workspace/phase10/10.2
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline summary --out workspace/phase10/10.2 --starts-dir workspace/phase10/10.2 --n-starts 33
```

---

## 10.3 Robust-grasp subset `S_task`

### Files changed (10.3)

| File | Change |
| --- | --- |
| `dkarma/experiments/robust_grasps.py` (new) | `stask`: for one design and the three seed configurations, the three-contact seeds, their 14 wrench capacities and their per-seed exact Full counts (`MC.evaluate_fixed_contacts_exact`), checked against the design's suite. `report`: `S_task` proportions, capacity distributions, the exact Full score on `S_task` beside the all-seeds score, the 10.2 endpoint on `S_task` (export seeds matched to suite seeds by joint configuration), and a post-hoc holding table |
| `dkarma/experiments/admissible_baseline.py` | `suite --per-seed-detail` (listed in 10.2): the 10.2 suites hold the seeds' states and capacities under every configuration, so `S_task` comes from the same seeds as the suites' counts |

### Procedure (10.3)

As pre-registered (P10.3): `S_task = {s : min_d α_s(d) ≥ 1}`, with `α_s(d)` the wrench capacity at the seed state in the
14 standard directions, at the primary ceiling of 0.108 N·m and the 35 g weight scale. The `stask` records of arm B, the
anthropomorphic start and the 16 Phase 9 designs (`workspace/phase10/10.3/stask/`, CPU, 2–8 min each) recompute the
seeds. They reproduce the stored suites exactly: every per-seed cell and bin count under every configuration where the
suite stored them, and the default-configuration capacities bit for bit. The 48 10.2 designs use their suites'
per-seed detail. The export seeds of all 66 designs matched their suite seeds one to one. Files:
`workspace/phase10/10.3/stask_report.{md,json}` (`robust_grasps report`).

### Results (10.3)

Fixed physical task, primary ceiling. **The subset results are not equivalent to the original criterion.** The
all-seeds results are reported beside them, unchanged.

| design | `\|S_task\|` / seeds found: default, `sobol512_rs7`, `lin7_rs13` | exact Full, all seeds (mean of 3 configs) | **exact Full, `S_task` only** (mean ± sd of 3 configs) | E, all seeds (default) | E on `S_task`: per robust seed / per budgeted seed |
| --- | --- | --- | --- | --- | --- |
| anthropomorphic start | 6/12, 9/12, 8/12 | 0.0174 | 0.0196 ± 0.0016 | – | – |
| arm B | 9/12, 10/12, 10/12 | 0.0146 | 0.0164 ± 0.0017 | 17.50 | 23.33 / 17.50 |
| Full-D18 start0 (9.5 candidate) | 10/12, 11/12, 10/12 | 0.0215 | 0.0241 ± 0.0028 | 27.17 | 32.60 / 27.17 |
| arm A′ start6 (9.5 candidate) | 11/12, 12/12, 12/12 | 0.0093 | 0.0093 ± 0.0010 | 15.00 | 16.36 / 15.00 |
| arm A′ start26 (best arm A′ of 10.2 by E) | 12/12, 12/12, 12/12 | 0.0141 | 0.0141 ± 0.0006 | 42.08 | 42.08 / 42.08 |
| Full-D18 start31 (best Full-D18 of 10.2 by E) | 12/12, 11/12, 12/12 | 0.0237 | 0.0241 ± 0.0024 | 45.00 | 45.00 / 45.00 |

Over the 24 new starts per arm (mean over designs and configurations):

| arm | mean `\|S_task\|` (of 12) | `\|S_task\|` / found | exact Full, all seeds | exact Full, `S_task` | designs with `\|S_task\|` ≥ 6 under every configuration |
| --- | --- | --- | --- | --- | --- |
| arm A′ | 4.78 | 0.76 | 0.0058 | 0.0063 | 8 of 24 |
| Full-D18 | 9.74 | 0.83 | 0.0187 | 0.0197 | 22 of 24 |

Restricting to `S_task` changes a design's exact Full score by −0.0007 to +0.0043 (56 designs with a nonzero score), usually
upward. The seeds it removes mostly fail the exact seed gate and score zero. It does not change the ordering of the
candidates. Every design and configuration,
with the per-seed `min_d α_s` distributions, is in `stask_report.md`.

**Post hoc** (not pre-registered): `S_task` membership against holding in simulation (central pad, force-floor, default
configuration, all 597 replayed seeds of the 66 designs). 392 of 467 robust seeds hold (84 %), against 9 of 130 others
(7 %). 11,889 of the 11,983 successful simulated primitives (99.2 %) are on `S_task` seeds. The exact wrench-capacity
test separates the grasps that the compliant simulation can execute from those it cannot. On every design, the 10.2
endpoint on `S_task` per budgeted seed is within 3.2 of the all-seeds value, and equal to it on 48 of the 54 replayed
designs that have seeds.

### Acceptance (10.3)

`S_task` proportions per seed configuration, capacity distributions, and the subset exact Full score and subset 10.2
endpoint for every listed design (the 9.5 candidates, arm B, every 10.2 design, the pilot designs and the
anthropomorphic start), beside the unchanged all-seeds numbers: **met**.

*Tests at the end of 10.3*: `dkarma/tests/test_simulation_replay.py` (which covers `robust_grasps.subset_measures` and
`seed_score`) 12 passed after the last change to `robust_grasps.py`; the full suites are rerun at the end of 10.4.

### Limitations (10.3)

* `S_task` uses one threshold (1 × weight in all 14 directions) at the primary ceiling. Its sensitivity to the threshold
  was not studied.
* The subset scores reuse the all-seeds formula on fewer seeds. No union-over-seeds coverage was computed (that would be
  a new formulation, SPEC §11.3).
* The 10.2 endpoint on `S_task` is available under the default configuration (and `sobol512_rs7` for the new designs)
  only, because the replays use those exports.

### How to reproduce (10.3)

```bash
for K in armB start/start0_nominal armAprime/start0_nominal ...; do   # the Phase 9 designs (see workspace/phase10/chain_103_p9.sh)
  JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.robust_grasps stask --design $K; done
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.robust_grasps report    # 10.2 designs from their suites' per-seed detail
```

---

## 10.4 Criterion v2 and re-selection

### Files changed (10.4)

| File | Change |
| --- | --- |
| `dkarma/experiments/selection_v2.py` (new) | criterion v2 exactly as frozen in P10.4 (R1–R5, decision, tie-break), the candidates by the frozen rule, and the original 9.5 criterion applied alongside through `candidate_selection`'s unchanged functions (rigid-contact replays from 9.3, 9.2 and 10.2). Also the two-finger cost, the export of the selected specification with its re-score (`candidate_selection.export_candidate`), and `phase10_numbers.json` beside the export |
| `results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25/` (new) | the selected specification: design JSON, URDF, robot YAML, actuator specification, run record and config; `selection_numbers.json` with the export check; `phase10_numbers.json` |
| `results/fabrication_candidates/README.md` | the Phase 10 outcome |
| `SPEC.md` §16 item 12 | criterion v2, recorded before the data (2026-10-14) as frozen by the session, subject to the owner's review |

### Procedure (10.4)

Criterion v2 and the candidate rule were frozen on 2026-10-14 at 15:50, before any Phase 10 data (P10.4). Its numeric
references were computed by their frozen rules:

* R3 reference: the anthropomorphic start's exact Full score on its own `S_task`, mean over the three configurations,
  **0.0196** (10.3).
* R5 reference: the median strict FP over the 13 eligible 9.2 libraries, central pad, force-floor controller, **0.2393**
  (10.1).

The candidates by the frozen rule are arm B, Full-D18 start0 and arm A′ start6 (the 9.5 candidates), plus the best design
of each arm from 10.2 by `E`: **arm A′ start26** (E 42.08) and **Full-D18 start31** (E 45.00), with no ties. Their load
sweeps (`admissible_baseline sweep`: 12 default-configuration seeds; stall, primary and continuous ceilings; 70 g) were
run for R4 and for the original criterion (`workspace/phase10/10.2/sweeps/`). Script:
`selection_v2 --out workspace/phase10/10.4 --export results/fabrication_candidates --rescore`; output
`workspace/phase10/10.4/selection_v2.{md,json}`.

### Results (10.4)

Fixed physical task. Criterion v2 and the original 9.5 criterion side by side (**not equivalent**):

| candidate | R1 | R2: `\|S_task\|` (default / `sobol512_rs7` / `lin7_rs13`) | R3: exact Full on `S_task` (≥ 0.0196) | R4: 70 g (seeds holding), retention | R5: compliant FP, force-floor (≤ 0.239; exactly valid) | **criterion v2** | original 9.5: req. 1 / 3 (min capacity) / 4 (rigid FP, position / force, vs 0.19 / 0.18) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| arm B | yes | 9 / 10 / 10 ✓ | 0.0164 ✗ | 9/12, 0.87 ✓ | 0.23 (194) ✓ | fails (R3) | yes / ✗ (0.00) / ✗ (0.63 / 0.50) |
| Full-D18 start0 | yes | 10 / 11 / 10 ✓ | 0.0241 ✓ | 10/12, 0.93 ✓ | 0.21 (297) ✓ | **passes** | yes / ✗ (0.00) / ✗ (0.78 / 0.67) |
| arm A′ start6 | yes | 11 / 12 / 12 ✓ | 0.0093 ✗ | 11/12, 0.93 ✓ | 0.15 (106) ✓ | fails (R3) | yes / ✗ (0.16) / ✗ (0.63 / 0.54) |
| arm A′ start26 | yes | 12 / 12 / 12 ✓ | 0.0141 ✗ | 12/12, 1.00 ✓ | 0.06 (269) ✓ | fails (R3) | yes / ✓ (4.83) / ✗ (0.36 / 0.22) |
| **Full-D18 start31** | yes | 12 / 11 / 12 ✓ | 0.0241 ✓ | 12/12, 1.00 ✓ | 0.07 (386) ✓ | **passes, selected** | yes / ✓ (1.92) / ✗ (0.49 / 0.44) |
| arm A (failure case) | no (17.8 mm bases) | – | – | – | – | ineligible | ineligible |

**Criterion v2 decision: Full-D18 start31 is selected.** Two candidates pass R1–R5. Their `S_task` exact Full scores
differ by 9 × 10⁻⁶ (0.02411 and 0.02410), far inside the larger seed-configuration standard deviation (0.0028), so the
frozen tie-break applies: the larger mean `|S_task|`, 11.67 for start31 against 10.33 for start0. **Original criterion:
nothing selected.** Every candidate fails requirement 4, which compares rigid-contact three-contact agreement with the
two-finger baseline's. Three also fail requirement 3. Arm A′ start26 and Full-D18 start31 are the first candidates of 9.5 and
10.4 to pass requirement 3.

*The selected design* (`workspace/phase10/10.2/full_d18/start31_sobol41_s0.25/`; Full KaRMA + Adam under D18, 40
iterations, optimizer seed 0, from Sobol point 41 at scale 0.25; exported to
`results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25/`):

* admissible: minimum base separation 47.9 mm (≥ 24.1), reach 115.8 mm (≤ 250);
* exact Full score 0.0237 ± 0.0019 over the three configurations (third of 48 10.2 designs; the best is Full-D18 start15,
  0.0251); 0.0241 on `S_task`;
* robust grasps 12 / 11 / 12 of 12, weakest seed 1.92 × weight (default configuration);
* load sweep: stall and primary ceilings and 70 g all score 0.0229 with 12 of 12 seeds holding; the continuous ceiling
  keeps 73 % (10 of 12 seeds hold);
* simulated three-contact manipulation: E = 45.0 primitives per budgeted seed under the default configuration and 45.0
  under `sobol512_rs7` (the best of the 48 designs); 26.8 under rigid contact; compliant three-contact FP 0.07 over 386
  exactly valid primitives;
* two-finger exact library score 0.2767. Two-finger cost: −0.049 (−15 %) against the best two-finger design of all 66
  scored (arm A′ start16, 0.3256, which has no three-contact seed); it is above every Phase 9 design (best 0.2374,
  arm A′ start6);
* original KaRMA: physical task cube fraction 0.0297 (all-finger gate); published KaRMA-T 0.0266.

*Export check:* the parameters of the exported copy are identical, and its re-score under the default configuration
reproduces the suite exactly: exact Full 0.022932 and two-finger 0.256531 (`selection_numbers.json`).

### Acceptance (10.4)

A selection under criterion v2 with the criterion's numbers, the two-finger cost of the choice, and the original
criterion's verdict beside it; the selected specification exported and re-scored from the exported copy: **met**.
Criterion v2 is a session decision, frozen before the data and recorded in SPEC §16 item 12, and **subject to the owner's
review**. The selection stands on it, not on the original criterion.

*Tests at the end of 10.4 and of Phase 10* (2026-10-23, all code final): `dkarma/tests` 205 passed, 6 skipped (CPU,
30 min; `workspace/phase10/dkarma_tests_10.4.log`). `karma_jax/tests` 101 passed, 1 skipped (GPU shared with other users'
jobs, `XLA_PYTHON_CLIENT_PREALLOCATE=false`; `karma_jax_tests_10.4.log`).

### Limitations (10.4)

* Criterion v2's thresholds are a judgement made by the session from the deployment requirements: 6 of 12 robust
  grasps; the anthropomorphic start as the R3 reference; the median design as the R5 reference. They were frozen before
  the data, but they are not the owner's. The owner may set different thresholds, which could change the outcome.
* The two passing designs tie on the primary measure of v2. The choice between them rests on the number of robust grasps
  (11.67 against 10.33), a secondary quantity.
* One optimizer run per start (optimizer seed 0). The selected design's rank among the 48 designs is stable over seed
  configurations (Spearman ρ ≥ 0.946), but its own optimizer variance is unmeasured.
* The simulation evidence is MuJoCo with an assumed silicone pad and the force-floor controller. It is not a measured
  fingertip or a hardware trial.

### How to reproduce (10.4)

```bash
for D in armAprime/start26_sobol33_s0.25 full_d18/start31_sobol41_s0.25; do
  JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline sweep --design $D --out workspace/phase10/10.2 --starts-dir workspace/phase10/10.2 --n-starts 33; done
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 --out workspace/phase10/10.4 --export results/fabrication_candidates --rescore
```

---

## Phase 10 in one page: established, refuted, remaining

Fixed physical task throughout, unless *published* is stated. D18 design box. Every endpoint, decision rule, threshold
and selection criterion was frozen in this report on 2026-10-14 at 15:50, before the data it applies to (addendum A1, the
position-servo re-tune, was dated before the data it governs). Post-hoc analyses are labelled as such. Tests at the end of
Phase 10: `dkarma/tests` 205 passed, 6 skipped (CPU, 30 min); `karma_jax/tests` 101 passed, 1 skipped (GPU).

### Established

1. **Full KaRMA produces better simultaneous-contact manipulation than a collision-valid original-KaRMA optimum** (10.2,
   the pre-registered claim). The setting: 24 new starts; the common benchmark (simulated three-contact manipulation with
   compliant fingertips under the force-floor controller, primitives executed to the first failure regardless of the exact
   verdict); each arm at its Phase 9 settings and budgets. Full-D18 is ahead on 22 of 24 starts: +20.2 primitives per
   budgeted seed, 95 % CI [+14.1, +26.3], Wilcoxon p = 9.1 × 10⁻⁵. Every secondary endpoint agrees: the
   `sobol512_rs7` robustness check (21 / 3, p = 2 × 10⁻⁵), rigid contact, pose-only, duty-cycle retention, and the exact
   Full score (24 of 24 under every seed configuration). Arm A′ used 3.4× Full-D18's wall clock, and its wall clock is about
   40× Full-D18's optimizer time.
   Post hoc, the advantage is not only arm A′'s nine designs without a found three-contact seed. On the 15 starts where
   both arms have seeds, Full-D18 wins 13 (p = 0.003). The claim covers the two methods as specified (objective plus
   optimizer) on starts drawn by the 8.4 procedure.
2. **Compliant fingertips close most of the three-contact simulation gap, but not all of it** (10.1, verdict *partial*,
   robust over the 3 × 3 grid of hardness and thickness under both controllers). Under force control, pooled strict
   three-contact success on the 12-seed libraries rises from 0.41 to 0.71, and pose success from 0.73 to 0.88. The
   two-finger reference is unchanged (0.82). The gap closes on arm B (+0.13 → −0.06) and narrows on the start (+0.49 →
   +0.26). Most remaining strict failures are end-of-primitive contact chatter: on the start, pose is 1.00 and duty-cycle
   retention 0.99, while strict success is 0.47.
3. **A technically defensible three-contact agreement reference exists**: a strict FP of 0.24 (force-floor, compliant;
   median of the 13 eligible 12-seed libraries; quartiles 0.18–0.52). The compliant two-finger baseline is at 0.18.
4. **The task-feasible grasp test predicts simulated holding** (10.3, post hoc). 84 % of `S_task` seeds hold in the
   compliant simulation, against 7 % of the other seeds, and 99.2 % of all successful simulated primitives are on
   `S_task` seeds. Full-D18 designs have 9.7 of 12 robust grasps on average, arm A′ designs 4.8.
5. **Criterion v2 selects Full-D18 start31** (10.4). Two designs pass R1–R5 (start31 and start0, tied on the `S_task`
   Full score). The frozen tie-break, more robust grasps, chose start31. The original 9.5 criterion still selects
   nothing.

### Refuted or not supported

* "Arm A′ is the better two-finger hand" (9.2: 7 of 8, p = 0.016) **does not replicate** on the 24 new starts. Full-D18
  is ahead on 15 of 24 (p = 0.66). Arm A′ still wins its own objective (published KaRMA-T, 20 of 24).
* "The three-contact simulation deficit of 8.6/9.3 is intrinsic to maintaining three point contacts" is **largely
  refuted**. Post hoc, most of it was an artefact of the rigid scene's friction model: a 1.8 mm torsional coefficient, i.e.
  a 3 mm patch, larger than a rigid capsule's or a silicone pad's. With the patch's friction (0.32 mm), strict
  three-contact success under force control rises from 0.39 to 0.64 on rotations even with the rigid normal law. The pad's
  normal compliance adds the rest (to 0.70).
* "Normal compliance is the main mechanism by which soft fingertips help" is **not supported** here (D1 against D3,
  post hoc), under the force-floor controller. Under the position servo the two act together.

### Not met, and why

* 10.1 is *partial*, not *resolved*. The start's force-controlled gap stays at +0.26 at the central parameters
  (+0.16 to +0.37 over the grid). Its remaining failures are instantaneous contact losses at the end of the primitive, the
  quantity the strict criterion measures.
* The original selection criterion selects nothing. Its requirement 4 compares rigid three-contact agreement with a
  two-contact reference. The selected design's rigid three-contact FP is 0.49 / 0.44 against 0.19 / 0.18; its compliant
  FP is 0.07.

### Claim status after Phase 10

| Research claim | After Phase 9 | After Phase 10 |
| --- | --- | --- |
| Differentiable rolling-contact morphology optimization is implementable | supported | supported |
| Actuation-aware feasibility can be calibrated against exact force constraints | supported within the evaluated conditions | supported; the exact wrench-capacity test also predicts compliant-simulation holding (10.3) |
| Simultaneous-contact reachability can be evaluated and optimized | supported | supported |
| Gradients are stable enough to run optimization experiments | substantially improved; clipping still required | unchanged (no non-finite iteration in 24 runs) |
| Gradient optimization universally beats CMA-ES | not supported | not addressed (the arms differ in objective too) |
| Full KaRMA consistently outperforms collision-valid original KaRMA | not established | **supported** on the common simulated benchmark: 24 new starts, p = 9 × 10⁻⁵, at the Phase 9 budgets |
| The exact three-contact metric predicts successful physical manipulation | partially supported by MuJoCo | **substantially supported in simulation**: compliant three-contact FP 0.06–0.23 for the five 10.4 candidates; residual chatter (10.1 partial); not hardware |
| Optimized morphologies are ready for hardware deployment | not yet established | **one design selected** by criterion v2 (session-frozen, subject to the owner's review); not by the original criterion |

### Fabrication recommendation, with its uncertainty

**Recommended for fabrication, subject to the owner's review of criterion v2: Full-D18 start31**
(`results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25/`), built with silicone fingertip pads of about
Shore 20A and 2 mm.

* Exact Full score 0.0237 ± 0.0019 (third of 48), 0.0241 on its robust grasps.
* Robust grasps 12 / 11 / 12 of 12, weakest grasp 1.92 × weight.
* 70 g held on all 12 seeds; 73 % of the score kept at the continuous ceiling.
* The best simulated three-contact manipulation of the 48 designs: 45 primitives per seed, under both seed
  configurations. Compliant FP 0.07.
* Two-finger score 0.2767: −15 % against the best two-finger design of the project, which cannot grasp with three
  fingers, and above every Phase 9 design.
* The runner-up, tied on v2's primary measure, is Full-D18 start0 (`full_d18_start0/`).

The uncertainty has five parts:

1. **Criterion v2 is the session's.** Its thresholds (6 of 12 robust grasps, the anthropomorphic start as the R3
   reference, the median design as the R5 reference) were frozen before the data but have not been reviewed. Under the
   original criterion nothing is selected.
2. **The evidence is simulated**, with an assumed pad (Gent's modulus, the bonded-layer law) and one controller. The pad's
   static law is reproduced in MuJoCo within about ±6 %, but not its dynamics.
3. **Selection effect.** The design was picked as the best E of 48. Its `sobol512_rs7` E (independent seeds) is
   identical, 45.0, which argues against a seed-sampling fluke. Its optimizer-seed variance is unmeasured.
4. **Actuation assumptions.** The torque ceiling (0.108 N·m) and the link masses are datasheet-based assumptions until a
   joint-side torque curve is measured.
5. **The start31 vs start0 choice** rests on the tie-break (11.67 against 10.33 robust grasps).

### Remaining

1. The owner's review of criterion v2 (SPEC §16 item 12), and with it the fabrication decision. The SPEC §14 bullets on
   the exact-vs-simulation error and on Full KaRMA vs original KaRMA can now cite 10.1 and 10.2; that text is left to the
   owner.
2. Before fabrication: measure the fingertip silicone (indentation test) and a joint-side torque curve, then re-run the
   10.1 replay of the selected design with the measured parameters.
3. The fingertip patch matters (post-hoc D3): the fabricated pad's torsional friction should be measured, since the
   simulation result depends on it more than on the pad's normal stiffness.
4. Hardware realization (SPEC §12 IV) of the selected design.
5. Open questions not addressed in Phase 10: why the original-KaRMA optimum lost the third finger on 9 of 24 new starts
   but 1 of 8 pilot starts; and the optimizer-seed variance of the selected design.
