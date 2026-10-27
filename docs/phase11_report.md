# Phase 11 report: re-optimization inside the servo envelope

Scope: SPEC.md §11.4 (added 2026-10-23 from the owner's decision D19, SPEC §8.1 and §16 item 14), items 11.1 to 11.5 in
order. D19 is the only change to the design space. The frozen task (`dkarma/configs/task_spec.yaml`, with its 11.8 mm
contact radius `link_radius_m`), the compliant pads (`dkarma/configs/compliant_pads.json`), the controllers, every metric,
kernel, contact model, solver and optimizer setting, and the Phase 9/10 settings (`kkt_method="reduced"`, `exp_so3_safe`,
norm-100 clipping, the cached exact evaluators, suites and exports on the CPU) are unchanged. `karma_jax` is not extended
for mixed radii, so original-KaRMA scores are not computed under D19. Phase 11 makes no comparative claim. Every table
states its task definition, which is the **fixed physical** task of SPEC §2 throughout. Every simulation result names its
controller, contact model and success criterion.

Raw outputs: `workspace/phase11/{11.1,11.2,11.3,11.4,11.5}/`; chain scripts `workspace/phase11/chain_*.sh`.

---

## Pre-registration (frozen 2026-10-23 23:59 EDT, before any Phase 11 data was produced)

Everything in this section was written before any D19 design was built, scored or simulated, before the regression of
11.1 ran, and before N11 or any start was evaluated. It is not edited after that point. Corrections, if any, are appended
below it with their date and apply only to data produced after them.

What was known when it was written: the Phase 7–10 numbers (`docs/phase10_report.md`), the D19 decision (SPEC §8.1,
§11.4, §16 item 14), and an audit of the code (which uses of a capsule radius exist, below). No measurement of a D19
design had been made.

### P11.1 D19 in code: the radius classification and the regression rule

**Radius classification.** Every use of a capsule radius in `dkarma` is one of two kinds:

* *contact surface*: the assigned fingertip capsules, whose radius is the task's `link_radius_m` (11.8 mm). This covers the
  rolling kinematics of the assigned contacts, the reference pinch IK's pair gaps and the contact-pair Jacobians, the
  seed-candidate reach filters on fingertips, the force and wrench problems' contact points, and the compliant pads.
* *collision envelope*: each capsule's own `cap_radius`. This covers the object–capsule collision gate, the
  capsule–capsule (finger–finger) clearance, the smooth clearance hinges, the MuJoCo collision geometry, the URDF's capsule
  geometry, the renders, and D18's base-separation rule.

Under D19 a fingertip's two radii coincide (11.8 mm), and links 0–2 carry 16.40 mm.

**The two-finger reference path.** The two-finger seed search (the reference pinch IK, its feasibility test and the
nullspace polish) and the exact two-finger validators run `karma_jax` kernels that apply one radius to every capsule. They
are not extended. With mixed radii, `dkarma` re-applies the kernels' own gates with per-capsule radii, at the same points
and in the reference order:

1. the seed pair's gaps are within `eps_g`;
2. the sphere clears every other capsule by `eps_col` (the object collision gate);
3. the thumb and index capsules clear each other by `eps_col` (the kernels' self-collision gate, unchanged in scope).

A seed is kept only if it passes, and a polished seed only if the polished state passes; otherwise the unpolished one is
kept, as the reference does. An exact primitive counts only if the kernel accepts it and these gates also pass at its end
state, and a failure takes the kernel's code for that gate (gap, collision, self-collision). With uniform radii equal to
`link_radius_m` the re-check is not run, so the Phase 7–10 path is unchanged.

**The D19 box** (SPEC §8.1, §11.4):

| quantity | value |
| --- | --- |
| length floors per slot | link 0: 12 mm, links 1 and 2: 34 mm, fingertip: 12 mm |
| length maxima | 90 mm |
| `base_xyz_max_m` | 0.10 |
| capsule radii per finger | 16.40 / 16.40 / 16.40 / 11.8 mm (links 0–2 / fingertip) |
| `r_servo` | 16.40 mm as written in D19; the half-diagonal of the 20 × 26 mm section is 16.401 mm, 1.2 µm more |
| smooth penalties | `Constraints(reach_max_m=0.25, base_sep_min_m=0.0333)` with the D18 weights (10, 10) |

*Hard admissibility under D19:*

* every pair of finger bases at least `r_0,a + r_0,b + eps_col` apart (D18's rule with each base capsule's own radius:
  33.30 mm);
* reach at most 250 mm per finger;
* the servo envelope:
  * links 0–2 have radius ≥ 16.40 mm;
  * links 1 and 2 are ≥ 34 mm long;
  * the fingertip's radius equals the task's contact radius (11.8 mm).

**Regression rule (acceptance of 11.1).** The D18 path is unchanged with uniform 11.8 mm radii and the D18 box when the
following reproduce **bit for bit** from Phase 10's design file of Full-D18 start31 (`full_d18/start31_sobol41_s0.25`):

| run | what must match the Phase 10 output exactly |
| --- | --- |
| (a) the 9.4 suite: three seed configurations, `--per-seed-detail`, original-KaRMA numbers included | every value of `suite.json` except the wall-clock fields (`wall_s`, `time_s`, `official_time_s`, `runtime_s`) |
| (b) its default-configuration three-contact export (12 seeds) | every array of the trajectory archive and every index entry |
| (c) the replays of that new export: central pad 20A/2 mm with the force-floor controller, and rigid contact with the position servo (0.5 mm) | every row of the replay JSON |

If any of them differs, the cause is found and fixed before 11.2 starts. Nothing beyond 11.1 runs until all three match.

### P11.2 Start set

**N11, the nominal start.** `anthropomorphic` on the three-finger 4-DOF topology with these changes, all else unchanged:

* links 1 and 2 of every finger at `max(L, 40 mm)`: thumb 45 / 40 mm, index and middle 40 / 40 mm;
* link 0 unchanged (12 / 12 / 15 mm) and fingertips unchanged (23 / 23 / 25 mm);
* the index–middle base pitch 40 mm instead of 30 mm (index base at the origin, middle at y = 40 mm);
* the thumb's base pose, every axis and every joint limit unchanged;
* the D19 radii.

**Fallbacks, in order, each recorded before any optimization:**

1. If N11 fails the hard D19 admissibility check, raise the pitch to 45, 50, 55 and 60 mm in turn, and take the first
   admissible value.
2. Count N11's three-contact seeds with `find_multi_seeds`: default seed configuration, budget 12, the frozen task at the
   primary ceiling. If there are fewer than 3, lengthen the three fingertips together by +5, +10, +15 and +20 mm (index and
   middle 28 / 33 / 38 / 43 mm, thumb 30 / 35 / 40 / 45 mm; the last step reaches 45 mm). Take the first step with
   ≥ 3 seeds, and report the attempt counts at every step (`seed_stats_3`: joint samples, reachable samples, candidates,
   IK solutions, kinematically valid, distinct, force-checked, in equilibrium).
3. If neither yields an admissible start with ≥ 3 seeds, stop Phase 11 and report the measurements. No claim that a
   three-contact grasp does not exist is made without the three 8.5 procedures.

**Sobol starts.** `make_starts` exactly as coded in `admissible_baseline starts`, run on the D19 box from N11 (the version
chosen above):

* scrambled Sobol, seed 321, `8 × n_starts` points in the 54 unconstrained coordinates;
* scale 0.25 for Sobol index `j < 4`, 0.5 for `4 ≤ j < 20`, and the default 0.25 for `j ≥ 20`;
* a point is accepted if its design has ≥ 3 three-contact seeds at budget 6 (the coded rule);
* `n_starts = 24`: start0 = N11 plus start1 … start23.

Rejected points are listed with their seed counts. Each start's D19 admissibility is recorded. As coded in 8.4 and 10.2,
it is not a rejection criterion: the optimizer enforces D19 from the first iteration. The 8.4 reproduction check of the
first eight starts applies to the D18 box only and is not run.

### P11.3 Optimization

`admissible_baseline full --box d19 --kkt-method reduced`, one process per start on the GPU, run sequentially, with a
20 GB RSS watchdog. The settings are those of the 10.2 Full-D18 arm:

* Full KaRMA + Adam, 40 iterations in 2 stages, 6 three-contact seeds found on the start design;
* `ADAM_DEFAULTS` (norm-100 clipping, skip non-finite, reset moments), learning rate 0.02, optimizer seed 0, warm-up
  compile;
* exact validation every 10 iterations and at stage ends (fresh seeds, exact SLSQP force check at every sub-step);
* the D19 box.

D19 is enforced throughout by two mechanisms:

* the smooth base-separation and reach penalties of `D19_CONSTRAINTS`;
* the hard D19 check before every exact validation, under which an inadmissible iterate scores zero.

Reported per run:

* compute: wall clock, optimizer time, validation time, gradient evaluations;
* non-finite and clipped iterations, and the returned iteration with its exact validation score;
* the returned design's admissibility and margins: base separation − 33.30 mm, 250 mm − reach, min(link 1, link 2) − 34 mm,
  link 0 and fingertip − 12 mm, 90 mm − longest link.

A returned design that fails the hard D19 check (possible only if no validated iterate was admissible) is reported and is
not a candidate.

### P11.4 Evaluation of every returned design

**Order.** N11's suite (three seed configurations, per-seed detail) runs first. Its `S_task` exact Full score, the R3
reference, is computed and written to `workspace/phase11/11.4/r3_reference.json` with a timestamp before any optimized
design's suite, export or replay starts. The CPU chain enforces this order.

**Per design** (CPU, at most three jobs at a time, RSS watchdog):

1. the 9.4 suite with `--per-seed-detail`: three seed configurations, 12 three-contact and 16 two-finger fresh seeds,
   exact Full scores (exact SLSQP check at every sub-step), exact two-finger library scores, wrench capacities, and
   `S_task`. No original-KaRMA numbers;
2. the default-configuration three-contact export, 12 fresh seeds, `CommandConfig(n_cells=4, n_rot_steps=10)`;
3. replays of that export, running to their end and stopping only at a drop:

   | contact model | force-floor controller | position servo |
   | --- | --- | --- |
   | central compliant pad (20A / 2 mm) | 9.3 parameters | 0.75 mm squeeze (addendum A1 of Phase 10) |
   | rigid contact (8.6/9.3 scene) | 9.3 parameters | 0.5 mm squeeze |

4. the load sweep (`admissible_baseline sweep`): stall, primary and continuous ceilings, and 70 g at the primary ceiling,
   on 12 fresh seeds.

**Endpoints, all descriptive** (Phase 11 makes no comparative claim):

* `E`, as defined in P10.2:
  * compliant 20A/2 mm, force-floor controller;
  * per command, strict success (centre within 4 mm and orientation within 0.2 rad of the plan, every assigned fingertip in
    contact at the primitive end) counted from the command start to the first failure, regardless of the exact verdict;
  * summed over seeds and commands and divided by 12.
* `E` split as `E = E_R + E_T`, where `E_R` sums the rotation commands and `E_T` the translation commands (each per
  budgeted seed).
* *Drop-criterion events* in the same replay: trajectories that meet the 9.3 drop criterion (centre > 20 mm from the plan,
  or no assigned fingertip in contact) at some primitive. They are split by what happened at that primitive:
  * *release*: no assigned fingertip in contact;
  * *> 20 mm deviation*: at least one fingertip still in contact.

  Each is reported per command type (72 translation and 72 rotation trajectories per design).
* The same quantities under rigid contact and under the position servo.
* The exact Full score per seed configuration and its mean, the exact two-finger library score, `S_task` and the exact Full
  score on `S_task` (the P10.3 formulas), the load sweep, and the compliant strict false-positive rate.
* *"Do the new designs do better on translations?"* This is a descriptive comparison with the Phase 10 selection's split
  (`E_R` 38.6, `E_T` 6.4): per design, `E_T` against 6.4 and `E_R` against 38.6, and their distribution over the Phase 11
  designs. No statistical claim is made.

### P11.5 Selection

**Criterion v2**, exactly as frozen in SPEC §16 item 12 and `docs/phase10_report.md` P10.4 (still subject to the owner's
review), with eligibility under D19:

| requirement | test |
| --- | --- |
| R1 eligibility | admissible under the hard D19 check; exact two-finger library score > 0 and ≥ 1 three-contact seed under each of the three seed configurations; the exported files reproduce the suite's default-configuration numbers |
| R2 robust grasps | `\|S_task\| ≥ 6` of 12 under every configuration |
| R3 manipulation from robust grasps | exact Full score on `S_task`, mean over the three configurations, at least N11's on its own `S_task` |
| R4 load robustness | at least one seed holds 70 g, and the 70 g score is ≥ 50 % of the 35 g score |
| R5 simulation agreement | compliant (20A/2 mm) force-floor strict false-positive rate ≤ 0.2393 (the 10.1 reference), with ≥ 10 exactly valid primitives on holding seeds |

*Decision.* Among the designs that pass R1–R5, the highest `S_task` Full score wins. A difference smaller than the larger
seed-configuration standard deviation of the two is a tie, broken by the larger mean `|S_task|`. The two-finger cost is
stated as in 9.5, against the best two-finger score among the Phase 11 designs. If nothing passes, nothing is selected and
the criterion is not loosened.

*Candidates.* The 24 designs returned by 11.3. N11 is the R3 reference and is not a candidate, as `start0_nominal` was not
in 10.4.

**The original 9.5 criterion alongside** (never presented as equivalent):

* requirement 1: eligibility under D19, as R1;
* requirement 3: minimum wrench capacity ≥ 1 × weight in all 14 directions at every default-configuration seed, plus the
  load sweep as R4;
* requirement 4: the rigid-contact three-contact strict false-positive rate below the 9.3 two-finger baseline's under the
  same controller (0.19 position servo, 0.18 force-floor), for at least one controller, on the Phase 11 rigid replays.

Its decision step (the higher mean exact Full score, ties broken by the minimum wrench capacity) is applied to the same
candidates.

**If a design is selected:**

* export it to `results/fabrication_candidates/selected_<arm>_<start>/`;
* re-score the exported copy (default configuration; it must reproduce the suite);
* render its figures with `visualize_design`;
* record the outcome in `results/fabrication_candidates/README.md`.

**Phase 10 comparison.** Descriptive only, as what the envelope costs: the selection (or the best Phase 11 design by the
`S_task` Full score if nothing is selected) against Full-D18 start31 on the exact Full score (all seeds and `S_task`),
`E`, `E_R` / `E_T`, robust grasps, the load sweep, the compliant false-positive rate and the two-finger score; and N11's R3
reference against the Phase 10 start's (0.0196).

### Amendments to the pre-registration (dated; each written before the data it governs)

**A1 (2026-10-24 00:14 EDT; owner's decision of about 00:15 relayed through the session that maintains SPEC §11.4; before
N11, any start or any D19 design was evaluated).** The owner withdrew the relaxation of link 0's length: link 0 also holds
the XL330 case along its axis and is at least 34 mm long ("I want the constraint to still be enforced by link 0"; SPEC §8.1
D19, §11.4, §16 item 14 as updated on 2026-10-24). Everything else in D19 is unchanged. This replaces, in the sections
above:

* *P11.1, the D19 box.* The length floors are 34 mm for links 0, 1 and 2 and 12 mm for the fingertip, with maxima of 90 mm.
  This is `Bounds(link_len_min_m=0.034, link_len_max_m=0.090, tip_len_min_m=0.012, tip_len_max_m=0.090,
  base_xyz_max_m=0.10)`. One floor serves every servo link, so `Bounds` needs no per-slot floor, and the one drafted for the
  relaxed box was removed before any data.
* *P11.1, the hard check.* The servo envelope requires links 0–2 to be ≥ 34 mm long, as well as ≥ 16.40 mm in radius.
* *P11.2, N11.* Links 0–2 of every finger at `max(L, 40 mm)`:
  * index and middle: 40 / 40 / 40 / 23 mm;
  * thumb: 40 / 45 / 40 / 25 mm (link 0 was 15 mm).

  The fingertips, the 40 mm pitch, the D19 radii and both fallbacks (with the same tip lengths) are unchanged.
* *P11.3, the margins.* The servo-link margin is min(links 0–2) − 34 mm; the fingertip margin is fingertip − 12 mm.

The regression rule of P11.1 (uniform radii, D18 box) is unaffected: its run was already in progress and does not involve
D19.

---

## 11.1 D19 in code

### Files changed (11.1)

| File | Change |
| --- | --- |
| `dkarma/core/morphology.py` | `capsule_radii(topo, link_r, tip_r)`, the `[C]` radii of a D19 hand. `anthropomorphic` takes per-capsule radii, the index–middle pitch (`pitch_m`), a floor for every link but the fingertip (`link_floor_m`) and a fingertip extension (`tip_extra_m`); the defaults reproduce the Phase 7–10 start. A per-slot floor was drafted for the relaxed box and removed when amendment A1 restored one floor for all servo links: `Bounds` is unchanged |
| `dkarma/core/admissibility.py` | D18's rule with each base capsule's own radius (`required_base_separations`; 24.1 mm for 11.8 mm capsules, 33.30 mm for 16.40 mm). The D19 constants (`R_SERVO_M` 16.40 mm, `SERVO_LEN_MIN_M` 34 mm). `ServoEnvelope` / `D19_ENVELOPE` and `envelope_check`: servo radius and 34 mm floor on links 0–2, and the fingertip at the contact radius. `admissibility(..., envelope=)` adds the envelope, its margins and the lengths. Without `envelope`, the record and its keys are the D18 ones |
| `dkarma/optimization/objectives.py` | `D19_CONSTRAINTS = Constraints(reach_max_m=0.25, base_sep_min_m=0.0333)`. `exact_validation(..., envelope=)`: an iterate outside D19 scores zero |
| `dkarma/core/contacts.py` | `mixed_radii(hm, link_radius)` and `reference_gates_per_capsule`: the reference two-finger kernels' gap, object-collision and self-collision gates with each capsule's own radius |
| `dkarma/solvers/exact.py`, `dkarma/metrics/exact_library.py` | the exact two-finger validators take `envelope`, the per-capsule gates at each accepted primitive's end state, with the kernel's failure codes. `evaluate_exact` turns it on for mixed radii and caches its own compiled programs, so the Phase 7–10 programs are untouched |
| `dkarma/solvers/seed_projection.py` | two-finger seeds and the polished seeds with mixed radii must also pass the per-capsule gates (`envelope_ok`, `n_envelope_rejected` in the stats). `multi_candidates`' reach filter uses each capsule's own radius, which matters only for the 8.5 option `centre_caps="all"`: the fingertips carry the contact radius. For equal radii the expression is bit-identical to the old one |
| `dkarma/experiments/three_contact_verification.py` | the 8.5 host solver's object and finger–finger clearances use each capsule's own radius; the contact links must carry the contact radius |
| `dkarma/experiments/simulation_replay.py` | `_mjcf_rigid` gives each capsule its own radius and ends the terminal capsule one fingertip radius before the tip (`_capsule_radii`). `mjcf_from_design` refuses compliant pads on a fingertip that is not 11.8 mm (the pads are fitted to it). The two-finger export passes `envelope` |
| `dkarma/export/urdf_export.py` | a design with mixed radii carries every link's capsule as URDF `<visual>` geometry (a cylinder and two spheres; the reference reads only `<collision>` cylinders). Uniform-radius URDFs are byte-identical to before |
| `dkarma/experiments/continuation_study.py` | `BOUNDS_D19`; `nominal_d19` (N11); `objective_spec("full3_d19", n11=...)`; `make_starts` takes the box from the spec and records every drawn Sobol point with its seed count (`points`) |
| `dkarma/experiments/admissible_baseline.py` | `--box d18\|d19` (`BOXES`: objective, bounds, constraints, envelope, whether original-KaRMA numbers are computed, arm name). The `nominal` stage (N11 and its fallbacks into `n11.json`, plus the N11 design files). `starts` under D19 records each start's admissibility and the rejected points. `full`, `suite` and `sweep` work in the box; `--runs-dir`. The D18 behaviour is the default and unchanged |
| `dkarma/experiments/{powered_comparison,robust_grasps,selection_v2,candidate_selection,visualize_design}.py` | generalized for a phase directory (used from 11.4 on; Phase 10 defaults unchanged): `endpoint_split` and `drop_events`; the suite and replay roots of `robust_grasps report`; `selection_v2 r3 / evaluate / select --phase-dir`; D19 in the eligibility text; `visualize_design --phase-dir`, with radii and the base-separation requirement taken from the data |
| `dkarma/tests/test_servo_envelope.py` (new, 9 tests), `dkarma/tests/test_simulation_replay.py` (+1) | below |

### Procedure (11.1)

*The audit.* Every use of a radius in `dkarma`, and in the `karma_jax` kernels it calls, was listed and classified as in
P11.1.

*Per-capsule radii already in Phase 10.* The following already used each capsule's own radius
(`dkarma.core.geometry` passes `hm.cap_radius` to the capsule pass):

* the capsule pass and the finger–finger clearance table (`r_i + r_j`);
* the contact pass and its surface Jacobian (the assigned capsule's radius);
* the rollout's exact object and self-collision gates;
* the projection;
* the three-contact seed IK residual and its verification (`_verify_multi`);
* the exact force and wrench problems (contact points from the contact pass; the 3 mm torsion patch is not a capsule
  radius);
* the gravity torques (the §2.1 masses are unchanged).

*Uniform-radius assumptions found and fixed.* Besides the four that SPEC lists (the MuJoCo scene, the 24.1 mm constants,
`anthropomorphic` and the start generation, the URDF and `visualize_design`), the audit found three more:

1. The two-finger reference path. The pinch IK's feasibility, the seed polish and the exact step gates are `karma_jax`
   kernels that apply `link_radius` to every capsule. They are not extended (rule (h)); `dkarma` re-applies their
   collision gates per capsule (P11.1).
2. The reach filter of the 8.5 `centre_caps="all"` candidates.
3. The 8.5 host solver's clearances.

*Not changed.* The frozen `dkarma/configs/task_spec.yaml` describes `link_radius_m` as the "capsule radius of every link".
Under D19 it is the contact radius only; the file is frozen and was not edited.

*Regression.* `workspace/phase11/chain_111_regression.sh` ran three CPU jobs on 2026-10-24 between 00:05 and 00:55, with
the Phase 10 design file copied byte for byte and the D18 defaults:

* the suite of `full_d18/start31_sobol41_s0.25` (three configurations, per-seed detail, original KaRMA);
* its default-configuration export;
* the two replays.

`workspace/phase11/compare_regression.py` compared them with the Phase 10 files. Amendment A1 arrived during the run, and
the code was edited after the jobs had started. The D18-path part of those edits restored `Bounds.len_bounds` to its
original text (the drafted per-slot floor defaulted to that same expression). The other edits touch only D19 branches
(`envelope`, `box d19`) and the phase-directory scripts.

### Results (11.1)

*Regression* (`workspace/phase11/11.1/regression/compare.json`): **bit for bit**.

* The suite: every value of `suite.json` equals Phase 10's, except the wall clocks. Exact Full 0.022932 / 0.026236 /
  0.021869 and two-finger 0.256531 / 0.270705 / 0.302788 under the three configurations; physical original KaRMA
  (all-finger) 0.029709.
* The export: all 1,297 arrays and the index are identical.
* The replays: the 144 rows of the compliant force-floor replay and of the rigid position-servo replay are identical, and
  both MuJoCo scene strings are identical.

*`visualize_design`* with no phase directory re-renders the seven static Phase 10 figures of the selected design
pixel for pixel (`hand_views`, `design_before_after`, `grasp_gallery` and the four charts; the MP4/GIF animations were not
re-encoded).

*The new tests* (`dkarma/tests/test_servo_envelope.py`):

* the D19 floors: 34 mm on links 0–2 and 12 mm on the fingertip; no decoded link 0, 1 or 2 shorter than 34 mm over 60
  random points up to 40 units outside the interior;
* N11's dimensions and its admissibility;
* D19 admissibility:
  * a 30 mm pitch passes 24.1 mm but fails 33.30 mm;
  * a 30 mm link 0 or link 1, an 11.8 mm servo link and a 16.40 mm fingertip each violate the envelope;
  * the Phase 10 start and the Phase 10 selection are outside D19;
* the smooth penalty;
* per-capsule gaps and clearances, against the uniform-radius model;
* **the collision gates with mixed radii**:
  * a sphere 14 mm from a servo link's axis clears 11.8 mm by 2.2 mm and is rejected at 16.40 mm (−2.4 mm). This holds
    for the object clearance, the three-contact seed verification and the reference two-finger path (whose kernel
    accepts it);
  * fingers 30 mm apart clear 11.8 mm by 6.4 mm and are rejected at 16.40 mm (−2.8 mm), in the all-finger gate and in
    the kernels' pair scope;
* the exact validators' failure codes under the per-capsule gates;
* **the MuJoCo scene against the dkarma capsules**, capsule by capsule, at three random configurations: radius to 1e-12,
  both endpoints to 1e-9 m;
* the pad guard;
* the Phase 10 rigid scene string unchanged;
* the URDF's capsule visuals: 9 servo cylinders, 18 servo and 6 fingertip spheres for N11. The reference loads the same
  joint tree and fingertips with and without them.

`test_simulation_replay.py` gained the `E_R` / `E_T` split and the drop-event split. On the Phase 10 selection they give
45.0 = 38.6 + 6.4 and, on translations, 43 releases and 12 deviations, as its figures README reports.

*A code-path check* (budget 2, CPU; not a result) ran N11's two-finger seeds, the two-finger exact validators with the
per-capsule gates, the three-contact seeds and the exact Full evaluator end to end. In it, the per-capsule check rejected
218 of the 267 two-finger IK solutions that the reference kernels accept. The kernels' IK keeps clear of 11.8 mm capsules,
not 16.40 mm ones, so under D19 the two-finger seed procedure finds fewer seeds than a per-capsule IK would. That is a
property of the procedure, not a claim about the hand.

*Test suites at the end of 11.1:* `karma_jax/tests` 101 passed, 1 skipped (GPU, `XLA_PYTHON_CLIENT_PREALLOCATE=false`:
the GPU is shared with another user's job; `workspace/phase11/karma_jax_tests_11.1.log`). `dkarma/tests` 215 passed,
6 skipped (CPU, 28 min; Phase 10's 205 plus the 10 new tests; `workspace/phase11/dkarma_tests_11.1.log`). The existing
finite-difference gradient tests pass unchanged.

### Acceptance (11.1)

D19 is in code with per-capsule radii classified throughout. The D19 box, its hard check and its penalty are in place,
and the required tests are added and pass. The D18 path reproduces a Phase 10 suite, export and two replays bit for bit:
**met** (2026-10-24 01:24).

### Limitations and findings (11.1)

* The two-finger exact library under D19 is the reference kernels plus a per-capsule re-check. It is conservative: every
  state it accepts satisfies D19's gates. It is not the result a per-capsule kernel would give, because the kernels' pair
  switching and IK still see 11.8 mm envelopes. It enters criterion v2 only through R1 (score > 0) and the two-finger cost.
* *Finding (pre-existing, not changed).* The exact two-finger library's self-collision gate is the reference's
  thumb–index check. `dkarma`'s `DeviceModel` carries no `cap_group`, so the middle finger is not tested against the pair
  in that evaluator, though it is in the object-collision gate. The 9.4 suites' two-finger scores of Phases 9–10 were
  computed this way. D18's text asks for the all-finger clearance in every evaluator's self-collision gate. Changing it
  would change Phase 9–10 numbers and is not a D19 matter; the per-capsule re-check keeps the kernels' scope. Reported for
  the owner.
* Link 0's capsule now must contain the case lengthwise (A1). Link 0 runs from joint 0 (in the palm) to joint 1. Whether
  joint 0's servo and the palm leave room for that is mechanical design, outside the model.

### How to reproduce (11.1)

```bash
workspace/phase11/chain_111_regression.sh            # CPU, ~50 min
~/mamba/envs/karma-jax/bin/python workspace/phase11/compare_regression.py
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m pytest dkarma/tests/test_servo_envelope.py -q
```

---

## 11.2 Start set

### Files changed (11.2)

None beyond 11.1. The `nominal` and `starts` stages of `admissible_baseline --box d19` (11.1) ran unchanged.

### Procedure (11.2)

`workspace/phase11/chain_113_gpu.sh` ran on the GPU (shared, `XLA_PYTHON_CLIENT_PREALLOCATE=false`), with a marker that
stopped the chain after the start set so this section could be written before any optimization:

* `admissible_baseline nominal --box d19` on 2026-10-24 at 01:24 (31 s): N11 and the frozen fallbacks of P11.2 / A1.
  Output: `workspace/phase11/11.2/n11.json`, plus N11's URDF, robot YAML and design JSON in `11.2/n11/`.
* `admissible_baseline starts --box d19 --n-starts 24`, 01:24 to 01:33. Output: `11.2/starts.json`, which lists every
  drawn point in `points`, and `starts_check.json`, with every start's D19 admissibility and the rejected points.

### Results (11.2)

**N11** (fixed physical task, D19 radii):

* *Dimensions.* Index and middle 40 / 40 / 40 / 23 mm; thumb 40 / 45 / 40 / 25 mm.
* *Base separations.* Index–middle 40.0 mm, index–thumb 47.2 mm, middle–thumb 83.8 mm, against 33.30 mm.
* *Fallback 1* was not needed: N11 is D19-admissible at the 40 mm pitch.
* *Fallback 2* was not needed: N11 has **12 of 12** three-contact seeds (default configuration, budget 12, GPU). The
  attempt counts:

  | stage | count |
  | --- | --- |
  | joint samples | 381 |
  | reachable | 105 |
  | IK candidates | 315 |
  | kinematically valid | 148 |
  | distinct | 15 |
  | in frictional equilibrium | 12 (3 rejected) |

  The start is N11 as defined, with the fingertips unextended.

**Starts** (the 8.4 procedure as coded, D19 box from N11, Sobol seed 321):

* 24 starts: start0 = N11 and 23 Sobol starts.
* Sobol points 0–46 were drawn; 23 were accepted and 24 rejected.
  * Rejected for fewer than 3 three-contact seeds at budget 6, with their seed counts: point 16 (scale 0.5) found 2;
    points 1, 4–10, 12, 13, 15, 17–21, 24, 25, 29, 31, 34, 36 and 38 found 0.
  * Every accepted point found the full 6. The seed counts are bimodal: 23 points at 6, 23 at 0 and one at 2.
* Of the 16 points at scale 0.5 (indices 4–19), 2 were accepted (11 and 14). Of the 31 at scale 0.25, 21 were accepted.
  The D19 start set is therefore almost all small perturbations of N11 (21 of 23).
* **Every start is D19-admissible**:

  | quantity | over starts 1–23 | N11 |
  | --- | --- | --- |
  | minimum base separation | 34.3–54.4 mm (margin 1.0–21.1 mm) | 40.0 mm |
  | shortest servo link | 37.8–39.3 mm (margin 3.8–5.3 mm over 34 mm) | 40 mm |
  | fingertips | 19.4–24.3 mm | |
  | reach | ≤ 154 mm | |

| start | Sobol point, scale | min base separation (mm) | shortest servo link (mm) | shortest fingertip (mm) |
| --- | --- | --- | --- | --- |
| start0_nominal | N11 | 40.0 | 40.0 | 23.0 |
| start1_sobol0_s0.25 | 0, 0.25 | 39.0 | 39.0 | 24.3 |
| start2_sobol2_s0.25 | 2, 0.25 | 41.5 | 39.1 | 22.9 |
| start3_sobol3_s0.25 | 3, 0.25 | 45.6 | 39.1 | 22.1 |
| start4_sobol11_s0.5 | 11, 0.5 | 47.2 | 37.8 | 19.4 |
| start5_sobol14_s0.5 | 14, 0.5 | 54.4 | 38.6 | 20.1 |
| start6_sobol22_s0.25 | 22, 0.25 | 39.0 | 38.9 | 22.2 |
| start7_sobol23_s0.25 | 23, 0.25 | 45.8 | 38.9 | 21.6 |
| start8_sobol26_s0.25 | 26, 0.25 | 45.1 | 38.8 | 21.4 |
| start9_sobol27_s0.25 | 27, 0.25 | 43.4 | 38.9 | 22.7 |
| start10_sobol28_s0.25 | 28, 0.25 | 37.4 | 39.0 | 21.9 |
| start11_sobol30_s0.25 | 30, 0.25 | 36.3 | 39.0 | 21.1 |
| start12_sobol32_s0.25 | 32, 0.25 | 40.5 | 38.9 | 21.2 |
| start13_sobol33_s0.25 | 33, 0.25 | 40.0 | 39.1 | 21.0 |
| start14_sobol35_s0.25 | 35, 0.25 | 47.9 | 39.1 | 22.9 |
| start15_sobol37_s0.25 | 37, 0.25 | 39.5 | 38.8 | 22.1 |
| start16_sobol39_s0.25 | 39, 0.25 | 42.9 | 38.9 | 23.9 |
| start17_sobol40_s0.25 | 40, 0.25 | 34.3 | 39.1 | 22.2 |
| start18_sobol41_s0.25 | 41, 0.25 | 44.3 | 39.3 | 21.1 |
| start19_sobol42_s0.25 | 42, 0.25 | 43.3 | 39.0 | 20.9 |
| start20_sobol43_s0.25 | 43, 0.25 | 44.6 | 39.0 | 23.8 |
| start21_sobol44_s0.25 | 44, 0.25 | 35.5 | 38.8 | 22.6 |
| start22_sobol45_s0.25 | 45, 0.25 | 41.6 | 38.8 | 22.3 |
| start23_sobol46_s0.25 | 46, 0.25 | 42.0 | 39.1 | 21.7 |

*Tests at the end of 11.2:* no code changed after the 11.1 runs (last edit 00:20, test run 00:55–01:23), so those results
stand: `dkarma/tests` 215 passed, 6 skipped; `karma_jax/tests` 101 passed, 1 skipped.

### Acceptance (11.2)

N11 is D19-admissible with 12 of 12 three-contact seeds, so neither fallback was used. The 24-start set was drawn by the
frozen procedure, with the rejected points listed. **Met** (2026-10-24 01:33).

### Limitations (11.2)

* "0 seeds" is what the procedure found at budget 6 (381 joint samples, default seed configuration) for 23 points. It is
  not a claim that those designs have no three-contact grasp (rule 13); no 8.5 procedure was run on rejected points, as in
  8.4 and 10.2.
* The start set has less spread than 10.2's (2 of 23 at scale 0.5, against 6 of 24). Two things contribute: the scale-0.5
  points nearly all lost the third finger, and the 33.30 mm separation is not a rejection rule while the seed count is. The
  set is as the frozen procedure produced it.
* The two seed searches (N11's fallback check and the start acceptance) ran on the GPU, as 8.4 and 10.2 did. The suites in
  11.4 run on the CPU, so N11's seed count there can differ (P10.2 notes the platform dependence of the seed IK).

### How to reproduce (11.2)

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline nominal --box d19 --out workspace/phase11/11.2 --starts-dir workspace/phase11/11.2 --n-starts 24
XLA_PYTHON_CLIENT_PREALLOCATE=false ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.admissible_baseline starts --box d19 --out workspace/phase11/11.2 --starts-dir workspace/phase11/11.2 --n-starts 24
```

---

## 11.3 Optimization

### Files changed (11.3)

| File | Change |
| --- | --- |
| `dkarma/experiments/selection_v2.py` | `runs --phase-dir`, the table below (`workspace/phase11/11.3/runs.{md,json}`). Also two fixes to the phase-directory code of 11.1, found in a dry run on copies of Phase 10 files before any Phase 11 evaluation: a key collision in the 11.4 table and the R5 reference's printed precision |

### Procedure (11.3)

`workspace/phase11/chain_113_gpu.sh` ran `admissible_baseline full --box d19 --kkt-method reduced --only K` for K = 0 … 23,
one process at a time on the GPU, from 2026-10-24 01:34 to 06:01, as frozen in P11.3. The settings:

* the 10.2 Full-D18 configuration: 40 iterations, 2 stages, 6 three-contact seeds found on the start, `ADAM_DEFAULTS`,
  learning rate 0.02, optimizer seed 0;
* exact validation (fresh seeds, exact SLSQP force check at every sub-step) every 10 iterations, each gated by the hard D19
  check;
* the D19 penalties of `D19_CONSTRAINTS`.

The GPU was shared throughout with another user's training job, which held 5.7 GB. Wall clocks include that contention.
The 11.4 CPU leg ran beside it (three workers).

### Results (11.3)

Per run (`workspace/phase11/11.3/runs.md`; designs in `workspace/phase11/11.3/full_d19/<start>/optimized_morphology/`). The
validations are the optimizer's own exact checks: 6 fresh seeds, GPU. The reported evaluation is the 11.4 suite.

| start | wall s | optimizer s | non-finite / clipped | exact validation at it 10 / 20 / 30 / 40 | returned (it, exact) | base separation (margin over 33.30) mm | shortest servo link (margin over 34) mm | shortest fingertip mm | longest link mm | reach mm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| start0_nominal (N11) | 839 | 87 | 0 / 0 | 0.0163 / 0.0193 / 0.0168 / 0.0216 | 40, 0.0216 | 48.8 (15.5) | 39.6 (5.6) | 26.5 | 45.3 | 156.2 |
| start1_sobol0_s0.25 | 527 | 87 | 0 / 0 | 0.0026 / 0.0076 / 0.0076 / 0.0076 | 20, 0.0076 | 50.5 (17.2) | 39.6 (5.6) | 25.6 | 44.4 | 153.2 |
| start2_sobol2_s0.25 | 900 | 87 | 0 / 5 | 0.0142 / 0.0144 / 0.0239 / 0.0272 | 40, 0.0272 | 53.9 (20.6) | 38.5 (4.5) | 25.7 | 44.7 | 151.8 |
| start3_sobol3_s0.25 | 420 | 78 | 0 / 0 | 0.0017 / 0.0017 / 0.0017 / 0.0017 | 10, 0.0017 | 49.9 (16.6) | 39.1 (5.1) | 22.1 | 46.3 | 153.8 |
| start4_sobol11_s0.5 | 750 | 76 | 0 / 0 | 0.0144 / 0.0174 / 0.0216 / 0.0198 | 30, 0.0216 | 53.0 (19.7) | 38.1 (4.1) | 22.5 | 44.8 | 151.1 |
| start5_sobol14_s0.5 | 600 | 75 | 0 / 0 | 0.0123 / 0.0104 / 0.0106 / 0.0104 | 10, 0.0123 | 54.8 (21.5) | 38.8 (4.8) | 21.4 | 48.1 | 155.0 |
| start6_sobol22_s0.25 | 854 | 75 | 0 / 0 | 0.0117 / 0.0208 / 0.0219 / 0.0263 | 40, 0.0263 | 45.8 (12.5) | 39.2 (5.2) | 27.7 | 45.1 | 159.8 |
| start7_sobol23_s0.25 | 287 | 76 | 0 / 0 | 0 / 0 / 0 / 0 (no seed found) | 10, 0.0000 | 47.2 (13.9) | 39.3 (5.3) | 22.4 | 45.9 | 154.4 |
| start8_sobol26_s0.25 | 675 | 75 | 0 / 4 | 0.0108 / 0.0178 / 0.0129 / 0.0145 | 20, 0.0178 | 46.7 (13.4) | 39.1 (5.1) | 23.9 | 46.0 | 153.9 |
| start9_sobol27_s0.25 | 748 | 75 | 0 / 1 | 0.0167 / 0.0216 / 0.0157 / 0.0162 | 20, 0.0216 | 49.0 (15.7) | 39.2 (5.2) | 25.4 | 44.0 | 155.0 |
| start10_sobol28_s0.25 | 599 | 68 | 0 / 1 | 0.0127 / 0.0118 / 0.0085 / 0.0126 | 10, 0.0127 | 47.4 (14.1) | 39.6 (5.6) | 23.5 | 43.4 | 149.2 |
| start11_sobol30_s0.25 | 692 | 68 | 0 / 1 | 0.0093 / 0.0133 / 0.0160 / 0.0181 | 40, 0.0181 | 52.6 (19.3) | 38.7 (4.7) | 24.2 | 42.9 | 151.3 |
| start12_sobol32_s0.25 | 701 | 68 | 0 / 1 | 0.0120 / 0.0171 / 0.0141 / 0.0183 | 40, 0.0183 | 43.2 (9.9) | 38.8 (4.8) | 25.8 | 44.1 | 153.0 |
| start13_sobol33_s0.25 | 631 | 68 | 0 / 0 | 0.0072 / 0.0121 / 0.0150 / 0.0184 | 40, 0.0184 | 49.7 (16.4) | 38.8 (4.8) | 27.8 | 44.9 | 155.0 |
| start14_sobol35_s0.25 | 387 | 68 | 0 / 0 | 0.0000 / 0.0029 / 0.0000 / 0.0007 | 20, 0.0029 | 40.6 (7.3) | 39.9 (5.9) | 24.6 | 49.1 | 155.8 |
| start15_sobol37_s0.25 | 838 | 68 | 0 / 0 | 0.0166 / 0.0197 / 0.0194 / 0.0240 | 40, 0.0240 | 43.3 (10.0) | 38.5 (4.5) | 27.0 | 45.6 | 155.3 |
| start16_sobol39_s0.25 | 672 | 68 | 0 / 0 | 0.0137 / 0.0207 / 0.0126 / 0.0151 | 20, 0.0207 | 45.4 (12.1) | 39.0 (5.0) | 28.0 | 44.5 | 155.4 |
| start17_sobol40_s0.25 | 712 | 68 | 0 / 0 | 0.0174 / 0.0157 / 0.0126 / 0.0220 | 40, 0.0220 | 47.7 (14.4) | 39.0 (5.0) | 24.8 | 44.7 | 151.0 |
| start18_sobol41_s0.25 | 714 | 68 | 0 / 3 | 0.0138 / 0.0168 / 0.0187 / 0.0153 | 30, 0.0187 | 42.2 (8.9) | 39.5 (5.5) | 22.8 | 46.0 | 155.8 |
| start19_sobol42_s0.25 | 411 | 68 | 0 / 0 | 0.0033 / 0.0000 / 0.0017 / 0.0017 | 10, 0.0033 | 49.8 (16.5) | 39.0 (5.0) | 20.9 | 43.1 | 149.1 |
| start20_sobol43_s0.25 | 731 | 68 | 0 / 0 | 0.0157 / 0.0133 / 0.0197 / 0.0214 | 40, 0.0214 | 57.0 (23.7) | 38.2 (4.2) | 27.6 | 44.8 | 152.2 |
| start21_sobol44_s0.25 | 412 | 68 | 0 / 0 | 0.0037 / 0.0030 / 0.0047 / 0.0054 | 40, 0.0054 | 51.5 (18.2) | 39.2 (5.2) | 23.4 | 46.3 | 151.8 |
| start22_sobol45_s0.25 | 789 | 68 | 0 / 2 | 0.0126 / 0.0139 / 0.0246 / 0.0255 | 40, 0.0255 | 56.2 (22.9) | 38.5 (4.5) | 28.0 | 43.3 | 154.7 |
| start23_sobol46_s0.25 | 636 | 68 | 0 / 0 | 0.0142 / 0.0124 / 0.0142 / 0.0169 | 40, 0.0169 | 46.3 (13.0) | 39.7 (5.7) | 25.8 | 47.0 | 155.5 |

*Compute.*

* Wall clock: 647 s mean, median 683, range 288–900 s; 4.31 h in total on the shared GPU.
* Optimizer time: 73 s mean, range 68–87 s; 29 min in total.
* Exact validation took 2.42 h of the wall clock.

Every run made 40 gradient evaluations, about 20 % below the 10.2 Full-D18 arm's 804 s mean wall clock.

*Gradient hygiene.* No non-finite gradient iteration in 24 runs. The norm-100 clip acted 18 times, in 8 runs.

*Admissibility of every returned design under D19: 24 of 24.*

| quantity | range over the returned designs |
| --- | --- |
| base-separation margin | 7.3–23.7 mm |
| servo-link margin over 34 mm | 4.1–5.9 mm (shortest servo link 38.1–39.9 mm) |
| fingertips | 20.9–28.0 mm |
| longest link | ≤ 49.1 mm (the 90 mm maximum is far away) |
| reach | ≤ 159.8 mm (≤ 250) |

*How the designs moved.* From start to returned design:

* the shortest servo link changed by −0.8 to +0.8 mm, so the 34 mm floor never bound;
* the shortest fingertip by 0 to +6.8 mm (median +2.8);
* the minimum base separation by −7.3 to +16.2 mm (median +6.2).

The optimizer lengthened fingertips and spread the bases. It did not shorten servo links.

*The returned iteration*, the best exact validation (P11.3): iteration 40 in 12 runs, 30 in 2, 20 in 5 and 10 in 5. The
returned exact validation (6 seeds, GPU) had a median of 0.0183 and a range of 0–0.0272.

*Start7 (`start7_sobol23_s0.25`).* Its start design had 6 three-contact seeds, and the smooth objective used them
re-projected. No exact validation of its iterates found a fresh seed: 381 joint samples, default configuration, budget 6,
at iterations 10, 20, 30 and 40. All four scored 0, and the first, iteration 10, was returned. That is what the procedure
found, not a claim that no grasp exists (rule 13). The design is evaluated in 11.4 like the others.

*Tests at the end of 11.3:* `karma_jax/tests` 101 passed, 1 skipped (GPU, right after the last run;
`workspace/phase11/karma_jax_tests_11.3.log`) and `dkarma/tests` 215 passed, 6 skipped (CPU, beside the 11.4
workers; `workspace/phase11/dkarma_tests_11.3.log`).

### Acceptance (11.3)

24 Full KaRMA + Adam runs at the 10.2 Full-D18 settings ran, with D19 enforced throughout (penalties and the gated exact
validation). Compute and every returned design's admissibility and margins are reported; all 24 are admissible. **Met**
(2026-10-24 06:01).

### Limitations (11.3)

* One run per start (optimizer seed 0). The optimizer-seed variance is not measured, as in 10.2.
* The returned design is the iterate with the best 6-seed exact validation on the GPU. Ties at a score of zero keep the
  earliest iterate (start7).
* The GPU was shared with another user's job, so the wall clocks are not comparable to a dedicated GPU.

### How to reproduce (11.3)

```bash
workspace/phase11/chain_113_gpu.sh 0 23          # GPU, ~4.3 h; skips the finished nominal and start stages
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 runs --phase-dir workspace/phase11 --out workspace/phase11/11.3
```

---

## 11.4 Evaluation of every returned design

### Files changed (11.4)

None beyond 11.1 and 11.3. The tools ran unchanged:

* `admissible_baseline suite --box d19 --per-seed-detail` and `sweep --box d19`;
* `simulation_replay export --libraries three_contact --budget 12` and `simulate`;
* `selection_v2 r3 / evaluate --phase-dir` (with `robust_grasps report`).

### Procedure (11.4)

`workspace/phase11/chain_114_cpu.sh` ran on the CPU from 2026-10-24 01:34 to 09:04, with three workers, a 16 GB memory
gate and a 14 GB RSS watchdog per process; no job failed or was killed. It ran in the frozen order (P11.4):

1. N11's suite under three configurations with per-seed detail (01:34–01:44).
2. N11's `S_task` Full score, written as the R3 reference: `workspace/phase11/11.4/r3_reference.json`, 01:44:25. Its
   record states that no other suite existed then.
3. Then, as each GPU run finished, per design:
   * the default-configuration export (12 seeds);
   * four replays: {central pad 20A/2 mm, rigid} × {force-floor at its 9.3 parameters, position servo with a 0.75 mm
     compliant / 0.5 mm rigid squeeze}, trajectories run to their end;
   * the suite;
   * the load sweep.

N11 itself was exported, replayed and swept as a descriptive reference; it is not a candidate.

Every number below is on the fixed physical task in the D19 box. Script: `selection_v2 evaluate --phase-dir
workspace/phase11` → `workspace/phase11/11.4/evaluation.{md,json}`, which reads the suites (`11.4/suite/`), the sweeps
(`11.4/sweeps/`), the replays (`11.4/sim/default/{20A_2mm,rigid}/`), the exports (`11.4/sim/default/*_trajectories.npz`)
and the `S_task` report (`11.4/stask_report.{md,json}`). Design files: `workspace/phase11/11.3/full_d19/<start>/
optimized_morphology/`. Optimizer seed 0. Seed configurations: `default`, `sobol512_rs7`, `lin7_rs13`, as in
`admissible_baseline.SEED_CONFIGS`.

### Results (11.4)

*Endpoints.* `E` is the P10.2 endpoint: compliant 20A/2 mm, force-floor controller, strict success (4 mm / 0.2 rad and
every assigned fingertip in contact at the primitive end) counted to the first failure regardless of the exact verdict,
per budgeted seed. `E_R` and `E_T` are its rotation and translation parts. A drop event is the 9.3 drop criterion met in
that replay: a *release* (no fingertip in contact) or a *deviation* (more than 20 mm off the plan while touching). There
are 72 translation and 72 rotation trajectories per design.

| design | three-contact seeds per config | exact Full, mean ± sd over configs | `\|S_task\|` per config | exact Full on `S_task` | two-finger exact | E | E_R | E_T | drops T: release / deviation | drops R: release / deviation | compliant FP (exactly valid) | E, rigid, force-floor | 70 g: retention (seeds holding) | seeds holding at the continuous ceiling |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| N11 (start design) | 12 / 12 / 12 | 0.0095 ± 0.0009 | 3 / 3 / 2 | 0.0197 | 0.1620 | 1.42 | 1.17 | 0.25 | 71 / 0 | 68 / 0 | 0.74 (47) | 1.17 | 0.27 (3/12) | 0 |
| start0_nominal | 12 / 12 / 12 | 0.0187 ± 0.0016 | 6 / 8 / 5 | 0.0251 | 0.2392 | 17.17 | 14.00 | 3.17 | 58 / 1 | 51 / 0 | 0.05 (169) | 9.08 | 0.69 (7/12) | 0 |
| start1_sobol0_s0.25 | 12 / 12 / 12 | 0.0074 ± 0.0015 | 1 / 2 / 0 | 0.0094 | 0.1565 | 0.00 | 0.00 | 0.00 | 69 / 1 | 70 / 0 | 1.00 (48) | 0.75 | 0.10 (1/12) | 0 |
| **start2_sobol2_s0.25** | 12 / 12 / 12 | 0.0211 ± 0.0013 | 10 / 11 / 8 | 0.0224 | 0.2621 | **29.75** | 24.08 | 5.67 | 57 / 3 | 33 / 0 | 0.10 (323) | 14.83 | 0.92 (11/12) | 1 |
| start3_sobol3_s0.25 | 11 / 12 / 12 | 0.0046 ± 0.0020 | 0 / 0 / 2 | 0.0070 | 0.1570 | 0.00 | 0.00 | 0.00 | 66 / 0 | 66 / 0 | – (0) | 0.00 | 0.00 (0/11) | 0 |
| start4_sobol11_s0.5 | 12 / 12 / 12 | 0.0170 ± 0.0015 | 11 / 7 / 10 | 0.0193 | 0.2072 | 16.92 | 14.75 | 2.17 | 53 / 7 | 34 / 0 | 0.22 (201) | 6.58 | 0.88 (11/12) | 5 |
| start5_sobol14_s0.5 | 12 / 12 / 12 | 0.0083 ± 0.0018 | 4 / 2 / 5 | 0.0134 | 0.1007 | 0.00 | 0.00 | 0.00 | 72 / 0 | 72 / 0 | – (0) | 0.00 | 0.42 (4/12) | 0 |
| start6_sobol22_s0.25 | 12 / 12 / 12 | **0.0282** ± 0.0009 | 9 / 10 / 8 | **0.0297** | 0.2744 | 27.00 | 20.08 | **6.92** | 49 / 2 | 35 / 0 | 0.35 (435) | 14.33 | 0.76 (9/12) | 1 |
| start7_sobol23_s0.25 | 0 / 0 / 0 | 0.0000 | 0 / 0 / 0 | 0.0000 | 0.0053 | 0.00 | 0.00 | 0.00 | – | – | – (0) | 0.00 | – | – |
| start8_sobol26_s0.25 | 12 / 12 / 12 | 0.0161 ± 0.0012 | 9 / 9 / 8 | 0.0161 | 0.2453 | 16.08 | 13.17 | 2.92 | 63 / 1 | 45 / 0 | 0.23 (169) | 9.92 | 0.67 (9/12) | 0 |
| start9_sobol27_s0.25 | 12 / 12 / 12 | 0.0167 ± 0.0007 | 8 / 8 / 9 | 0.0204 | 0.2555 | 18.17 | 15.08 | 3.08 | 61 / 3 | 47 / 0 | 0.18 (209) | 7.33 | 0.81 (9/12) | 1 |
| start10_sobol28_s0.25 | 12 / 12 / 12 | 0.0121 ± 0.0008 | 2 / 4 / 3 | 0.0185 | 0.1909 | 5.08 | 4.08 | 1.00 | 68 / 2 | 66 / 0 | 0.08 (53) | 2.58 | 0.24 (2/12) | 0 |
| start11_sobol30_s0.25 | 12 / 12 / 12 | 0.0142 ± 0.0002 | 11 / 10 / 12 | 0.0137 | 0.1797 | 8.00 | 7.00 | 1.00 | 66 / 3 | 52 / 0 | 0.55 (167) | 6.42 | 0.85 (11/12) | 2 |
| start12_sobol32_s0.25 | 12 / 12 / 12 | 0.0180 ± 0.0007 | 11 / 8 / 10 | 0.0176 | 0.2550 | 11.83 | 10.00 | 1.83 | 62 / 5 | 48 / 0 | 0.40 (178) | 6.42 | 0.78 (11/12) | 0 |
| start13_sobol33_s0.25 | 12 / 12 / 12 | 0.0181 ± 0.0022 | 10 / 9 / 11 | 0.0180 | 0.2433 | 22.33 | 17.67 | 4.67 | 46 / 6 | 39 / 0 | 0.11 (239) | 14.33 | 0.84 (10/12) | 2 |
| start14_sobol35_s0.25 | 12 / 12 / 12 | 0.0024 ± 0.0013 | 0 / 1 / 0 | 0.0034 | 0.0996 | 0.00 | 0.00 | 0.00 | 72 / 0 | 72 / 0 | – (0) | 0.00 | 0.00 (0/12) | 0 |
| start15_sobol37_s0.25 | 12 / 12 / 12 | 0.0190 ± 0.0009 | 9 / 11 / 11 | 0.0197 | 0.2314 | 18.42 | 14.83 | 3.58 | 56 / 3 | 46 / 0 | 0.07 (182) | 10.92 | 0.80 (9/12) | 5 |
| start16_sobol39_s0.25 | 12 / 12 / 12 | 0.0192 ± 0.0012 | 11 / 11 / 9 | 0.0196 | 0.2651 | 27.83 | 23.50 | 4.33 | 53 / 8 | 42 / 0 | 0.04 (282) | 10.42 | 0.87 (11/12) | 3 |
| start17_sobol40_s0.25 | 12 / 12 / 12 | 0.0167 ± 0.0006 | 9 / 12 / 12 | 0.0179 | 0.2105 | 20.50 | 17.25 | 3.25 | 62 / 1 | 49 / 0 | 0.23 (256) | 9.17 | 0.91 (9/12) | 0 |
| start18_sobol41_s0.25 | 12 / 12 / 12 | 0.0178 ± 0.0021 | 10 / 10 / 8 | 0.0193 | 0.2125 | 25.17 | 21.58 | 3.58 | 54 / 5 | 43 / 0 | 0.05 (256) | 9.75 | 0.86 (10/12) | 1 |
| start19_sobol42_s0.25 | 12 / 12 / 12 | 0.0018 ± 0.0003 | 0 / 0 / 0 | 0.0000 | 0.1337 | 0.00 | 0.00 | 0.00 | 72 / 0 | 72 / 0 | – (0) | 0.00 | 0.00 (0/12) | 0 |
| start20_sobol43_s0.25 | 12 / 12 / 12 | 0.0183 ± 0.0016 | 11 / 12 / 11 | 0.0190 | 0.2763 | 21.08 | 18.25 | 2.83 | 58 / 7 | 41 / 0 | 0.16 (251) | 12.50 | 1.00 (11/12) | 2 |
| start21_sobol44_s0.25 | 7 / 12 / 7 | 0.0046 ± 0.0009 | 0 / 0 / 0 | 0.0000 | 0.1667 | 0.00 | 0.00 | 0.00 | 42 / 0 | 42 / 0 | – (0) | 0.00 | 0.00 (0/7) | 0 |
| start22_sobol45_s0.25 | 12 / 12 / 12 | 0.0191 ± 0.0024 | 9 / 10 / 9 | 0.0216 | 0.2578 | 19.42 | 15.83 | 3.58 | 61 / 2 | 47 / 0 | 0.30 (264) | 14.42 | 0.83 (10/12) | 0 |
| start23_sobol46_s0.25 | 12 / 12 / 12 | 0.0166 ± 0.0016 | 9 / 11 / 10 | 0.0174 | 0.2315 | 14.50 | 12.83 | 1.67 | 64 / 3 | 52 / 0 | 0.15 (158) | 9.83 | 0.73 (9/12) | 0 |

The compliant position-servo and rigid position-servo values are in `evaluation.md`.

*Distribution over the 24 optimized designs (descriptive):*

* `E`: mean 13.30, median 16.50, best 29.75 (start2), IQR [0.00, 20.65];
* `E_R`: mean 11.00, median 13.58, best 24.08;
* `E_T`: mean 2.30, median 2.50, best 6.92 (start6);
* the exact Full score: mean 0.0140, median 0.0167, best 0.0282 (start6);
* the two-finger exact score: mean 0.202, best 0.2763 (start20).

Seven designs have `E = 0`: start1, 3, 5, 7, 14, 19 and 21. Five of them (start3, 5, 14, 19, 21) hold none of their
export's grasps in simulation, or have none. These are also the designs whose own optimizer validations were near zero
(11.3). N11 itself scores `E = 1.42`, with 2 of its 12 grasps holding.

*Drop events* (compliant pad, force-floor, pooled over the designs with an export): 1,620 translation and 1,620 rotation
trajectories.

* Translation: 1,384 releases and 63 deviations.
* Rotation: 1,164 releases and no deviation.

Nearly every drop is a release: the sphere leaves all three fingertips. A deviation of more than 20 mm while still
touching occurs only on translations. On the selected design, translations give 57 releases and 3 deviations of 72;
rotations give 33 releases.

**Do the new designs do better on translations? No.** The Phase 10 selection (Full-D18 start31, D18 box) had
`E = 45.0 = 38.6 + 6.4`.

* Over the 24 Phase 11 designs, `E_T` averages 2.3. One design exceeds 6.4 (start6, 6.92); the selected start2 has 5.67.
* No design reaches 38.6 on rotations (best 24.08).
* Translations remain the weak part: they are 17 % of the pooled `E` (2.30 of 13.30), against 14 % for start31. On the
  selected design they are 19 % (5.67 of 29.75).

This is a descriptive comparison and makes no claim.

*Two-finger seeds under D19* (the per-capsule re-check of 11.1): every design except start7 found the full 16 two-finger
seeds under every configuration, so the re-check did not starve the two-finger library.

*Start7.* The suite found no three-contact seed under any configuration:

| configuration | joint samples | reaching | candidates | IK, kinematically valid |
| --- | --- | --- | --- | --- |
| default | 381 | 0 | 0 | – |
| `sobol512_rs7` | 669 | 1 | 3 | 3 IK attempts, 0 valid |
| `lin7_rs13` | 341 | 0 | 0 | – |

Its two-finger seeds were 0 / 0 / 2. This is what the procedure found, with these attempt counts. No claim is made that
the design has no grasp (rule 13), and the 8.5 procedures were not run, because nothing depends on it: start7 is
ineligible under R1 either way.

*Tests at the end of 11.4:* no code changed after the 11.3 test runs, so those results stand: `dkarma/tests` 215 passed,
6 skipped; `karma_jax/tests` 101 passed, 1 skipped.

### Acceptance (11.4)

Every returned design and N11 have the suite with per-seed detail, the default-configuration export, the four replays and
the load sweep. `E` is reported with its rotation/translation split and the drop events with their release/deviation
split. The translation question is answered descriptively. **Met** (2026-10-24 09:04).

### Limitations (11.4)

* The simulation is MuJoCo with the assumed 10.1 pad and the 9.3 controllers. It is not hardware.
* R3 rests on N11's robust grasps, which are few: 3 / 3 / 2 seeds, with `S_task` Full 0.0159 / 0.0179 / 0.0255 per
  configuration. The reference is therefore noisier than Phase 10's (0.0196 from 11 / 12 / 12 robust seeds of the Phase 10
  start).
* The Phase 10 comparison is across two design boxes, with different starts, radii and floors, so it measures what the
  envelope costs to this procedure, not to the best possible D19 hand.

### How to reproduce (11.4)

```bash
workspace/phase11/chain_114_cpu.sh 0 23          # CPU, 3 workers, ~7.5 h beside the GPU leg
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 evaluate --phase-dir workspace/phase11 --out workspace/phase11/11.4
```

---

## 11.5 Selection

### Files changed (11.5)

| File | Change |
| --- | --- |
| `dkarma/experiments/visualize_design.py` | the label of the `full_d19` arm, and the phase name in the standing chart's title, both in the phase-directory branch. Also a layout fix to `wrench_capacity.png`: its wrapped subtitle is anchored at its top, so it no longer runs into the title. The overlap was present in Phase 10's copy too; that kept copy was not regenerated, and regenerating it now would differ from it only in the subtitle's position |
| `results/fabrication_candidates/selected_full_d19_start2_sobol2_s0.25/` (new) | the selected specification: design JSON (per-capsule radii), URDF (with the capsule visuals), robot YAML, actuator specification, run record and config. `selection_numbers.json` holds the export check, `phase11_numbers.json` the criterion's rows and the 11.4 evaluation, and `figures/` the renders, replay videos and charts |
| `results/fabrication_candidates/README.md` | the Phase 11 outcome |

### Procedure (11.5)

`selection_v2 select --phase-dir workspace/phase11 --out workspace/phase11/11.5 --export results/fabrication_candidates
--rescore` (2026-10-24) applied the criteria to the 24 designs returned by 11.3:

* **criterion v2**, exactly as frozen (SPEC §16 item 12, P10.4; still subject to the owner's review), with eligibility
  under D19. The R3 reference is N11's `S_task` Full score, 0.01974, read from `workspace/phase11/11.4/r3_reference.json`
  and consistent with the `S_task` report. The R5 reference is 0.2393 (10.1);
* **the original 9.5 criterion** alongside: requirements 1, 3 and 4, with the 9.3 two-finger baseline's rigid
  false-positive rates of 0.185 (position servo) and 0.182 (force-floor) and the Phase 11 rigid replays.

Output: `workspace/phase11/11.5/selection.{md,json}`. Figures: `visualize_design --phase-dir workspace/phase11 --key
full_d19/start2_sobol2_s0.25 --seed 4`. Grasp seed 4 is shown because it is the design's best seed in the 11.4 replay
(69 of 84).

### Results (11.5)

Fixed physical task, D19 box. **Criterion v2** and the original criterion side by side. They are **not equivalent**.

| requirement | designs passing (of 24) | failures |
| --- | --- | --- |
| R1 eligibility (D19-admissible, two-finger > 0 and a three-contact seed under every configuration, export reproduces) | 23 | start7 (no three-contact seed found; two-finger 0 / 0 / 0.016) |
| R2 `\|S_task\|` ≥ 6 of 12 under every configuration | 15 | — |
| R3 exact Full on `S_task` ≥ N11's 0.0197 | 5: start0, 2, 6, 9, 22 | — |
| R4 70 g held and ≥ 50 % retained | 16 | — |
| R5 compliant force-floor FP ≤ 0.2393 with ≥ 10 exactly valid | 13 | — |
| **all of R1–R5** | **2: start2, start9** | start0 fails only R2 (lin7_rs13: 5 robust grasps); start6 and start22 fail only R5 (FP 0.35 and 0.30) |

| candidate (passing v2) | R2: `\|S_task\|` | R3: Full on `S_task` (± sd over configs) | R4: 70 g retention (seeds) | R5: compliant FP (exactly valid) | original 9.5: 1 / 3 (min capacity) / 4 (rigid FP, position / force) |
| --- | --- | --- | --- | --- | --- |
| **full_d19/start2_sobol2_s0.25** | 10 / 11 / 8 | **0.0224 ± 0.0009** | 0.92 (11/12) | 0.10 (323) | yes / ✗ (0.75) / ✗ (0.66 / 0.59) |
| full_d19/start9_sobol27_s0.25 | 8 / 8 / 9 | 0.0204 ± 0.0009 | 0.81 (9/12) | 0.18 (209) | yes / ✗ (0.00) / ✗ (0.73 / 0.63) |

**Criterion v2 decision: Full-D19 start2 (`full_d19/start2_sobol2_s0.25`) is selected.** Its `S_task` score exceeds
start9's by 0.0020, which is more than the larger seed-configuration standard deviation (0.0009). It is not a tie, so the
tie-break is not used.

**Original criterion: nothing is selected.** Every eligible candidate fails both requirements:

* *requirement 3*: some default-configuration grasp carries less than 1 × weight in some direction (the best minimum,
  0.97, is start12's);
* *requirement 4*: the rigid three-contact FP exceeds the two-finger baseline's under both controllers.

*Two-finger cost.* The selection's exact two-finger score is 0.2621, against 0.2763 for the best Phase 11 design (start20):
−0.014, −5 %. There is no arm A′ in Phase 11.

*Export check* (`selection_numbers.json`). The exported parameters are identical to the scored design's. The re-score from
the exported copy (default configuration) reproduces the suite exactly: exact Full 0.021995, two-finger 0.275267. The URDF
carries the per-capsule radii as visual capsules: 36 visual elements, 9 servo links at 16.40 mm and 3 fingertips at
11.8 mm.

*The selected design* (`workspace/phase11/11.3/full_d19/start2_sobol2_s0.25/`; Full KaRMA + Adam under D19, 40 iterations,
optimizer seed 0, from Sobol point 2 at scale 0.25; returned at iteration 40):

| finger | base (mm, palm frame) | links 0 / 1 / 2 / fingertip (mm) |
| --- | --- | --- |
| index | (−4.3, −8.9, 11.6) | 40.4 / 40.4 / 40.5 / 30.4 |
| middle | (9.7, 47.7, 1.0) | 40.2 / 41.6 / 38.5 / 31.1 |
| thumb | (−10.9, −44.4, −28.3) | 39.6 / 44.7 / 39.0 / 25.7 |

* Radii 16.40 mm (links 0–2) and 11.8 mm (fingertips).
* Minimum base separation 53.9 mm (≥ 33.30); shortest servo link 38.5 mm (≥ 34); reach 151.8 mm (≤ 250).
* Exact Full score 0.0211 ± 0.0013 over the three configurations, 0.0224 on `S_task`.
* Robust grasps 10 / 11 / 8 of 12; the weakest grasp carries 0.75 × weight.
* Load sweep (12 default-configuration seeds):

  | condition | exact Full | seeds holding |
  | --- | --- | --- |
  | stall and primary ceilings | 0.0220 | 12 / 12 |
  | 70 g at the primary ceiling | 0.0202 | 11 / 12 |
  | continuous ceiling | **0.0008** | **1 / 12** |

* Simulated three-contact manipulation: `E` = 29.75 = 24.08 (rotations) + 5.67 (translations), the best of the 24
  designs. Under rigid contact `E` = 14.83. The compliant FP is 0.10 over 323 exactly valid primitives.

**Phase 10 comparison: what the envelope costs.** Descriptive, not a claim. The two designs differ in box, start and run,
so this is the cost to this procedure.

| measure | Phase 10 selection: Full-D18 start31 (D18 box, 11.8 mm capsules) | Phase 11 selection: Full-D19 start2 (D19 box) | change |
| --- | --- | --- | --- |
| exact Full score, mean over 3 configurations | 0.0237 | 0.0211 | −11 % |
| exact Full score on `S_task` | 0.0241 | 0.0224 | −7 % |
| robust grasps (per configuration) | 12 / 11 / 12 | 10 / 11 / 8 | −2 on average |
| weakest default-configuration grasp | 1.92 × weight | 0.75 × weight | below 1 |
| `E`: total = rotations + translations | 45.0 = 38.6 + 6.4 | 29.75 = 24.1 + 5.7 | −34 % (rotations −38 %, translations −12 %) |
| `E`, rigid contact (force-floor) | 26.8 | 14.8 | −45 % |
| `E`, compliant, position servo | 42.2 | 25.1 | −41 % |
| compliant FP (exactly valid) | 0.07 (386) | 0.10 (323) | +0.03 |
| 70 g: seeds holding, retention | 12 / 12, 1.00 | 11 / 12, 0.92 | |
| continuous ceiling (0.043 N·m): score kept, seeds holding | 73 %, 10 / 12 | **3 %, 1 / 12** | the largest cost |
| two-finger exact score | 0.2767 | 0.2621 | −5 % |
| reach | 115.8 mm | 151.8 mm | +36 mm |
| R3 reference (each phase's unoptimized start) | 0.0196 | 0.0197 (N11) | |

Across the phases, the 24 Phase 11 designs average `E` 13.3 (median 16.5), against 26.6 (median 29.5) for the 24 Full-D18
designs of 10.2. Seven of the 24 Phase 11 designs have `E = 0`, against none of the 24 Full-D18 designs of 10.2.

*Tests at the end of 11.5 and of Phase 11* (final code): `dkarma/tests` 215 passed, 6 skipped (CPU, 28 min;
`workspace/phase11/dkarma_tests_11.5.log`); `karma_jax/tests` 101 passed, 1 skipped (GPU;
`workspace/phase11/karma_jax_tests_11.5.log`).

### Acceptance (11.5)

**Met.** Criterion v2 selected a design, reported with the criterion's numbers and with the original criterion's verdict
(nothing selected) beside it. The selected design is exported and reproduces the suite from the exported copy; its figures
are rendered with `visualize_design`; `results/fabrication_candidates/README.md` records the outcome. Criterion v2 is the
session's (SPEC §16 item 12) and is **subject to the owner's review**.

### Limitations (11.5)

* Criterion v2's thresholds were set by the Phase 10 session and have not been reviewed by the owner.
* R3 compares with N11, whose robust grasps are few (2–3 per configuration), and it is the requirement most designs fail
  (19 of 24). A different reference would change the candidate set.
* The selection is the best of 24 on the `S_task` score, which carries a selection effect. Its optimizer-seed variance is
  unmeasured.
* The selected design's continuous-duty margin is thin, and its weakest grasp is below 1 × weight. Both are visible only
  in the measures the original criterion and the sweep report, not in v2's requirements.

### How to reproduce (11.5)

```bash
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.selection_v2 select --phase-dir workspace/phase11 --out workspace/phase11/11.5 --export results/fabrication_candidates --rescore
MUJOCO_GL=osmesa JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.visualize_design --phase-dir workspace/phase11 --key full_d19/start2_sobol2_s0.25 --seed 4 --out results/fabrication_candidates/selected_full_d19_start2_sobol2_s0.25/figures
```

---

## Phase 11 in one page: the selection, what the envelope costs, the recommendation

Fixed physical task throughout (15 mm / 35 g sphere, ±100 mm cube, XL330 primary ceiling 0.108 N·m; SPEC §2). The D19 box
applies: links 0–2 are 16.40 mm capsules at least 34 mm long; fingertips are 11.8 mm and 12–90 mm; finger bases are at
least 33.30 mm apart. The rules were frozen on 2026-10-23 at 23:59, before any Phase 11 data. Amendment A1 (the owner's
withdrawal of the link-0 relaxation, 2026-10-24 00:14) also predates all D19 data. Phase 11 makes no comparative claim.

Tests at the end of Phase 11 (final code): `dkarma/tests` 215 passed, 6 skipped (205 + 10 new); `karma_jax/tests` 101
passed, 1 skipped (`karma_jax` unchanged).

### What was done

* **11.1 D19 in code.** Every radius use is classified as contact surface or collision envelope. Per-capsule radii now
  reach the MuJoCo scene, the URDF, the 8.5 verifier and the two-finger reference path (by a per-capsule re-check, since
  `karma_jax` is not extended). The D19 box, its hard check and its penalty are in place. With uniform radii and the D18
  box, a Phase 10 suite, export and two replays reproduce **bit for bit**.
* **11.2 Start set.** N11 (40 / 40 / 40 / 23 mm fingers, a 40 / 45 / 40 / 25 mm thumb, 40 mm pitch) is D19-admissible with
  12 of 12 three-contact seeds, so neither fallback was used. There are 24 starts, all admissible; 24 of the 47 Sobol
  points drawn were rejected for lack of seeds.
* **11.3 Optimization.** 24 Full KaRMA + Adam runs at the 10.2 settings. All 24 returned designs are D19-admissible; no
  run had a non-finite gradient. Wall clock 4.3 h on a shared GPU, optimizer time 29 min.
* **11.4 Evaluation.** Every design and N11 got the suite, export, four replays and load sweep. `E` has a mean of 13.3,
  median 16.5 and best 29.75; seven designs have `E = 0`. Nearly every drop event is a release; deviations occur only on
  translations (63 of 1,447 translation drops). **The new designs do not do better on translations**: `E_T` averages 2.3,
  and one design exceeds the Phase 10 selection's 6.4.
* **11.5 Selection.** Criterion v2 passes 2 of 24 designs and **selects Full-D19 start2** without a tie (`S_task` Full
  0.0224 against 0.0204). The original 9.5 criterion selects nothing: requirement 3, the weakest grasp below 1 × weight,
  and requirement 4, rigid three-contact FP above the two-finger baseline, fail for every candidate.

### The selected design (criterion v2, subject to the owner's review)

**Full-D19 start2** (`results/fabrication_candidates/selected_full_d19_start2_sobol2_s0.25/`, with `figures/`):

| quantity | value |
| --- | --- |
| links 0 / 1 / 2 / fingertip | index 40.4 / 40.4 / 40.5 / 30.4 mm; middle 40.2 / 41.6 / 38.5 / 31.1 mm; thumb 39.6 / 44.7 / 39.0 / 25.7 mm |
| base separation | 53.9 mm |
| reach | 151.8 mm |
| exact Full score | 0.0211 ± 0.0013 (all seeds), 0.0224 (`S_task`) |
| robust grasps | 10 / 11 / 8 of 12 |
| weakest grasp | 0.75 × weight |
| simulated three-contact manipulation `E` (compliant pad, force-floor) | 29.75 = 24.08 rotations + 5.67 translations, the best of the 24 |
| compliant FP | 0.10 (323 exactly valid) |
| 70 g | 11 / 12 seeds hold, 92 % retained |
| continuous ceiling (0.043 N·m) | **1 of 12 seeds hold, 3 % retained** |
| two-finger exact score | 0.2621, −5 % against the best Phase 11 design |
| export check | the exported copy reproduces the suite exactly |

### What the servo envelope costs, against the Phase 10 selection (descriptive)

Against Full-D18 start31 (D18 box):

| measure | change under D19 |
| --- | --- |
| exact Full score | −11 % (−7 % on `S_task`) |
| simulated manipulation `E` | −34 % (45.0 → 29.75); rotations −38 %, translations −12 % |
| `E` under rigid contact | −45 % |
| robust grasps | 11.7 → 9.7 per configuration |
| weakest grasp | 1.92 → 0.75 × weight |
| continuous-ceiling retention | 73 % → 3 % (10 → 1 of 12 seeds) |
| two-finger exact score | −5 % |
| reach | +36 mm |

The thicker, longer servo links leave less free space around the 30 mm sphere and lengthen the moment arms. The clearest
cost is torque headroom: the D19 hand holds every grasp at the primary ceiling and at stall, but almost none at the
continuous ceiling. Across all 24 designs the procedure's typical result falls by 44 % (`E` median 29.5 → 16.5), and seven
starts end without simulated manipulation.

### Fabrication recommendation, with its uncertainty

**Recommended, subject to the owner's review of criterion v2: Full-D19 start2**, built with silicone fingertip pads of about
Shore 20A and 2 mm, as in Phase 10. It supersedes the Phase 10 selection, which D19 excludes. The runner-up under v2 is
Full-D19 start9.

The uncertainty has six parts:

1. **Criterion v2 is the session's**, not yet reviewed. Under the original criterion nothing is selected. Its R3 reference
   (N11's 0.0197) rests on 2–3 robust grasps per configuration.
2. **Torque headroom.** At the continuous estimate (0.043 N·m, 20 % of stall) one grasp of 12 holds. A sustained hold
   therefore relies on intermittent-duty torque near the 0.108 N·m primary ceiling. Before fabrication: a measured
   joint-side torque curve and a thermal check of a sustained hold.
3. **The weakest grasp is 0.75 × weight.** Two of the 12 default grasps are not robust (10 of 12 in `S_task`). A grasp
   planner must avoid them.
4. **The evidence is simulated**, with the assumed 10.1 pad and the 9.3 force-floor controller. Under rigid contact `E`
   halves (14.8), and the rigid three-contact FP (0.59–0.66) is far above the two-finger baseline. The real pad's
   torsional friction matters (Phase 10).
5. **Selection effect and variance.** The design is the best of 24 by v2's measure (and by `E`), with one optimizer run per
   start.
6. **Packaging beyond D19.** The capsules contain the servo cases, but horn placement, cabling, joint 0's servo in the palm
   and the palm itself are out of scope (D19). Link 0 must now hold a case along its 40 mm length.

### Remaining

1. The owner's review of criterion v2 (SPEC §16 item 12), and with it the fabrication decision between Full-D19 start2 and
   nothing.
2. Before fabrication: the torque curve and a thermal hold test (part 2 above); an indentation test of the pad silicone;
   and a re-run of the selected design's compliant replay with the measured parameters.
3. The finding of 11.1: the exact two-finger library's self-collision gate is thumb–index only, in every phase. Changing
   it to all-finger would change Phase 9–10 numbers; the owner's call.
4. Not addressed: optimizer-seed variance; why half of the Sobol points around N11 have no three-contact seed (11.2); and
   whether a 34 mm floor on link 0 is the right model of a knuckle servo.
5. SPEC §14 and the status line of CLAUDE.md still describe Phase 11 as current. They are maintained by the owner's SPEC
   session and were not edited here.
