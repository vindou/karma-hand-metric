# What does KaRMA reward? An adversarial-optimization audit of a kinematic dexterity metric

**Status:** draft, results arriving. Every slot marked **[TBD …]** is filled when the corresponding
queue job finishes (`tail optimize/queue_done.log`). Machine-generated tables for all arms are
regenerated into `RESULTS_AUTO.md` by `python optimize/report.py` and are the source for the
numbers below.

**Audience:** our group, with an eye to a workshop note. Readers are assumed not to have read the
KaRMA paper; Section 1 summarises it with pointers. Paper references use the arXiv v2 HTML
numbering (sections, equations, tables) and should be checked against the PDF before external use.
Code pointers are `file:line` in this repository at the commit the experiment ran on.

---

## Abstract

**[TBD final; provisional]** KaRMA (Peticco & Agrawal, IROS 2026, arXiv:2605.15548) scores a robot
hand's fine-manipulation ability from its kinematic tree alone by searching for the poses a small
sphere can reach when rolled in a thumb–index pinch. We ask what the metric actually rewards by
optimizing a free two-finger kinematic tree *against* it with CMA-ES, re-scoring the winners with the
official pipeline, and running controlled sweeps over degrees of freedom, joint-range caps,
objective (translation vs rotation), object size, and the metric's own normalisation. In the first
run, 89 generations from an anthropomorphic start produced a hand whose official KaRMA-T is **0.575**,
5.9× the best published hand (LEAP, 0.097), by collapsing the palm to 16 mm between knuckles and
turning both fingers into long styli; its KaRMA-R (0.277) did not exceed the best real hands and its
KaRMA-S (0.795) exceeded all of them. We also show that the normalising length L_ref can be driven
down by vestigial fingers that never touch the object, inflating KaRMA-T by at least 27× on an
unchanged pinch geometry. **[TBD: recurrence across restarts, DOF and range scaling, T-vs-R
divergence, radius sensitivity, and whether a pinch-only L_ref preserves the published ranking.]**
We close with recommendations for the metric and for what, if anything, is worth building.

---

## 1. Introduction

### 1.1 What KaRMA measures

KaRMA ("Kinematic Rolling Manipulation Ability") answers one question about a hand: *holding a
small sphere in a two-finger thumb–index precision pinch, how far can the hand translate and
reorient that sphere by rolling it, without regrasping, finger gaiting, or letting go?* Everything
is kinematic. There is no controller, no policy, no dynamics, no actuator model; the paper is
explicit that this is "a standardized lower bound on thumb–index rolling-pinch dexterity, not a
universal dexterity score" (paper Sec. VII). The motivation (paper Sec. I–II) is that the usual
hand metrics (workspace volume, Yoshikawa manipulability, grasp-quality indices such as GCI,
opposability) are static: they describe where a fingertip can be or how well a grasp resists
wrenches, but not whether the hand can *move an object while holding it*, which is what dexterity
means in practice.

**The three scores** (paper Sec. III-D, Eqs. 1–3; code `karma/metric.py:615-679`,
`karma/seed_selection.py:834-839`):

| score | definition | reading |
| --- | --- | --- |
| **KaRMA-T** | number of distinct sphere-centre voxels reached by a translation search, times voxel volume, divided by L_ref³ (Eq. 1). With voxel edge h = 0.05 L_ref this is simply voxels/8000, i.e. the fraction of an L_ref-sided cube the reachable set fills. Averaged over the three best starting pinches. | translational reach of the pinch, size-normalised |
| **KaRMA-R** | at each reached voxel, tilt the pinch and record which of 228 orientation bins of the pinch axis (in the sphere's body frame) are reached; KaRMA-R is the mean coverage of the five best voxels (Eq. 2), again averaged over the three best pinches. | rotational reach at the best spots |
| **KaRMA-S** | median over all evaluated starting pinches of the voxel count, divided by the best (Eq. 3). | how much the score depends on finding one special grasp; 1.0 means every pinch is as good as the best |

The paper notes KaRMA-S "should be interpreted jointly with KaRMA-T" (Sec. III-D3): a hand that
reaches almost nothing from every pinch scores S near 1 trivially.

**Non-dimensionalisation by L_ref** (paper Sec. III-B; `karma/lref.py`, `karma/config.py:191-208`;
`docs/scale_invariance_nondim.md`). Hands differ in size, so every length in the problem is a fixed
fraction of a characteristic hand length

> L_ref = mxPr + mF

where mxPr is the maximum pairwise distance between finger *knuckle* joint origins (palm spread) and
mF is the *median* knuckle-to-fingertip chain length over all finger chains (finger reach). The paper
justifies the sum as capturing "both palm spread and finger reach, the two geometric factors that
bound the envelope of feasible pinch locations". At the nominal L_ref = 200 mm the sphere radius is
10 mm, the finger capsule radius 11.8 mm, the voxel edge 10 mm, the contact-gap tolerance 0.35 mm,
the collision clearance 0.5 mm and the slip gate 0.5 mm (`karma_config.yaml:16-44`); all are scaled
by L_ref/200 mm for each hand. Two details matter later: the knuckle is taken as the first
non-mimic revolute joint of each chain, and a rule advances it distally when that joint's range is
under 25° or the chain is a length outlier (>1.35× the median), provided this *reduces* mxPr
(`karma/lref.py:41-45, 308`). Chains shorter than 10 mm are discarded as "hardware stubs"
(`karma/lref.py:41`).

**Geometry model** (paper Sec. IV-A; `karma/robot.py:187-436`). Finger links are capsules built
procedurally from the joint tree, not from the URDF meshes: each link's capsule runs from its
inbound joint to its child joint, and terminal links get a user-specified tip length. The palm has
no geometry at all. Contact between a capsule and the sphere reduces to a signed gap
g = ‖p − c‖ − r_s − r_ℓ (Eq. 4), which must stay within ε_g at both contacts.

**Constraints enforced at every step** (paper Sec. IV-B/C):

- joint limits from the URDF, plus a per-step cap of 10° per joint (`max_joint_step_deg`);
- no sphere penetration into any non-contact capsule, and no thumb–index self-collision, each with
  a clearance ε_col (`karma/collisions.py:9, 114`);
- antipodal force feasibility: the two inward contact normals must lie within 2·atan(μ) of
  anti-parallel, μ = 0.6 (paper Sec. V-B; `karma/grasp.py:55`);
- rolling contact: each step solves a QP (OSQP) that moves the sphere by a commanded displacement
  while minimising tangential slip under Montana's rolling kinematics (Eq. 5), followed by a
  Gauss–Newton projection back onto the two-contact manifold; a step is rejected outright if the
  residual tangential slip exceeds the gate (`docs/slip_tolerance_study.md`,
  `karma/seed_selection.py:356`).

**The search** (paper Sec. IV-D and III-D; `karma/seed_selection.py`, `karma/metric.py`):

1. *Seeds.* Sample joint configurations (per-joint sweeps, 512 Sobol points, 200 pseudo-random
   draws; `seed_selection.py:140-162`), keep those where the two fingertip capsules are within reach
   of each other, propose sphere centres between them, and solve a least-squares IK that places both
   contacts on the sphere. Every feasible solution is a candidate pinch; each is polished toward
   joint-range centre in the grasp nullspace.
2. *Phase 1, translation.* From each seed, a breadth-first search over a voxel grid aligned with the
   seed's manipulability ellipsoid, using six ±step primitives, eight sub-steps per voxel, each
   sub-step a rolling QP plus projection plus the full constraint check. All seeds are searched to
   the budget (10 000 voxels); the best three by voxel count form the ensemble.
3. *Phase 2, rotation.* At every voxel of the best seeds, tilt the pinch axis in four directions in
   8.6° steps until a constraint fails (`metric.py:143-191`), binning the axis with HEALPix
   (nside = 6, 432 pixels, 228 after identifying antipodes; paper Sec. III-C, `karma/orientation.py`).

**What the paper reports** (Sec. VI, Tables I–III). Sixteen hands span KaRMA-T from 0.097 (LEAP)
to 0.0003 (Ability). KaRMA-T correlates with workspace intersection (ρ = 0.93), total joint range
(ρ = 0.80) and DOF count (ρ = 0.71) but not with Yoshikawa manipulability (ρ = −0.20) or GCI
(ρ = −0.08); opposability misranks 4 of 16 hands by two or more places. T and R are strongly
rank-correlated (ρ = 0.96) yet no static baseline predicts where they diverge. The scores are
bit-reproducible and exactly invariant to translating, rotating or uniformly scaling the URDF
(Table III); joint limits are the most sensitive input, moving a hand's voxel count by up to 20%
(34% in one case), without reordering well-separated hands.

### 1.2 Why optimize against it

A metric intended to guide design will be used as an objective, explicitly or not. The fastest way
to learn what it then produces is to make it the objective now, with a design space wide enough
that the optimizer is not forced to produce a hand. This is a Goodhart test: the result is a
sketch of the metric's implied ideal, and the places where that ideal diverges from what the
authors intend are the places the metric needs a fix or a caveat. It also gives a second, independent
answer to the paper's validation question. The paper validates by correlation with baselines on
sixteen real hands; here we ask whether the features that separate those hands are the same ones
an optimizer saturates when it is free to choose.

---

## 2. Questions and pre-stated predictions

Each arm was specified, with the outcome that would support the metric and the one that would
undermine it, before its results were seen.

| arm | question | supports the metric | undermines it |
| --- | --- | --- | --- |
| **A. Recurrence** | Do independent restarts converge on the same features? (5 runs, 4+4 DOF, 150° range cap, objective T) | the same few features recur in every winner | winners share a score but not a shape |
| **B. DOF scaling** | How does the optimum scale with joints per finger (2+2, 3+3, 4+4, 5+5)? | saturates by 4+4 | keeps growing, so T partly counts joints |
| **C. Range pricing** | How does the optimum scale with the permitted joint range (90°, 120°, 150°, 180° total)? | weak dependence above typical hardware ranges | near-linear growth, so T partly measures permitted range |
| **D. Objective** | Are the T-optimum and R-optimum the same hand? (objectives T, R, normalised T+R) | similar hands | each optimum scores poorly on the other |
| **E. Object scale** | Are rankings stable across sphere radius (0.03, 0.05, 0.08 L_ref)? | stable | optimized hands collapse away from the training radius while real hands do not |
| **F. Normalisation** | Does a pinch-only L_ref (and disabling knuckle advancement) preserve the published ranking and close the stub exploit? | at most adjacent swaps; stubs no longer help | the published order depended on the artifact |

Two cross-cutting checks: a **proxy audit** (does the reduced-cost evaluator the optimizer sees rank
designs like the official pipeline?) and **KaRMA-S** on every winner (is the score won through one
privileged pinch?).

---

## 3. Methods

### 3.1 The hand family (`optimize/hand_family.py`)

A design is a thumb and an index finger, each a serial chain of N revolute joints on a palm with no
geometry. Per joint: a unit axis in the link frame (two angles, so flexion, abduction, twist and
anything oblique are all available), a link length, and asymmetric limits [−lo, +hi]. Per finger: the
chain lengths and a tip length. The thumb knuckle's pose on the palm (position and rotation vector)
is free.

Three things are pinned, each to remove a direction the metric does not see:

- the **index knuckle** sits at the palm origin with identity orientation, because KaRMA is exactly
  invariant to rigid transforms of the whole hand;
- the **index wire** (sum of its link and tip lengths) is 100 mm, because KaRMA is exactly
  scale-invariant; thumb lengths are absolute, so the thumb/index length ratio is free;
- the **topology** is exactly two fingers, for the reason demonstrated in Section 4.5.

Bounds: thumb knuckle within ±80 mm of the index knuckle; thumb links 12–80 mm; index links as
weights normalised to the 100 mm wire; joint limits 15°–75° per side (30°–150° total), so q = 0 is
always inside the limits and no joint has a range small enough to trigger L_ref's knuckle
advancement. Every link is declared a contact link, so contact may switch to any phalanx and
self-collision is checked over both whole fingers. With 4+4 joints the genotype has 46 real
coordinates; angles are unbounded (periodic), the rest are box-bounded.

The starting point for run T is a hand-drawn anthropomorphic design: index with an abduction
joint then three flexion joints (12/40/25 mm links, 23 mm tip), thumb with a CMC abduction and three
flexions (15/45/30 mm, 25 mm tip) set 47 mm proximal-ulnar and rolled into opposition. Restarts A2
and A3 instead begin from the best of 48 uniformly random designs.

### 3.2 The proxy evaluator (`optimize/fast_eval.py`, `karma_jax/fast.py`)

Official scoring costs minutes per hand on the CPU (52 min for the CPU pilot's winner), too slow
for an optimizer. The proxy calls the metric's own IK, nullspace polish, manipulability frame,
rolling-QP BFS and Phase-2 kernels unchanged, so the physics per step is identical; it is cheaper
only in search policy. Two presets were used: `loop` on the CPU pilot (run T) and `rich` on the GPU
for every arm, where batching over seeds makes the richer setting cost about the same:

| | official | proxy `loop` (CPU pilot) | proxy `rich` (GPU arms) |
| --- | --- | --- | --- |
| joint samples for seeds | 1 + 5/joint + 512 Sobol + 200 random | 1 + 3/joint + 128 Sobol + 32 random | 1 + 5/joint + 256 Sobol + 64 random |
| sphere-centre proposals per sample | 9 (5 offsets) | 5 (3 offsets) | 9 (5 offsets) |
| seed dedup | 0.01 rad | 0.20 rad, max 24 seeds | 0.10 rad, max 48 seeds |
| translation BFS | every seed, to 10 000 voxels | every seed to 15 voxels; top 2 (later 1) to a cap | every seed to 15 voxels; top 3 to 10 000 |
| voxel cap | 10 000 | 1200 → 2500 → 6000 (raised when it bound) | 10 000 (= official) |
| KaRMA-T | mean of top 3 seeds | mean of finalists | mean of top 3 finalists |
| Phase 2 | all voxels of top-3 seeds | skipped for objective T; 12 core voxels for R, T+R | same |

Cost per design: CPU `loop` about 17 s on the baseline, growing linearly with voxels reached (about
8 ms per primitive attempt) to several minutes for the best hands; GPU `rich` 8 to 16 s regardless of
reach, and 6 s effective with eight worker processes sharing the RTX 4090. On the CPU pilot's winner
the `rich` proxy gives T = 0.464 against an official 0.568 (`loop`: 0.416); on the baseline 0.021
against 0.029. The proxy is a conservative estimator because fewer pinches are tried; whether it
*ranks* designs like the official pipeline is the proxy audit (Section 4.3).

Infeasible designs (no pinch found) get a small negative objective proportional to how far apart the
fingertips are at their closest, so the optimizer has a gradient toward feasibility.

**GPU backend.** The repository also contains `karma_jax`, a batched JAX port of the metric
(`docs/jax_rewrite_plan.md`). Scores are not bit-identical to the reference (different QP and IK
solvers) but close: on the baseline T = 0.0288 vs 0.0285 and on the CPU pilot's winner 0.568 vs
0.575 (4602 vs 4648 voxels), in 64 s and 148 s instead of 130 s and 52 min. Using it on this hand
family exposed two padding defects (`optimize/karma_jax_padding_repro.py`): a joint table padded to
exactly the joint count, the loader's default, gives wrong forward kinematics under `jit` for an
8-joint two-root hand (first-run symptom: 2 seeds instead of 211), and padded active-joint lanes
corrupt the Gauss-Newton projection through a duplicate-index scatter (86 of 200 projection verdicts
flip). Neither is reachable from the 16 shipped hands as the package tests load them. All GPU
results here use one spare joint lane and exact sizes otherwise (`optimize/backend.py`), under which
the package's own per-kernel oracle tests pass on these hands to 1e-15.

### 3.3 The optimizer (`optimize/run_optimization.py`)

CMA-ES (`cma` 4.0.0) with population 12 on the CPU pilot and 16 on the GPU arms, per-coordinate initial standard
deviations of 20 mm for positions, 12 mm for thumb lengths, 0.15 for index weights, 0.25 rad for
limits and 0.4–0.6 rad for angles; box constraints via cma's boundary transform. The objective is the
proxy KaRMA-T (arm A, B, C), the proxy top-5 rotational coverage (D_R), or T/0.0969 + R/0.335,
each term normalised by the best published hand (D_TR). All random draws inside the proxy are
seeded, so the objective is deterministic per design.

### 3.4 Official scoring and features

Every winner is written as a URDF plus robot YAML and scored by `karma.metric.compute_metric` with
the shipped `karma_config.yaml`, exactly as `run_metric.py` would (`optimize/evaluate_full.py`), and
rendered with its reachable set (`optimize/render_hand.py`). Arm E re-scores with
`sphere_radius_m` changed to 0.006 and 0.016 (0.03 and 0.08 L_ref). Arm F re-scores the sixteen
bundled hands with `compute_lref` replaced by the pinch-only or no-advancement variant
(`optimize/lref_variants.py`); the shipped definition is also re-run on this machine as the control.

For the feature comparison (`optimize/features.py`) each hand, real or optimized, gets: knuckle
distance, thumb and index wire, L_ref, knuckle distance over mean wire (palm compactness), wire over
L_ref, distal-segment fraction of each finger, mean and max joint range, mean axis obliquity per
finger (angle between the joint axis and the plane perpendicular to the link: 0° is a pure
flexion/abduction axis, 90° a twist), DOF count, thumb–index opposition angle at neutral, and
tip-to-tip gap at neutral over L_ref.

### 3.5 Compute and deviations from plan

All runs on one 12-core AMD EPYC (no SMT) with an RTX 4090. The CPU pilot and arm F use the
pinned Python 3.10 environment from `environment.yml` (plus `cma`); the GPU arms use a Python 3.12
environment with the same pinned Pinocchio and JAX 0.7.1 (CUDA 12). The shipped Inspire check and the
full 16-hand sweep reproduce the published numbers on this machine (Section 4.4). Jobs run from a
sequential queue (`optimize/queue_runner.sh`) so nothing shares the GPU or the cores.

Deviations, stated so they can be judged: (i) the CPU pilot (run T) ran with the `loop` proxy whose
voxel cap was 1200 for generations 1–35 and 2500 for 36–89, and bound at both values; it is kept as
a pilot and backend cross-check, and arm A was re-run from the same start on the GPU as A1 with the
uncapped `rich` proxy; (ii) a self-matching `pgrep` stalled the CPU queue for 12 h between run T and
its scoring, with no effect on results; (iii) the CPU queue was stopped after arm F's control and
14 of 16 pinch-only hands, and the remaining arms were re-queued on the GPU with 60 generations and
population 16 each.

---

## 4. Results

### 4.0 The starting hand

The anthropomorphic start scores, officially, **T = 0.0285, R = 0.230, S = 0.519** (243 voxels,
L_ref 154.7 mm, 211 candidate pinches). That alone would place it 6th of 17, between Wuji
(0.034) and Shadow (0.013), which says something about how low the bar is: a sketch with eight
plain hinge joints and no engineering out-scores eleven commercial hands on this metric.
Figure: `optimize/runs/baseline/hand.png`.

### 4.1 Run T: the first T-optimum

**Trajectory.** 89 generations, 1068 proxy evaluations, 16.9 min to generation 35 and about 5.5 h in
total as evaluations slowed with reach. The proxy objective rose in plateaus (0.03 → 0.067 at gen 3
→ 0.133 at 19 → 0.149 at 35 → 0.180 at 51 → 0.27 at 70 → 0.3125 at 78), each a jump to a new
basin rather than a smooth climb; the last value is the 2500-voxel cap. Figure:
`optimize/runs/T/progress.png`.

**Official score of the winner:**

| hand | L_ref | mxPr | mF | voxels (top-3) | KaRMA-T | KaRMA-R | KaRMA-S | pinches |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **run T winner** | 117.5 mm | 15.6 mm | 101.9 mm | 4648 / 4608 / 4543 | **0.575** | 0.277 | **0.795** | 193 |
| LEAP (published best) | 249.4 | | | 792 | 0.097 | 0.304 | 0.448 | 254 |
| Allegro | 222.5 | | | 445 | 0.055 | 0.335 | 0.542 | 513 |

**The design** (`optimize/runs/T/final/opt_T.json`, figure `optimize/runs/T/final/hand.png`):

- Thumb knuckle 15.6 mm from the index knuckle. The palm, as the metric sees it, has vanished.
- Thumb: links 14.5 / 15.4 / 12.7 mm then a **61 mm tip**; all four axes oblique (none within 30° of a
  pure flexion or twist axis); ranges 133°, 69°, 118°, 116°.
- Index: links 4.6 / 41.1 / 10.8 mm then a **44 mm tip**; two axes oblique; ranges 144°, 145°,
  143°, 97°.
- Reachable set: a near-complete shell about 180 mm across centred on the fused knuckles (the hand
  is a pair of 100 mm styli on a common pivot, and the sphere can be carried almost anywhere at full
  reach). The sphere radius for this hand is 5.9 mm.

**Reading.** Three points. First, the proxy under-reported but did not mislead: the official
pipeline, with 193 pinches instead of 24 and no cap, finds an even larger set. Second, KaRMA-S of
0.795 is higher than any published hand: the *median* pinch reaches 3694 voxels, so this is not a
cherry-picked grasp but a hand on which nearly every pinch works. Third, KaRMA-R did not rise with
T. The Phase-2 failure log is dominated by the slip gate (10 529 rejections), thumb–index
self-collision (3039) and the friction cone (2924): two nearly parallel styli can carry the sphere
far but cannot tilt it much before they slide or collide. T and R are already pulling apart here;
arm D tests it directly.

### 4.2 Arm A: recurrence across restarts

**[TBD: A2, A3 and the run T continuation. Table: per run, start, generations, official T/R/S,
voxels, L_ref, knuckle distance, thumb/index tip fraction, mean range, obliquity. Then the
recurrence statement: which features have IQR small relative to the published spread, which do not.
Renders: `optimize/runs/A2/final/hand.png`, `optimize/runs/A3/final/hand.png`.]**

Provisional, from run T alone against the sixteen published hands (Section 4.10): the winner sits
outside the published range on palm compactness (knuckle distance / mean wire = 0.15, published
minimum 0.36) and on wire/L_ref (0.87, published maximum 0.68), and at the high end on range and
obliquity. Whether that is *the* optimum or *an* optimum is what A2/A3 decide.

### 4.3 Proxy audit

**[TBD: 20 designs from run T, stratified by proxy objective, scored officially. Report Spearman ρ,
median official/proxy ratio, top-5 agreement, and a scatter. If ρ < 0.8 or top-5 agreement ≤ 2/5,
the restarts are re-run with doubled seed sampling.]**

What is known: on the baseline the ratio was 1.55 (0.029 vs 0.018); on the run T winner the proxy
was capped so the ratio is not informative.

### 4.4 Arm F: the normalisation

**Control.** Re-running the shipped definition on this machine reproduces the published Table I
ranking exactly: KaRMA-T Spearman ρ = 1.000 with zero pairwise inversions over 16 hands, KaRMA-R
ρ = 0.997 with one inversion. Voxel counts differ by at most eight (LEAP 797 vs 792, DLR 64 vs 72),
the cross-machine variation the README warns about. So any change seen under a variant is the
variant's doing.

**Which hands the rules touch.** The knuckle-advancement rule fires only for Shadow (L_ref 184.8 →
195.4 mm without it) and SVH (182.9 → 210.5 mm). The pinch-only definition moves every hand, in
both directions: Orca 168.9 → 132.9 mm, Allegro 222.5 → 201.4, LEAP 249.4 → 227.6, SVH 182.9 →
233.4, Sharpa 188.3 → 196.3, Inspire 155.4 → 164.8.

**[TBD: pinch-only re-scoring of all 16 hands (table in RESULTS_AUTO.md): ρ and inversions vs
published; which pairs swap; whether the five "floor" hands stay on the floor. No-advancement
re-scoring of Shadow and SVH.]**

### 4.5 The stub-finger degeneracy

Because mF is a *median over all finger chains* and the stub threshold is an absolute 10 mm, a
hand can shrink L_ref, and with it the sphere, the voxel and the capsule radius, by adding fingers
that never touch anything. On the baseline hand, three one-joint "fingers" of 11 mm placed at the
index knuckle (`optimize/exploits.py`):

| | mxPr | mF | L_ref | voxel | capsule radius | proxy KaRMA-T |
| --- | --- | --- | --- | --- | --- | --- |
| two-finger baseline | 47.2 mm | 107.5 mm | 154.7 mm | 7.7 mm | 9.1 mm | 0.018 |
| + 3 stubs of 11 mm | 28.6 mm | 11.0 mm | 39.6 mm | 2.0 mm | 2.3 mm | ≥ 0.50 (hit the 4000-voxel cap) |

An inflation of at least 27× from an unchanged thumb and index. Two rules compound: the median
collapses to the stub length, and the length-outlier clause of the advancement rule then also moves
the thumb knuckle distally, cutting mxPr from 47 to 29 mm. The same mechanism acts, more gently, on
real hands whose non-pinch fingers are shorter than the pinch pair. This is why the main
optimization pins the topology at two fingers, and why arm F exists: **[TBD: under the pinch-only
L_ref the stub hand scores … (expected: identical to the two-finger baseline).]**

### 4.6 Arm D: translation versus rotation

**[TBD: D_R and D_TR winners. Table: official T, R, S for the T-optimum, R-optimum and T+R-optimum,
each also scored on the other objectives. Renders. Statement of whether the R-optimum keeps a palm
and conventional flexion axes, and of the price in T of asking for R.]**

Provisional: the T-optimum's R of 0.277 is below LEAP's 0.304 and Allegro's 0.335 despite a
reachable set six times larger; the Phase-2 failure log says why (slip gate and self-collision of
two near-parallel styli).

### 4.7 Arm B: degrees of freedom

**[TBD: 2+2, 3+3, 4+4 (run T), 5+5. Table and a plot of official T vs DOF. Statement: saturating or
not. Note the 2+2 case: whether a feasible rolling pinch exists at all with four joints.]**

### 4.8 Arm C: joint-range pricing

**[TBD: caps 90°, 120°, 150° (run T), 180° total. Plot of official T vs cap; note whether the
winners actually sit at the cap (run T's did not: its mean ranges were 120–125° under a 150° cap).]**

### 4.9 Arm E: object scale

**[TBD: eight real hands and five winners at r = 0.03, 0.05, 0.08 L_ref. Table of T and R at each
radius; rank correlation across radii for real hands vs optimized hands.]**

### 4.10 Features: published hands versus optimized designs

Spearman correlation of each feature with official KaRMA-T across the sixteen published hands
(`optimize/runs/features.csv`), with the run T winner's value:

| feature | ρ with T (16 hands) | published median [min, max] | run T winner |
| --- | --- | --- | --- |
| index wire [mm] | 0.84 | 97 [73, 221] | 100 (pinned) |
| L_ref [mm] | 0.71 | 184 [155, 327] | 117.5 |
| DOF (thumb + index) | 0.71 | 7.5 [3, 9] | 8 (pinned) |
| index axis obliquity [°] | 0.65 | 0 [0, 30] | **[TBD]** |
| thumb wire [mm] | 0.44 | 122 [88, 221] | 104 |
| max joint range [°] | 0.44 | 120 [76, 184] | 145 |
| wire / L_ref | 0.32 | 0.61 [0.52, 0.68] | 0.87 |
| mean joint range [°] | 0.27 | 90 [59, 155] | 121 |
| knuckle distance / wire | **−0.61** | 0.62 [0.36, 0.94] | **0.15** |
| index distal fraction | −0.43 | 0.33 [0.26, 0.69] | 0.44 |
| thumb distal fraction | 0.02 | 0.35 [0.19, 0.70] | 0.59 |

**[TBD: add A2, A3, D and B/C winners as columns; a scatter figure of the two or three strongest
features with the winners marked.]**

Provisional reading: among real hands, the single strongest predictor of KaRMA-T after sheer finger
length is a *compact palm* (ρ = −0.61 for knuckle distance over finger length), and the optimizer
drives exactly that feature past the end of the real-hand range. That is agreement in direction
between the two ways of interrogating the metric. The disagreement is on distal fraction: real hands
with long last segments on the index score *worse* (ρ = −0.43), while the optimizer makes both
fingers mostly tip.

---

## 5. Discussion

**[TBD: final version after arms A–F. Provisional structure and content below.]**

**What KaRMA-T rewards.** Long pinch fingers relative to L_ref, a palm as narrow as self-collision
allows, large ranges, and distal segments long enough that the fingertip capsule itself does most of
the work. The first three are also what separates LEAP and Allegro from the rest of the published
table, so in direction the metric's gradient and the real-hand ranking agree. The optimizer goes
further than any real hand on all of them because nothing in the metric prices a palm, actuators,
or the loss of every other grasp type.

**What KaRMA-T does not reward.** Rotation. **[TBD: arm D]**

**The L_ref problem.** The normalisation does the job it was designed for on the sixteen hands (the
ranking is insensitive to cross-machine noise and, **[TBD]**, to the pinch-only redefinition), but it
has two exploitable clauses and a hidden coupling: any change that shrinks L_ref also shrinks the
*object* and the *finger thickness*, so a "smaller" hand is handed an easier problem. This is by design
for comparing geometrically similar hands at different scales; it becomes a loophole as soon as the
pinch pair and the rest of the hand are not similar.

**Is the T-optimum a dexterous hand?** It is a very good two-point compass. Its KaRMA-S says it is
robust *for this task*: almost any pinch works. It would be useless for anything else, because there
is nothing to grasp with and no palm to grasp against. That is not a flaw in the optimization; it is
the metric's definition of the task, made visible.

**Limits of this study.** One hand family (serial chains, capsule links, no palm geometry), one
optimizer, caps chosen by us, a proxy in the loop (audited in 4.3), and official scores from a single
machine. **[TBD: number of restarts actually completed.]** We did not vary friction, the slip gate, or
the step sizes, all of which the paper flags as sensitive.

---

## 6. Recommendations

### 6.1 For the metric

1. **Compute L_ref from the pinch pair only** (knuckle distance plus mean of the two wires), or at
   least exclude chains that are not in the pinch from the median. Closes the stub degeneracy by
   construction. **[TBD: arm F shows this changes the published ranking by …]**
2. **Drop or bound the knuckle-advancement rule.** It fires on two real hands and is the second half
   of the stub exploit. If a low-range proximal joint should not count as the knuckle, say so with a
   fixed rule on that joint, not one conditioned on whether it shrinks mxPr.
3. **Report T, R and S together, always.** The T-optimum has below-LEAP R; the paper's T–R
   correlation of 0.96 across real hands does not hold once a design is pushed. **[TBD: arm D.]**
4. **Average over sphere radii** (for example 0.03, 0.05, 0.08 L_ref) or report the curve. A single
   radius invites specialisation. **[TBD: arm E says whether this matters in practice.]**
5. **Price joint range** or report it alongside. Range is the paper's most sensitive input (Sec.
   VI-C), it correlates 0.80 with T, and nothing stops a URDF author from widening limits.
   **[TBD: arm C quantifies the dependence.]**
6. **State the palm assumption.** The palm has no geometry in the model. For hands whose pinch
   happens against or near the palm this is generous; for the optimizer it is an invitation to delete
   the palm.

### 6.2 What to build

**[TBD: revise once arms A and D are in. Current view:]**

- **Do not build the T-optimum.** It is not buildable as drawn (4.6 mm and 10.8 mm joint-to-joint
  links, eight oblique axes, 16 mm between knuckles), and a physical copy would mostly test
  force control and friction, which the metric excludes by design.
- **Build-worthy candidates will come from arms B and D**: the 3+3 or 2+2 winner if it still scores
  far above LEAP, or the T+R winner if asking for rotation brings back a palm and conventional axes.
  Scale invariance lets us print at 2× (200 mm fingers, 12 mm ball) to fit actuators.
- **Before printing, check the design in a physics simulator** (the URDF and capsule bodies drop into
  MuJoCo): gravity, μ = 0.6, position-controlled joints. The question is whether the kinematic
  reachable set survives contact physics at all. One day of work.
- **If it survives, instrument the print.** The experiment worth doing with hardware is to measure
  the achieved reachable set of a tele-operated pinch and overlay it on the predicted voxel cloud.
  That tests the metric's actual claim. An un-instrumented print is a talk prop, which is also fine,
  but should be called that.
- **The one feature worth copying into a real hand**, if arm A confirms it, is the long, thick
  distal segment with a near-palm thumb base. LEAP already has the former. Whether the latter
  survives a palm with other fingers on it is a design question the metric cannot answer.

---

## 7. Reproduction

```bash
conda activate karma-hand-metric            # environment.yml + pip install cma==4.0.0
python optimize/run_optimization.py --out optimize/runs/T --objective T --hours 2.5     # main run
python optimize/evaluate_full.py --design optimize/runs/T/best.json --out optimize/runs/T/final
python optimize/render_hand.py --run-dir optimize/runs/T/final
python optimize/run_arm_f.py --variant pinch --out optimize/runs/armF_pinch
python optimize/exploits.py --design optimize/runs/baseline/baseline_hand.json --out optimize/runs/exploit_stubs
python optimize/features.py --designs optimize/runs/T/best.json
python optimize/report.py                    # regenerates RESULTS_AUTO.md
```

Queue of every job in this report: `optimize/queue.txt` (consumed) and `optimize/queue_done.log`.

| path | contents |
| --- | --- |
| `optimize/hand_family.py` | design parametrisation, URDF/YAML writer |
| `optimize/fast_eval.py` | proxy evaluator |
| `optimize/run_optimization.py` | CMA-ES driver, checkpoints, `--resume` |
| `optimize/evaluate_full.py`, `render_hand.py`, `render_best.py`, `plot_progress.py` | official scoring and figures |
| `optimize/lref_variants.py`, `run_arm_f.py`, `run_arm_e.py`, `exploits.py`, `proxy_audit.py`, `features.py` | the arms |
| `optimize/runs/<arm>/` | `best.json`, `evals.jsonl`, `progress.png`, `final/summary.json`, `final/hand.png`, `final/current.pkl` (open with `run_viser_app.py --result`) |

## Appendix: machine-generated tables

See `RESULTS_AUTO.md` in this directory (regenerated by `python optimize/report.py`).
