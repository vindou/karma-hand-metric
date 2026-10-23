# Phase 7 report: scientific experiments, simulation legs (2026-10-12)

Phase 7 of `SPEC.md` §11, Experiments I to IV of §12 without the hardware realization
(out of scope by the owner's instruction). Everything here was produced by the scripts in
`dkarma/experiments/` from the frozen task of §2 (15 mm / 35 g sphere, XL330 ceilings,
±100 mm cube); raw outputs are JSON under `workspace/phase3/` and `workspace/phase7/`.
Reported numbers come from the exact evaluators (CLAUDE.md rule 2); smooth values are
labelled as such.

## Experiment I: is Diff-KaRMA a useful surrogate?

**Dataset leg** (`workspace/phase3/surrogate.json`; details in `docs/phase3_report.md`
§3.2). Nine two-finger designs: baseline, the CMA-ES winners T and A1 of the Phase 0
study, six perturbations of the baseline. Spearman rank correlations of the smooth
coverages with the exact evaluators:

| pair | rho |
| --- | --- |
| smooth K_T vs exact-library K_T (same commands, reference kernels) | +0.987 |
| smooth K_T vs the original KaRMA BFS on the physical task (top-3 voxel fraction) | +0.979 |
| smooth K_R vs exact-library K_R | +0.971 |
| smooth K_R vs physical KaRMA-R | +0.937 |
| smooth K_T vs the published (L_ref-normalized) KaRMA-T | −0.333 |

The surrogate ranks designs as the exact evaluators do on the physical task. The
published metric ranks the opposite way on this set because its two top scorers (T:
0.558, A1: 0.429; 5 to 6 times LEAP) cannot roll the 15 mm sphere at all (one cell under
every physical evaluator): the `L_ref` normalization that those designs exploited does
not exist in the physical task, which is exactly why SPEC §2 fixed the object.

**Local-improvement leg** (`workspace/phase3/surrogate_local.log`). From three starts,
six accepted normalized gradient steps on the smooth objective (step 0.05 in the
unconstrained coordinates, halved until the smooth value rises), the candidate
re-scored exactly with fresh seeds:

| start | smooth | exact-library K_T | physical KaRMA cube fraction | published KaRMA-T |
| --- | --- | --- | --- | --- |
| baseline | 0.241 → 0.587 | 0.00067 → 0.00066 | 0.01071 → **0.01167** (+9 %) | 0.0298 → 0.0290 |
| pert00 | 0.071 → 0.126 | 0.00014 → 0.00014 | 0.00025 → **0.00033** (+32 %) | 0.00017 → 0.00021 |
| pert01 (one seed) | 0.105 → 0.105 (no step accepted) | unchanged | unchanged | unchanged |

Two of three starts improved the exact physical KaRMA; none worsened it; the start with
a single feasible seed had no ascent direction among the tried step sizes. The smooth
gain is mostly rotational (the balanced objective weights `K_R / 0.30`), which the
exact library's translation count does not see; the BFS count does.

**Optimization-performance leg** (`workspace/phase7/exp1_adam/`, `exp1_cmaes/`; same
two-finger baseline start, bounds, 8 seeds, Diff-KaRMA objective, exact-library
validation with fresh seeds; wall clock on the RTX 4090 including validations):

| optimizer | budget | wall clock | start exact score | best exact score (where) | final iterate's exact score |
| --- | --- | --- | --- | --- | --- |
| Adam, 3 continuation stages | 60 iterations | 477 s | 0.213 | **0.250** (iteration 40) | 0.196 (iteration 60, strict stage) |
| CMA-ES | 20 generations × 8 (160 evaluations) | 359 s | 0.213 | 0.244 (generation 4) | 0.104 (generation 19) |

Both improve the exact score by 15 to 18 % at similar wall clock; the gradient method
reaches its best after 40 iterations (40 gradient evaluations) where CMA-ES used 40 full
evaluations to reach 0.244 and then **drifted into surrogate exploitation**: its final
incumbent has the highest smooth value of the run (0.331) and an exact score below the
start (0.104). The exact-selection rule of SPEC §8.5 is what returns a sound design in
both runs; the smooth objective alone would have reported the worst one. Adam's strict
third stage also lowered the exact score (0.250 → 0.196), the same effect as in
Experiment III; the continuation end-point is too tight for these budgets and should be
validated per stage. Single runs, one start: a pilot of the comparison, not its result.

## Experiment II: does actuation awareness change the prediction?

`dkarma/experiments/actuation_awareness.py`, `workspace/phase7/exp2.json`. Kinematics
fixed (baseline and one perturbed two-finger design, 6 seeds each), torque ceiling swept
from stall to a tenth of the continuous estimate, object mass doubled twice. "Exact ok"
counts, among the primitive-end states the reference kernels accept kinematically, those
whose exact force allocation (SLSQP, Phase 4) is feasible.

**Table replaced in Phase 8.1** (`docs/phase8_report.md` §8.1, `workspace/phase8/exp2/exp2.json`):
the Phase 7 table (`workspace/phase7/exp2.json`) was produced with the Phase 4 force solver,
which did not converge at infeasible states, and with a credit mapping that kept 75-90 % of
the kinematic coverage where the exact solver rejected every state. Smooth values are smooth.

| design | setting | kinematic K_T~ (smooth) | actuation-aware K_T~ (smooth) | smooth max torque usage | exact force-feasible states (incl. seeds) |
| --- | --- | --- | --- | --- | --- |
| baseline | τ̄ 0.215 N·m (stall) | 0.00603 | 0.00603 | 1.01 | 36 / 36 |
| baseline | τ̄ 0.108 (primary) | 0.00603 | 0.00599 | 1.02 | 35 / 36 |
| baseline | τ̄ 0.043 (continuous) | 0.00603 | 0.00000 | 1.08 | 0 / 36 |
| baseline | τ̄ 0.020 | 0.00603 | 0.00000 | 1.29 | 0 / 36 |
| baseline | τ̄ 0.010 | 0.00603 | 0.00000 | 1.51 | 0 / 36 |
| baseline | mass 70 g | 0.00603 | 0.00396 | 1.07 | 23 / 36 |
| baseline | mass 140 g | 0.00603 | 0.00001 | 1.17 | 1 / 36 |
| pert | τ̄ 0.215 | 0.00390 | 0.00390 | 1.01 | 23 / 23 |
| pert | τ̄ 0.108 | 0.00390 | 0.00390 | 1.02 | 23 / 23 |
| pert | τ̄ 0.043 | 0.00390 | 0.00000 | 1.05 | 0 / 23 |
| pert | τ̄ 0.010 | 0.00390 | 0.00000 | 1.15 | 0 / 23 |
| pert | mass 70 g | 0.00390 | 0.00309 | 1.07 | 19 / 23 |
| pert | mass 140 g | 0.00390 | 0.00000 | 1.14 | 1 / 23 |

Findings:

1. The kinematic metric is blind to the load by construction; the actuation-aware
   evaluator separates the same kinematics by what the XL330 can hold. At the primary
   ceiling the 35 g sphere is held throughout (29/30, 17/17); at the continuous rating
   (0.043 N·m) no reached state can be held exactly on either hand, so a design that
   must hold the object continuously needs either a lighter object or the stall-level
   duty. That is a concrete, checkable design conclusion the published metric cannot give.
2. Specific states accepted by the original KaRMA and rejected by Act-KaRMA are listed
   per (seed, command, primitive) in the JSON (`rejected`).
3. **Calibration of the surrogate near the margin** (closed in Phase 8.1). In Phase 7 the
   smooth coverage kept 75 % (baseline) and 90 % (pert) of the kinematic value at the
   continuous rating while the exact check rejected every state. With the Phase 8.1
   calibration (converged force solver, hinge 0.005, hard-knee credit, seed state gated) it
   keeps 0 %; the exact evaluator, not the smooth one, decides feasibility either way.
4. The "max torque usage" of the smooth allocation exceeds 1 even where the exact
   allocation is feasible (1.10 at the primary ceiling): the smooth solution is the
   minimizer of a penalized energy, not a feasible point, so its margins are indicative
   only. Reported margins should come from the exact allocation.

## Experiment III: does simultaneous multi-contact evaluation add information?

`dkarma/experiments/codesign_comparison.py`, `workspace/phase7/exp34/results.json`. Four
3×4 hands optimized from the same anthropomorphic start, bounds and budget (Adam, 40
iterations, 2 continuation stages, 6 seeds, exact selection every 20 iterations) with
(1) thumb–index Diff-KaRMA, (2) mean pairwise KaRMA, (3) simultaneous three-contact
MC-KaRMA, (4) Full KaRMA; the start and all four evaluated on the common suite with fresh
seeds. Counts are means over the 6 three-contact seeds: cells of the ±100 mm cube reached
by exactly valid trajectories, and distinct 4608-bin SO(3) orientations; "Full" columns
apply the XL330 torque gates; wrench capacity is in multiples of the object's weight at
the three-contact seeds (14 directions); the last columns are the two-finger exact library
and the original KaRMA on the physical task.

| design | MC cells | MC SO(3) bins | Full cells | Full SO(3) bins | capacity mean / min | fraction ≥ 1 | 2-finger exact K_T | 2-finger exact K_R | physical KaRMA cube fraction | physical KaRMA-R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| start | 4.0 | 21.8 | 1.5 | 5.7 | 3.75 / 1.44 | 0.85 | 0.00063 | 0.041 | 0.0105 | 0.112 |
| (1) thumb–index Diff | 2.3 | 14.7 | 1.7 | 8.8 | 4.99 / 1.78 | 0.95 | 0.00067 | 0.061 | 0.0116 | 0.091 |
| (2) pairwise | **4.5** | 21.2 | **2.7** | **13.7** | 4.10 / 1.41 | 0.82 | 0.00071 | 0.050 | 0.0097 | 0.098 |
| (3) MC | 4.3 | 17.7 | 1.7 | 5.0 | 2.92 / 1.11 | 0.85 | **0.00081** | 0.050 | **0.0156** | 0.093 |
| (4) Full | 3.3 | 16.3 | 2.0 | 10.3 | **4.65 / 2.24** | 0.83 | 0.00073 | **0.065** | 0.0113 | 0.106 |

Optimization traces: smooth objective first → last iteration and best exact validation:
(1) 1.13 → 0.15, 0.209; (2) 1.05 → 0.22, 0.175; (3) 0.64 → 0.13, 0.018; (4) 0.64 → 0.012,
0.010 (the smooth values of different stages and variants are on different scales and are
not comparable across arms; the exact validation is the quantity selected on).

What the data say, and what they cannot:

1. **Pairwise dexterity does not predict simultaneous-contact manipulation**, in the
   direction the spec anticipated: the thumb–index optimum (1) is the best two-finger
   hand of the set by the exact library and improved the physical KaRMA by 11 %, yet it
   has the *fewest* exactly reachable three-contact cells and orientations (2.3 cells,
   14.7 bins against the start's 4.0 and 21.8). Optimizing for one pair moved the third
   finger out of useful reach.
2. **No arm dominates the three-contact suite at this budget.** The pairwise arm has the
   most actuated three-contact cells and orientations; the Full arm has the largest and
   most uniform wrench margins (min 2.24 × weight) and the best two-finger rotation
   coverage; the MC arm, tuned for unactuated three-contact rolling, gained the most on the
   original physical KaRMA (+49 %) but lost margin. With one 40-iteration run per arm and
   six seeds the differences of one to two cells are within what a different seed set
   produces; this is a pilot, not a decision.
3. **The actuated three-contact objective is the hardest to optimize**: arm (4)'s smooth
   value collapsed in the strict stage and its exact validation fell from 0.010 to 0.004,
   so the selection rule returned the iteration-20 design. The force energy's gentle
   calibration (Experiment II, finding 3) and the kernel tightening interact badly; the
   obvious next steps are a larger force weight, re-seeding per stage, and the hybrid
   multistart.
4. Torque gates remove 40 to 75 % of the three-contact cells on every design: holding the
   35 g sphere with three XL330-driven fingers at the primary ceiling is the binding
   constraint of this task, not kinematics.

## Experiment IV: end-to-end co-design (simulation leg)

Arm A: CMA-ES against the original KaRMA on the 3×4 topology (its fast proxy as fitness,
12 generations × 8, as the Phase 0 study did; final scoring by the original metric), from
the same anthropomorphic start and bounds as Experiment III. Arm B: the Full-KaRMA design
of Experiment III (4). Both scored on the physical suite; both exported with their torque
specification (`workspace/phase7/exp34/armA_design/`, `run_full/optimized_morphology/`,
`actuator_spec.yaml`) as the kinematic plus joint-torque specification for mechanical
realization. Hardware realization is out of scope by instruction.

| design | three-contact seeds found | MC cells | Full cells | Full SO(3) bins | three-contact capacity mean / min | 2-finger exact K_T | physical KaRMA cube fraction | published-style KaRMA-T (own L_ref) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A: original KaRMA + CMA-ES | **0** | – | – | – | – | 0.00050 | **0.0142** | 0.0416 |
| B: Full KaRMA + Adam | 6 | 3.3 | 2.0 | 10.3 | 4.65 / 2.24 | 0.00073 | 0.0113 | – |
| start | 6 | 4.0 | 1.5 | 5.7 | 3.75 / 1.44 | 0.00063 | 0.0105 | – |

Arm A is the best two-finger hand on this topology by every two-finger measure, the
original metric included (+35 % physical cube fraction over the start), and it has lost
the third finger: the three-contact seed search finds no configuration in which all three
fingertips meet the sphere with a frictional equilibrium. The CMA-ES moved the middle
finger out of the thumb's reach because nothing in its objective used it. Arm B keeps
three-contact grasps with the largest and most uniform wrench margins of the set, at a
7 % two-finger cost against A. This is the "same hardware resources, underutilization
made explicit" case of SPEC §12 IV in its sharpest form: the two-finger objective does
not merely ignore the third finger, it removes its usefulness.

What this leg does not establish: which of A or B performs better physically. That needs
the fabrication protocol and the three-way error decomposition (surrogate vs exact,
exact vs simulation, simulation vs hardware). The first term is quantified in Experiments
I and II; the other two remain open.

## What is complete and what is not

| SPEC §16 component | State |
| --- | --- |
| Differentiability | validated morphology gradients through the trajectory evaluator (Phases 1-3) |
| Reachability | exact validation by the reference kernels (two contacts) and the fixed-assignment replay (three) |
| Actuation | equilibrium, friction, torsion and torque feasibility verified by an independent exact solver (Phase 4) |
| Multi-contact | simultaneous three-finger trajectories and SO(3) coverage (Phase 5) |
| Co-design | optimizer module with exact selection (Phase 6); matched-budget Adam-vs-CMA-ES comparison not yet run |
| Hardware | out of scope |
