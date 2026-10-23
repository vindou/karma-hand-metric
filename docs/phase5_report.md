# Phase 5 report: pairwise and simultaneous three-contact reachability, MC-KaRMA (2026-10-11)

Phase 5 of `SPEC.md` §11. Deliverable: masked three-contact kernels, three-contact seeds,
SO(3) coverage, the pairwise baseline. Exit criterion: the two-contact reduction test, the
extra-contact monotonicity, and three-finger seeds found on the 3×4 family. All hold (§3).

## 1. Files and purpose

| File | Purpose |
| --- | --- |
| `dkarma/solvers/seed_projection.py` | `assignment(topo, n_lanes, fingers)`; `solve_multi_ik` (projected LM on the multi-contact residual: gaps, soft object/self clearance hinges, regularization), `multi_candidates` (joint samples whose fingertips can all meet one sphere, centroid centres ± along the triangle normal), `find_multi_seeds` (IK → exact kinematic verification → dedup → exact frictional-equilibrium check with unlimited torque → fixed budget) |
| `dkarma/solvers/rollout.py` | `RollConfig.grasp_mode`: `"antipodal"` (reference two-contact cone test) or `"equilibrium"` (smooth frictional equilibrium of all lanes under the task load, torque unlimited: the multi-contact grasp gate) |
| `dkarma/metrics/multicontact.py` | `evaluate_pairs` (two-contact pipeline per finger pair with its own seeds; `K_pair` mean, min, max), `multi_library` (6 translations, 6 rotations about the seed-frame axes), `evaluate_fixed_contacts` (MC-KaRMA / Full KaRMA: one object state, one joint configuration, one shared force problem, SO(3) coverage over the 576 Hopf targets), `evaluate_fixed_contacts_exact` (hard replay with every exact gate including SLSQP equilibrium at primitive ends; counts cells and 4608 SO(3) bins) |
| `dkarma/core/morphology.py` | `anthropomorphic` spaces non-thumb fingers 30 mm apart (22 mm interpenetrated the 11.8 mm capsules) |
| `dkarma/tests/test_multicontact.py` (5) | §3 |

Already general from earlier phases and reused unchanged: the `Cn`-lane rolling problem and
solvers (`rolling_qp`), the masked projection, `contact_pass`, clearances, the `[f, m]`
force problem, the Hopf SO(3) grids and kernels.

## 2. What was implemented

**Simultaneous contacts.** Three valid lanes in the same step: nine rolling rows, three
gap bands, one joint vector, one sphere pose; the projection closes all three gaps; the
grasp gate is the smooth frictional equilibrium of all three contacts (plus the two-finger
torsion model of Phase 4) under the object's weight with torque unlimited; with
`actuation=True` the XL330 ceilings gate every step as well (Full KaRMA). No finger
gaiting, no contact addition or removal (D4, SPEC §6.5).

**Three-contact seeds.** The reference's seed recipe has no three-contact analogue, so:
deterministic joint samples (linspace, Sobol, `RandomState(42)`), pairwise-reach filter on
the three fingertip capsules, sphere-centre proposals at the centroid of the closest
points and ±`r_s/2` along the triangle normal, batched projected Levenberg-Marquardt on
`[q, dp/r_s]` with the gap, clearance-hinge and regularization residuals (autodiff
Jacobian), exact verification of gaps, clearances and limits, dedup, exact
frictional-equilibrium check by SLSQP, fixed budget. The seed frame is the pinch pair's
manipulability frame (shared convention with the two-finger variants). Seeds are
re-projected per morphology inside the differentiable evaluator.

**Orientation.** Full rotation: `exp(-(3 - tr(R^T T))/(2 sigma^2))` over 576 targets
(smooth) and nearest-of-4608 binning (exact). Not compared numerically to the
two-finger 228-bin coverage (SPEC §6.4).

**Pairwise baseline.** The two-contact evaluator per permitted pair, each pair with its
own seed set; reported as mean, min, max over pairs; fixed averaging at equal finger count.

## 3. Tests and results (CPU fp64)

| Test | Result |
| --- | --- |
| two-contact reduction | with the third lane masked, the 3-lane hard solve (`dq`, slip) and projection equal the 2-lane ones to 1e-10 / 1e-12 |
| extra-contact monotonicity | at a three-contact seed, the optimal rolling residual of the 3-lane soft problem is ≥ the 2-lane one for 4 commands, strictly for ≥ 1 |
| three-contact seeds consistent | 3×4 anthropomorphic hand, small budget: seeds found; all three gaps within `eps_g`; limits respected; exact frictional equilibrium feasible |
| MC and Full evaluators | finite scores; gates `[S, 6, H, 8]`; `K_SO3 ∈ [0,1]`; jit equals eager; Full score ≤ MC score (torque gates only remove credit); exact counterpart returns fractions in `[0,1]`; Full-objective gradient through the morphology finite |
| pairwise baseline | three pairs of the 3×4 hand: `K_min ≤ K_pair ≤ K_max`, seeds found for the thumb–index pair |

`dkarma/tests/test_multicontact.py`: 5 passed (204 s). Seed pipeline on the GPU for the
3×4 start: 252 candidates, 123 kinematically valid, 4 of 4 requested seeds in 12 s.

## 4. Known limitations and numerical issues

- **No reference kernel exists for three contacts**, so the exact path is the
  fixed-assignment hard replay with exact gates, not an independent implementation. Its
  two-contact specialization agrees with the reference where the reference keeps the pair
  (Phase 2); that is the evidence it rests on.
- **Seed frame for three contacts** uses the pinch pair's manipulability ellipsoid; a
  three-contact object Jacobian would be the natural generalization (open).
- **Candidate generation** proposes centres from the three fingertip capsules only; grasps
  using proximal phalanges are not sought (same restriction as the reference's distal seeds).
- **Cost.** The three-contact force problem has 12 variables and is solved at every step;
  a Full-KaRMA evaluation is several times a Diff-KaRMA evaluation.
- **Seed count varies with morphology** (budget fixed, validity masked), so optimization
  from a start with few three-contact seeds has little signal; the hybrid optimizer's
  multistart mitigates this.

## 5. Minimal usage

```python
from dkarma.core import Topology, anthropomorphic, build_hand_model, load_task_spec
from dkarma.solvers.seed_projection import find_multi_seeds, SeedConfig
from dkarma.metrics import act_karma as AK, multicontact as MC
from dkarma.metrics.config import MC_KARMA, FULL_KARMA

task = load_task_spec(); sp = task.step_params()
topo = Topology.three_finger(4); hm = build_hand_model(topo, anthropomorphic(topo)); act = AK.actuation_for(hm, task)
fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
seeds = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=8))
mc = MC.evaluate_fixed_contacts(hm, topo, fingers, seeds, sp, task, act, MC_KARMA)
full = MC.evaluate_fixed_contacts(hm, topo, fingers, seeds, sp, task, act, FULL_KARMA)
exact = MC.evaluate_fixed_contacts_exact(hm, topo, fingers, seeds, sp, task, act, MC_KARMA)
pairs = MC.evaluate_pairs(hm, topo, [("thumb", "index"), ("thumb", "middle"), ("index", "middle")], sp, task)
```

## 6. Comparison with Phase 4

Phase 4 evaluated two maintained contacts. Phase 5 evaluates three with one shared
object state and force equilibrium, finds the seeds such grasps need, and measures full
orientation coverage; the pairwise baseline makes Experiment III's comparison possible.

## 7. Remaining work before Phase 6

None blocking; the optimizer consumes any of the four variants through one `Objective`.
