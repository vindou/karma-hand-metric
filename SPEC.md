# Differentiable, actuation-aware, multi-contact KaRMA for robotic-hand co-design

Status: project specification, 2026-10-11. Supersedes the advisor conversation
archived at `docs/spec_source_conversation_2026-10-11.md` (kept locally, not in the repository), which is the source
of the methodology below; this document grounds it in what the repository
actually contains and freezes the decisions needed to start building.

Audience: whoever implements this (human or coding agent). Assumes familiarity
with the KaRMA paper (arXiv:2605.15548, IROS 2026), `karma/` and `karma_jax/`.

---

## 0. One-paragraph summary

KaRMA scores a hand's fine-manipulation ability by searching, with a two-finger
pinch on a sphere, how far the sphere can be rolled without regrasping. It is
discrete (voxel and orientation-bin counts), kinematic only, and not
differentiable in the hand's morphology. This project builds, on top of the
existing JAX port, one differentiable rolling-trajectory engine with modular
feasibility constraints, and expresses four metric variants as configurations
of that engine: **Diff-KaRMA** (smooth surrogate of the original), **Act-KaRMA**
(adds contact-force equilibrium, friction and joint-torque feasibility),
**MC-KaRMA** (adds simultaneous three-finger contact and full SO(3) orientation
coverage), and **Full KaRMA** (all of the above). A morphology optimizer
(gradient-based, CMA-ES, and hybrid) then co-designs hand skeletons against
these objectives, every candidate is validated by exact (non-smooth)
evaluators, and the result is a kinematic plus joint-torque specification to be
realized mechanically and tested on hardware. The original discrete KaRMA stays
untouched as the reference evaluator.

| Variant | Contacts | Rolling | Force equilibrium + torque limits | Orientation space |
| --- | --- | --- | --- | --- |
| Original KaRMA (reference, exact) | 2 | hard gate | no | pinch axis, 228 HEALPix line bins |
| Diff-KaRMA | 2 | smooth residual | no | pinch axis, smooth kernel |
| Act-KaRMA | 2 | smooth residual | yes | pinch axis, smooth kernel |
| MC-KaRMA | 3 | smooth residual | no | SO(3), smooth kernel |
| Full KaRMA | 3 | smooth residual | yes | SO(3), smooth kernel |

Each smooth variant has an exact counterpart (hard gates, discrete counts) used
only for validation. Smooth scores are surrogates for optimization; they are
never reported as the metric.

---

## 1. What exists today (Phase 0 audit, 2026-10-11)

### 1.1 Repository state

One commit (`c8d3698`, the public KaRMA release) plus uncommitted work:
`karma_jax/` (JAX/CUDA port, ~4,400 lines, 76 oracle tests), `optimize/`
(CMA-ES adversarial optimization of KaRMA, ~3,500 lines plus 430 MB of runs),
`docs/jax_rewrite_plan.md`, and small edits to the entry points that add
`--backend jax|reference --precision fp32|fp64` (JAX is now the default when
importable). Published results (`results/`) and the reference package
(`karma/`) are unchanged.

Two environments (see `CLAUDE.md`): `karma-hand-metric` (Python 3.10, the
paper's pinned CPU stack) and `karma-jax` (Python 3.12, jax 0.7.1 CUDA 12,
same Pinocchio/OSQP pins, cma 4.0.0). Hardware: one RTX 4090, 12 cores.

### 1.2 The reference evaluator (`karma/`), facts that bind this project

- **Morphology** comes from a URDF through Pinocchio (`karma/robot.py:load_robot`);
  per-hand YAML adds active joints, contact links, seed links, tip lengths and
  mimic couplings. Links are capsules built procedurally from the joint tree
  (`_build_link_capsules`), with **one global capsule radius** (11.8 mm nominal)
  for every link; the palm has no geometry. Only 1-DoF revolute joints occur.
- **Scale normalization**: every length in `karma_config.yaml` is nominal for a
  200 mm hand and rescaled by `L_ref/0.2`, `L_ref = mxPr + mF` (`karma/lref.py`).
  With voxel edge `0.05 L_ref`, KaRMA-T = `mean_top3_voxels / 8000`. There is
  **no `V_ref`** in the code; KaRMA-T is normalized by `L_ref^3`.
- **Contact and rolling**: gap `g = |c - p_axis| - r_s - r_l`; the rolling
  constraint is a **first-order velocity no-slip residual**
  `J_surf dq - r_s [n]x dtheta = dp` (`karma/rolling_qp.py`), **not** the
  Montana contact equations. The translation QP minimizes that residual with a
  1e5-weighted prior on the sphere rotation, a linearized gap band
  `|g| <= eps_g`, joint-limit and 10°/step boxes, and `|dtheta| <= 0.2` rad;
  solver OSQP (`eps 1e-7`). Each sub-step is followed by a damped Gauss-Newton
  projection back onto the two-contact manifold (`karma/projection.py`). A
  per-sub-step **slip gate** (0.5 mm nominal, 40% of the 1.25 mm sub-step)
  rejects steps whose tangential residual is too large.
- **Hard gates at every step**: gap, sphere-vs-other-capsule clearance
  (`eps_col`), thumb-index clearance, antipodal friction cone
  (`dot(-n_t, n_i) >= cos(2 atan mu)`, mu = 0.6), joint limits, slip. Contact
  pair = nearest capsule per finger by |gap|, with hysteresis `eps_g/2`.
- **Search**: seeds from 713 + 5·n_active joint samples × 9 sphere-centre
  proposals, solved by scipy TRF least squares (gap, collision hinge, antipodal
  and regularization residuals), deduplicated at 0.01 rad, polished toward
  joint-range centre in the grasp nullspace. **Every** seed runs a trial BFS to
  the 10,000-voxel budget; the three best form the ensemble. Phase 1 is a FIFO
  BFS over a voxel grid aligned with the seed's manipulability frame, 6
  primitives × 8 sub-steps. Phase 2 is **not** a BFS: at each voxel, four
  straight-line tilts of the pinch axis (±0.15 rad about two perpendicular
  axes, up to 50 steps, stop at first failure or efficiency < 0.3).
- **Scores**: KaRMA-T as above; KaRMA-R = mean over the ensemble of the mean
  coverage of each seed's five best voxels, coverage = reached bins / 228,
  where the tracked quantity is the pinch axis `R^T(-n_thumb)` in the sphere
  body frame, antipodally identified, HEALPix nside 6; KaRMA-S = median / best
  trial-BFS voxel count over all seeds (computed in `seed_selection.py`).
- **Determinism and invariance** are by construction (base canonicalization,
  placement snapping, non-dimensional QP). Everything is Python control flow,
  NumPy, Pinocchio, OSQP, scipy and healpy; nothing is differentiable.

### 1.3 The JAX backend (`karma_jax/`), what can be reused

Pure-JAX kernels under `vmap`, driven by a host-side level-synchronous BFS.
Reproduces the paper's ranking exactly (Spearman 1.0 on KaRMA-T, median 2.2%
deviation per hand), 16-hand sweep in 590 s warm. Reusable as-is or with a
traced-morphology refactor:

| Capability | Function | Notes |
| --- | --- | --- |
| FK, analytic frame Jacobians | `kinematics.forward_kinematics`, `frame_jacobian`, `fold_active` | `fori_loop` over joints; gradients would flow through `R_joint, t_joint, axis, R_frame, t_frame` if these were traced inputs |
| Capsule-sphere pass, contact kinematics | `geometry.capsule_pass`, `contact_kinematics` | `argmin` pair selection, `clip`/`where` endpoint switches |
| Collision, self-collision, friction cone | `geometry.check_*` | boolean outputs |
| Box QP (dense Mehrotra IPM) | `qp.solve_box_qp` | fixed iteration budget, masked convergence; no `custom_vjp` or implicit differentiation |
| Rolling and rotation QPs | `rolling.build_rolling_qp`, `solve_rolling_step`, `build_rotation_qp` | non-dimensionalized by `r_s` |
| Manifold projection | `projection.project_onto_contacts` | Gauss-Newton with `argmax` backtracking |
| One translation / rotation step with all gates | `steps.attempt_primitive_step`, `rotation_step` | returns fail codes |
| Seed IK (scipy TRF port) | `ik.solve_pinch_ik`, `trf.py`, `ik.ChunkedIK` | fp64, `while_loop` (forward-mode only) |
| Exact batched metric | `metric.compute_metric`, `fast.evaluate` (proxy) | returns the reference `MetricResult` |

**Barriers to morphology gradients found in the audit** (each is a Phase 1 or
Phase 2 work item):

1. The hand model is extracted from Pinocchio as NumPy and the `DeviceModel`
   is **closed over inside `jax.jit`**, so XLA bakes morphology in as constants.
   There is no path from a parameter vector to the FK inputs without writing a
   URDF and reloading. There is no `jax.grad`, `custom_vjp` or implicit
   differentiation anywhere in the package.
2. Capsule radius and sphere radius are global scalars in `StepParams`, not
   model arrays. Procedural capsule construction (tip length minus radius,
   phantom-capsule removal, tip-direction heuristics) happens in `karma/robot.py`.
3. The score is a host-side count gated by booleans: piecewise constant in
   morphology by construction.
4. Padding defects: `J_pad == n_joints` gives wrong jitted FK on the optimizer's
   two-root hands (workaround `J_pad = n_joints + 1`), and `M_pad > n_active`
   corrupts the projection because padded lanes scatter into joint 0
   (`optimize/backend.py:4-9`, `optimize/karma_jax_padding_repro.py`). Both
   must be fixed before many designs can share one compiled program.
   `run_metric.py`, `robust_tests/run_tests.py` and `fast_eval._evaluate_jax`
   still use the defective default. *Fixed in Phase 0 (2026-10-11): root
   cause was an XLA miscompile of constant duplicate-index scatters; the
   kernels are now gather-only, exact padding is correct everywhere and
   padded sizes reproduce it (`docs/phase0_report.md`,
   `karma_jax/tests/test_padding.py`).*
5. Seed IK uses `lax.while_loop` and host-side dedup/sort; it is not
   reverse-differentiable and seeds are chosen by discrete rules.
6. One design per call: `fast.evaluate` and `compute_metric` have no
   cross-design batching (the remaining Phase E item of the JAX plan).

### 1.4 The existing optimization work (`optimize/`), what it tells us

CMA-ES over a two-finger, 4+4-DOF family (`optimize/hand_family.py`: free
thumb-base pose, per-joint axis direction (2 angles), link lengths, asymmetric
limits; index knuckle pinned at the origin and index wire pinned at 100 mm by
invariance; 46 genotype dimensions) against a reduced-search proxy of KaRMA-T,
re-scored officially. Findings that shape this project:

- Original KaRMA-T is exploitable: the run-T winner scores **0.575** (LEAP, the
  best real hand: 0.097) by collapsing the palm to 16 mm and growing long
  styli; vestigial stub fingers drive `L_ref` down and inflate T by ≥ 27×
  without changing the pinch. The actuation-aware metric must therefore use a
  **fixed physical object and a fixed task volume**, never an `L_ref`
  normalization that the design can move.
- The optimizer-generated URDFs carry **placeholder `effort="1"`** limits and
  dummy inertials. Half the bundled URDFs have placeholder efforts too (ARMS,
  SVH, Orca, DLR, D'Claw, Inspire; Dex5 has `effort="0"`). Torque ceilings must
  come from a separate actuator spec, never from URDF effort fields.
- A CMA-ES baseline pipeline, proxy-vs-official audit tooling, feature
  extraction, rendering and a sequential GPU job queue already exist and are
  the Experiment I baseline. Arms B, C, D, E of that study are still queued.

---

## 2. Decisions frozen for the first implementation

These follow the advisor's recommendations; change them here, not in code.

| ID | Decision | Value |
| --- | --- | --- |
| D1 | Debug topology | the existing two-finger 4+4 DOF family (`optimize/hand_family.py`), for direct comparison with the CMA-ES results |
| D2 | Primary experimental topology | three fingers, 4 DOF each (thumb + two fingers), fixed finger and joint count per run |
| D3 | Torque treatment | fixed per-joint joint-side torque ceilings, identical for corresponding joints across all candidate designs; shared-budget optimization is an ablation only |
| D4 | Contact mode | maintained contacts, fixed contact assignment per rollout, no finger gaiting, no contact addition or removal |
| D5 | Motion model | quasi-static: contact-force equilibrium, friction, torque limits, gravity; no inertial, thermal or transmission effects |
| D6 | Main objective | balanced translation + rotation, each normalized by a fixed reference scale; translation-only and rotation-only are run as ablations |
| D7 | Object | sphere, radius 15 mm, mass 35 g (a 30 mm glass marble), mu 0.6, under gravity; **fixed in physical units** across designs (not scaled by `L_ref`); sized so the nominal two-finger hold uses about 40% of the D13 torque ceiling (§2.2) |
| D8 | Task volume | cube of sphere-centre positions, side 200 mm (±100 mm) centred on the nominal grasp, 10 mm grid (20³ = 8,000 cells), `V_ref` = 8.0 L; derived from the actuator in §2.2 |
| D9 | Precision | fp32 kernels on GPU with fp64 available as a flag; all exact validation in fp64; seed IK fp64 |
| D10 | Exact vs smooth | two evaluation paths, never merged: smooth for gradients, exact for reported numbers |
| D11 | Reference | `karma/` is frozen; `karma_jax/` stays the exact batched evaluator and is extended, not forked |
| D12 | Package | new code lives in `dkarma/` (differentiable KaRMA); see §9 |
| D13 | Actuator | ROBOTIS DYNAMIXEL **XL330-M077-T**, one per joint, direct drive (joint-side torque = servo output torque, efficiency 1.0); ceilings in §2.1 |
| D14 | Symmetry | none: every finger's base pose, link lengths and axes are independent parameters; no tying across fingers in any run |
| D15 | Git | the current working tree is **not** committed; work continues on top of it uncommitted until the owner says otherwise. **Superseded 2026-10-23** (§16 item 13): committed on branch `dkarma-phase0-10` for the owner's fork |
| D16 | SO(3) targets | 576 Hopf-grid rotations with `sigma_R = 0.15 rad` for the smooth three-finger coverage; 4608 bins for the exact counting evaluator (§4.5) |
| D17 | Gravity | palm facing up: `g = [0, 0, -9.81]` m/s² in the palm frame for the primary task; other orientations are load-case ablations |

### 2.1 Actuator specification (XL330-M077-T)

Datasheet values (ROBOTIS e-Manual and product page, retrieved 2026-10-11):

| Quantity | Value |
| --- | --- |
| Stall torque | 0.180 N·m at 3.7 V (1.11 A); **0.215 N·m at 5.0 V (1.47 A)**; 0.228 N·m at 6.0 V (1.74 A) |
| Continuous torque | 0.043 N·m, ROBOTIS's own estimate at 20% of stall; no measured continuous-duty rating is published |
| No-load speed | 383 rev/min at 5.0 V |
| Gear ratio | 77.5 : 1 |
| Case | 20.0 × 34.0 × 26.0 mm, 18 g |
| Encoder | 12-bit contactless absolute, 4096 steps/rev |
| Thermal | operating −5 to 70 °C; default temperature limit 70 °C; sustained high current damages the motor |
| Operating voltage | 3.7–6.0 V, recommended 5.0 V |

Torque ceilings used by Act-KaRMA and Full KaRMA, per joint, identical for
every joint of every finger (D3):

| Setting | `tau_bar` | Role |
| --- | --- | --- |
| `primary` | **0.108 N·m** (50% of stall at 5 V) | principal experiments: a quasi-static hold is intermittent duty, well below stall but above the 20% continuous estimate |
| `continuous` | 0.043 N·m | ablation: the conservative sustained-duty bound |
| `stall` | 0.215 N·m | ablation and the "ideal actuation" limit check |

Both are assumptions until a joint-side torque curve is measured on the
actual transmission; the spec file names them as such. Link gravity is
modelled with each non-terminal link carrying one servo: link mass 0.028 kg
(18 g servo + 10 g structure, estimate) at the link midpoint, terminal link
0.010 kg.

```yaml
actuation:
  model: joint_side_constant_torque
  servo: XL330-M077-T
  supply_voltage_v: 5.0
  stall_torque_nm: 0.215
  torque_limit_nm: 0.108                       # primary; ablations use 0.043 and 0.215
  torque_limit_basis: "0.5 x stall at 5 V, intermittent duty; replace with measured joint-side curve"
  transmission: direct_drive
  transmission_efficiency_assumed: 1.0
  include_link_gravity: true
  link_mass_kg: 0.028                          # per non-terminal link, at the midpoint (estimate)
  terminal_link_mass_kg: 0.010
object:
  radius_m: 0.015
  mass_kg: 0.035
  friction_coefficient: 0.6
  gravity_m_s2: [0, 0, -9.81]                  # palm frame, palm facing up (D17); ablations rotate this vector
task:
  cube_side_m: 0.200
  grid_m: 0.010
  centre: nominal_grasp_sphere_centre
orientation:
  pinch_axis_bins: 228                         # two-finger, as the reference (HEALPix nside 6, antipodal)
  so3_targets_smooth: 576                      # Hopf grid level 1, sigma_R = 0.15 rad (D16)
  so3_bins_exact: 4608                         # Hopf grid level 2
```

### 2.2 Why these numbers: scale set by the actuator

- **Hand scale.** A direct-drive serial chain of XL330 cases has a joint
  pitch of roughly 35–40 mm, so a 4-DOF finger is 150–170 mm long and a
  three-finger palm 60–80 mm wide. LEAP, built from XC330 servos with the
  same 20 × 34 × 26 mm case, has `L_ref` = 249 mm. New designs will be in
  this range; the spec does not force a minimum link length (remote actuation
  stays allowed), it only uses the scale to size the task. *Superseded for
  Phase 11 by D19 (§8.1, owner's decision 2026-10-23): links 0–2 each hold an
  XL330 case, so they are at least 34 mm long and 32.8 mm thick.*
- **Task cube.** LEAP's published reachable set (792 voxels of 12.5 mm) spans
  175 × 162 × 150 mm and voxel indices −8..8 about the seed, i.e. ±100 mm.
  Allegro spans 122 × 100 × 111 mm, D'Claw 213 × 180 × 180 mm. A ±100 mm cube
  contains the LEAP set with margin on every axis; the 10 mm grid matches the
  reference's nominal voxel. Smooth occupancy over 8,000 cells per design is
  cheap next to the rollouts.
- **Object load vs torque.** Holding a 35 g sphere (0.34 N) in a two-finger
  pinch at mu = 0.6 needs ≥ 0.29 N normal force per contact, about 0.33 N
  resultant; at a 0.12 m moment arm the proximal joint sees ≈ 0.04 N·m, i.e.
  ~40% of the primary ceiling and ~90% of the continuous estimate. The load is
  therefore informative: feasible at rest, binding as the arm grows or the
  disturbance wrench rises, and infeasible under the continuous ablation for
  long fingers. The reference metric's nominal sphere at LEAP scale is 12.5 mm
  radius, so a 15 mm sphere is also geometrically comparable.

---

## 3. Common mathematical model

All variants share one representation of hand, object, contacts and
trajectories, so differences between objectives cannot be confounded by
differences in geometry or numerics.

### 3.1 Morphology vector

```
phi = [phi_base, phi_length, phi_axis, phi_limits, phi_tip]
```

- `phi_base`: position and orientation of each finger's first joint in the
  palm frame (thumb free; one finger pinned at the origin by rigid invariance
  when optimizing kinematic-only objectives; **not** pinned in scale when the
  object is physical).
- `phi_length`: inter-joint link lengths, one positive scalar per segment,
  parameterized through a bounded transform (e.g. sigmoid between `l_min` and
  `l_max`). `l_min` exists only to avoid degenerate zero-length links; it is
  not a bearing-envelope constraint.
- `phi_axis`: revolute axis per joint as a unit vector in the parent link
  frame, parameterized by two angles (as `hand_family.py` already does) or by a
  normalized 3-vector.
- `phi_limits`: joint lower and upper bounds, **fixed** in the principal
  experiments (D3 analogue for range); ordered when optimized.
- `phi_tip`: fingertip capsule length; fixed initially.

Finger count and joint count are fixed per run. Capsule radius `r_l` is fixed
(per-link array in the model, uniform value initially).

### 3.2 Kinematics

For finger `i`, link `k`: `T_{i,k}(q, phi) in SE(3)`. The module provides, with
autodiff through both `q` and `phi`:

- joint origins and world axes,
- contact points `p_c(q, phi)` on capsule surfaces,
- contact Jacobians `J_c = d p_c / d q` (closest-point Jacobian, as
  `geometry.contact_kinematics` computes today),
- spatial twist Jacobians.

Implementation: a JAX-native constructor `phi -> DeviceModel arrays`
(`R_joint, t_joint, axis, R_frame, t_frame, cap_tip`) that bypasses URDF and
Pinocchio, with Pinocchio kept as the oracle (FK of the exported URDF must match
to 1e-9 in fp64).

### 3.3 Contact geometry

Capsule links, spherical object, as in the paper. Gap for capsule `i` with
centreline `S_i(q, phi)`, sphere centre `p`, radii `r_l`, `r_s`:

```
g_i(q, p, phi) = d(p, S_i(q, phi)) - r_s - r_l
```

Positive is separation. Two contact modes:

1. **Fixed contact assignment** (first target): the link in contact at the
   seed stays the active link for the whole rollout.
2. **Candidate contact assignments**: a fixed set of assignments evaluated
   separately and aggregated with a smooth maximum.

No continuous soft nearest-link switching: it creates artificial contacts
across physically distinct surfaces. Final (exact) evaluation uses discrete,
geometrically valid assignments with the reference's nearest-link rule.

### 3.4 Object and task model

```
E = (r_s, m_s, mu, g, w_ext, X_task)
```

`w_ext` is the applied object wrench (gravity at minimum), `X_task` the
evaluated pose domain: the position grid of D8 and, for orientation, either
the 228 antipodal pinch-axis bins (two-finger) or a fixed set of SO(3)
targets (three-finger). Values are those of §2.1 (object 15 mm / 35 g /
mu 0.6, cube ±100 mm at 10 mm). For the co-design experiment all of `E` is
fixed across designs.

### 3.5 Units and frames

SI throughout the new code (m, kg, N, N·m, rad). Palm frame is the world frame
of the model; gravity direction is part of the task spec. No `L_ref` scaling
in `dkarma/`; the reference evaluator keeps its own.

---

## 4. The trajectory-library evaluator (Diff-KaRMA)

### 4.1 Principle

A pose earns reachability credit only through a feasible continuous rolling
path from an initial grasp. The differentiable evaluator does **not**
differentiate through the BFS. It evaluates a **fixed-size library** of
rollouts:

```
Gamma_{m,s,u}(phi) = Rollout(phi, mode m, seed s, command u),   u in U_s
```

Commands are sequences of desired object twists or pose increments covering
± translations along the seed frame axes, ± rotations about the pinch-axis
perpendiculars (two-finger) or about three body axes (three-finger), and
combinations. Shapes are fixed: `S` seeds × `U` commands × `H` steps.

Discretized trajectory: `Gamma = {q_k, x_k = (p_k, R_k), c_k}_{k=0..H}`.

### 4.2 Differentiable rolling solver

Per step, the existing rolling QP in parametric form:

```
z* = argmin_z  1/2 z^T H(phi,q) z + c(phi,q,u)^T z
      s.t.     A(phi,q) z <= b(phi,q)
```

with `z = [dq, dtheta]`, the same objective, gap band and boxes as
`rolling.build_rolling_qp`. Options for derivatives, to be chosen at Phase 2
by measured gradient accuracy: (a) softly constrained unconstrained solve
(penalty on gap band and boxes) differentiated directly, (b) implicit
differentiation of the KKT system of `qp.solve_box_qp` at the active set via
`jax.custom_vjp`, (c) an off-the-shelf differentiable QP layer (JAXopt /
qpax-style) if it matches the IPM on the recorded reference QPs. Start with
(a), add hard constraints progressively. Validate every option against finite
differences of the smooth rollout objective, and check KKT residuals and
gradient behaviour near active-set changes.

After each step, the Gauss-Newton projection is kept as a fixed-iteration,
differentiable map (no `argmax` backtracking in the smooth path; fixed damping).

### 4.3 Residuals and trajectory weight

Per step `k`, nonnegative smooth residuals:

- rolling: `r_roll,k = J_surf dq_k - r_s [n]x dtheta_k - dp_k` (the reference's
  velocity residual; tangential part is the slip),
- gap: `softplus(|g_k| - eps_g)` per active contact,
- collision: `softplus(eps_col - clearance_k)` over non-contact capsules and
  finger pairs (smooth min over capsules via log-sum-exp),
- limits: `softplus(q_k - q_hi) + softplus(q_lo - q_k)`.

Trajectory energy and validity weight:

```
E^(j) = lambda_r E_roll + lambda_g E_gap + lambda_c E_col + lambda_l E_lim
w_j   = exp(-E^(j) / sigma_E)
```

Energies are **normalized per step or taken as a smooth worst-step penalty**
(log-sum-exp over steps), so lengthening `H` cannot shrink the weight. `w_j`
is a surrogate, not a probability of success. Per-constraint residuals are
reported individually, not only the aggregate.

A trajectory prefix is credited only up to the first step where the exact
gates would fail by a margin (soft truncation: cumulative product of per-step
weights), so a rollout that leaves the manifold at step 3 does not earn
coverage at step 30.

### 4.4 Smooth translational coverage

Fixed grid `V` of sphere-centre positions (D8), spacing `h`:

```
a_{jkv} = exp(-|p_{jk} - v|^2 / (2 sigma_p^2))
o_v     = 1 - prod_{j,k} (1 - w_jk a_{jkv})          (computed in log space)
K_T~    = (h^3 / V_ref) * sum_v o_v
```

`V_ref` is the fixed task volume. `K_T~` is smoothed coverage by the evaluated
library, not the BFS reachable volume.

### 4.5 Smooth rotational coverage

Two-finger: pinch axis `a_jk = R_jk^T(-n_thumb)` in the body frame, bin
centres `b` from the same HEALPix nside-6 line bins as the reference; kernel
`k_R(a,b) = exp(-(1 - (a^T b)^2) / (2 sigma_R^2))` (no `|.|` cusp, no arccos).
Aggregate as in 4.4 over trajectories; evaluate at a **fixed set of spatial
regions** (e.g. the grid cells nearest the seed and the library end-points),
not through an exact top-five selection.

Three-finger: full orientation, `e_R = log(R_t^T R)^vee`,
`k(R, R_t) = exp(-|e_R|^2 / (2 sigma_R^2))` over a fixed, deterministic,
approximately uniform set of SO(3) targets (D16): the **576-point** level of
the Hopf-fibration grid (Yershova, Jain, LaValle, Mitchell 2010: HEALPix
nside 2 on S² × 12 points on the fibre), neighbour spacing ≈ 30°, each target
owning a ball of ≈ 18° radius, with kernel width `sigma_R = 0.15 rad` (about
half the spacing, so coverage is smooth without gaps). The exact three-finger
counting evaluator uses the next level, **4608 bins** (≈ 15° spacing, ≈ 9°
radius), comparable to the reference's ≈ 10° pinch-axis bins. Both sets are
generated once, stored as constant arrays with their generator and checked
by a test for uniformity (nearest-neighbour spacing within 10% of the mean).
Rationale: rolling reaches tens of degrees from a seed, so at 72 targets only
a handful ever fall inside the reachable region and the gradient comes from
very few kernel terms; at 576 the kernel cost (about 2 M log-maps per design
at 8 seeds × 12 commands × 40 steps) is still negligible next to the rollouts.
Not numerically comparable to the two-finger 228-bin coverage; reported
separately.

### 4.6 Seeds and seed robustness

A fixed budget of feasible seeds per design (same count for every design),
found deterministically (Sobol candidates, contact projection, exact
feasibility check; the reference's seed pipeline for two fingers). Seeds are
**reprojected per morphology** (differentiably where numerically possible,
otherwise outside the gradient loop with the gradient reported as conditional
on the seed set). Aggregate:

```
K_seed = (1 - beta) mean_s K_s + beta softmin_s K_s
```

### 4.7 Interface and acceptance

```python
score, diag = dkarma.metrics.evaluate(morphology, seeds, commands, metric_config)
grad = jax.grad(lambda phi: dkarma.metrics.objective(phi, seeds, commands, metric_config))(phi)
```

`diag` contains: translation coverage, rotation coverage, per-constraint
residuals (per trajectory and worst-step), rollout success flags, coverage per
seed, gradient norms per parameter group. Acceptance: finite, finite-difference
validated gradients that improve the smooth objective under small steps
without driving rollouts into grossly infeasible states, and a positive
Spearman correlation between `K_T~` and exact KaRMA-T across a morphology
dataset (Experiment I).

---

## 5. Act-KaRMA: force- and torque-feasible manipulation

Adds quasi-static force feasibility to the same rollouts. A trajectory must
satisfy motion compatibility **and** force feasibility.

### 5.1 Model

Point contacts with Coulomb friction. **Amended in Phase 4:** each contact also
transmits a bounded torsional moment `|m_c| <= mu a f_n` (soft-finger limit, patch
radius `a` = 3 mm, configurable; `a = 0` is the pure point contact). Without it
every two-finger pinch is infeasible: the weight's micro-N·m moment about the
contact line lies outside the range of the rank-5 grasp matrix
(`docs/phase4_report.md` §2.1). For `N_c` active
contacts with positions `r_i` relative to the sphere centre:

```
G = [ I ... I ; [r_1]x ... [r_Nc]x ]           (6 x 3N_c)
G f + w_ext = 0                                 (object equilibrium)
tau = J_c^T f + tau_hand,grav                   (joint torques; link gravity included)
|tau_j| <= tau_bar_j                            (per-joint ceiling from the actuator spec)
|f_t,i| <= mu f_n,i,   f_n,i >= 0               (friction cone)
```

`tau_bar` is identical for every joint of every design (D3, D13: one
XL330-M077-T per joint, primary ceiling 0.108 N·m). URDF `effort` values are
never read; `tau_hand,grav` uses the link masses of §2.1.

### 5.2 Exact force allocation (validation path)

```
f* = argmin_f 1/2 |f|_W^2  s.t.  G f + w_ext = 0, cones, -tau_bar <= J_c^T f + tau_g <= tau_bar
```

A convex SOCP at fixed geometry; solved on the host with a verified solver for
validation. Because `f = 0` is feasible when `w_ext = 0`, every tested state
carries a nonzero load (gravity at least) or a prescribed disturbance.

### 5.3 Smooth feasibility margin (optimization path)

```
E_F* = min_f  lambda_w |G f + w_ext|^2 + lambda_tau E_tau(f) + lambda_mu E_mu(f) + lambda_n E_n(f)
E_tau = sum_j softplus((sqrt(tau_j^2 + eps^2) - tau_bar_j) / eps_tau)^2
w_F   = exp(-E_F* / sigma_F)
```

A small inner problem per step, solved by a fixed number of projected-gradient
or Newton iterations inside the rollout (differentiable through unrolling or
implicit differentiation). The step weight becomes the product of kinematic
and force weights, or preferably one combined worst-step residual. `E_F*` is
evaluated at **every** step, not only the terminal pose.

### 5.4 Wrench capacity as a separate diagnostic

For sampled unit wrench directions `d` on a scaled 6-D sphere:
`alpha*(d) = max alpha s.t. G f + w_0 + alpha d = 0, cones, torque limits`.
Report mean and minimum capacity and the fraction of directions resisting a
specified magnitude. Not part of the default objective (it would double-count
feasibility already gating the trajectories); studied as an ablation term.

### 5.5 Interface

```python
force = dkarma.solvers.contact_forces.solve(morphology, state, w_ext, actuator_limits, contact_config)
score, diag = dkarma.metrics.act_karma.evaluate(morphology, trajectories, actuator_limits, load_cases, metric_config)
```

Diagnostics: max normalized torque usage, friction-cone and normal-force
margins, equilibrium residual, wrench capacity by direction,
actuation-feasible translation and rotation coverage, worst-step force margin.

---

## 6. MC-KaRMA: multi-contact reachability

### 6.1 Pairwise baseline

For fingers `{A, B, C}`: `K_pair = (K_AB + K_AC + K_BC) / 3`, plus min and max.
Uses the two-contact pipeline unchanged. Measures multi-pair versatility, not
simultaneous three-contact ability. Fixed averaging convention; compare at
equal finger count only.

### 6.2 Simultaneous three-contact rolling

One object state, one joint configuration, one shared force equilibrium.
`J_c = blockdiag(J_A, J_B, J_C)` (independent chains), `G = [G_A G_B G_C]`.
The per-step solver minimizes the commanded-motion error plus the three
rolling and gap residuals under joint-limit and collision constraints; the
QP grows to `z = [dq (all three fingers), dtheta]` with six gap rows. Never
intersect three independent two-finger solutions. Expectation to verify: for a
fixed morphology and seed, adding a mandatory third contact shrinks or
preserves the feasible set; three-finger advantages must come from wrench
authority, full rotation, and better seeds.

### 6.3 Three-contact seeds (highest-risk item)

Deterministic Sobol joint candidates → contact-site proposal from
sphere-centre and link proximity → nonlinear IK/contact projection on three
gaps with normal-geometry and collision terms → exact grasp and wrench
feasibility check → diverse fixed-budget seed set, stored with keys.
Reprojection per morphology as in §4.6.

### 6.4 Contact modes

Two-finger and three-finger maintained-contact motion, multiple fixed seeds,
no gaiting, no contact switching within a rollout. Contact switching is a
hybrid-systems problem and out of scope.

### 6.5 Interface

```python
pairs = dkarma.metrics.multicontact.evaluate_pairs(morphology, finger_pairs, metric_config)
three = dkarma.metrics.multicontact.evaluate_fixed_contacts(morphology, active_fingers=(0,1,2), seeds, actuator_limits, metric_config)
```

Array layout uses a fixed maximum number of contacts (3) with boolean active
masks, so two- and three-finger grasps share kernels.

---

## 7. Full objective

```
J(phi) = lambda_T K_T~(phi) + lambda_R K_R~(phi)        (default)
J(phi) = ... + lambda_W K_W~(phi) - lambda_C C(phi)    (ablation: wrench reward, design regularizers)
```

with actuation feasibility inside every trajectory weight. `lambda_T, lambda_R`
normalize each term by a fixed reference scale (e.g. the hand-drawn baseline's
exact scores). Three settings are run: translation-focused, rotation-focused,
balanced (D6). A Pareto front over translation, rotation and load capacity is
a target result. `C(phi)` holds only explicit design-space constraints
(§8.1) and regularization, no mechanical-packaging terms.

Metric variants are configurations of one evaluator:

```python
DIFF_KARMA = MetricConfig(contacts=2, rolling=True, torque_limits=False, force_equilibrium=False, orientation_space="pinch_axis")
ACT_KARMA  = MetricConfig(contacts=2, rolling=True, torque_limits=True,  force_equilibrium=True,  orientation_space="pinch_axis")
MC_KARMA   = MetricConfig(contacts=3, rolling=True, torque_limits=False, force_equilibrium=False, orientation_space="SO3")
FULL_KARMA = MetricConfig(contacts=3, rolling=True, torque_limits=True,  force_equilibrium=True,  orientation_space="SO3")
```

Each has `exact=True/False`; the exact form replaces smooth residuals by the
reference's hard gates and smooth coverage by counts.

---

## 8. Morphology optimizer

Own module, agnostic to metric internals.

### 8.1 Problem

`phi* = argmax_{phi in Phi} J(phi)` with `Phi`: link-length boxes, unit axes,
finger-base regions, an overall reach bound `reach(phi) <= L_max`, and fixed
limits and torque ceilings. No bearing or structural constraints; from Phase 11 the servo envelope
(D19, below) is the one packaging constraint.

**Admissibility (D18, added 2026-10-13 after the Phase 8 review).** A design is admissible only
if no two modelled fingers occupy the same space: every pair of capsules of different fingers
must clear by `eps_col` in every joint configuration that any evaluator visits, and, as the
configuration-independent part of that condition, any two finger bases must be at least
`2 r_l + eps_col` apart (24.1 mm for the 11.8 mm capsules). Phase 8 found that the original
KaRMA objective, which checks self-collision only among the fingers of its contact pair, placed
arm A's index and middle bases 17.8 mm apart so that its proximal capsules overlap by ≥ 5.8 mm in
every configuration (`docs/phase8_report.md` §8.5). Every optimizer, baseline included, enforces
D18 from Phase 9 on: the base-separation condition as a smooth penalty and a hard check, the
all-finger clearance as the self-collision gate of every evaluator. The reach bound is 250 mm
per finger and is the global hand envelope (owner's decision, §16 item 10).

**Servo envelope (D19, owner's decision 2026-10-23, §16 item 14; applies from Phase 11).** In the
direct-drive chain, joint 0's servo sits in the palm (not modelled) and the servos of joints 1–3 sit in
links 0–2; the fingertip link holds none, as §2.1's mass model already assumes (28 g for links 0–2,
10 g for the fingertip). Each of links 0–2 must contain one XL330-M077-T case, 20.0 × 34.0 × 26.0 mm
(datasheet, §2.1; the owner's measurement of 34 × 20 × 23 mm fits inside it), with its 34 mm side along
the link axis and its 20 × 26 mm cross-section centred on the axis. The link's capsule contains the case
when:

* its radius is at least half the cross-section diagonal, `r_servo = sqrt(20² + 26²) / 2 = 16.40 mm`
  (diameter 32.80 mm);
* its length (joint to joint) is at least the case length, `L_min = 34 mm`.

Links 0–2 all use `r_servo` and the 34 mm floor. A relaxation of link 0's length (joint 1's servo
mounted across the finger at the knuckle, link 0 back to 12–90 mm) was set on 2026-10-23 and withdrawn by
the owner on 2026-10-24, before any Phase 11 start was evaluated: link 0 keeps the 34 mm floor.

The fingertip link keeps `r_l = 11.8 mm` and its length bounds (12–90 mm). It is the contact surface,
so the frozen task's contact radius (`link_radius_m`, §2) is unchanged. D18's rule then gives a base
separation of `2 r_servo + eps_col = 33.30 mm`, because the base capsule is a servo link. The all-finger
clearance uses each capsule's own radius. These are packaging bounds, not a mechanical design: horn
placement, cabling, bearings and the palm stay out of scope. None of the 70 designs of Phases 7–10
satisfies D19: in every one, link 0 of the index or middle finger sits at the old 12 mm floor, and a
link 1 or 2 is shorter than 34 mm (link 2 in 193 of their 210 fingers).

### 8.2 Parameterization

Start structured and low-dimensional: finger bases (xyz + orientation), one
length per segment, two angles per axis, ranges / tip radii / torques fixed.
No parameters are tied across fingers (D14): a three-finger 4-DOF hand has
3 × (6 + 4 × 3) = 54 free coordinates with fixed limits, 78 if limits are
optimized in an ablation. The 4+4 two-finger family from
`optimize/hand_family.py` is the first instance; the 3×4 topology the second.

### 8.3 Algorithms

- Gradient: Adam on the smooth objective, optional L-BFGS refinement.
- Gradient-free: CMA-ES, reusing `optimize/run_optimization.py`'s pipeline
  (checkpointing, `evals.jsonl`, box bounds, `CMA_stds`).
- Hybrid: Sobol or CMA-ES initial designs → gradient refinement → exact
  evaluation → best physically feasible candidate.

### 8.4 Continuation and selection

Kernel widths `sigma_p, sigma_R` and penalty weights follow a schedule from
broad to tight; the final stage uses strict constraints followed by exact
verification. Two scores are tracked: the training (smooth) score and the
validation (exact) score, the latter computed every `k` iterations with the
original discrete KaRMA (via `karma_jax.metric.compute_metric`) and the exact
actuation-aware evaluator. The saved candidate is the best by a prespecified
validation criterion. All parameters and PRNG keys are logged so every design
regenerates exactly.

```python
result = dkarma.optimization.optimize_morphology(initial_params, parameterization, objective, optimizer_config, validation_fn)
result.best_params, result.best_urdf, result.training_history, result.validation_history, result.constraint_history
```

---

## 9. Architecture

### 9.1 Packages

```
karma/            frozen reference (paper numbers); never modified
karma_jax/        exact batched evaluator; extended for traced morphology and population batching, oracle-tested against karma/
dkarma/           this project
  core/           morphology.py (phi -> model arrays, bounded transforms), kinematics.py (thin wrappers over karma_jax.kinematics), geometry.py, contacts.py, object_model.py
  solvers/        rolling_qp.py (parametric QP + derivative strategy), rollout.py (lax.scan rollout), contact_forces.py (soft + exact), seed_projection.py
  metrics/        smooth_coverage.py, diff_karma.py, act_karma.py, multicontact.py, full_karma.py, exact.py (discrete counterparts), config.py (MetricConfig)
  optimization/   parameterization.py, objectives.py, constraints.py, adam.py, cmaes.py (wraps optimize/), hybrid.py, checkpoint.py
  experiments/    configs/, gradient_validation.py, surrogate_validation.py, optimize_morphology.py, ablations.py, robustness.py
  export/         urdf_export.py (phi -> URDF + robot YAML, roundtrip-tested), actuation_spec.py
  tests/          test_morphology.py, test_kinematics.py, test_contacts.py, test_gradients.py, test_rollout.py, test_wrench_feasibility.py, test_multicontact.py, test_optimization.py
optimize/         existing CMA-ES study; its hand family and queue tooling are reused, not rewritten
```

### 9.2 Reuse map

| Need | Reuse | Change required |
| --- | --- | --- |
| FK, Jacobians | `karma_jax.kinematics` | pass `DeviceModel` as a traced pytree argument instead of closing over it; fix `J_pad` and `M_pad` scatter defects |
| Model arrays | `karma_jax.model.FlatModel` | add a JAX constructor from `phi` (no Pinocchio); per-capsule radius array |
| Contact pass | `karma_jax.geometry` | smooth variants (soft-min clearance, fixed assignment) alongside the discrete ones |
| QP | `karma_jax.qp`, `rolling` | derivative strategy (§4.2); unchanged for the exact path |
| Projection | `karma_jax.projection` | fixed-iteration smooth variant |
| Step gates | `karma_jax.steps` | reused as the exact validator of smooth rollouts |
| Seeds | `karma_jax.seeds`, `ik` | two-finger as-is; three-finger extension |
| Exact metric | `karma_jax.metric.compute_metric`, `karma.metric.compute_metric` | validation callbacks only |
| Design family, URDF writer, CMA-ES loop, queue | `optimize/hand_family.py`, `run_optimization.py`, `queue_runner.sh` | generalize to N fingers; keep the 2-finger instance bit-compatible |
| HEALPix bins | `karma.orientation`, `karma_jax.healpix` | bin centres exported once as a constant array for the smooth kernel |

### 9.3 Array layout and JAX rules

```
B: designs  S: seeds  U: commands  H: steps  Q: joints (padded)  C: contacts (max 3)
q.shape = (B, S, U, H, Q)   p.shape = (B, S, U, H, 3)   R.shape = (B, S, U, H, 3, 3)
contact_active.shape = (B, S, C)   f.shape = (B, S, U, H, C, 3)
```

Pure functions; PyTree dataclasses for parameters; `lax.scan` for rollouts;
`vmap` over seeds, commands and designs; `jit` around the batched objective;
`value_and_grad` for updates; explicit PRNG keys; masks and padding for
variable contact counts. No Python lists of arrays, host callbacks, shape
changes or geometry construction inside compiled code. Keep per-design
variability in array values, never in shapes. Avoid dense fp32 `@` with
contraction width > 3 (TF32); use `karma_jax.linalg` helpers.

---

## 10. Tests and acceptance

### 10.1 Numerical tests

| Test | Expected |
| --- | --- |
| FK consistency | `dkarma` FK of `phi` equals Pinocchio FK of the exported URDF, 1e-9 fp64 |
| Jacobian finite differences | contact and twist Jacobians match central differences to 1e-6 relative |
| Axis normalization | all axes unit length after any parameter update |
| Joint-limit enforcement | no accepted exact rollout state exceeds limits (1e-6 slack) |
| Contact gap | gap sign and magnitude match `karma.contacts` on recorded states |
| Rolling residual | exact rollouts satisfy the slip gate; smooth residuals agree with the reference residual on the same step |
| Force equilibrium | `|G f + w_ext| < 1e-8 N` for exact allocations |
| Friction cone, torque | exact solutions satisfy cones and ceilings to 1e-9 |
| Two-contact reduction | the 3-contact solver with one contact masked reproduces the two-contact QP solution |
| Gradient check | autodiff vs finite differences, relative error < 1e-3 away from active-set changes, on a two-link planar finger with known analytic derivatives first, then spatial hands |
| jit/vmap agreement | batched results equal per-sample results |
| Determinism | identical inputs and keys give identical outputs |
| URDF roundtrip | export → reload through Pinocchio → FK matches |
| Reference regression | `karma_jax.metric.compute_metric` on the 16 hands and the run-T, A1, baseline designs stays within the tolerances documented in `docs/jax_rewrite_plan.md` |

### 10.2 Physical sanity tests (property-based, toy hands first)

- Torque monotonicity: raising a torque ceiling never shrinks the exact feasible wrench set.
- Friction monotonicity: raising `mu` never shrinks the exact feasible force set.
- Load behaviour: heavier objects make gravitational support harder (reachability may be non-monotonic; the force set is).
- Three-contact consistency: a reported three-contact state satisfies all three gaps and one shared object motion.
- Scale: kinematic KaRMA stays scale-invariant; actuation-aware scores at fixed physical torque and load do not.
- Ideal-actuation limit: with permissive torque and force limits, Act-KaRMA approaches Diff-KaRMA; with permissive residual tolerances, Diff-KaRMA's exact counterpart approaches original KaRMA on the trajectory library.
- Extra-contact monotonicity: adding a mandatory contact at fixed morphology and seed never enlarges the kinematic feasible set.

---

## 11. Roadmap

Each phase ends with working code, tests, and a short report (files changed,
formulation actually implemented, tests run with results, limitations, a
minimal usage example, comparison with the previous phase, remaining work).
Do not advance because code runs; advance when the acceptance tests pass.

| Phase | Deliverable | Exit criterion |
| --- | --- | --- |
| **0 Audit and baseline freeze** | §1 of this document; regression fixtures (16 hands + optimizer designs: baseline, T, A1) with their exact scores under both backends; padding-defect fixes in `karma_jax` with tests; `J_pad` default corrected in all callers | reference and JAX scores reproduce the recorded numbers within documented tolerances; `karma_jax/tests` green |
| **1 Differentiable morphology and geometry** | `dkarma.core`: `phi -> DeviceModel`, traced-model FK/Jacobians, per-capsule radii, smooth contact pass, URDF export | FK/Jacobian/roundtrip tests; `jax.grad` of a contact gap w.r.t. `phi` matches finite differences |
| **2 Differentiable rolling rollout** | `dkarma.solvers.rollout` with soft constraints, then hard constraints via the chosen QP-derivative strategy; `lax.scan` over `H`; exact validator from `karma_jax.steps` | gradient tests pass on the planar toy and on the 4+4 family; rollouts validated by the exact gates agree with reference sub-steps on recorded states |
| **3 Smooth reachability objective** | Diff-KaRMA: coverage kernels, trajectory weights, seed aggregation, diagnostics, continuation schedule | Experiment I surrogate checks: positive rank correlation with exact KaRMA-T on a morphology dataset; local gradient steps improve exact KaRMA on a majority of starts |
| **4 Quasi-static force and torque** | Act-KaRMA: grasp matrix, exact SOCP allocation, smooth margin, wrench capacity, actuator spec loader | physical sanity tests; exact and smooth evaluators agree on feasibility away from the margin |
| **5 Pairwise and three-contact** | MC-KaRMA: masked 3-contact kernels, three-contact seeds, SO(3) coverage, pairwise baseline | two-contact reduction test; extra-contact monotonicity; three-finger seeds found on the 3×4 family |
| **6 Optimization module** | Adam, CMA-ES wrapper, hybrid, checkpointing, validation callbacks, experiment artifact layout (§13) | reproducible multistart runs on the 4+4 family under matched wall-clock budgets against the existing CMA-ES results |
| **7 Experiments and realization** | Experiments I–IV, export of optimized skeletons and torque specs, hardware protocol | per §12 |
| **8 Calibration and validation** (§11.1) | actuation-penalty recalibration, continuation repair, seed-policy study, statistically meaningful optimizer comparison, verification of the three-contact failure, independent simulation, fabrication-candidate selection | per §11.1; no new metric formulation |
| **9 Admissible baselines and controller-independent validation** (§11.2) | KKT-derivative repair with a three-way derivative comparison, the collision-valid original-KaRMA baseline and the key comparison against Full KaRMA on the shared admissible design space, force/impedance MuJoCo replay with separated success criteria, 16-seed final validation, fabrication-candidate selection | per §11.2; no new metric formulation |
| **10 Compliant contact, powered admissible comparison, deployment-relevant grasps** (§11.3) | compliant-fingertip MuJoCo replay with separated outcomes, the A′ vs Full-D18 comparison on ≥ 16 new starts with a pre-registered common manipulation endpoint, the task-feasible grasp subset `S_task`, a revised selection criterion frozen before re-selection | per §11.3; framework fixed, no new metric formulation |
| **11 Re-optimization inside the servo envelope** (§11.4) | D19 in code (per-capsule radii, the 34 mm floor on links 0–2, the 33.30 mm base separation) with a bit-identical regression, a new start set, Full KaRMA + Adam on 24 starts, the 10.2 evaluation pipeline, selection by criterion v2 with the original criterion alongside, the export of the selected design | per §11.4; D19 is the only change, no new metric formulation |

Phase 0 was delivered on 2026-10-11 (`docs/phase0_report.md`): fixtures in
`dkarma/fixtures/`, padding fixes with tests in `karma_jax/`, and the GPU
re-score of all fixtures on the fixed kernels (recorded; see the Phase 1 report
for the fp32 shifts it revealed). Phase 1 was delivered the same day
(`docs/phase1_report.md`): `dkarma.core` and `dkarma.export`, 103 tests. Phase 2
was delivered the same day (`docs/phase2_report.md`): `dkarma.solvers` (soft and
hard-KKT rolling steps, masked projection, scan rollout, exact validator), 26
tests. Phases 3-6 followed on 2026-10-11/12 (`docs/phase{3,4,5,6}_report.md`):
Diff-KaRMA with physical-unit seeds and the exact library counterpart, Act-KaRMA
with the soft-finger torsion amendment (§5.1 note), MC/Full KaRMA with
three-contact seeds and SO(3) coverage, and the optimizer module. Phase 7's
simulation legs ran on 2026-10-12 (`docs/phase7_report.md`: Experiment I rho ≈ 0.98
vs the exact evaluators on the physical task; Experiment II torque/mass sweeps;
Experiments III/IV 3×4 co-design pilots; hardware out of scope). Phase 8 ran on 2026-10-11/13 (`docs/phase8_report.md`): force-credit recalibration with a converged
force solver, continuation repair (gradient hygiene and stage selection as the Adam default),
seed-policy study (re-projection default, exact-ranked union for two-finger objectives), the
8-start optimizer comparison (CMA-ES ahead on the two-finger objective, Adam on the three-contact
actuated one, matched optimizer time), the three-procedure verification of arm A's three-contact
failure, the write-ups, and the MuJoCo exact-vs-simulation confusion matrices (8.6, MuJoCo 3.15 installed on the owner's confirmation); arm A's index and middle fingers were found to overlap in every configuration (17.8 mm base separation). The owner's review of Phase 8 (`docs/phase8_review_2026-10-13.md`) set Phase 9 (§11.2). Phase 9 ran on
2026-10-13/14 (`docs/phase9_report.md`): the non-finite gradients traced to `exp_so3` at a zero rotation
(bit-identical repair, 107 → 0 non-finite iterations), D18 enforced in every optimizer, the collision-valid
original-KaRMA baseline (arm A′) compared with Full KaRMA under D18 on eight starts (Full ahead on the exact
Full score 5 of 8, p = 0.38; on simulated three-contact execution 6 of 8, p = 0.062 / 0.148; arm A′ the better
two-finger hand, 7 of 8, p = 0.016), a force-floor controller in MuJoCo (three-contact discrepancy *mixed*),
stable design rankings across three seed sets, and no fabrication candidate passing the written criterion.
The owner's review of Phase 9 (`docs/phase9_review_2026-10-14.md`) sets Phase 10 (§11.3). The working
tree stays uncommitted (D15). Phase 10 ran on 2026-10-14/23 (`docs/phase10_report.md`): compliant pads close
most of the three-contact simulation gap (verdict partial); Full-D18 beats arm A′ on the pre-registered
manipulation endpoint (22 of 24 new starts, p = 9 × 10⁻⁵); criterion v2 (§16 item 12, pending the owner's
review) selects Full-D18 start31 and the original criterion selects nothing. On 2026-10-23 the owner set
the servo envelope (D19, §16 item 14), which no Phase 10 design satisfies, and Phase 11 (§11.4).

### 11.1 Phase 8: calibration and validation before more optimization (added 2026-10-12)

The Phase 7 pilots were reviewed. The review's verdict, adopted here: preserve the
existing modules, introduce **no further metric formulation**, and move to validation,
optimizer calibration and physical realization. The metric-agnostic optimizer is the
architectural feature to build the ablations on. Phase 8 is that work, as seven items
with acceptance criteria. Items 8.1 to 8.3 gate every later optimization run; 8.4 and
8.5 gate every comparative claim; 8.6 and 8.7 gate fabrication.

**Review findings this phase answers**

| finding (Phase 7 pilots) | consequence |
| --- | --- |
| at the 0.043 N·m continuous ceiling the exact solver finds no holdable state on either design, yet the smooth score keeps 75–90 % of its kinematic value | the differentiable force penalty does not track exact feasibility near the margin (8.1) |
| Adam's strict continuation stage and the Full arm's stage 1 lowered the exact score; CMA-ES's last incumbent was a smooth-only gain | the schedule and the selection cadence are miscalibrated (8.2) |
| gradients are conditional on the start design's seeds; reprojection alone may not recover good grasps after large morphology moves | seed policy untested (8.3) |
| every optimizer comparison is one run from one start | no statistical weight (8.4) |
| "zero three-contact seeds" for the original-KaRMA design came from one seed-search procedure | not a proof of infeasibility (8.5) |
| the exact evaluators are feasibility checks, not a performance model under an independent simulator or controller | the exact-vs-simulation error term is unmeasured (8.6) |
| rho = −0.33 between smooth K_T and the published KaRMA-T | needs the written explanation of the two task definitions, not just the number (8.7) |

**8.1 Recalibrate actuation feasibility.** Make the smooth score respond sharply to
violations of the specified torque ceiling and friction cone, validated against the
independent exact solver (`contact_forces.solve_exact`) rather than tuned by eye.
Procedure: on a held-out set of states (every primitive-end state of Experiment II's
libraries plus states sampled around the margin by scaling the load and the ceiling),
compute the exact feasibility label and the smooth energy; choose `EnergyWeights.force`,
the hinge scale and the step-credit mapping so that the per-state credit `exp(-e_F)`
separates the labels. Candidates: a larger weight (20–50), a tighter hinge, a credit
mapping with a hard knee at the exact tolerance, or a two-term energy (soft inside the
margin, steep outside). *Acceptance:* on the held-out states, AUC of smooth credit against
the exact label ≥ 0.95 and, at the continuous ceiling, actuation-aware coverage ≤ 10 % of
the kinematic value wherever the exact evaluator rejects every state; the gradient tests
of Phase 4 still pass; Experiment II re-run with the new calibration and its table
replaced in the Phase 7 report.

**8.2 Repair the continuation schedule.** Find out why stricter stages reduce the exact
score. Instrument a run: per stage, exact score of the stage-start and stage-end designs,
the smooth-vs-exact gap, the share of steps gated by each residual. Test, separately and
combined: stage-specific validation (select the best validated design *within* each
stage and warm-start the next stage from it), re-projection or re-generation of seeds at
every stage boundary (8.3), a less aggressive final `sigma`, and skipping the strict
stage. *Acceptance:* on the two-finger baseline and the 3×4 start, the exact score of the
final iterate is ≥ the best intermediate exact score minus 2 % in at least 4 of 5 starts;
the chosen schedule is the new default and is documented with the measurements.

**8.3 Seed-policy study.** Controlled comparison, same starts and budgets, of (a)
reproject the start seeds (current), (b) regenerate seeds every stage, (c) regenerate
every `k` iterations, (d) a union of reprojected and regenerated seeds with the fixed
budget enforced by exact-score ranking. Report the exact score of the final design, the
number of feasible seeds along the run, and wall clock. The gradient's conditionality on
the seed set must be stated in every report that uses policy (a). *Acceptance:* a policy
is adopted as default with the data, and `make_objective`/`adam` expose it as a parameter.

**8.4 Statistically meaningful optimizer comparison** (Experiment I's optimization leg,
done properly). For the two-finger Diff-KaRMA objective and the 3×4 Full objective: ≥ 8
independent starts per optimizer (Sobol-perturbed around the anthropomorphic start plus
random designs that have seeds), Adam / CMA-ES / hybrid at **matched wall clock** on the
RTX 4090 (measured, validations excluded from the budget and run afterwards on every
checkpoint), exact evaluation of every returned design with fresh seeds and the physical
original KaRMA. Report mean, median, best and interquartile range; paired comparison per
start; the fraction of runs whose returned design beats the start. *Acceptance:* the
report states a conclusion with its uncertainty; "differentiable optimization is
superior to CMA-ES" is claimed only if it holds on the paired comparison at the matched
budget, otherwise the honest statement is written instead.

**8.5 Verify the three-contact failure of the original-KaRMA design.** Before claiming
the third finger is unusable: (i) a 10× larger seed search (more joint samples, centres
proposed from all phalanges not only the fingertips, several LM restarts per candidate),
(ii) alternative initializations (joint-space random restarts, warm starts from the other
designs' seeds projected onto this design), (iii) an independent constrained grasp solver
on the host (SciPy `trust-constr` or SLSQP on the gaps-plus-clearances problem, from many
starts) that does not share code with `solve_multi_ik`. Run the same procedure on the
designs that *do* have seeds as a control. *Acceptance:* the claim in the Phase 7 report
is upgraded to "no feasible three-contact grasp found by three independent procedures
with N attempts" with N stated, or withdrawn and replaced by the found seeds and the
re-scored suite.

**8.6 Independent physical-task performance in simulation.** The exact evaluators are
quasi-static feasibility checks. Establish performance under a separate contact-dynamics
simulator and controller: install MuJoCo in the `karma-jax` environment (owner to confirm
the package), build the scene from the exported URDF with capsule collision geometry, the
15 mm / 35 g sphere, Coulomb friction 0.6 with a torsional coefficient consistent with the
3 mm patch (§5.1 amendment), joint position actuators with the XL330 ceiling as the force
limit; a simple controller replays each trajectory's joint path from the exact library;
success = sphere stays in contact (no drop, slip below a stated bound) and reaches the
commanded pose within a stated tolerance. Report, per design, the fraction of
exactly-valid trajectories that succeed in simulation and the fraction of exactly-rejected
ones that nevertheless succeed, i.e. the exact-vs-simulation error term of §12 IV.
*Acceptance:* the two-finger baseline, the 3×4 start, arm A and arm B of Experiment IV
are evaluated; the confusion matrix is in the report; disagreements are examined (which
gate, which assumption).

**8.7 Write-ups and candidates.** (a) A careful section explaining the negative
correlation between the smooth/physical scores and the published KaRMA-T: the two task
definitions (`L_ref`-scaled nominal object vs fixed physical object), what each rewards,
and why the Phase 0 exploit designs separate them; include the per-design table and the
scale of each hand relative to the object. (b) Preserve the two-finger-optimized (arm A)
and the full-method (arm B) designs as fabrication candidates with their exported
specifications; select nothing for fabrication until 8.1 to 8.6 are done and the
selection criterion (§12 IV) is written down first.

**Not in Phase 8:** new metric variants, new kernels, new contact models, changes to
`karma/` or to the frozen task of §2. Any such change needs an explicit owner decision
recorded in §16.

---

### 11.2 Phase 9: admissible baselines and controller-independent validation (added 2026-10-13)

Scope from the owner's Phase 8 review (`docs/phase8_review_2026-10-13.md`): the framework is
sufficient for the paper's main experimental phase; what remains is numerical robustness,
independent task validation on a mechanically admissible design space, and the selection of
fabrication candidates. **No further metric extensions.** The question Phase 9 must answer:
*can Full KaRMA produce a collision-valid three-finger hand that outperforms a collision-valid,
kinematic-only optimized hand on the same physical manipulation benchmark?* Items in priority
order; each ends with a section of `docs/phase9_report.md` and a green `dkarma/tests`.

**9.1 Repair the KKT derivative at degenerate active sets.** `rolling_qp.kkt_resolve` returns
non-finite or exploding derivatives when the identified active set makes the KKT matrix
singular (Phase 8.2: 26 to 54 non-finite iterations per five runs under reseeding, norms 1e4 to
1e8; the optimizer-level hygiene treats the symptom). Deliverable: a comparison, on a corpus of
recorded degenerate and well-conditioned states from the Phase 8 runs, of (a) the current exact
active-set derivative, (b) a regularized KKT derivative (several regularization levels), (c) a
least-squares / pseudoinverse derivative, and (d) the active-set derivative with the dependent
rows removed, each against central finite differences of the QP solution where the derivative
exists and against the directional derivatives on both sides where it does not. Do not add
damping until the NaNs disappear: report finite-difference agreement on well-conditioned sets
(must stay within the Phase 2 tolerances), the behaviour at degeneracies, and the optimization
effect (the 8.2 study's `hygiene_select` variant re-run with the chosen derivative on the same
five starts, non-finite and spike counts, final and returned exact scores). *Acceptance:* the
chosen derivative passes the existing gradient tests unchanged on well-conditioned states,
produces no non-finite gradient on the corpus, and the re-run matches or improves the Phase 8
returned scores (paired, 5 starts, reported with the usual caveat on noise).

**9.2 Collision-valid original-KaRMA baseline and the key comparison.** Arm A stays in the paper
as the failure-mode case. Add arm A′: CMA-ES against the original KaRMA (the `optimize/` proxy
and final scoring as in Phase 7) on the 3×4 topology with D18 enforced throughout the
optimization (base separation as a hard constraint on the candidate before evaluation, all-finger
clearance in the fitness's self-collision check, infeasible candidates scored zero), same
start, bounds and evaluation budget as arm A. Then the comparison the review names: arm A′ against
the Full-KaRMA optimum (arm B, and a fresh Full-KaRMA run under the same D18 constraint if arm B
violates it — it does not: 13.5 mm clearance bound) on the common three-contact physical suite
(`codesign_comparison.suite`: exact MC and Full counts with the 8.6 exact path, wrench margins,
two-finger exact coverage, the physical original KaRMA) and in MuJoCo (9.3). CLAUDE.md rule 12
applies: ≥ 8 independent starts per arm, matched evaluation budgets, fresh seeds, paired
statistics. *Acceptance:* the comparison is reported with its uncertainty, whichever way it
goes; the claim "Full KaRMA produces better simultaneous-contact manipulation than a
collision-valid original-KaRMA optimum" is made only if the paired data support it.

**9.3 Force/impedance-controlled MuJoCo replay with separated criteria.** Keep the 8.6 scene,
morphologies and target trajectories; add a contact-force / impedance controller (per-fingertip
normal-force regulation around the exact allocation, or joint impedance with the squeeze as a
force set-point) and report, for the same trajectories, three separated outcomes per primitive:
object-pose success (position and orientation within the 8.6 tolerances), three-contact
retention, and drop or gross instability. Compare the position servo of 8.6 and the force
controller on the two-finger baseline, the 3×4 start, arm B and arm A′. *Acceptance:* the
confusion matrices of both controllers on the four designs; a statement of whether the
three-contact discrepancy of 8.6 is controller-induced (success rises substantially under force
control) or model-induced (it does not), with the numbers.

**9.4 Final validation with more seeds.** Phase 8.2 measured the exact validation's one-step
sensitivity at 10 % (8 seeds) and 4.3 % (16 seeds). Every number that ranks candidates in 9.2
and 9.5 is computed with ≥ 16 fresh seeds (two-finger) and ≥ 12 three-contact seeds, and reported
with its seed-set spread (three seed-search configurations). *Acceptance:* the candidate
rankings of 9.2 are stable across the three seed sets.

**9.5 Fabrication-candidate selection.** Apply the criterion written in `docs/phase8_report.md`
§8.7(b), amended for D18 (eligibility requires admissibility), to arm A′, arm B and any Phase 9
Full-KaRMA optimum, using 9.2 to 9.4. Arm A is ineligible (not buildable as three fingers) and is
kept as the failure case. *Acceptance:* a selection with the criterion's numbers, the two-finger
cost of the choice stated, and the exported specifications under `results/fabrication_candidates/`.

**Design-box values** (owner's decision, §16 item 10): minimum base separation 24.1 mm
(`2 r_l + eps_col`), reach bound 250 mm per finger as the global hand envelope; long fingers
within it are admissible, not an optimization error.

**Paper framing adopted** (review §7): three contributions — the differentiable object-centric
dexterity objective (rolling reachability and actuation feasibility); the morphological
insight that local or pairwise dexterity conflicts with coordinated multi-finger manipulation,
established on collision-valid baselines; validation across exact reachability, contact
dynamics and hardware. The optimizer comparison is an additional result: gradient-based
optimization performs differently depending on the constraint structure (Phase 8.4: ahead on
the narrow three-contact feasible region, behind CMA-ES on the two-finger objective at matched
time). Working title: *Differentiable Dexterity Optimization for Robotic Hands*. Every table and
plot states which task definition (published `L_ref`-scaled vs fixed physical) it reports.

### 11.3 Phase 10: compliant contact, a powered admissible comparison, and deployment-relevant grasps (added 2026-10-14)

Scope from the owner's Phase 9 review (`docs/phase9_review_2026-10-14.md`). Phase 9
(`docs/phase9_report.md`) fixed the gradient failure, which came from `exp_so3` at a zero rotation
and not from the KKT derivative. It added the collision-valid original-KaRMA baseline and separated
object-pose tracking from contact retention in MuJoCo. It left the central comparative claim and the
fabrication criteria unresolved. On eight starts, Full KaRMA under D18 is ahead of arm A′ on the exact
Full score on 5 of 8 (+26 %, p = 0.38, bootstrap CI includes zero). It is ahead on simulated
three-contact execution on 6 of 8 (p = 0.062 / 0.148). Arm A′ is the better two-finger hand (7 of 8,
p = 0.016). No candidate passed the selection criterion: every one has a weak grasp among its 12 seeds,
and every one has a three-contact simulation false-positive rate far above the two-finger reference.

**The framework stays fixed.** No metric variant, kernel, contact model of any evaluator, solver or
optimizer change, and no change to the frozen task (§2) or the D18 design box. The norm-100 gradient
clipping, the `reduced` KKT derivative, `exp_so3_safe` and the cached exact evaluators stay as they are.
The large finite gradients near binding contact constraints, and the interior-point QP failures on
17–24 % of hard-stage translation sub-steps, are known properties of the landscape. They are not Phase 10
work items. Phase 10 answers three questions in this order, then re-runs the selection. Each item ends
with a section of `docs/phase10_report.md` and green `dkarma/tests` and `karma_jax/tests`.

**10.1 Does compliant contact resolve the remaining simulation mismatch?**

* *Model.* A defensible silicone-fingertip approximation in the MuJoCo scene of 8.6/9.3. It applies to
  the capsules that carry assigned contacts, the terminal links. It is a simulation model only: it lies
  within the existing capsule envelope, as the outer `t` millimetres of the 11.8 mm capsule radius, so the
  zero-load geometry, the morphologies, the exported trajectories and every exact verdict are unchanged.
  The compliance comes from a stated material assumption: a silicone hardness range, e.g. Shore 10A–40A;
  a pad thickness range, e.g. 1–3 mm; and the resulting contact stiffness and damping for the contact
  patch. It is mapped to MuJoCo contact parameters (`solref` / `solimp`, or a soft pad geom) with the
  torsional and rolling friction of the resulting patch. The parameters are documented as assumptions
  until they are measured on fabricated fingertips. The central parameter set is fixed before any
  replay. A small sensitivity grid (at least 3 stiffness × 2 thickness values) and the rigid 8.6 model are
  run as bounds.
* *Same trajectories and controllers.* The 9.3 archives and the 9.2 12-seed libraries; the 8.6 position
  servo and the 9.3 force-floor controller with their 9.3 parameters. A controller parameter may be
  re-tuned only if compliance makes it unstable, and only on the two-finger baseline, as in 9.3, never
  on three-contact data.
* *Outcomes, separated:* object-pose tracking (4 mm / 0.2 rad), contact retention at the primitive end
  (the 9.3 strict criterion) and over the primitive (the 9.3 duty-cycle reading, now pre-registered as
  a secondary criterion), and drop or gross instability. The rates are reported per library on holding
  seeds, with the confusion matrices.
* *Decision rule, frozen in the Phase 10 report before the first compliant replay.* The residual gap is
  the same design's two-finger strict rate minus its three-contact strict rate, under the same
  controller on holding seeds. It is *resolved* if, at the central parameters, the gap is ≤ 0.10 on
  every three-contact design. It is *not resolved* if the gap shrinks by less than half of its rigid value
  on every design. Otherwise it is *partial*. The verdict is *robust* if it holds over the sensitivity grid.
* *Second deliverable:* a technically defensible three-contact simulation-agreement reference for 10.4.
  This is the three-contact false-positive rate a design achieves under compliant contact, measured on
  the designs of 9.3 and 9.2, so that three-contact agreement is no longer judged against a two-contact
  reference. Its numeric threshold is derived and frozen in 10.4, before re-selection.
* *Acceptance:* the separated outcomes and confusion matrices under both controllers at the central
  parameters and over the grid, with a verdict by the frozen rule.

**10.2 Does the full objective outperform an admissible baseline over a larger sample?**

* *Arms:* arm A′ and Full-D18 exactly as in Phase 9 (`dkarma/experiments/admissible_baseline.py`). Arm A′
  is original KaRMA + CMA-ES with D18 throughout: 96 proxy evaluations, population 8, σ₀ 0.3, CMA seed =
  start index. Full-D18 is Full KaRMA + Adam: 40 iterations, 2 stages, 6 three-contact seeds,
  `ADAM_DEFAULTS`, optimizer seed 0. The design box is D18.
* *Starts:* at least 16 **new** starts, the review's minimum, target 24. They are drawn by the 8.4
  procedure (Sobol seed 321, scales 0.25 and 0.5, each start required to have ≥ 3 three-contact seeds),
  continuing the sequence past the eight Phase 9 starts. The Phase 9 starts are the pilot. They are
  reported alongside but are not part of the primary test, since the endpoint below was chosen with
  knowledge of them.
* *Primary endpoint, frozen in the Phase 10 report before the first new run.* It is a common manipulation
  benchmark measured identically for both arms, not either objective's own score: simulated three-contact
  manipulation with the 10.1 compliant fingertips at the central parameters under the force-floor
  controller. The quantity is the number of commanded three-contact primitives executed successfully
  (pose and the 10.1 retention criterion), counted along each command until its first failure,
  regardless of the exact verdict, per budgeted seed. It uses 12 fresh seeds, the default seed
  configuration, and the shared hard-mode rollout as the planner for both arms. Statistic: two-sided
  Wilcoxon signed-rank on the paired per-start differences, α = 0.05, with the bootstrap 95 % CI of the
  mean difference and the sign counts. The claim "Full KaRMA produces better simultaneous-contact
  manipulation than a collision-valid original-KaRMA optimum" is made only if p < 0.05.
* *Secondary endpoints, reported without separate claims:*
  * the 9.2 measure (exactly valid primitives that succeed in simulation);
  * pose-only success, and the rigid-contact replay;
  * the exact Full score (9.4 suite: 12 three-contact and 16 two-finger seeds × three seed
    configurations), the exact two-finger library score, and the original KaRMA on the physical and
    published tasks with the all-finger gate;
  * the primary endpoint repeated under the `sobol512_rs7` seed configuration as a pre-registered
    robustness check.
* *Power, from the Phase 9 pilot* (8 starts, rigid contact; indicative only). The exact Full score has
  paired effect size d ≈ 0.26, which needs about 116 starts for 80 % power; at 16 starts the power is
  0.17. Simulated strict execution per budgeted seed has d ≈ 0.89 (position servo) and d ≈ 0.70 (force
  controller), giving 0.92 / 0.75 power at 16 starts and 0.99 / 0.91 at 24. This is why the exact score is
  secondary and why the target is 24 starts.
* *Budgets:* the Phase 9 settings, with compute reported. Arm A′ takes about 2,300 s per run on the GPU,
  against Full-D18's 69 s of optimizer time (about 820 s with validations). The difference favours arm A′
  by more than 30×, which makes a Full-KaRMA win conservative. A Full-KaRMA loss is attributed to the
  objective only after a budget-matched control. *Cost, measured in Phase 9:* about 0.9 GPU-hours and
  about 2 CPU-worker-hours per start, the latter for the suites and exports, which run on the CPU, three
  at a time (CLAUDE.md compute hygiene). 24 starts take about 21 GPU-hours, with the CPU work running in
  parallel.
* *Acceptance:* the paired comparison on ≥ 16 new starts, with the primary endpoint, its CI and the
  frozen decision applied, whichever way it goes; the secondary endpoints and the robustness check
  reported.

**10.3 Which candidate has the best deployment-relevant set of robust grasps?**

* *Task-feasible grasp subset* (review §5), at the primary torque ceiling over the 14 standard wrench
  directions of §5.4:
  `S_task = { s ∈ S : min_d α_s(d) ≥ 1 }`.
* *Report, for every candidate of 9.5 and 10.4 and every 10.2 design:*
  * the proportion `|S_task| / |S|` under each seed configuration, and the per-seed capacity
    distribution;
  * the exact Full score and the 10.2 primary endpoint computed on the seeds in `S_task` only. These use
    the same formulas restricted to a subset of seeds; any union-over-seeds coverage would be a new
    formulation and needs an owner decision (§16).
* *Transparency:* the all-seeds results are reported alongside, unchanged. Subset results are never
  presented as equivalent to the original criterion.
* *Acceptance:* the subset proportions and subset measures for every listed design, beside the all-seeds
  numbers.

**10.4 Revised selection criterion and re-selection.**

* *Criterion v2.* It is written from stated deployment requirements: the frozen task (15 mm / 35 g
  sphere, primary ceiling, the ±100 mm task cube), the intended manipulation (three-fingertip rolling of
  the sphere) and the controller to be used on hardware. It is frozen, dated, in the Phase 10 report and
  recorded in §16 as frozen by the session and subject to the owner's review, **before any candidate is
  evaluated against it**.
  * Robustness is stated over `S_task`: for example, a minimum number of robust grasps out of 12 under
    every seed configuration, and a minimum exact Full score on them.
  * Simulation agreement is stated against the compliant three-contact reference of 10.1, not against
    the two-finger reference.
  * The original §8.7(b) criterion as applied in 9.5 is applied alongside and reported unchanged.
* *Candidates:* arm B; the Phase 9 candidates (Full-D18 start0, arm A′ start6); and, from 10.2, the best
  design of each arm by the primary endpoint, ties broken by the exact Full score. Arm A stays the
  failure case.
* *Acceptance:* a selection, or none, under criterion v2 with the criterion's numbers, the two-finger cost
  of the choice, and the original criterion's verdict beside it. A selected design's specification is
  exported to `results/fabrication_candidates/selected_<design>/` and re-scored from the exported copy.

**Claim status after Phase 9** (review §6; the paper reports claims at this status until Phase 10
changes them):

| Research claim | Status after Phase 9 |
| --- | --- |
| Differentiable rolling-contact morphology optimization is implementable | supported |
| Actuation-aware feasibility can be calibrated against exact force constraints | supported within the evaluated conditions |
| Simultaneous-contact reachability can be evaluated and optimized | supported |
| Gradients are stable enough to run optimization experiments | substantially improved (9.1); clipping still required |
| Gradient optimization universally beats CMA-ES | not supported (8.4) |
| Full KaRMA consistently outperforms collision-valid original KaRMA | not established (9.2); the object of 10.2 |
| The exact three-contact metric predicts successful physical manipulation | partially supported by MuJoCo (9.3); the object of 10.1 |
| Optimized morphologies are ready for hardware deployment | not yet established; the object of 10.3–10.4 |

The paper's working title stays *Differentiable Dexterity Optimization for Robotic Hands*. The decisive
result the review asks for is robust manipulation improvement under a common, mechanically admissible
design space, measured on a common benchmark (10.2's primary endpoint), not a better score on Full
KaRMA's own metric.

### 11.4 Phase 11: re-optimization inside the servo envelope (added 2026-10-23)

Scope from the owner's decision of 2026-10-23 (§16 item 14). Phase 10 (`docs/phase10_report.md`) selected
Full-D18 start31 under criterion v2, pending the owner's review. Its links are 12–47 mm long and every
capsule has the 11.8 mm radius, so the XL330 cases the hand is to be built from do not fit in it. D19
(§8.1) makes the servo envelope part of the design box. Phase 11 re-runs the optimization inside D19 and
selects a new candidate by the same procedure. It makes no comparative claim: the 10.2 result stands for
the D18 box and is not re-tested. The new floor will bind: in every one of the 70 designs of Phases 7–10,
link 0 of the index or middle finger sits at the old 12 mm floor and a link 1 or 2 is shorter than 34 mm
(nearly always link 2). The 16.40 mm proximal capsules also leave less free space around the 30 mm
sphere.

**D19 is the only change.** It brings per-capsule radii (16.40 mm for links 0–2, 11.8 mm for the
fingertip), a 34 mm floor on links 0–2, and the 33.30 mm base separation that follows from D18's rule.
The frozen task stays (§2, `dkarma/configs/task_spec.yaml`), including the 11.8 mm contact radius
`link_radius_m`. Contacts stay on the fingertip, whose radius is unchanged. The rolling kinematics, the
10.1 compliant-pad sets (`dkarma/configs/compliant_pads.json`, fitted to the 11.8 mm fingertip) and the
force-floor controller therefore stay valid. There is no metric, kernel, contact-model, solver or
optimizer change. The Phase 9/10 settings stay: `kkt_method="reduced"`, `exp_so3_safe`, the norm-100
clipping, the cached exact evaluators, and suites and exports on the CPU. `karma_jax`'s capsule pass
applies one radius to every capsule and is not extended. Original-KaRMA scores (the suite's `official_*`
numbers) are therefore not computed under D19, and nothing in Phase 11 depends on them. Each item ends
with a section of `docs/phase11_report.md` and green `dkarma/tests` and `karma_jax/tests`. Endpoints,
rules and thresholds are written into the report, dated, before the data they apply to.

**11.1 D19 in code.**

* *Per-capsule radii throughout.* Classify every use of a capsule radius as either the contact surface
  (the assigned fingertip capsules: the task's `link_radius_m`, 11.8 mm) or the collision envelope (each
  capsule's own `cap_radius`). Known uniform-radius assumptions:
  * `simulation_replay._mjcf_rigid` uses `cap_radius[0]` for every capsule and for the fingertip's
    length;
  * `admissibility.MIN_BASE_SEPARATION_M` and `objectives.D18_CONSTRAINTS` hard-code 24.1 mm;
  * `anthropomorphic` and the start generation take one radius;
  * `visualize_design` and the URDF export must carry per-capsule radii.

  Audit the seed search, the contact pass, the exact gates (object–capsule collision, all-finger
  clearance), the exact force and wrench problems, and the admissibility checks.
* *The D19 box.*
  * Length floors: 34 mm for links 0–2 and 12 mm for the fingertip; maxima 90 mm; `base_xyz_max_m=0.10`.
    This is `Bounds(link_len_min_m=0.034, link_len_max_m=0.090, tip_len_min_m=0.012,
    tip_len_max_m=0.090)`; a per-slot floor, where already implemented, takes 34 / 34 / 34 / 12 mm.
  * Radii 16.40 / 16.40 / 16.40 / 11.8 mm per finger.
  * `Constraints(reach_max_m=0.25, base_sep_min_m=0.0333)` with the D18 weights.
* *Tests.*
  * Per-capsule gaps, and the object–capsule collision gate with mixed radii: a configuration that
    clears 11.8 mm but not 16.40 mm is rejected.
  * D19 admissibility.
  * The length floors: no decoded link 0, 1 or 2 is shorter than 34 mm.
  * The MuJoCo scene's capsules match the dkarma model, capsule by capsule.
  * The existing finite-difference gradient tests pass unchanged.
* *Regression.* With uniform 11.8 mm radii and the D18 box, a Phase 10 design's suite numbers and one
  10.2 replay reproduce bit for bit.

**11.2 Start set, frozen before any run.**

* *Nominal start N11.* The anthropomorphic start with links 0–2 at `max(L, 40 mm)` (the §2.2 pitch of a
  chain of XL330 cases, strictly inside the encoded box). The fingertips (23 / 23 / 25 mm) are unchanged,
  and the index–middle base pitch is raised from 30 to 40 mm. Two fallbacks apply in order, recorded before any
  optimization:
  1. If N11 is not admissible under D19, raise the pitch in 5 mm steps up to 60 mm.
  2. If N11 has fewer than 3 three-contact seeds (default configuration, budget 12), lengthen the three
     fingertips together in 5 mm steps up to 45 mm, reporting the attempt counts at every step.

  If neither yields a start, stop and report. Under rule 13, no claim that a grasp does not exist is made
  without the three 8.5 procedures.
* *Starts.* N11 plus Sobol perturbations by the 8.4 procedure as coded in `admissible_baseline starts`:
  the D19 box, its per-point scales, and points with fewer than 3 three-contact seeds rejected and
  listed. 24 starts in total.

**11.3 Optimization.** Full KaRMA + Adam with the 10.2 Full-D18 settings: 40 iterations in two stages,
6 three-contact seeds, optimizer seed 0, exact validation every 10 iterations. D19 is enforced
throughout. Run one process per start on the GPU, and report compute and every returned design's
admissibility.

**11.4 Evaluation of every returned design** (CPU, the 10.2 pipeline):

* the 9.4 suite with per-seed detail: exact Full and two-finger scores under the three seed
  configurations, wrench capacities, and `S_task`;
* the default-configuration export of 12 three-contact seeds, replayed with the 10.1 central pads and
  with rigid contact, each under the force-floor controller and the position servo;
* the load sweep.

Report E per design, split into rotations and translations, and the drop-criterion events, split into
releases and > 20 mm deviations. The Phase 10 selection's E = 45.0 came 38.6 from rotations and 6.4 from
translations (`results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25/figures/`).

**11.5 Selection.**

* *Criterion v2* as frozen in §16 item 12, still subject to the owner's review. Its references are
  computed by their frozen rules:
  * R3: N11's exact Full score on its own `S_task`, mean over the three configurations, computed before
    any optimized design is evaluated;
  * R5: 0.2393, the 10.1 reference (same fingertip, pads and controller).

  Every Phase 11 design is a candidate, and the frozen tie-break applies.
* *The original 9.5 criterion* (requirements 1, 3 and 4, with eligibility under D19) is applied
  alongside and never presented as equivalent.
* *Phase 10.* No Phase 10 design is admissible under D19. The Phase 10 selection is superseded for
  fabrication, and its results stand for the D18 box. The report compares the Phase 11 selection with
  it descriptively (what the envelope costs in exact Full score and E), not as a claim.
* *Acceptance.* A selection, or none, with the criterion's numbers and the original criterion's verdict
  beside it. A selected design is:
  * exported to `results/fabrication_candidates/selected_<design>/`;
  * re-scored from the exported copy;
  * given figures with `dkarma/experiments/visualize_design.py`.

  `results/fabrication_candidates/README.md` records the outcome.

Phase 11 ends with the report's one-page summary: the selected design (or none) and its numbers, what the
envelope costs against Phase 10, and the fabrication recommendation with its uncertainty.

---

## 12. Experiments

**I. Is Diff-KaRMA a useful surrogate?** Morphology dataset from the 4+4
family plus other templates; measure Spearman correlation smooth vs exact,
coverage bias, gradient finite-difference error, local-improvement success
rate (gradient steps re-scored exactly), optimization performance vs
wall-clock for Adam, CMA-ES and hybrid, and constraint violation of the
optimized designs. The existing `optimize/` runs are the CMA-ES arm.

**II. Does actuation awareness predict better?** Synthetic designs with
controlled distal length, axis orientation and torque; hold kinematics fixed
and lower torque; compare with independent quasi-static or dynamic simulation
(MuJoCo) task success; show states original KaRMA accepts and Act-KaRMA rejects.

**III. Does simultaneous multi-contact add information?** Optimize 3×4 hands
with (1) thumb–index KaRMA, (2) mean pairwise, (3) simultaneous three-contact,
(4) Full KaRMA, same bounds, torques, object; evaluate all on one three-contact
task suite including rotation about the original pinch axis and wrench margins.

**IV. End-to-end co-design and hardware.** Fabricate A′ (original KaRMA +
CMA-ES on the admissible design space, D18; the Phase 7 arm A is kept as the
failure case and is not buildable as three fingers) and B (Full KaRMA), same
topology, torque ceilings and envelope; the designs to fabricate are those the
Phase 10 re-selection (§11.3 item 10.4) names, none having passed in Phase 9;
optionally C (pairwise or actuation-only) and D (conventional baseline).
Protocol: export skeleton and torque spec → mechanical design preserving
joint kinematics → compliant fingertips → measure as-built axes, ranges,
torques → as-built model → same pipeline on designed and as-built → identical
physical trials (tracked sphere, then one non-spherical object). Outcomes:
task success, pose error, maximum displacement, slip, torque saturation,
mass/friction sweeps, predicted vs actual reachable poses. Decompose error
into surrogate-vs-exact, exact-vs-simulation, simulation-vs-hardware.

---

## 13. Reproducibility artifacts

Per optimization run:

```
experiment/<name>/
  config.yaml  initial_morphology.urdf  optimized_morphology.urdf  actuator_spec.yaml
  parameter_history.npz  objective_history.csv  validation_history.csv
  final_reachable_poses.npz  contact_force_diagnostics.npz  constraint_diagnostics.json
  random_seeds.json  results_summary.json
```

Fabricated hands add the as-built URDF, measured actuator specs, controller
config, trial-level outcomes and synchronized logs. Headline plots are
generated from these artifacts only.

---

## 14. Claims this project does not make

- The surrogate explores a finite trajectory library; it is not differentiable
  exact global reachability and may miss disconnected reachable regions.
- Quasi-static wrench feasibility is not dynamic executability.
- Smooth scores are not KaRMA scores; only exact evaluators produce reported numbers.
- Three-finger SO(3) coverage is not comparable to two-finger pinch-axis coverage.
- Capsule links and a spherical object remain the geometric abstraction.
- Differentiable optimization is not claimed superior to evolutionary search in
  general: Phase 8.4 found it behind CMA-ES on the two-finger objective and
  ahead on the three-contact actuated objective at matched optimizer time.
- Exact quasi-static feasibility is not simulated manipulation success. The
  exact-vs-simulation error term is partly controller-dependent (Phase 9.3:
  force control raises strict three-contact success by 13–15 points; object
  poses follow the commands in 0.69–0.81 of exactly valid primitives; the
  residual is fingertip contact maintenance under rigid capsules, the object of
  §11.3 item 10.1).
- Full KaRMA is not claimed to outperform a collision-valid original-KaRMA
  optimum: on eight starts the difference was not significant (Phase 9.2); the
  claim waits for §11.3 item 10.2.
- Optimized morphologies are not claimed ready for hardware: no candidate passed
  the written selection criterion (Phase 9.5).

---

## 15. Completion criteria

| Component | Done when |
| --- | --- |
| Differentiability | stable, finite-difference-validated morphology gradients through the trajectory evaluator |
| Reachability | exact validation confirms continuous contact-constrained rolling for credited poses |
| Actuation | equilibrium, friction and torque feasibility independently verified by the exact solver |
| Multi-contact | true simultaneous three-finger trajectories with full orientation coverage |
| Co-design | reproducible multistart optimization under matched budgets, with exact validation of every reported design |
| Calibration (Phase 8) | smooth force credit tracks exact feasibility (AUC ≥ 0.95); continuation preserves exact gains; seed policy chosen with data; optimizer comparison with ≥ 8 starts per method; three-contact claims verified by independent procedures; exact-vs-simulation error measured |
| Admissible baselines and controller-independent validation (Phase 9) | KKT derivative without non-finite gradients and with finite-difference agreement; the collision-valid original-KaRMA baseline compared with Full KaRMA on the shared admissible space with ≥ 8 starts; force-controlled replay with separated criteria; candidate ranking stable across seed sets; selection by the written criterion |
| Compliant contact, powered comparison, deployment-relevant grasps (Phase 10) | compliant-fingertip replay with separated outcomes and a verdict by a pre-registered rule; the A′ vs Full-D18 comparison on ≥ 16 new starts with a pre-registered common manipulation endpoint, reported whichever way it goes; `S_task` proportions and subset measures beside the all-seeds results; selection, or none, by a criterion frozen before re-selection |
| Hardware | a fabricated hand substantially preserves the optimized performance envelope, deviations quantified |

---

## 16. Decisions taken by the project owner (2026-10-11)

1. Actuator: DYNAMIXEL XL330-M077-T (D13, §2.1). Torque ceilings and link
   masses are derived from its datasheet and flagged as assumptions until a
   joint-side torque curve is measured.
2. Task volume: estimated relative to the actuator (D8, §2.2): ±100 mm cube at
   10 mm, sized from the reachable set of LEAP, which uses the same servo case.
3. No symmetry anywhere (D14).
4. The working tree stays uncommitted (D15).
5. SO(3) targets: 576 for the smooth kernel, 4608 bins for exact counting
   (D16, §4.5).
6. Gravity: palm facing up, gravity along −z of the palm frame, for the
   primary task (D17). Other palm orientations are a load-case ablation.

No decisions remain open before Phase 1.
7. Review of the Phase 7 pilots (2026-10-12): no new metric formulation; next
   work is calibration and validation as §11.1 Phase 8.
8. MuJoCo (2026-10-13): installed into `~/mamba/envs/karma-jax` (3.15) on the
   owner's instruction; the simulation leg of 8.6 ran with it.
9. Review of Phase 8 (2026-10-13, `docs/phase8_review_2026-10-13.md`): the
   framework is sufficient for the main experimental phase; no further metric
   extensions; Phase 9 is §11.2 in its priority order; the paper framing of
   §11.2 is adopted; arm A is retained as the failure-mode case and a
   collision-valid original-KaRMA baseline (arm A′) is added.
10. Admissibility (D18, §8.1), confirmed 2026-10-13: no two modelled fingers may
    occupy the same space; finger bases at least `2 r_l + eps_col` = 24.1 mm
    apart. The design box's reach bound stays at 250 mm per finger (the global
    hand envelope; long fingers within it are admissible and are not treated as
    an optimization error).
11. Review of Phase 9 (2026-10-14, `docs/phase9_review_2026-10-14.md`): the
    framework stays fixed (no metric, kernel, solver or task change); Phase 10
    is §11.3 (compliant-fingertip simulation; the A′ vs Full-D18 comparison on at
    least 16 new starts with a fixed evaluation protocol; the task-feasible grasp
    subset `S_task = {s : min_d α_s(d) ≥ 1}` reported beside the all-seeds
    results; a revised selection criterion justified by the deployment
    requirements and frozen before re-selection, never presented as equivalent to
    the original one). Arm A stays the failure-mode case and is not the
    principal evidence of superiority. Phase 9.5's decision (nothing selected) and
    its unchanged criterion stand.
12. **Selection criterion v2** (frozen by the Phase 10 session on 2026-10-14 15:50 EDT, before any
    candidate was evaluated against it and before any Phase 10 data was produced; **subject to the
    owner's review**; full text in `docs/phase10_report.md` "P10.4"). Written from the deployment
    requirements (frozen task, three-fingertip rolling of the sphere, compliant silicone fingertips,
    the 9.3 force-floor controller on XL330 servos, a planner choosing among grasps). R1 eligibility
    as 9.5; R2 `|S_task| ≥ 6` of 12 seeds under every seed configuration, `S_task = {s : min_d α_s(d) ≥ 1}`
    at the primary ceiling; R3 exact Full score on `S_task` (mean over the three configurations) at
    least the anthropomorphic start's on its own `S_task`; R4 the 9.5 load sweep (70 g held, ≥ 50 %
    retention); R5 three-contact strict false-positive rate under the force-floor controller with the
    central compliant fingertips at most the median of the 9.2 12-seed libraries under the same model
    (≥ 10 exactly valid primitives required). Highest `S_task` Full score among passing candidates
    wins. The original 9.5 criterion is applied alongside, unchanged, and the two are never presented
    as equivalent.
13. Git (2026-10-23): the working tree is committed on branch `dkarma-phase0-10` for the owner's public fork
    `vindou/karma-hand-metric` (upstream `mfpeticco/karma-hand-metric`), superseding D15. The owner's
    working notes (the phase prompts, the phase reviews and the source conversation) stay local and out
    of the repository, as do `workspace/` and the bulky optimizer artefacts (`optimize/runs/**/*.pkl`,
    `optimize/runs/**/*.mp4`, the marimo session cache).
14. **Servo envelope (D19, 2026-10-23).** The hand is built from XL330-M077-T servos in a direct-drive
    chain, with the servos of joints 1–3 in links 0–2. Each of links 0–2 holds one case,
    20.0 × 34.0 × 26.0 mm (datasheet; the owner's measurement of 34 × 20 × 23 mm fits inside), and their
    capsules have radius `sqrt(20² + 26²) / 2 = 16.40 mm`. Each case lies along its link's axis, and
    links 0–2 are at least 34 mm long. (A relaxation of link 0's length, set on 2026-10-23, was withdrawn
    by the owner on 2026-10-24, before any Phase 11 start was evaluated.) The fingertip link holds no servo
    and keeps 11.8 mm, its length bounds and 10 g (§2.1's mass model is unchanged). D18's base separation becomes 33.30 mm. The frozen task is unchanged. Phase 11
    (§11.4) re-runs the optimization inside D19 to produce a new candidate; none of the 70 Phase 7–10
    designs is admissible under it.
