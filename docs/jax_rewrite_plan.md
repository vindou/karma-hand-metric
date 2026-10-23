# KaRMA on JAX/CUDA: rewrite plan

Status: proposal, 2026-10-10. Audience: whoever implements the port (assumes
familiarity with `karma/` and with JAX `jit`/`vmap`/`lax`).

The goal is throughput: evaluate many more hands per hour, both for the 16-hand
sweep and for optimization loops such as `optimize/` that score hundreds of
designs per run. Scores must stay equivalent to the reference implementation
(same ranking, voxel counts within the tolerance the README already grants
across machines), not bit-identical to it.

---

## 1. Where the time goes today

Measured single-process on LEAP (`/tmp/.../profile_kernels.py`, 12-core EPYC,
`OMP_NUM_THREADS=1`):

| kernel | per call | notes |
| --- | --- | --- |
| `solve_rolling_step` (one QP sub-step) | 1.0 ms | OSQP `setup` 0.41 ms, of which 0.30 ms is a Python `import_module` call inside `osqp.interface.algebra_available`; the actual OSQP `solve` is 0.09 ms |
| `_project_onto_contacts` (Gauss-Newton + backtracking) | 0.40 ms | up to 10 iterations, 3 backtracks each, one Pinocchio FK per trial |
| `update_kinematics` (Pinocchio FK + Jacobians) | 0.08 ms | 65% of it is `_precompute_endpoints`, a Python loop over capsules |
| `contact_kinematics` | 0.04 ms | pure Python/numpy on 3-vectors |
| `_attempt_primitive_step` (8 sub-steps, full validation) | 14.5 ms | |
| `_solve_for_centre` (scipy TRF seed IK) | 7.2 ms | |
| `_explore_rotations_at_voxel` (Phase 2, one voxel) | 154 ms | includes `load_robot` (URDF re-parse) per voxel, per task |

The numerical content of a sub-step is tiny: an 11-variable QP with 13
constraints, a 2x8 Gauss-Newton system, FK over 16 to 24 revolute joints. The
cost is Python and library call overhead, roughly 10x the arithmetic. That is the
ideal shape for batching: thousands of identical tiny problems per kernel launch.

How many sub-steps a run needs (LEAP, from `results/16_hand_batch/summary.yaml`):

| phase | work | share of the 292 s reference runtime |
| --- | --- | --- |
| seed IK | ~750 joint samples x up to 9 centres = ~6,800 TRF solves | small |
| seed ranking: trial BFS on every feasible seed at `max_states` | 254 seeds x median 355 voxels x ~5 attempts x 8 sub-steps = ~3.6M sub-steps | dominant (~90%) |
| Phase 1 on the best seed | 792 voxels x ~5 x 8 = ~32k sub-steps | small |
| Phase 2 | 792 voxels x 4 directions x up to 50 steps = up to 160k rotation steps | ~5% |
| ensemble seeds #2, #3 | Phase 1 + Phase 2 again | small |

So for capable hands the budget is the rolling sub-step kernel, run several
million times, and seed ranking is where it is spent. Low-DOF hands (Inspire,
SVH) are already fast; the rewrite does not need to help them.

---

## 2. Design in one paragraph

Keep `karma/` as the reference and oracle. Add a new package (`karma_jax/`)
that represents a hand as a flat, padded array model (parents, joint axes and
placements, capsule table, mimic map), runs forward kinematics, contact
geometry, the rolling QP, manifold projection, collision and grasp checks as
`jit`-compiled functions over a leading batch axis, and drives them with a
level-synchronous breadth-first search that expands the frontiers of all seeds
(and, in optimization loops, all designs of a population) in one batch. All
data-dependent control flow in the kernels (sub-step early exit, backtracking,
Gauss-Newton convergence, rotation-stream termination, contact-pair hysteresis)
becomes masked fixed-shape computation; the only host round-trips are once per
BFS level, for voxel deduplication. The search order is reproduced exactly, so
the only sources of divergence from the reference are the floating-point
solvers themselves.

---

## 3. Hard constraints discovered

1. **Python version.** Every JAX release with CUDA wheels requires Python
   >= 3.11 (`jax 0.7.x`, `0.8.x`); the latest requires >= 3.12. The repo's
   `<3.11` cap is a reproducibility pin, not a wheel limitation: `pin==2.7.0`
   and `hpp-fcl==2.4.4` both ship cp311 and cp312 Linux wheels (checked on
   PyPI). So the JAX env is Python 3.12 with the same pinned Pinocchio, and
   the runtime keeps Pinocchio for model building, canonicalization and
   snapping, exactly as today. Only the per-step compute leaves Pinocchio.
   Verify once that `pin 2.7.0` on 3.12 gives bit-identical FK to the 3.10
   env (it should; same C++ binary built against the same Eigen).
2. **All joints are revolute.** Across the 16 URDFs: 262 revolute, 139 fixed,
   5 `contact` (ignored), zero prismatic/continuous/floating. Pinocchio maps
   them to `RX/RY/RZ/RevoluteUnaligned`; one generic "revolute about unit axis
   `a` in the joint frame" covers all of them. No prismatic support is needed
   (the template mentions it, but nothing uses it).
3. **Shapes are small and bounded.** Pinocchio `nq` <= 24, active joints
   `m` <= 9, frames <= 85, capsules roughly 10 to 45, contact links <= 4 per
   finger, mimic joints <= 5. Pad everything to one bucket (e.g. `nq=24`,
   `m=10`, `n_caps=48`, `n_contact_links=4`) so one compiled program serves
   every hand and every design in a family. Dead lanes are masked; the waste is
   negligible against launch overhead.
4. **fp32 is the default (decision).** The GPU is an RTX 4090, where fp64
   runs at 1/64 of fp32 rate, and fp32 halves memory traffic for the batched
   solves. The geometric tolerances (0.35 mm gap, 0.5 mm clearance, 0.5 mm slip
   gate on 0.1 m geometry) leave fp32 a ~10^3 margin. Three places need
   explicit care, detailed in Section 5.1: the QP stopping tolerance (the
   reference's 1e-7 is below fp32 epsilon), the sphere position accumulated
   over thousands of sub-steps, and the sphere rotation accumulated as a
   product of exponentials. fp64 remains available as a flag and is the
   validation mode (Section 7).
5. **Seed IK uses scipy's bounded TRF.** There is no JAX equivalent that
   reproduces its iterates. This is the largest unavoidable numerical
   divergence (Section 7).

---

## 4. Module-by-module port

Unchanged (CPU, numpy): `config.py`, `lref.py`, `results.py`, `storage.py`,
`viser_*`, `orientation.healpix_npix_lines`, all scripts under `experiments/`,
`robust_tests/`, `baselines/`, `tools/` (they only call `load_config`,
`load_robot`, `compute_metric`, `compute_lref_breakdown`, and read
`MetricResult`).

### 4.1 `robot.py` -> `karma_jax/model.py` (flat model, numpy, built once per hand)

Replace `RobotContext` with an immutable pytree of padded arrays:

```
FlatModel
  parent[J]            int   parent joint index (-1 = world), topological order
  M_joint[J,4,4]       SE3   parent-joint frame -> joint frame (URDF <origin>), after
                              canonicalization + grid snap of root joints
  axis[J,3]            unit  revolute axis in the joint frame
  q_lo[J], q_hi[J]
  frame_parent[F], M_frame[F,4,4]   link/fixed frames hanging off joints
  anc[F,J]             bool  joint J is an ancestor of frame F (for Jacobians)
  cap_frame[C], cap_distal[C] (-1 = terminal), cap_tip[C,3], cap_valid[C]
  active_q[m]          int   q indices of thumb+index active joints (sorted, as today)
  S[J,m], s0[J]        mimic fold: q_full = S @ q_active + q_fixed, s0 carries offsets
  thumb_caps[4], index_caps[4], all_contact_caps[8]   capsule ids (padded with -1)
  seed_thumb_cap, seed_index_cap
```

Build it by extraction from the existing `RobotContext`: call `load_robot`
unchanged (URDF parse, canonicalization, grid snap, `_build_link_capsules`,
mimic resolution all stay in Pinocchio/numpy), then read `model.parents`,
`model.jointPlacements`, each joint's axis (`RX/RY/RZ` -> unit axis,
`RevoluteUnaligned` -> its `axis`), `model.frames[*].parent/placement`,
position limits, and the `LinkCapsule` table, into padded arrays. This is a
one-time ~1 ms step per hand, guarantees the same joint order and `q` layout as
the reference pickles and viser, and keeps `KARMA_SNAP_DECIMALS` and every
loader heuristic untouched. A Pinocchio-free loader is not needed (Section 3,
item 1) and is not planned.

Mimic joints become a linear map. Today `update_kinematics` writes
`q[target] = mult * q[source] + offset` and `fold_mimic_jacobian` adds
`mult * J[:, target]` into the source column. With `q_full = S q_active + s0`
both are `J_active = J_full @ S`, which is exact and removes a special case
from every kernel. `baselines/robot_loader.py` keeps its own loader; leave it.

Oracle test (same env, Pinocchio in-process): for all 16 hands and 200 random
`q`, FK frame positions/rotations and `LOCAL_WORLD_ALIGNED` frame Jacobians
agree with Pinocchio to 1e-12 in fp64 and to 1e-6 relative in fp32, and
capsule endpoints agree with `capsule_endpoints_world`.

### 4.2 `kinematics.py` (batched FK and Jacobians)

- FK: `lax.scan` over joints in topological order, each step a batched 4x4
  compose with the joint rotation `exp(q_j a_j)`. Depth is `J <= 24`, trivially
  cheap; `vmap` over the batch. Frame placements are one more batched compose.
- Jacobians analytically, not by autodiff. For revolute joint `j` with world
  axis `w_j` and world origin `o_j`, the `LOCAL_WORLD_ALIGNED` Jacobian of
  frame `f` has column `j` equal to `[w_j x (p_f - o_j); w_j]` if `anc[f,j]`,
  else zero. That is one batched outer product masked by `anc`, gives exactly
  what Pinocchio returns, and is cheaper and more transparent than
  `jax.jacfwd` through the scan. Keep a `jacfwd` version in the tests only.
- Output per batch element: `p_frame[F,3]`, `R_frame[F,3,3]`, `Jv[F,3,J]`,
  `Jw[F,3,J]`. Only the capsule frames need Jacobians; gather them.

### 4.3 `geometry.py` (contacts and collisions in one pass)

The reference computes the sphere-vs-capsule gap in three places with three
code paths (`contact_kinematics`, `contact_gap_only` for pair selection,
`check_collisions` with an AABB prefilter). On the GPU, compute every capsule at
once: endpoints `[B,C,2,3]`, closest-point parameter `t[B,C]`, gap `[B,C]`,
normal `[B,C,3]`. From that single tensor:

- **Contact pair selection**: `argmin |gap|` over the thumb capsule ids and
  over the index capsule ids; apply the hysteresis rule with `jnp.where`
  (`cur_gap - best_gap < 0.5 eps_g` keeps the current link). Exact.
- **Object collision**: `min` over all capsules with the two active ones masked
  out, compared with `eps_col`. The AABB prefilter is unnecessary.
- **Contact kinematics** for the two selected capsules: gather their endpoints
  and Jacobians, then the same closest-point Jacobian formula as
  `contact_kinematics` (the three `t` regimes become `jnp.where` on
  `t <= 1e-12`, `t >= 1 - 1e-12`); the degenerate-normal branch becomes a
  `where` too.
- **Self-collision**: Ericson segment-segment distance for the thumb x index
  capsule pairs (<= 16), branches rewritten as `where`; `min` masked by
  validity, compared with `eps_col`.
- **Grasp**: `dot(-n_t, n_i) >= cos(2 atan(mu))`.

### 4.4 `qp.py` (batched rolling and rotation QP)

Both QPs are `min 1/2 x'Px + c'x  s.t.  l <= Ax <= u` with `A = [A_gap; I]`,
`n = m+3` (rolling) or `m` (rotation), `nc = n + 2`. The build (surface
Jacobian `Jv_origin - skew(r) Jw`, `-r_s skew(n)` block, `lambda_theta = 1e5`
prior on `delta_theta`, non-dimensionalization by `r_s`, the `0.2 rad` theta
bound, joint-step and joint-limit boxes) is straight batched linear algebra.

Solver choice. Two viable options, both to be implemented against an OSQP
oracle before deciding:

1. **Primal-dual interior point (recommended first).** Convert to `Gx <= h`
   with `G = [A; -A]` (`2nc <= 28` rows), Mehrotra predictor-corrector, a fixed
   iteration count (15 to 25) with a convergence mask, Newton step via the
   reduced normal equations `(P + G' diag(z/s) G) dx = r` (an `n x n <= 12`
   dense Cholesky, batched). IPM reaches 1e-8 in ~10 iterations on problems
   this small regardless of conditioning, so a fixed count is safe, and every
   lane does identical work, which is what `vmap` wants. Report `success` from
   the final KKT residual; the reference treats OSQP status `solved` and
   `solved_inaccurate` as success and anything else (chiefly primal
   infeasibility, when the gap band and the joint box conflict) as `qp_fail`.
2. **ADMM replicating OSQP.** Same `sigma`, `rho`, `alpha = 1.6`, termination
   criteria and adaptive-rho schedule as OSQP would give solutions closest to
   the reference, with one KKT factorization per problem and ~50 to 200 cheap
   iterations. Harder to make fixed-shape (variable iteration count, rho
   updates that refactor). Keep as fallback if IPM parity is unsatisfying.

Note for the parity discussion: OSQP's default `adaptive_rho_interval = 0`
derives the interval from measured setup time when profiling is on, so the
reference's own bit-reproducibility already rests on that being disabled or
stable in the pinned build. Do not try to match OSQP bit-for-bit.

Measured (Phase B, 400 recorded LEAP rolling QPs, fp64): the IPM converges in
6 to 9 iterations and lands within a median 1.6e-8 (95th percentile 4.6e-7) of
a tight OSQP optimum. The reference's own OSQP answers are off by a median
3.6e-6 and a 95th percentile of 1.6e-2, because OSQP's relative termination
criterion is scaled by the `1e5` rotation penalty, which makes `eps_rel = 1e-7`
an effective tolerance of ~1e-3 on the dual residual. So the JAX path solves
the same QPs more accurately than the reference, and element-wise agreement of
`dq` with the reference is bounded by the reference's noise, not ours. The
oracle test therefore compares optimality on the identical assembled problem;
this noise is also a floor on how closely any reimplementation can track the
reference's BFS trajectories.

Post-solve bookkeeping (residual split into tangential slip and spin, slip gate
in mm, achieved-rotation back-solve with the `1e-4` damping, efficiency clip)
ports directly.

### 4.5 `projection.py` (batched Gauss-Newton with backtracking)

`_project_onto_contacts` has three data-dependent constructs. All three have
exact masked equivalents:

- The `break` on `max|g| <= eps_g`: run the fixed `max_iters = 10` with a
  `converged` mask that freezes `q` and `total_dtheta`.
- Backtracking "first alpha in {1, 1/2, 1/4} that reduces `|g|`": evaluate all
  three candidates in one FK over a `[B,3]` batch and select the first that
  improves, defaulting to `alpha = 1/8` x 2... precisely: the reference tries
  `alpha = 1, 0.5, 0.25` and, if none improves, applies `0.125`
  (the loop halves after the last failed try). Reproduce that exactly with
  `jnp.where` chains.
- The `2 x m` damped solve and `_compute_rolling_dtheta` are batched
  `jnp.linalg.solve`.

### 4.6 `steps.py` (one primitive step, one rotation step)

`attempt_primitive_step(state[B], delta_p[B,3]) -> (ok[B], new_state[B],
fail_code[B])`: `lax.fori_loop` over the 8 sub-steps with an `alive` mask
(reference: return on first failure; here lanes go dead and skip their work
only in the sense of being masked). Each sub-step is: contact kinematics for
the current pair, rolling QP, apply `dq`, advance `p`, `R <- exp(dtheta) R`,
pair re-selection, projection. After the loop: gap check, object collision,
self-collision, grasp, joint-limit check, in the reference's order, producing a
small integer fail code so the `fail_reasons` tallies survive.

`rotation_step` is the Phase 2 body (`solve_rotation_step`, efficiency and slip
gates, apply, pair re-select, projection, validations, pinch-axis-in-body
output).

`exp_so3` via Rodrigues with the small-angle guard; `_reorthogonalize` via
batched 3x3 SVD (or a polar-decomposition Newton iteration, cheaper and
adequate).

### 4.7 `search.py` (level-synchronous multi-seed BFS)

This is where batching changes the algorithm's shape without changing its
result. The reference is FIFO BFS per seed with "first discoverer wins" for a
voxel's representative configuration, which matters because later expansion
from that voxel starts from that configuration. FIFO order equals
level-by-level order, and within a level the order is (queue position,
primitive index). So:

```
for each BFS level (all seeds that still have a frontier):
    attempts = [(seed, node, prim) for node in frontier[seed] in queue order
                                   for prim in 0..5 if prim != node.arrival_prim ^ 1]
    ok, new_state, code = attempt_primitive_step(batched attempts)      # GPU
    voxel = round(R_seed' (p - origin) / h)                               # GPU
    -> host: keys (seed, voxel); keep the first occurrence in attempt order
       that is not already discovered; truncate per seed at max_states
       exactly as the sequential loop would (it stops inserting when the
       count hits the cap, mid-level); append to per-seed storage and to
       next frontier in that order; tally fail codes and revisits
```

With this tie-break the discovered set, representative configs, parents,
`arrival_prim` and `unique_boundary`/`revisits` statistics are identical to the
sequential BFS given identical kernels. One host sync per level; a 355-voxel
BFS has a few dozen levels. Per-level batch size across 254 LEAP seeds is in the
thousands to low tens of thousands of attempts, enough to fill the GPU.

Storage: per-seed node arrays (`q`, `p`, `R`, pair, parent, arrival_prim) in
growing host-side numpy pools (hundreds of voxels x 2 KB; do not preallocate
`n_seeds x max_states`). Upload the frontier's states each level (small).

The best seed's Phase 1 and the ensemble seeds #2 and #3 run as one three-seed
BFS instead of three sequential ones. `_run_trial_bfs(return_configs=True)` and
Phase 1's `VoxelConfig` graph are the same output with different bookkeeping;
unify them.

For the optimizer (`optimize/fast_eval.py`), the batch axis also spans
designs: every design of a `HandFamily` population has the same padded shape,
so one BFS call ranks every seed of every design in the generation together.
That is the single largest throughput multiplier available and costs nothing
beyond carrying a `design` index alongside `seed`.

### 4.8 `rotation.py` (Phase 2)

Streams = (voxel, direction), 4 per voxel, batched across all voxels of all
seeds being scored. `lax.while_loop` (or `fori_loop` to 50) with an `alive`
mask; each iteration one `rotation_step`. Record the achieved pinch axis in the
body frame per (stream, step) into a `[S,50,3]` buffer with a validity mask;
bin on the host afterwards with `healpy.ang2pix` vectorized (the bin set is
order-independent, so no exactness concern), or port `ang2pix` RING to JAX later
if the transfer shows up in profiles (it will not: it is 50x3 floats per
stream). Representative configs per `(voxel, bin)` are "first to reach" in the
reference; reproduce by taking, per bin, the minimum `(direction, step)` key.
This removes the per-voxel `load_robot` and the process pool entirely.

Compaction: streams die at different times (median well under 50 steps); every
10 iterations drop dead lanes and re-pack if the alive fraction falls under
~50%. Optional; measure first.

### 4.9 `ik.py` (batched seed IK)

Replace `scipy.optimize.least_squares(method="trf")` with a batched bounded
Levenberg-Marquardt in JAX: same residual vector (gap x `0.35/eps_g`, hinge
collision x `0.25`, antipodal x `200`, `0.1` regularization), the same
hand-written Jacobian with its deliberately zeroed collision and antipodal rows,
the same non-dimensional `dp/r_s` variable and `+-3(1 + r_l/r_s)` bounds, box
handling by projection after each accepted step, fixed iteration budget
(e.g. 60) with per-lane damping update and convergence mask. The per-iteration
work is the same geometry pass as everything else plus an `(m+3)x(m+3)` solve.

This will not reproduce TRF's iterates or, in multi-modal cases, its local
minimum. Mitigation in Section 7. Candidate generation (`_candidate_sphere_
centres`, closest points between the two seed capsules, reach filter) is a
batched version of code already written in numpy.

**Outcome (Phase D, measured).** A projected Levenberg-Marquardt on the
reference's residual stalled where TRF succeeds (the antipodal term, weight
200 with a zero Jacobian row, dominates the cost; LM's damping blows up after
~25 rejected steps). Giving LM the full analytic Jacobian found 1.7 to 2.5x
*more* feasible pinches than the reference but still missed specific ones,
including the best Inspire seed, and recall of the reference's set was only
0.65 to 0.9. The adopted solution is `karma_jax/trf.py`: a batched,
`vmap`-able port of scipy's `trf_bounds` (exact trust-region subproblem on an
SVD of the augmented Jacobian, Moré's secular iteration, reflected and Cauchy
step selection, scipy's radius update and termination). Run in fp64 on the
same residual and the same deliberately incomplete Jacobian, it reproduces
scipy's solutions to ~1e-12 and its feasibility verdicts on 97 to 99% of
candidates (the rest are borderline roundoff), at a median of 15 to 17
function evaluations. The seed IK therefore always runs in fp64 regardless of
the search precision; it is a few seconds per hand. The LM remains available
(`IKSettings(method="lm")`) for experiments that want a superset of pinches.

Seed polishing (`_polish_one_seed`: 5 nullspace steps with `pinv(J_obj)`) is
batched `jnp.linalg.pinv` on `3 x m`; re-validation reuses the geometry pass.
Dedup (`0.01 rad` greedy) stays on the host; it is O(n^2) on a few hundred
seeds.

### 4.10 `metric.py` (orchestrator)

Same signature and the same `MetricResult`, `StateNode`, `voxel_ori_bins`,
`seed_sensitivity` (including `ensemble_t/r`, `karma_s`), `phase1/2_fail_
reasons`. `precomputed_seed` keeps working and becomes the main parity tool
(Section 7). `rotation_workers` becomes a no-op knob.

Backend selection (decision): JAX on CUDA in fp32 is the default for
`run_metric.py`, `run_all_hands.py`, `run_viser_app.py` and
`optimize/fast_eval.py` as soon as Phase D's parity gate passes. A
`--backend reference` flag (or `KARMA_BACKEND=reference`) runs the numpy /
Pinocchio / OSQP code path unchanged, for audits and for regenerating the
paper's numbers; `--precision fp64` runs the JAX path in double; and
`JAX_PLATFORMS=cpu` runs the same kernels on the CPU backend when no GPU is
present (the entry points print which device and precision they used, and the
result YAML records both under `summary`). The README's reproducibility section
is rewritten to describe fp32 CUDA as the shipped configuration and the
reference path as the archival one.

### 4.11 `ablation.py`

Monkeypatching `collisions.check_collisions` cannot reach a jitted kernel.
Make the ablation an explicit `Constraints(joint_limits, collisions, antipodal)`
static argument threaded into `attempt_primitive_step` / `rotation_step`
(static -> three extra compiled variants, only when used). Keep the
`ablation_context` API and the `KARMA_ABLATION` env var reading so
`experiments/run_constraint_ablation.py` is untouched.

---

## 5. Compilation, shapes and memory

- One padded bucket for all shipped hands (`nq 24, m 10, C 48, F 96`) and one
  per `HandFamily` topology. Static per-compile: bucket sizes and the
  `Constraints` flags. Dynamic: everything else including batch size. Pad batch
  sizes to powers of two (or a small ladder) to bound the number of compiles,
  and enable the persistent compilation cache
  (`jax.config.update("jax_compilation_cache_dir", ...)`) so a cold start pays
  compile once per machine.
- Expected compile time per variant: tens of seconds for the step kernels (the
  8-sub-step `fori_loop` with a 10-iteration `fori_loop` inside and a 20-iteration
  IPM inside that). Keep loops as `lax` loops, not Python-unrolled, or compile
  time explodes.
- Memory at `B = 65k` states in fp64: states ~2 KB each (130 MB), geometry pass
  `[B,48,...]` ~50 MB, IPM working set with `28 x 12` constraint matrices and
  `12 x 12` normal matrices ~ 500 MB. Comfortable on 24 GB; `B` up to ~250k is
  feasible if ever useful.
- Determinism on GPU: set `XLA_FLAGS=--xla_gpu_deterministic_ops=true` in the
  entry points. Reductions in these kernels are over fixed small axes and
  should be deterministic anyway; the flag guards it. Also consider
  `--xla_gpu_enable_command_buffer` (CUDA graphs) once kernels stabilize; it
  attacks launch overhead, which is the expected bottleneck.
- CPU fallback: the same code runs on the JAX CPU backend. It still removes
  the Python overhead and should beat the reference on a laptop; benchmark it
  so the README can state a number.

### 5.1 Making fp32 safe

Default dtype is `float32` throughout (`jax_enable_x64` off), with these
targeted measures:

- **QP tolerance.** The reference stops OSQP at `eps_abs = eps_rel = 1e-7` on
  the non-dimensional (O(1)) problem; fp32 epsilon is 6e-8, so that target is
  unreachable. Set the IPM convergence test to ~1e-5 on the non-dimensional
  residuals. In physical units that is 1e-7 m of contact-velocity residual,
  four orders below the 0.5 mm slip gate and 0.35 mm gap band, so the gates
  behave the same; the Phase B oracle compares QP objectives at 1e-5 rather
  than 1e-6 in fp32.
- **Sphere position without drift.** The reference accumulates
  `p_cur += sub_dp` across sub-steps and voxels and rounds the accumulated
  `p` to a voxel; in fp32 that accumulates ~1e-5 m over a long BFS and can
  flip boundary voxels. Instead carry the integer voxel index and sub-step
  count as state and reconstruct `p = origin + R_seed @ (h * voxel) + k * sub_dp`
  from them in fp64 on the host (origin and `R_seed` are host-side fp64), then
  cast to fp32 for the kernels. Voxel assignment becomes exact by
  construction; the kernels never see accumulated error in `p`.
- **Sphere rotation.** `R_sphere` is a product of hundreds of `exp_so3` steps.
  Re-orthonormalize after every primitive step and every rotation step (the
  reference does it once per stored voxel), which is a batched 3x3 polar
  decomposition, essentially free. The orientation bin depends on `R` only
  through one unit vector, so residual fp32 error of ~1e-6 rad cannot move a
  bin of ~10 degrees except on a bin boundary.
- **Gauss-Newton and nullspace solves.** `JtJ + 1e-2 r_s^2 I` and the
  `pinv(J_obj)` in polishing are well conditioned by their damping and fine
  in fp32. `jnp.linalg` dispatches to cuSOLVER batched routines in fp32.
- **Host-side scalars stay fp64.** `L_ref` scaling, canonicalization, snap,
  voxel bookkeeping, scoring and the YAML output are numpy fp64, unchanged.
- **Invariance is unaffected.** Translation, rotation and scale invariance come
  from canonicalization and non-dimensionalization, both done in fp64 before
  the kernels see anything, and the kernels are deterministic, so the
  `robust_tests` invariance suite is expected to pass exactly in fp32 too.

**Measured (Phase D).** Three findings shaped the fp32 recipe:

1. **TF32 was on.** XLA runs fp32 cuBLAS GEMMs on NVIDIA GPUs in TF32 (10-bit
   mantissa) by default. Under `vmap`, every per-lane `a @ b` with a contraction
   wider than a few elements becomes a batched GEMM, so the Jacobian folds, the
   QP assembly and, worst of all, the mimic coupling `A_mimic @ q` (a 16-wide
   contraction applied to every configuration) were rounded to ~1e-3. Forward
   kinematics itself was unaffected (3-wide products are lowered to fused
   elementwise code). Symptoms: the Shadow BFS from a fixed seed swung between
   21 and 38 voxels with QP settings, and spurious joint-limit failures appeared.
   Forcing `jax_default_matmul_precision="highest"` fixed the accuracy but cost
   2.2x in throughput.
2. **Exact products without the flag.** `karma_jax/linalg.py` provides
   `mm/mv/vm/mtm` as broadcast-multiply-reduce (exact fp32, fused, no library
   call) and the kernels use them for every product with a contraction wider
   than 3; the mimic coupling and Jacobian fold are gather/scatter
   (`kinematics.apply_mimic`, `fold_active`). With TF32 left at XLA's default,
   the fp32 Shadow BFS now reproduces the fp64 voxel set and effort statistics
   exactly, and throughput is back to 500k sub-steps/s at 8k lanes (585k at
   32k). `KARMA_JAX_MATMUL_PRECISION=highest` remains as a belt-and-braces
   switch.
3. **QP precision.** With exact products, an fp32 QP at tolerance 1e-6
   reproduces fp64 search results on Shadow and Inspire (1e-5 does not), so
   pure fp32 is the default (`qp.fp32_settings()`). Mixed precision
   (`qp.mixed_settings()`, the QP solved in fp64 inside the fp32 pipeline) is
   available and costs 1.7x.

Run-to-run determinism holds within a process and across processes that load
the same compiled executables from the persistent cache. A fresh compile can
pick different cuBLAS/cuSOLVER algorithms (XLA autotuning), and the search is
chaotic enough that this moved LEAP's best-seed count from 906 to 896 between
two compiles of the same code. That is the same class of deviation the README
documents across machines; `--xla_gpu_autotune_level=0` would remove it at some
cost and is left as an option.

---

## 6. Throughput targets (estimates, to be confirmed by the Phase B gate)

Reference on this 12-core box: ~1.8 ms per sub-step per core, ~7k sub-steps/s
at perfect parallel efficiency; LEAP's ~4M sub-steps take ~500 s here (292 s on
the paper's 32-thread i9).

A batched sub-step at `B ~ 10k` is roughly 300 to 500 kernel launches after XLA
fusion (20 IPM iterations, 10 Gauss-Newton iterations each with a 3-candidate
FK, geometry passes, validations). Launch-bound at 10 to 20 us each gives 5 to
10 ms per batched sub-step, i.e. 1 to 2M sub-steps/s when the batch is full.
Targets:

| workload | reference (this box) | target |
| --- | --- | --- |
| LEAP full metric | ~500 s | 10 to 25 s (seed ranking fully batched; Phase 1 of the best seed alone runs at small B and is launch-bound but only ~32k sub-steps) |
| 16-hand sweep | ~30 min | 2 to 4 min including one-time compiles |
| `fast_eval` per design in `optimize/` | tens of seconds, 12 in parallel on 12 cores | a generation of 12 designs batched together in a few seconds; larger populations nearly free |

The honest caveat: a single hand's best-seed Phase 1 has frontiers of tens to
hundreds of voxels and cannot fill a GPU; it benefits only from the removed
Python overhead (still ~10x). Everything that is embarrassingly parallel today
(seed IK, seed ranking, Phase 2, population evaluation) is where the 50 to 100x
comes from. Phase B produces real numbers before anything downstream is built.

**Phase B gate, measured** (`karma_jax/bench_step.py`, LEAP states, RTX 4090,
full 8-sub-step primitive step with projection and all validations):

| dtype | batch | per batched call | sub-steps/s | attempts/s |
| --- | --- | --- | --- | --- |
| fp32 | 1,024 | 68 ms | 121k | 15k |
| fp32 | 8,192 | 115 ms | 568k | 71k |
| fp32 | 32,768 | 398 ms | 659k | 82k |
| fp64 | 1,024 | 208 ms | 39k | 4.9k |
| fp64 | 8,192 | 1,041 ms | 63k | 7.9k |
| fp64 | 32,768 | 4,262 ms | 62k | 7.7k |

Compile time is about 8 s per shape. At 8k lanes fp32 costs 14 ms per batched
sub-step, inside the 20 ms gate; fp64 is 10x slower, so the kernels are not
purely launch-bound at scale and fp32 as the default is confirmed. Against the
reference's ~7k sub-steps/s on this 12-core box, full batches run 80 to 90x
faster; LEAP's ~4M seed-ranking sub-steps take about 7 s of kernel time.

---

## 7. Parity and validation

Layered, from exact to statistical:

1. **Kernel oracles** (reference env with Pinocchio/OSQP/scipy, random states
   from real BFS runs dumped to `.npz`):
   FK and Jacobians 1e-12; contact kinematics, collision distances,
   segment-segment distances 1e-12; pair selection and all boolean gates
   identical; QP: both solvers satisfy constraints to 1e-7 and objectives
   agree to 1e-6 (not `x` itself when the problem is near-degenerate);
   projection: identical convergence decisions on the dumped states, `q`
   within 1e-8; HEALPix bins identical; LM vs TRF: same feasibility verdict on
   >= 95% of dumped candidates, and the surviving seed sets produce the same
   best-seed voxel count (next layer).
2. **Same seed, same BFS** via `compute_metric(precomputed_seed=...)` with the
   reference's selected seed, `R_seed` and `top_seeds` injected (the plumbing
   exists for the ablation experiment). Expect voxel sets to match exactly or
   to differ by boundary voxels only; investigate any difference above the
   README's "a voxel or two" with the per-level trace (the level-synchronous
   BFS makes the first divergent attempt easy to localize).
3. **End to end on all 16 hands** against `results/16_hand_batch/summary.yaml`:
   Spearman rank correlation 1.0 on KaRMA-T and KaRMA-R, KaRMA-T relative
   deviation under ~3% on hands above the low-DOF floor, KaRMA-S within 0.05.
   The seed IK is the expected source of any larger deviation; if it is, add
   a few more IK restarts per candidate (cheap on the GPU) so the feasible seed
   set is a superset of the reference's.
4. **Invariance and determinism**: `robust_tests/run_tests.py` translation,
   rotation, scale and 3-run determinism must hold exactly, as today. They
   should, because canonicalization and snapping are kept and the kernels are
   deterministic; this is the regression test for the loader port.
5. **Precision isolation**: run layers 2 and 3 first with `--precision fp64`
   to separate algorithmic differences (solver, IK) from precision, then in
   the fp32 default. Both must pass the layer-3 budget; a hand that passes in
   fp64 and fails in fp32 points at one of the Section 5.1 items. Report
   fp32-vs-fp64 voxel counts per hand in the results directory alongside the
   reference numbers.

---

## 8. Roadmap

Each phase ends in a runnable artifact and a measured number.

**Phase A: foundation (about a week).** New env (Python 3.12, `jax[cuda12]`,
the same pinned `pin==2.7.0`, `hpp-fcl==2.4.4`, `osqp`, numpy, scipy, healpy,
PyYAML), with a one-off check that Pinocchio FK is bit-identical to the 3.10
env. `FlatModel` extraction from `RobotContext`. Batched FK, Jacobians,
geometry pass. Oracle tests for all 16 hands in fp64 and fp32. Deliverable:
`karma_jax.model`, `kinematics`, `geometry` with tests green.

**Phase B: kernels and the benchmark gate (one to two weeks).** IPM QP with
the fp32 tolerance, projection, `attempt_primitive_step`, `rotation_step`,
constraints flags, per-step re-orthonormalization. Oracle tests on dumped
states. Micro-benchmark of a batched sub-step at `B = 1k, 10k, 50k` in fp32
(primary) and fp64 on the 4090 and on the CPU backend. Decision point: if the
fp32 batched sub-step is slower than ~20 ms at `B = 10k`, profile with the XLA
profiler and fix (fusion, loop structure, command buffers) before building the
search on top.

**Phase C: search (one to two weeks).** Level-synchronous multi-seed BFS,
Phase 2 streams, host-side dedup and bookkeeping, `MetricResult` assembly.
Layer-2 parity on LEAP, Allegro, Shadow, Dex3 with injected seeds. First
end-to-end timing with reference seeds.

**Phase D: seeds and full pipeline (about a week).** Batched candidate
generation, LM IK, polish, dedup, ranking, KaRMA-S. Layer-3 parity on all 16
hands in fp64 then fp32, `robust_tests` invariance suite in fp32, ablation
flags, JAX-CUDA fp32 made the default in the entry points with the reference
and fp64 flags, docs and README numbers.

**Phase E: optimizer integration and polish.** `optimize/fast_eval.py` on the
JAX backend with cross-design batching (one BFS per generation), proxy-vs-
official audit rerun, compaction in Phase 2 if profiles call for it, optional
JAX `ang2pix`.

---

### Status (2026-10-10)

Phases A to D are implemented in `karma_jax/` with 76 oracle tests passing
(`../.venv/bin/python -m pytest karma_jax/tests`), and Phase E is partly done:

| phase | state |
| --- | --- |
| A foundation | done: flat model from `RobotContext`, batched FK/Jacobians, geometry pass; exact vs Pinocchio on all 16 hands |
| B kernels | done: IPM QP (more accurate than the reference's OSQP settings), projection (matches to 5e-15), step kernels (376/377 outcome agreement on recorded attempts); benchmark 500 to 585k sub-steps/s fp32 |
| C search | done: level-synchronous multi-seed BFS (same order, stats and voxel sets to within boundary voxels), batched Phase 2 (identical bin sets on tested voxels), vectorized host bookkeeping |
| D seeds and pipeline | done: scipy-TRF port reproduces the reference's seeds; `karma_jax.metric.compute_metric` returns the reference `MetricResult`; `run_metric.py --backend jax` (default when importable), `run_all_hands.py`, `robust_tests` via `KARMA_BACKEND=jax` |
| E integration | partly: `optimize/fast_eval.py` gets `backend="jax"` (`karma_jax.fast`); cross-design batching of a whole population and fp32-vs-fp64 per-hand report remain |

End-to-end on this machine (RTX 4090, fp32 search, fp64 seed IK): Inspire
reproduces the reference exactly (4 voxels, top-3 [4,3,3], identical T/R/S,
254 vs 256 seeds). LEAP: ~810 to 900 best-seed voxels against 792, median seed
392 to 397 against 355, KaRMA-T 0.10 against 0.097, KaRMA-R 0.29 against 0.30,
in 87 s (with compiles) against ~500 s for the reference on this 12-core box.
The surplus voxels trace to the more accurate QP solutions (Section 4.4); the
best-seed count is the chaotic quantity, the median and top-3 mean are stable.

**16-hand sweep (layer 3), JAX fp32 vs the paper's table**
(`results/all_hands_20261010_181104.yaml` vs `results/16_hand_batch/summary.yaml`;
reference runtimes are the paper's 32-thread i9, JAX runtimes this RTX 4090 and
include a first-time compile of every batch shape, about 40 to 60 s per hand):

| hand | voxels ref / jax | KaRMA-T ref / jax | dT | KaRMA-R ref / jax | KaRMA-S ref / jax | time ref / jax (s) |
| --- | --- | --- | --- | --- | --- | --- |
| leap | 792 / 814 | 0.0969 / 0.1012 | +4.4% | 0.304 / 0.286 | 0.448 / 0.488 | 292 / 84 |
| allegro | 445 / 461 | 0.0547 / 0.0560 | +2.4% | 0.335 / 0.347 | 0.542 / 0.531 | 307 / 183 |
| dclaw | 315 / 301 | 0.0364 / 0.0359 | -1.5% | 0.231 / 0.231 | 0.225 / 0.282 | 50 / 120 |
| sharpa | 290 / 307 | 0.0340 / 0.0358 | +5.0% | 0.247 / 0.257 | 0.369 / 0.358 | 134 / 156 |
| wuji | 310 / 277 | 0.0339 / 0.0327 | -3.7% | 0.233 / 0.223 | 0.352 / 0.390 | 92 / 140 |
| shadowhand | 110 / 105 | 0.0131 / 0.0129 | -1.6% | 0.161 / 0.156 | 0.209 / 0.219 | 44 / 139 |
| ARMS_skel | 96 / 93 | 0.0118 / 0.0114 | -3.2% | 0.149 / 0.145 | 0.344 / 0.355 | 43 / 80 |
| dex5 | 77 / 79 | 0.0093 / 0.0093 | +0.9% | 0.173 / 0.177 | 0.195 / 0.190 | 33 / 123 |
| DLR | 72 / 61 | 0.0081 / 0.0075 | -6.7% | 0.159 / 0.163 | 0.556 / 0.689 | 16 / 52 |
| dex3 | 34 / 33 | 0.0040 / 0.0040 | 0.0% | 0.094 / 0.092 | 0.059 / 0.061 | 12 / 81 |
| xhand1 | 32 / 32 | 0.0035 / 0.0035 | +2.4% | 0.097 / 0.104 | 0.094 / 0.094 | 13 / 86 |
| orcahand | 25 / 24 | 0.0029 / 0.0025 | -13.0% | 0.137 / 0.118 | 0.320 / 0.375 | 9 / 67 |
| xhandlite | 17 / 16 | 0.0020 / 0.0020 | -2.1% | 0.061 / 0.058 | 0.059 / 0.062 | 9 / 82 |
| inspire | 4 / 4 | 0.00042 / 0.00042 | 0.0% | 0.0044 / 0.0044 | 0.250 / 0.250 | 6 / 81 |
| svh | 3 / 3 | 0.00038 / 0.00038 | 0.0% | 0.0102 / 0.0097 | 0.333 / 0.333 | 7 / 87 |
| ability | 3 / 3 | 0.00029 / 0.00029 | 0.0% | 0.0044 / 0.0044 | 0.333 / 0.333 | 6 / 59 |

Spearman rank correlation with the paper: 1.000 on KaRMA-T (zero pairwise
inversions), 0.994 on KaRMA-R. KaRMA-T deviates by a median 2.2% (max 13% on
OrcaHand, 24 vs 25 voxels); above the low-DOF floor the median is 2.4% and the
max 6.7% (DLR). The five floor hands match exactly.

Timing: the cold sweep took 1,621 s because every hand compiled its own batch
shapes (40 to 60 s per hand). With the persistent cache warm
(`results/all_hands_20261010_182148.yaml`, identical voxel counts on all 16
hands) the sweep takes **590 s** against the paper's 1,072 s on a 32-thread
desktop and roughly 2,500 s for the reference on this 12-core machine:

| hand | ref (32-thread i9) | JAX cold | JAX warm |
| --- | --- | --- | --- |
| leap | 292 s | 84 s | 44 s |
| allegro | 307 s | 183 s | 60 s |
| sharpa | 134 s | 156 s | 50 s |
| wuji | 92 s | 140 s | 44 s |
| dclaw | 50 s | 120 s | 40 s |
| shadowhand | 44 s | 139 s | 39 s |
| low-DOF hands (each) | 6 to 16 s | 59 to 87 s | 18 to 36 s |

Capable hands are 2 to 7x faster than the paper's machine; low-DOF hands are
still 2 to 5x slower, because every run pays a fixed 15 to 30 s of fp64 seed IK
over ~3,000 to 7,000 candidates plus launch-bound small batches in the ranking
BFS and Phase 2. The structural fix for small hands is the remaining Phase E
item: batching many designs (or hands) into the same kernel calls so the
per-call floor is amortized, which is exactly the optimizer's workload.

**Same seed, same BFS (layer 2) on LEAP.** The reference's own seed selection
on this machine picked a best seed worth 797 voxels (paper: 792; top-3 [797,
762, 762], median 371, 252 seeds). Our BFS from that identical seed and grid
frame reaches 826 voxels in fp64 and 768 in fp32. So from one seed, the solver
difference is worth +3.6% and fp32 costs 7% relative to fp64 on this hand; the
individual attempt outcomes agree at the 99% level (Section 7, layer 1) but
LEAP's 4,000-attempt BFS compounds the boundary flips. Averaged quantities are
steadier: in the full pipeline, fp32 LEAP gives top-3 mean 809 and median seed
397 against the reference's 774 and 371. A per-hand fp32-vs-fp64 report over the
whole sweep remains on the list.

Invariance (layer 4): `KARMA_BACKEND=jax robust_tests/run_tests.py` on Inspire
with two translations, two rotations and 0.5x/2x scaling reports exact matches
for every variant in fp32 (voxels 4 -> 4, KaRMA-R 0.0044 -> 0.0044).

Known gaps and follow-ups:

- First run of a new hand pays ~10 s of compile per batch bucket (4 buckets,
  two kernels); the persistent cache removes it afterwards. Padding to shared
  bucket shapes across hands would trade that for ~1.5x more kernel work.
- The seed IK now runs as `ik.ChunkedIK`: 16 trial steps per call with host-side
  compaction of finished lanes (2.7x faster than a vmapped `while_loop` on
  LEAP's 2,880 candidates, identical verdicts).
- `--xla_gpu_deterministic_ops=true` makes the TRF port find far fewer feasible
  seeds (2 instead of 16 on a LEAP sample) although the batched SVD is
  accurate under both modes; the cause is not understood. The flag is not used.

## 9. Risks and open decisions

- **Seed IK divergence** (medium likelihood, medium impact): changes which
  pinches exist, hence possibly the best seed. Mitigations: superset sampling,
  layer-2 isolation, reporting deviations per hand rather than hiding them.
- **QP solver semantics on infeasible steps** (medium): the reference's
  `qp_fail` rate depends on OSQP's infeasibility detection; an IPM declares
  failure differently. Measure `fail_reasons` histograms side by side in
  layer 2; tune the residual threshold, or fall back to the ADMM variant.
- **Compile time and recompiles** (low after bucketing): watch for shape
  leaks (e.g. varying `n_seeds` in a non-batch position). Keep all per-hand
  variability in array values, never in shapes.
- **Launch-bound floor at small batches** (certain, bounded): the best-seed
  Phase 1 will not see the headline speedup. Acceptable given its share of the
  runtime; if it matters later, interleave it with the ensemble seeds and with
  Phase 2 of already-finished seeds.
- **fp32 boundary flips** (medium likelihood, low impact): even with Section
  5.1, a voxel or orientation bin that sits within fp32 noise of a feasibility
  threshold can flip relative to the fp64 reference. This is the same class of
  deviation the README already documents across BLAS backends. If a hand's
  fp32 count deviates beyond the layer-3 budget, the fallback is mixed
  precision: fp64 in the QP and projection solves only, fp32 elsewhere, which
  keeps most of the throughput.
- **Reference-path drift**: the numpy path stays in the repo and must keep
  producing the paper's numbers. Keep it under the same tests, and run the
  16-hand reference sweep in CI occasionally rather than on every change.
- **Legacy pickles and viser**: unaffected because the model is extracted from
  the same Pinocchio `RobotContext`; add a joint-name map to `MetricResult`
  metadata anyway so a future loader change cannot silently misread saved `q`
  vectors.

Open decisions to make at the Phase B gate: IPM vs ADMM, whether to compact
Phase 2 streams, and whether the fp32 QP tolerance of 1e-5 needs tightening to
~3e-6 (the practical fp32 floor) for any hand.

---

## 10. Third-party kinematics: PyRoKi assessment

[PyRoKi](https://github.com/chungmin99/pyroki) (Kim et al., IROS 2025, MIT)
is a JAX toolkit for kinematic optimization: differentiable URDF forward
kinematics, capsule/sphere/halfspace/heightmap collision primitives, and costs
solved by Levenberg-Marquardt through `jaxls`. Reviewed from source on
2026-10-10 (`src/pyroki/_robot.py`, `_robot_urdf_parser.py`, `collision/*`).

What it would cover, and how well:

| KaRMA need | PyRoKi | fit |
| --- | --- | --- |
| Flat joint model, topological order, batched FK | `JointInfo` (twists, `parent_transforms`, `_topo_sort_inv`), `Robot.forward_kinematics(cfg[*batch, n_act]) -> [*batch, n_links, 7]` via `fori_loop`, `jdc.jit`, leading batch axes | good; same design as Section 4.1/4.2 |
| Mimic joints | linear map from actuated config (`get_full_config`), single level only | partial: PyRoKi reads mimic tags from the URDF; KaRMA takes couplings from the robot YAML and deliberately ignores URDF mimic tags (Inspire has 6 in the URDF, 3 in the config). Would need overriding |
| Frame Jacobians (`Jv`, `Jw`, `LOCAL_WORLD_ALIGNED`) | none exposed; costs use autodiff through FK, with one hand-written analytic pose Jacobian as a documented example | gap: we would write the analytic Jacobian ourselves either way |
| Sphere-capsule and capsule-capsule signed distance | `sphere_capsule`, `capsule_capsule` in `_geometry_pairs.py`, broadcastable | good, but ~60 lines and we also need the closest-point parameter `t`, the normal and its Jacobian, which they do not return |
| Procedural capsules from the joint tree with `tip_length_m` and phantom removal | `RobotCollision.from_urdf` fits one capsule per link from collision meshes or primitives | mismatch: a different capsule definition changes the metric; we would construct `Capsule` objects from our own endpoints anyway |
| Rolling QP with hard gap and box constraints | `jaxls` LM with augmented-Lagrangian constraints | mismatch for the inner loop: it is a nonlinear least-squares framework with penalty-based constraints and its own termination, not a QP with OSQP-like semantics |
| Bounded least-squares seed IK | `jaxls` LM | plausible, see below |
| Canonicalization, grid snap, Pinocchio joint order for saved `q` | none | we keep Pinocchio for these regardless |
| Pinned, reproducible dependencies | version `0.0.0`, no PyPI release, `jaxls` as an unversioned git dependency, plus `jaxlie`, `jax_dataclasses`, `jaxtyping`, `loguru`, `tyro`, `robot_descriptions`, `trimesh`, `viser`, `yourdfpy` | poor fit for a metric whose README promises a pinned, bit-reproducible env |

Verdict: not defensible as the foundation, defensible as a reference and as a
candidate for one component.

- The parts PyRoKi does well (URDF to flat arrays, batched FK) are the easiest
  and smallest part of this port, roughly 300 to 500 lines that we get almost
  for free by extracting from the Pinocchio model we must load anyway. Adopting
  PyRoKi would add a second URDF parser (`yourdfpy`) and a joint-order mapping
  between it and Pinocchio, which is more glue, not less.
- The hard parts (Jacobians of the closest point, the QP, the projection, the
  level-synchronous BFS, the batched IK, the single geometry pass) are not in
  PyRoKi, and its conventions (URDF-driven mimics, mesh-fitted capsules,
  quaternion FK output, no Jacobian API) would be fought at every interface.
- The dependency footprint and the unreleased, git-pinned `jaxls` are at odds
  with the repository's reproducibility stance.

What to take from it:

1. **FK oracle in tests.** Install PyRoKi in a test-only extra, pinned to a
   commit hash, and assert our FK against both Pinocchio and PyRoKi. Cheap
   insurance against a convention error that two independent implementations
   would not share.
2. **Design confirmation.** Its `JointInfo` (twists + parent transforms +
   topological permutation, batch axes leading, `fori_loop` FK under `jit`) is
   the same shape as `FlatModel`; keep ours, crib their topological-sort and
   padding conventions where useful.
3. **`jaxls` for the seed IK (Phase D experiment).** The seed IK is a bounded
   nonlinear least-squares problem, exactly what `jaxls` solves. Try it, pinned
   to a commit, against the hand-rolled LM of Section 4.9 on the layer-1 IK
   parity test. Accept it only if it is at least as close to the scipy TRF
   verdicts and `vmap`s cleanly over the ~6,800 candidates; otherwise keep the
   hand-rolled LM, which is ~150 lines.

## Appendix: cheap wins in the reference implementation

Independent of the port, three changes would speed up the reference by roughly
2x and make it a fairer baseline: cache the OSQP algebra check (it is 30% of
each solve), reuse `solver.update` instead of `setup` per step, and load the
robot once per Phase 2 worker (`initializer=`) instead of once per voxel. They
do not change results.
