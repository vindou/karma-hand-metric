# Phase 0 report: audit and baseline freeze (2026-10-11)

Phase 0 of `SPEC.md` §11. The audit itself is `SPEC.md` §1; this report covers
the remaining deliverables: regression fixtures with recorded scores under both
backends, the two `karma_jax` padding defects fixed at the root with tests, and
the `J_pad` workaround removed from the callers. Everything here ran on the CPU
(`JAX_PLATFORMS=cpu`) because the optimize queue owned the GPU; the GPU
re-score of all fixtures on the fixed kernels is the one item left open.

## 1. Files changed and purpose

| File | Change |
| --- | --- |
| `karma_jax/model.py` | `FlatModel`/`DeviceModel` gain five per-joint gather tables (`jm_source`, `jm_mult`, `jm_offset`, `is_active`, `active_lane`); `build_flat_model` fills them and rejects duplicate, chained or active mimic targets; docstring states that exact padding is correct |
| `karma_jax/kinematics.py` | `apply_mimic`, `set_active`, `fold_active` rewritten gather-only (no `.at[].set/add`), `fold_active` adds mimic contributions in the reference's list order |
| `karma_jax/qp.py` | `solve_box_qp(..., row_mask=None, var_mask=None)`: padded rows/variables are left out of `mu`, the fraction-to-boundary ratio tests, the tolerance scales and the residual checks; unmasked arithmetic unchanged |
| `karma_jax/rolling.py` | both steps pass the masks built from `model.active_valid`; padding docstring updated |
| `karma_jax/tests/test_padding.py` | new, 24 tests (see §3) |
| `dkarma/__init__.py`, `dkarma/fixtures/__init__.py` | new package; fixture registry, loaders, `scores_from_result`, `compare` with the documented tolerance levels |
| `dkarma/fixtures/scores.yaml` | frozen scores for 16 hands + baseline/T/A1 under `reference` and `jax_fp32`, with provenance |
| `dkarma/fixtures/designs/{baseline,T,A1}/` | byte copies of the scored URDF, robot YAML and `HandSpec` JSON from `optimize/runs/` |
| `dkarma/fixtures/freeze.py` | rebuilds `scores.yaml` from the original result files |
| `dkarma/fixtures/rescore.py` | CLI: re-score fixtures on a backend, compare, write a report, optionally record |
| `dkarma/tests/test_fixtures.py` | new, 21 tests + 6 opt-in re-score tests |
| `optimize/backend.py` | `J_pad = n_joints + 1` workaround removed (exact default); docstring explains why results are bitwise unchanged |
| `optimize/karma_jax_padding_repro.py` | header rewritten: root cause, fixed state; still runnable, prints zero error |
| `CLAUDE.md` | rule 6 now describes the fix and the scatter rule; `dkarma/` row and reference-numbers pointer updated |
| `SPEC.md` | §1.3 item 4 and §11 Phase 0 line annotated as fixed/delivered (no other spec change) |

Not touched: `karma/` (verified with `git diff -- karma/`: only the pre-existing
`results.py` edit from the JAX work), `results/`, the queue files.

## 2. What was actually implemented

### 2.1 Root cause of "J_pad == n_joints breaks jitted FK"

The defect was never about the joint table. `apply_mimic` wrote the mimic
couplings with `q.at[mimic_target].set(...)`. The mimic table is padded to
`K_pad = max(8, n_mimic)` with target index 0 in the padded lanes, so on a hand
with no mimics it is eight writes of `q[0]` into `q[0]` (harmless by itself).
XLA (jax 0.7.1, CPU and GPU alike) rewrites a `set`-scatter whose indices are
compile-time constants and whose update count equals the operand length into a
plain gather, ignoring that the indices are duplicates. With the model closed
over inside `jit` (every production kernel does that) and `J_pad == 8` the
rewrite fires and every joint receives `q[0]`. Minimal form:

```python
idx = jnp.zeros(8, jnp.int32)                      # closed-over constant
jax.jit(lambda q: q.at[idx].set(q[idx]))(q8)       # returns q8[0] everywhere
```

Length 7 or 9, an `add`-scatter, or `idx` passed as an argument are all
correct. That is why only the optimizer's 8-joint hands (and no shipped hand)
were affected, why `J_pad = 9` "fixed" it, and why the existing oracle tests
(which pass the model as a jit argument) never saw it.

Fix: `apply_mimic` is a gather over per-joint tables,
`q_c[j] = jm_mult[j] * q[jm_source[j]] + jm_offset[j]`, with
`(j, 1.0, 0.0)` for non-mimic joints (`1.0 * x + 0.0 == x` exactly). Mimic
joints get the same expression the reference evaluates in
`karma.robot.update_kinematics`.

### 2.2 Root cause of "M_pad > n_active corrupts the projection"

`set_active` used `q.at[q_ids].set(where(valid, q_active, q[q_ids]))`. Padded
lanes carry `q_ids == 0`, so they scattered the old `q[0]` into joint 0
together with the real lane-0 update; XLA leaves the winner of duplicate writes
undefined. Fix: `set_active` is `where(is_active, q_active[active_lane], q)`,
a gather that never reads padded lanes.

`fold_active` had a benign scatter-add (`J.at[..., source].add(contrib)`),
replaced by a static loop over the mimic table that adds each coupling's
contribution to the matching active lane in list order. This reproduces
`karma.robot.fold_mimic_jacobian` bit-for-bit, including on Inspire and SVH
where two mimics share one source (the previous scatter-add left the order of
those two additions to XLA).

### 2.3 Padded lanes in the QP

Padded active-joint variables are inert in the rolling QPs (zero cost, unit
curvature, `[-1, 1]` box; their Newton block is diagonal and decoupled), but
they entered the complementarity average `mu`, the ratio tests and the
relative tolerance scales, shifting the central path by a tolerance-sized
amount. `solve_box_qp` now takes `row_mask`/`var_mask`; the rolling and
rotation steps pass masks from `model.active_valid`. With an all-true mask the
arithmetic is the same expression as before (the masked dot keeps the same
reduction with exact zeros in the masked terms), so exact-padding results are
bitwise unchanged.

### 2.4 Callers

`build_flat_model`'s exact defaults are now correct, so `run_metric.py`,
`robust_tests/run_tests.py` and `optimize/fast_eval._evaluate_jax` needed no
change; `optimize/backend.py` dropped its `n_joints + 1` workaround. Both
J_pad choices give bit-identical FK (test below), so the queued arms B–E stay
comparable with A1–A3 scored before this change.

### 2.5 Fixtures

`dkarma/fixtures/scores.yaml` records, per fixture and backend, the score
fields of the paper's table (`n_voxels`, `KaRMA_T/R/S`, `best_voxels`,
`median_voxels`, `n_seeds`, plus `ensemble_t/r`, `L_ref_mm`, runtime) with
`source`, `machine`, `date` and a code-state note:

| Fixture set | `reference` | `jax_fp32` |
| --- | --- | --- |
| 16 shipped hands | paper table, 32-thread i9, 2026-07-01 | warm sweep on this RTX 4090, 2026-10-10 (`results/all_hands_20261010_182148.yaml`, gitignored, so copied) |
| baseline | this machine, 2026-10-09 (`optimize/runs/baseline/summary.json`) | not yet recorded |
| T | this machine, 2026-10-10, 12 rotation workers | not yet recorded |
| A1 | queued (`optimize/queue.txt`, `final_reference`) | this RTX 4090, 2026-10-10 |

Tolerance levels (`dkarma.fixtures.TOLERANCES`): `exact` (same machine and
stack), `practical` (the robustness harness's practical match: ≤ 1 voxel,
|ΔT| < 1e-3, |ΔR| < 0.01, |ΔS| < 0.1), `jax` (5 % + 3 voxels on counts, 5 % on
KaRMA-T, 0.02 on R, 0.1 on S, exact counts on hands at or below 4 voxels).
The `jax` level is provisional until the GPU re-score below is recorded.

## 3. Tests run and results

All on CPU, `~/mamba/envs/karma-jax` unless noted.

| Suite | Before the fixes | After |
| --- | --- | --- |
| `karma_jax/tests` (existing 78 oracle tests) | 77 passed, 1 skipped, 241 s | 77 passed, 1 skipped, 259 s |
| `karma_jax/tests/test_padding.py` (new) | n/a (two of them fail on the old code: the repro) | 24 passed, 47 s |
| `dkarma/tests` | n/a | 21 passed, 6 skipped (opt-in re-score), 4 s |
| `optimize/karma_jax_padding_repro.py` | J_pad=8: FK error 3.3e-2; M_pad=10: 86/200 verdict mismatches, |Δq| 0.32 | both zero (|Δq vs reference| 5e-13) |

Padding tests, measured agreement between exact and padded models
(`J_pad ∈ {nJ, nJ+1, nJ+5}`, `M_pad ∈ {nM, nM+3}`):

| Kernel | Hands | Padded vs exact |
| --- | --- | --- |
| `apply_mimic`, closed-over jit | baseline, T, A1, inspire, svh, leap | equals the dense `A_mimic q + b` (atol 1e-15) |
| `forward_kinematics`, closed-over jit | baseline, T, A1, leap, inspire | matches Pinocchio to 1e-10; bit-identical across padding |
| `set_active`/`get_active` | inspire, T | bit-identical, padded lanes ignored |
| `fold_active` | inspire, svh, leap, T | bit-identical to `fold_mimic_jacobian` |
| `project_onto_contacts` | baseline, T, A1 (100 random pinch states each) | verdicts identical to the reference, |Δq| ≤ 7e-12; padded vs exact within rtol 1e-11 (dense GN solve rounding) |
| rolling + rotation QP steps | leap (524 + 221 recorded calls), inspire (6 + 4) | flags identical, max |Δx| 1.2e-13 |
| `attempt_primitive_step` (8 sub-steps, all gates) | leap, 40 recorded nodes | ok/code/pair/p identical, max |Δq, ΔR| 1.1e-13 |

Re-score of the floor hands against the frozen numbers
(`python -m dkarma.fixtures.rescore`):

| Fixture | Backend | voxels | KaRMA-T | KaRMA-R | KaRMA-S | vs recorded |
| --- | --- | --- | --- | --- | --- | --- |
| inspire | reference (this machine) | 4 | 0.00042 | 0.0044 | 0.250 | exact |
| svh | reference | 3 | 0.00038 | 0.0102 | 0.333 | exact |
| ability_hand_right | reference | 3 | 0.00033 | 0.0044 | 0.333 | practical (ensemble [3,3,2] vs the paper's [3,2,2]; one ensemble voxel on a different machine, |ΔT| 3.8e-5) |
| inspire | jax fp32, CPU | 4 | 0.00042 | 0.0044 | 0.250 | exact |
| svh | jax fp32, CPU | 4 | 0.00042 | 0.0097 | 0.250 | one voxel above the GPU record (3); the pre-fix kernels on CPU give the identical 4-voxel result (same 627 seeds, ensemble [4,3,3]), so this is CPU-vs-GPU fp32 compilation, not the fixes |
| ability_hand_right | jax fp32, CPU | 3 | 0.000292 | 0.0044 | 0.333 | exact |

Reports: `workspace/phase0/rescore_reference_floor.yaml`,
`workspace/phase0/rescore_jax_cpu_floor.yaml`.

## 4. Known limitations and numerical issues

- **Padding is not bitwise invariant through dense solves.** FK, mimic folding
  and `set_active` are bit-identical across padding; the Gauss-Newton solve
  and the QP Cholesky see a larger block-diagonal matrix and round
  differently (≤ 1e-11 relative, ≤ 1.2e-13 absolute on the recorded steps). A
  boundary voxel can in principle flip under a different `M_pad`; the metric
  tolerance that already covers fp32-vs-fp64 covers this.
- **The XLA bug is worked around, not fixed.** Any new `.at[].set` with a
  constant index vector whose length can equal the operand length is at risk
  (CLAUDE.md rule 6). `.at[].add` and traced indices are safe.
- **Chained mimics raise** in `build_flat_model` (none of the 16 hands has
  one); the previous code silently mis-evaluated them.
- **Reference backend across machines**: ability_hand_right differs from the
  paper by one ensemble voxel on this machine (see table), so the `exact`
  level only applies to numbers recorded on the same machine.
- **Scores not yet recorded**: `jax_fp32` for baseline and T, `reference` for
  A1 (queued). The GPU re-score of all 19 fixtures on the fixed kernels is
  pending; the `jax` tolerance level is provisional until then.
- **JAX CPU and GPU are different compiled programs.** SVH reaches 4 voxels
  on CPU and 3 on the GPU with the same source (old and new kernels agree
  bit-for-bit with each other on CPU), so the `jax_fp32` records are
  GPU-specific and the `exact` floor rule of the `jax` level applies to GPU
  runs. Floor hands take about 1 min each on CPU; capable hands are
  impractical there, so the full JAX regression is GPU-only.

## 5. Minimal usage

```python
from dkarma import fixtures as FX
import jax.numpy as jnp
from karma_jax import kinematics as K

flat, cfg, ctx = FX.load_flat("T")             # exact padding, now correct
flat8, _, _ = FX.load_flat("T", J_pad=16, M_pad=12)   # padded: same answers
dm = flat8.to_device(jnp.float64)
fk = K.forward_kinematics(dm, jnp.zeros(16))   # joints >= 8 are inert

row = FX.scores_from_result("T", result, elapsed_s)        # after compute_metric
print(FX.compare(row, FX.fixtures()["T"].recorded["reference"], "practical"))
```

```bash
# re-score, compare, and (once the GPU is free) record the JAX numbers
~/mamba/envs/karma-jax/bin/python -m dkarma.fixtures.rescore --backend jax --all --record --note "Phase 0 kernels"
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m pytest karma_jax/tests dkarma/tests -q
```

## 6. Comparison with the previous state

Before: `karma_jax` was correct only at the exact padding the 16 hands used,
produced wrong FK on the optimizer's hands at exact padding (so every caller
except `optimize/backend.py` was wrong on those hands), and could not pad
`M_pad` at all; the recorded numbers lived in four unrelated files, one of
them gitignored. After: all paddings agree to roundoff, the callers share one
correct default, the three designs that drove the study are frozen with their
files, and every recorded score has a loader, a tolerance and a re-score path.
The oracle suite and the optimize queue see bitwise-identical numbers.

## 7. Remaining work (Phase 0 exit criterion)

1. When the queue is idle: `python -m dkarma.fixtures.rescore --backend jax --all --record`
   on the GPU (about 10 min warm) and tighten `TOLERANCES["jax"]` to what it
   shows; record `jax_fp32` for baseline and T, and `reference` for A1 when
   the queued `final_reference` job finishes (`python -m dkarma.fixtures.freeze`).
2. Optionally re-score the 16 hands on the reference backend on this machine
   (about 40 min, all cores) to turn the paper's `practical` comparison into
   an `exact` same-machine record.
3. Phase 1 (`SPEC.md` §11) can start: traced-model FK is already exercised by
   `test_fk_closed_over_jit_padding_invariant`'s argument-passing twin in the
   oracle suite; the gather-only kernels have clean VJPs.
