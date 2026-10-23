# Phase 3 report: smooth reachability objective, Diff-KaRMA (2026-10-11/12)

Phase 3 of `SPEC.md` §11. Deliverable: Diff-KaRMA (coverage kernels, trajectory weights,
seed aggregation, diagnostics, continuation) on top of the Phase 2 rollout, with seeds in
physical units and the exact library counterpart. Exit criterion (Experiment I surrogate
checks): positive rank correlation with exact KaRMA-T on a morphology dataset, and local
gradient steps improving exact KaRMA on a majority of starts. The first holds strongly
(§3.2); the second is reported in §3.3.

## 1. Files and purpose

| File | Purpose |
| --- | --- |
| `dkarma/configs/task_spec.yaml`, `dkarma/core/object_model.py` | `rolling` block: the fixed physical tolerances (capsule 11.8 mm, `eps_g` 0.35 mm, `eps_col` 0.5 mm, 1.25 mm sub-step, 0.5 mm slip gate, 10° joint cap); `TaskSpec.step_params()` builds the physical `StepParams` (no `L_ref` scaling) |
| `dkarma/metrics/orientation.py` | the 228 pinch-axis line-bin centres (HEALPix nside 6, RING 0..227) and the reference binning; Hopf SO(3) grids (72 / 576 / 4608) with rotation matrices, nearest-neighbour spacing, exact binning |
| `dkarma/metrics/smooth_coverage.py` | log-space occupancy `o = 1 - prod(1 - w k)`; translation, pinch-axis and SO(3) kernels and coverages; region-restricted orientation coverage; the soft seed aggregate |
| `dkarma/metrics/commands.py` | fixed-size command libraries from the seed frame and pinch axis (6 translations × `n_cells` cells, 4 tilts × `n_rot_steps`); differentiable `seed_frame` equal to the reference's |
| `dkarma/solvers/seed_projection.py` | physical-unit two-finger seeds: reference sampling, reach filter, batched LM IK on the reference residual, dedup, fixed budget with validity masks, nullspace polish, frame; differentiable per-morphology `reproject` |
| `dkarma/metrics/config.py` | `SmoothConfig` (kernel widths, seed aggregation, objective weights and reference scales), `MetricConfig` and the four variants |
| `dkarma/metrics/diff_karma.py` | `evaluate_seed`, `evaluate` (jit-able; per-seed `K_T~`, `K_R~`, region coverages, energies, gates, credits), `objective(phi, ...)`, `summarize` |
| `dkarma/metrics/exact_library.py` | exact counterpart on the same library (reference step kernels; counted cells and bins); `official_karma` (published semantics) and `official_karma_physical` (the original BFS on the physical task through a per-design metric YAML) |
| `dkarma/metrics/continuation.py` | kernel/penalty schedule, soft solver first |
| `dkarma/experiments/surrogate_validation.py` | Experiment I |
| `dkarma/tests/test_orientation_coverage.py` (10), `test_diff_karma.py` (8) | §3.1 |

## 2. What was implemented

Per seed `s` (fixed budget `S`, invalid lanes score 0) and command `u`: the Phase 2 rollout
gives states and residuals; `energies` gives prefix log-weights `log W_k`. Coverage:

    K_T~ = mean_v [1 - prod_{j,k} (1 - W_jk a_jkv)],   a = exp(-|p - v|^2 / 2 sigma_p^2),   v in the ±100 mm cube at the seed
    K_R~ = mean_b [1 - prod (1 - W k)],                 k = exp(-(1 - (a_body . b)^2) / 2 sigma_R^2),  228 bins

plus region-restricted `K_R` at the seed cell and the translation end cells, `K_s = K_T/ref_T
+ K_R/ref_R`, and `(1 - beta) mean_s K_s + beta softmin_s K_s`. Everything after the seed
set is one jitted function of the morphology; seeds are re-projected inside it. The exact
counterpart replays the same commands with the reference kernels and counts.

Seeds are found deterministically in physical units on the `dkarma` model (no URDF): on the
baseline design 168 feasible pinches from 1,629 candidates in 9 s on the GPU.

## 3. Tests and results

### 3.1 Unit tests (CPU fp64)

`test_orientation_coverage` 10 passed: the 228 centres map to themselves under the
reference binning (nudged off the equator ring's z = 0), Hopf grids unit and uniform
(nearest-neighbour spacing within 10 % of the mean at 72 and 576, 12 % at 4608), Haar
fractions within 0.03, occupancy bounds and monotonicity, coverage limits, region
coverage ≤ global, soft seed aggregate bounds, command library and seed frame equal to
the reference's. `test_diff_karma` 8 passed: physical parameters, seed pipeline
deterministic and exactly feasible, fixed budget padding, evaluator shapes/bounds/jit,
zero score without seeds, objective gradient vs FD (soft mode, 8 coordinates, rel 1e-3),
hard-mode gradient finite with an ascent direction, exact library counts, schedule.

GPU (baseline design, 8 seeds, 6 translation × 40 steps + 4 rotation × 12 steps):
evaluation 0.49 s, objective + gradient 1.1 s after a 22 / 76 s compile; a normalized
gradient step of 1e-3 raises the smooth objective (0.2405 → 0.2436).

### 3.2 Experiment I, dataset leg (`workspace/phase3/surrogate.json`)

Nine two-finger designs (baseline, the CMA-ES winners T and A1, six perturbations of the
baseline), 8 seeds each, scored by: smooth `K_T~`, `K_R~`; the exact library (counted
cells / bins over the same commands); the **original KaRMA BFS on the physical task**
(`official_karma_physical`: SPEC §2 object and tolerances, top-3 voxel fraction of the
cube and KaRMA-R); and the published metric (`L_ref`-scaled nominal task).

| design | smooth K_T | exact-lib K_T | physical KaRMA cube fraction | published KaRMA-T | smooth K_R | exact-lib K_R | physical KaRMA-R |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | 0.00588 | 0.00067 | 0.01071 | 0.0298 | 0.106 | 0.053 | 0.109 |
| T | 0.00163 | 0.00013 | 0.00013 | 0.558 | 0.008 | 0.004 | 0.004 |
| A1 | 0.00163 | 0.00013 | 0.00013 | 0.429 | 0.008 | 0.004 | 0.004 |
| pert00 | 0.00205 | 0.00014 | 0.00025 | 0.0002 | 0.022 | 0.014 | 0.024 |
| pert01 | 0.00213 | 0.00025 | 0.00108 | 0.0001 | 0.025 | 0.018 | 0.023 |
| pert02 | 0.00446 | 0.00047 | 0.00762 | 0.0200 | 0.082 | 0.044 | 0.094 |
| pert03 | 0.00433 | 0.00047 | 0.00258 | 0.0028 | 0.100 | 0.046 | 0.063 |
| pert04 | 0.00163 | 0.00014 | 0.00079 | 0.0213 | 0.008 | 0.006 | 0.017 |
| pert05 | 0.00392 | 0.00028 | 0.00229 | 0.0008 | 0.067 | 0.030 | 0.032 |

Spearman rank correlations:

| pair | rho |
| --- | --- |
| smooth K_T vs exact-library K_T | **+0.987** |
| smooth K_T vs physical KaRMA cube fraction | **+0.979** |
| exact-library K_T vs physical KaRMA | +0.992 |
| smooth K_R vs exact-library K_R | +0.971 |
| smooth K_R vs physical KaRMA-R | +0.937 |
| smooth score (balanced) vs physical KaRMA | +0.803 |
| smooth K_T vs published KaRMA-T | −0.333 |
| physical KaRMA vs published KaRMA-T | −0.259 |

The surrogate ranks designs as the exact evaluators do on the physical task. The
negative correlation with the published number is the Phase 0 finding seen from the
other side: the CMA-ES winners that score 0.56 and 0.43 on the `L_ref`-normalized metric
(5 to 6× the best real hand) cannot roll the physical 15 mm sphere at all (one cell, the
seed, under every evaluator), while the modest baseline (0.03 published) is the most
capable physical hand of the set. Surrogate bias: smooth coverage is 5 to 10× the exact
library count (the 6 mm kernel credits neighbouring cells) and the library reaches far
fewer cells than the BFS (a handful of commands versus thousands of branches); rank order,
not magnitude, is what the optimizer uses.

### 3.3 Experiment I, local-improvement leg

Running at the time of writing (`workspace/phase3/surrogate_local.log`): from three
starts, six accepted normalized gradient steps on the smooth objective, each candidate
re-scored by the exact library and the physical KaRMA. The outcome is appended to this
section and to `docs/phase7_report.md` when it completes.

## 4. Known limitations and numerical issues

- **Energies are steep.** With the default weights a slip of ten gates gives `e = 100`,
  so credit effectively truncates at the first bad step; the continuation schedule's
  `sigma` relaxes this early in a run. Reported, not hidden: `summarize` lists per-command
  worst and total energies and the fraction of steps passing each exact gate.
- **Trajectory library ≠ reachable set.** Six straight commands and four tilts per seed;
  the BFS explores branches. Rank agreement is high (§3.2); magnitudes are not comparable.
- **Fixed assignment.** Where the reference would switch contact links the smooth path
  keeps the seed assignment; `pair_switch_frac` was 0 on every design here.
- **Kernel widths are continuation knobs** (`sigma_p` 6 mm, `sigma_R` 0.09 rad final).
- **Two-finger only** in this phase; three contacts and SO(3) are Phase 5.
- **Run interruption.** The dataset run stopped after nine designs without a traceback
  (session interruption); the script now supports `--resume` and `--local-only`.

## 5. Minimal usage

```python
from dkarma.core import build_hand_model, load_task_spec, Bounds, encode, fixed_from_params
from dkarma.solvers.seed_projection import find_seeds
from dkarma.metrics.config import DIFF_KARMA
from dkarma.metrics import diff_karma as DK
from dkarma.metrics.exact_library import evaluate_exact, official_karma_physical

task = load_task_spec(); sp = task.step_params()
hm = build_hand_model(topo, params); seeds = find_seeds(hm, topo, sp)
d = DK.evaluate(hm, topo, seeds, sp, task, DIFF_KARMA); print(DK.summarize(d))
grad = jax.grad(lambda phi: DK.objective(phi, topo, bounds, fixed_from_params(params), seeds, sp, task, DIFF_KARMA))(encode(params, topo, bounds))
exact = evaluate_exact(hm, topo, seeds, sp, task, DIFF_KARMA)                  # counted cells / bins
official = official_karma_physical(topo, params, task, "workspace/tmp")        # the original BFS on the physical task
```

## 6. Comparison with Phase 2

Phase 2 differentiated one trajectory. Phase 3 turns a library of trajectories into a
scalar design objective whose ranking agrees with the exact evaluators at rho ≈ 0.98,
with seeds and tolerances in physical units and the `L_ref` exploit closed by
construction.

## 7. Remaining work

The local-improvement numbers (§3.3) and the matched-budget optimizer comparison
(Experiment I, Phase 7).
