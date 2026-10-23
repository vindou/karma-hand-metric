# CLAUDE.md

Working guide for this repository. Read `SPEC.md` for what we are building and
why; this file is about how to work here without breaking things.

## What this repo is

KaRMA (Kinematic Rolling Manipulation Ability) is a kinematics-only metric for
robotic-hand dexterity (Peticco & Agrawal, IROS 2026, arXiv:2605.15548). The
repo holds three generations of code:

| Package | Role | Status |
| --- | --- | --- |
| `karma/` | the paper's reference implementation (NumPy, Pinocchio, OSQP, scipy, healpy) | **frozen**; produces the published numbers in `results/`; never edit |
| `karma_jax/` | exact batched JAX/CUDA port of the same metric, oracle-tested against `karma/` | done through Phase D of `docs/jax_rewrite_plan.md`; extend, do not fork |
| `optimize/` | CMA-ES adversarial optimization of KaRMA over a two-finger design family, with a GPU job queue | experiment running; arms B–E still queued |
| `dkarma/` | the differentiable, actuation-aware, multi-contact extension and morphology optimizer specified in `SPEC.md` | Phases 0-10 delivered (`docs/phase{0..10}_report.md`; reviews `docs/phase8_review_2026-10-13.md`, `docs/phase9_review_2026-10-14.md`); **Phase 10 (SPEC §11.3) awaits the owner's review**: 10.1 compliant pads, verdict partial; 10.2 claim made (Full-D18 ahead of arm A′ on 22 of 24 new starts, p = 9 × 10⁻⁵); criterion v2 (SPEC §16 item 12, frozen by the session, not yet accepted) selects Full-D18 start31; the original criterion selects nothing. Framework fixed, no new metric formulations. `dkarma/tests`: 205 passed, 6 skipped. Only one session may write under `dkarma/` at a time |

Git: the work is committed on branch `dkarma-phase0-10` on top of the upstream commit
`c8d3698` (owner's decision, 2026-10-23, SPEC.md §16 item 13, superseding D15).
`origin` is the owner's public fork `vindou/karma-hand-metric`; the original is
`mfpeticco/karma-hand-metric`. Commit or push only when asked. Kept out of the
repository: `workspace/`, the optimizer pickles and videos under
`optimize/runs/`, the marimo session cache, and the owner's working notes (the
phase prompts, the phase reviews and the source conversation), which stay local.

Frozen hardware assumptions (SPEC.md §2): one DYNAMIXEL XL330-M077-T per
joint, direct drive, primary torque ceiling 0.108 N·m (half of the 0.215 N·m
stall at 5 V); object is a 15 mm, 35 g sphere at mu 0.6; task cube ±100 mm at
10 mm around the nominal grasp; palm facing up with gravity along −z of the
palm frame; SO(3) coverage over 576 Hopf-grid targets (smooth, σ_R 0.15 rad)
and 4608 bins (exact); no parameter tying across fingers. Admissibility (SPEC
§8.1 D18, confirmed 2026-10-13): no two modelled fingers may occupy the same
space; finger bases ≥ 2 r_l + eps_col = 24.1 mm apart; reach bound 250 mm per
finger is the global hand envelope.

## Environments and commands

Two interpreters, both under `~/mamba/envs` (created with mamba, not where
the README's `conda env create` would put them):

| Env | Python | Use for |
| --- | --- | --- |
| `~/mamba/envs/karma-hand-metric/bin/python` | 3.10, the paper's pinned stack | reference backend, bit-reproducible numbers, `robust_tests/` |
| `~/mamba/envs/karma-jax/bin/python` | 3.12, jax 0.7.1 CUDA 12, same `pin==2.7.0`/`osqp==1.1.1`/`numpy==1.26.4`, cma 4.0.0 (scipy is 1.17.1, unpinned), mujoco 3.15.0 (installed 2026-10-13 on the owner's confirmation) | everything JAX, `optimize/`, `dkarma/`, MuJoCo replays |

Run everything from the repo root.

```bash
# one hand, JAX backend (default when karma_jax imports) or the paper's reference
~/mamba/envs/karma-jax/bin/python run_metric.py --config robots/robot_inspire.yaml            # ~20 s warm
~/mamba/envs/karma-hand-metric/bin/python run_metric.py --config robots/robot_inspire.yaml --backend reference   # ~10 s

# JAX oracle tests (101 passed, 1 skipped as of Phase 9; needs the GPU free, see below)
~/mamba/envs/karma-jax/bin/python -m pytest karma_jax/tests -q
KARMA_JAX_TEST_ALL_HANDS=1 ~/mamba/envs/karma-jax/bin/python -m pytest karma_jax/tests -q   # all 16 hands

# dkarma tests (CPU fp64; the FD-heavy rollout gradient tests take ~12 min, the rest ~15 min)
JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m pytest dkarma/tests -q

# MuJoCo replay of the exact trajectory libraries (Phase 8.6/9.3; export and simulate on the CPU: JAX_PLATFORMS=cpu)
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay export --out workspace/phase8/8.6
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay simulate --out workspace/phase8/8.6

# robustness harness on either backend
~/mamba/envs/karma-hand-metric/bin/python robust_tests/run_tests.py --config robust_tests/test_config_invariance.yaml
KARMA_BACKEND=jax KARMA_PRECISION=fp32 ~/mamba/envs/karma-jax/bin/python robust_tests/run_tests.py --config robust_tests/test_config_invariance.yaml
```

Reference numbers to check against: `dkarma/fixtures/scores.yaml` (frozen
Phase 0 fixtures: 16 hands + designs baseline/T/A1 under both backends, with
tolerances; re-score with `python -m dkarma.fixtures.rescore`), which copies
`results/16_hand_batch/summary.yaml` (paper), the JAX-vs-paper table in
`docs/jax_rewrite_plan.md` "Status", and `optimize/runs/*/final/summary.json`.

## Compute hygiene (12 cores, one RTX 4090)

- **The optimize queue is stopped** (owner's decision, 2026-10-11 01:00:
  pure development for now). The runner was killed, the 21 unrun jobs (arms
  B, C, D, E, the CPU re-scores and the gallery) are parked in
  `optimize/queue.paused_2026-10-11.txt`, `queue.txt` is empty and
  `queue.STOP` is present. Do not restart it unless asked. The D_R run was
  killed at generation 44 and can be resumed from its `cma_state.pkl` with
  `--resume`. If the queue is ever running again: `tail optimize/queue_done.log`
  shows the current job; jobs use all cores and the GPU, do not overlap.
- A process from another conda env (`designmodule`, pid 348815) holds about
  400 MiB of GPU memory; it is not ours, leave it.
- **One JAX process per GPU.** A second JAX process with default preallocation
  (75%) breaks cuSolver in the first. If a second process is unavoidable, set
  `XLA_PYTHON_CLIENT_PREALLOCATE=false` on it.
- **TF32 silently corrupts fp32 dots.** XLA runs fp32 cuBLAS GEMMs in TF32
  (1e-3 rounding) for contractions wider than ~3. Use `karma_jax/linalg.py`
  helpers or gather/scatter instead of dense `@` in fp32 kernels; do not fix it
  by setting `jax_default_matmul_precision=highest` (2x slower).
- **Compile cache** lives at `~/.cache/karma_jax_xla` (set in
  `karma_jax/__init__.py`). First run of a new batch shape costs ~10 s; warm
  up before timing anything, and report compile time separately.
- `--xla_gpu_deterministic_ops=true` makes the seed IK find far fewer seeds
  for reasons not understood; do not enable it.
- **Host memory is 43 GB.** One Python process running many optimizer runs
  (new jitted objectives per run) grows past 24 GB after ~20 runs and is
  OOM-killed after swapping silently (no log output, GPU 0 %): run long study
  matrices as one process per variant or per start (`workspace/phase8/chain_*.sh`
  pattern, `--only`), and never run a second JAX job, even on CPU, beside a big
  GPU study.
- Phase 9 measurements: a `continuation_study` process grows to ~30 GB over 5
  runs; a 9.2 suite (`admissible_baseline suite`) holds ~8 GB on the CPU; a
  `simulation_replay export` ~9 GB. Run suites and exports on the CPU: the
  per-state eager loops of the export (contact pass, force problem per sub-step)
  pay a kernel launch per op on the GPU and were ~5x slower there. Scripts that
  vmap a fresh closure per seed recompile every call; the exact evaluators now
  use cached programs (`multicontact.CACHED_ROLLOUTS`,
  `exact_library.CACHED_VALIDATORS`, bit-identical, verified).
- Bit-reproducibility matters: unchanged code reproduces a Phase 8 Adam run to
  2e-16, but a one-ulp change in a single op (e.g. `sqrt(sum(x**2))` instead of
  `norm(x)`) diverges the soft stage by iteration 4. Keep replacements of
  reference ops bit-identical (`dkarma.core.geometry.exp_so3_safe`).
- Long jobs: chain with `nohup` scripts that wait on a PID or on a `done` marker
  in the previous log; watch logs with `tail -F` (files that do not exist yet).
  Do not `sed -i` a waiting chain script (bash keeps reading the old inode):
  kill and relaunch it.
- Shell gotchas: `pgrep -f <pattern>` / `pkill -f` from this shell match the
  shell itself (use `ps -eo args | grep '[r]un_optimization'` or test for an
  output file); in zsh, unquoted `echo ====` fails, quote separator strings.

## Rules for new work

1. **Never modify `karma/`.** It is the oracle. If a reference behaviour looks
   wrong, document it in `SPEC.md` §1 and reproduce it exactly in the exact
   path anyway.
2. **Two evaluation paths, kept distinct.** Smooth/differentiable code is for
   gradients; exact code (hard gates, discrete counts) produces every reported
   number. Never present a smoothed score as KaRMA, and never credit a
   trajectory whose solver failed or whose exact feasibility is unverified.
3. **SI units, explicit frames, no `L_ref` scaling in `dkarma/`.** The
   reference scales lengths by `L_ref/0.2`; the actuation-aware metric uses a
   fixed physical object and task volume (see `SPEC.md` §2). Torque limits come
   from an actuator spec file, never from URDF `effort` fields (most are
   placeholders).
4. **Determinism.** Explicit PRNG keys, fixed seed budgets per design, logged
   configs. Identical inputs must give identical outputs within a process.
5. **JAX discipline.** Pure functions, PyTree dataclasses, `lax.scan`/`vmap`/
   `jit`, fixed shapes, masks for variable contact counts. Per-design
   variability goes in array values, never in shapes. Pass the model as a
   traced argument; do not close over `DeviceModel` inside `jit` in new code.
6. **Padding in `karma_jax`** is fixed (Phase 0, 2026-10-11,
   `docs/phase0_report.md`): the exact defaults of `build_flat_model` are
   correct and padded `J_pad`/`M_pad` give the same results
   (`karma_jax/tests/test_padding.py`). Keep `kinematics.apply_mimic`,
   `set_active` and `fold_active` gather-only: XLA miscompiles a
   constant-index `set`-scatter with duplicate indices when the index count
   equals the operand length, which padded lanes always produce. Pass
   `row_mask`/`var_mask` to `qp.solve_box_qp` for any new QP with padded
   variables.
7. **Gradients are validated, not assumed.** Every differentiable component
   ships with a finite-difference test, first on a two-link planar toy with
   known derivatives, then on spatial hands. Check behaviour near active-set
   changes.
8. **Phase discipline.** Implement only the approved phase of `SPEC.md` §11.
   Do not start the next phase because the code runs; the acceptance tests
   decide. Do not add mechanical-packaging, bearing or structural constraints.
9. **Per-phase report** (in the final message and, for larger phases, under
   `docs/`): files changed and purpose, formulation actually implemented,
   tests run with results, known limitations and numerical issues, a minimal
   usage example, comparison with the previous phase, remaining work.
10. **Tests before trust.** Run `karma_jax/tests` after touching `karma_jax/`
    and the relevant `dkarma/tests` after touching `dkarma/`. Reference
    regression (16-hand sweep) is expensive; run it when kernels change, not
    on every edit, and compare against the Status table.
11. **No new metric formulations (Phases 8-10).** No new metric variants,
    kernels or contact models of any evaluator; no solver or optimizer changes in
    Phase 10; changes to the frozen task of SPEC §2 or to the design box need an
    owner decision in SPEC §16. Phase 10 (SPEC §11.3) keeps the framework fixed.
    Its rules:
    (a) The compliant fingertip is a MuJoCo model only. It lies within the
    existing capsule envelope, so morphologies, trajectories and exact verdicts
    are unchanged. Its material parameters are stated assumptions with a
    sensitivity grid, and the rigid 8.6/9.3 result is kept as a bound.
    (b) Every endpoint, decision rule, threshold and selection criterion is
    frozen in writing, dated in `docs/phase10_report.md`, before the data it
    applies to are produced. Phase 9's eight starts are the pilot and are not
    part of the 10.2 primary test. The 10.2 primary endpoint is a common
    manipulation benchmark, not either arm's own score.
    (c) Controller parameters are never tuned on three-contact data; re-tuning,
    if compliance forces it, uses the two-finger baseline only.
    (d) Results on the task-feasible subset `S_task` are always reported beside
    the unchanged all-seeds results and are never presented as equivalent to
    the original 9.5 criterion, which is applied alongside criterion v2.
    (e) The Phase 9 settings stay: `kkt_method="reduced"`, `exp_so3_safe`,
    norm-100 clipping, cached exact evaluators, and suites and exports on the CPU.
    Phase 9 rules, still binding: (a) the KKT
    derivative repair is a three-way derivative comparison with
    finite-difference tests on well-conditioned and degenerate states, never
    "add damping until the NaNs stop" (SPEC §11.2 item 9.1); (b) every
    optimizer, the original-KaRMA baseline included, enforces admissibility
    (D18) throughout the run; (c) exact Full-KaRMA numbers use the exact SLSQP
    force check at every sub-step (`multicontact.evaluate_fixed_contacts_exact`,
    Phase 8.6), never the smooth force gate; (d) simulation results state the
    controller and the success criterion (pose, contact retention, drop) they
    were measured with.
12. **Claims need statistics.** A comparative statement about optimizers or
    objectives requires ≥ 8 independent starts per method, matched wall clock
    measured on this GPU with validations outside the budget, exact evaluation
    of every returned design with fresh seeds, and mean/median/best/IQR plus a
    paired comparison. One run is a pilot and is labelled as such.
13. **Infeasibility is never concluded from one search.** "No seed found" means
    the procedure found none; say so with the attempt count. Before claiming a
    grasp or trajectory does not exist, run the independent procedures of SPEC
    §11.1 item 8.5 and report all of them.
14. **Experiment provenance.** Every reported number names its script, its
    JSON under `workspace/`, the design files, the seed config and the
    optimizer seed. Re-running the script must reproduce the table; if it
    cannot (GPU autotuning, interrupted runs), say which numbers moved.

## Where things are

- Metric config: `karma_config.yaml` (nominal lengths for a 200 mm hand).
  Per-hand configs: `robots/robot_*.yaml`, template with field-by-field
  comments in `robots/robot_template.yaml`. URDFs: `robots/urdfs/`.
- Methodology notes: `docs/scale_invariance_nondim.md`,
  `docs/slip_tolerance_study.md`, `docs/jax_rewrite_plan.md` (port design,
  measured parity, risks), `docs/spec_source_conversation_2026-10-11.md`
  (advisor conversation the spec was distilled from; local only, like the
  phase prompts and reviews).
- Optimizer design family and URDF writer: `optimize/hand_family.py`; proxy
  evaluator `optimize/fast_eval.py` and `karma_jax/fast.py`; official re-score
  `optimize/evaluate_full.py`; draft write-up `optimize/runs/REPORT.md`, auto
  tables `optimize/runs/RESULTS_AUTO.md` (regenerate with `optimize/report.py`).
- Phase 8 results: `docs/phase8_report.md` (sections 8.1-8.7 and the one-page
  summary), raw outputs under `workspace/phase8/{8.1,8.2,8.4,8.5,8.6,exp2}`
  (8.2 holds the continuation and seed-policy studies for `diff2` and `full3`,
  one run directory per variant and start with the §13 layout), the chain
  scripts `workspace/phase8/chain_*.sh`; fabrication candidates (nothing
  selected) under `results/fabrication_candidates/`. Phase 8 settings now in
  code: `ForceWeights` (hinge 0.005, damped Newton, envelope gradient),
  `EnergyWeights.force_knee/force_steep` (the knee credit), the seed-state force
  gate, `ADAM_DEFAULTS` (clip 100, skip non-finite, reset moments, stage
  selection), `seed_policy="reproject"` default with `"union_stage"` for
  two-finger objectives, `multi_candidates(centre_caps="all", n_restarts=...)`.
- Phase 9 results: `docs/phase9_report.md` (9.1-9.5 and the one-page summary;
  owner's review `docs/phase9_review_2026-10-14.md`, which sets Phase 10),
  raw outputs under `workspace/phase9/{9.1,9.2,9.3,9.5}` (9.1: traces, corpus,
  derivative comparison, re-runs `rerun_reduced/`; 9.2: arm A' and Full-D18 runs,
  suites `suite/<design>/suite.json`, sweeps, MuJoCo leg `sim/`, `summary.md`;
  9.3: augmented 8.6 archives and `controller_comparison.md`), chain scripts
  `workspace/phase9/chain_*.sh`. Phase 9 settings now in code:
  `RollConfig.kkt_method="reduced"`, `exp_so3_safe` in the rollout, D18
  (`dkarma.core.admissibility`, `Constraints.base_sep_min_m`/`D18_CONSTRAINTS`,
  `build_flat_model(self_collision_all=True)` / `DeviceModel.cap_group` in
  `karma_jax`), the force-floor controller of `simulation_replay` (`--controller
  force`), the cached exact evaluators.
- `dkarma/` layout: `core/` (topology, morphology, kinematics, contacts,
  actuation, task spec), `solvers/` (rolling QP, projection, rollout, exact
  validator, contact forces, seeds), `metrics/` (coverage kernels, commands,
  Diff/Act/MC/Full evaluators, exact library, continuation), `optimization/`
  (objectives, Adam/CMA-ES/hybrid, checkpoint), `experiments/`
  (`surrogate_validation.py` = Exp I, `actuation_awareness.py` = Exp II,
  `codesign_comparison.py` = Exp III/IV, `optimize_morphology.py` = CLI,
  `force_calibration.py` = 8.1, `continuation_study.py` = 8.2/8.3,
  `validation_noise.py`, `optimizer_comparison.py` = 8.4,
  `three_contact_verification.py` = 8.5, `simulation_replay.py` = 8.6 and 9.3, `kkt_derivative.py` = 9.1,
  `admissible_baseline.py` = 9.2/9.4, `candidate_selection.py` = 9.5),
  `fixtures/`, `tests/`.
- Phase 10 results: `docs/phase10_report.md` (pre-registration and addendum A1, 10.1-10.4, one-page summary), raw outputs
  under `workspace/phase10/{10.1,10.2,10.3,10.4}`, chain scripts `workspace/phase10/chain_*.sh`; the stop and restart
  note `workspace/phase10/STOPPED_2026-10-15.md`. Phase 10 code: `dkarma/experiments/{compliant_pad,compliant_contact_study,
  powered_comparison,robust_grasps,selection_v2}.py`, frozen pad sets `dkarma/configs/compliant_pads.json`
  (`simulation_replay simulate --pad 20A_2mm` replays with the central compliant fingertips; the compliant position servo
  uses a 0.75 mm squeeze, `workspace/phase10/10.1/retune.json`). Selected (criterion v2, pending review):
  `results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25/`, with renders, replay videos and charts in
  its `figures/` (`dkarma/experiments/visualize_design.py`; MuJoCo renders need `MUJOCO_GL=osmesa`, EGL is unavailable).
- Phase 7 results: `workspace/phase3/surrogate.json`, `workspace/phase7/exp2.json`,
  `workspace/phase7/exp34/results.json` (+ `run_*/`, `armA_design/`),
  `workspace/phase7/exp1_{adam,cmaes}/`; all read by `docs/phase7_report.md`.
  `workspace/` is gitignored: copy anything worth keeping into `results/`
  deliberately, never by default.
- Heavy jobs now are the `dkarma/experiments` scripts (GPU, minutes to hours).
  Run them one at a time with `nohup ... > workspace/<phase>/<name>.log 2>&1 &`,
  chain follow-ups with an `until grep -q "^done" <log>` watcher, and monitor
  the log; the optimize queue stays stopped.
- Scratch outputs go to `workspace/` (gitignored) or the session scratchpad,
  never into `results/`.
