# Phase 6 report: morphology optimizer (2026-10-11)

Phase 6 of `SPEC.md` §11. Deliverable: an optimizer module that knows nothing about the
metric internals, with Adam, CMA-ES and the hybrid, continuation, exact validation
callbacks, checkpointing and the §13 artifact layout. Exit criterion: reproducible
multistart runs on the 4+4 family under matched budgets. The module and tests are
delivered (§3); the matched-budget comparison itself is Experiment I's optimization leg
and runs in Phase 7.

## 1. Files and purpose

| File | Purpose |
| --- | --- |
| `dkarma/optimization/objectives.py` | `Objective` (topology, bounds, fixed parameters, seeds, task, metric config, actuation, pairwise seed sets): `value`, jitted `value_and_grad`, `params_of`, `with_config`; `make_objective` (finds the seeds on the start design for the chosen variant: two-contact, three-contact, or per pair); `Constraints` and `reach_penalty` (SPEC §8.1: a reach bound only); `exact_validation` (fresh seeds on the candidate, exact library coverage, optionally the original KaRMA on the physical task) |
| `dkarma/optimization/optimizers.py` | `adam` (ascent over the continuation schedule: soft solver first, kernels tightening; validation every `k` iterations; result = best by validation), `cmaes` (the `cma` package on the same unconstrained coordinates, fitness batched by `vmap` on the GPU, validation of the incumbent), `hybrid` (Sobol-perturbed starts, Adam refinement, exact selection) |
| `dkarma/optimization/checkpoint.py` | `save_run` writes `config.yaml`, initial and optimized morphology (URDF + robot YAML + design JSON), `actuator_spec.yaml`, `parameter_history.npz`, `objective_history.csv`, `validation_history.csv`, `random_seeds.json`, `seeds.npz`, `constraint_diagnostics.json`, `results_summary.json`; `load_run` |
| `dkarma/metrics/config.py` | `PAIRWISE_KARMA` (mean two-contact score over all finger pairs, each with its own seeds) |
| `dkarma/experiments/optimize_morphology.py` | CLI: `--metric diff|act|mc|full|pairwise --topology 2x4|3x4 --optimizer adam|cmaes|hybrid --objective balanced|translation|rotation --torque ...` |
| `dkarma/tests/test_optimization.py` (4) | §3 |

## 2. What was implemented

The optimizer works on the unconstrained `Phi` of Phase 1 (lengths and base positions
through sigmoid boxes, angles raw), so every iterate is an admissible design; the only
additional constraint is the smooth reach penalty. The smooth objective is the chosen
variant's score minus the penalty; `jax.value_and_grad` of it is one compiled function.
Seeds are found once on the start design and re-projected inside the objective (SPEC
§4.6); validation re-finds seeds on the candidate, so the reported numbers never depend
on the optimizer's seed set. Continuation (`metrics/continuation.py`) runs the first stage
with the soft solver and broad kernels. CMA-ES evaluates its population in one batched
call. Two scores are tracked throughout (training = smooth, validation = exact), and the
returned design is the best by validation (SPEC §8.5).

## 3. Tests and results (CPU, baseline design, tiny configurations)

| Test | Result |
| --- | --- |
| objective value and gradient finite; reach penalty zero inside the bound, positive beyond | pass |
| Adam, 6 iterations over 2 stages, validation every 3: histories complete, selection by the validation criterion | pass |
| CMA-ES 2 generations × 4, hybrid with 2 starts | pass |
| exact validation callback (fresh seeds, exact library) and the full artifact round trip (`save_run` → `load_run`, 11 files) | pass |

`dkarma/tests/test_optimization.py`: 4 passed (212 s). On the GPU an Adam iteration of the
two-finger Diff-KaRMA objective with 8 seeds and 10 commands takes about 1.1 s; the exact
validation with the original KaRMA on the physical task takes 1 to 3 minutes per call, so
`--validate-every` sets the run time.

## 4. Known limitations

- **Matched-budget comparison pending** (Experiment I optimization leg, Phase 7).
- **Seed-set conditioning.** The gradient is conditional on the start design's seed set
  (re-projected); large morphology moves can leave those seeds in poor grasps even when
  the candidate has better ones. The hybrid's multistart and the validation's fresh seeds
  mitigate this; re-seeding every stage is a one-line extension.
- **No L-BFGS stage** yet; Adam with continuation is the only gradient method.
- **CMA-ES bounds.** The unconstrained coordinates are unbounded by construction; CMA-ES
  sees no bound handling and relies on the sigmoid boxes.

## 5. Minimal usage

```bash
~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.optimize_morphology --metric full --topology 3x4 \
    --optimizer adam --iters 60 --validate-every 10 --out workspace/phase7/full_3x4_adam
```

```python
obj = make_objective(topo, params0, bounds, task, DIFF_KARMA)
res = adam(obj, encode(params0, topo, bounds), n_iters=60, validation_fn=exact_validation(obj, "workspace/val"))
save_run("workspace/run", topo, params0, res, config)
```

## 6. Comparison with Phase 5

Phases 1 to 5 produced differentiable and exact evaluators; Phase 6 closes the loop that
turns gradients into designs, with the exact evaluators deciding what is reported.
