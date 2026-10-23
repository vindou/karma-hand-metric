# Phase 2 report: differentiable rolling rollout (2026-10-11)

Phase 2 of `SPEC.md` §11. Deliverable: `dkarma.solvers` with a softly constrained
rollout and the hard-constrained rollout through a QP-derivative strategy, a
`lax.scan` over the horizon, and the exact validator built on `karma_jax.steps`.
Exit criterion: gradient tests pass on the planar toy and the 4+4 family, and
rollouts validated by the exact gates agree with reference sub-steps on recorded
states. Both hold (§3).

## 1. Files and purpose

| File | Purpose |
| --- | --- |
| `dkarma/solvers/rolling_qp.py` | the step problem for `Cn` masked contacts (`assemble`), the soft closed-form solve (`solve_soft`), the hard solve (`solve_hard`: reference IPM value + KKT re-solve derivative), `kkt_resolve`, `dtheta_achieved` |
| `dkarma/solvers/projection.py` | masked-lane Gauss-Newton projection onto the contact manifold, equal to `karma_jax.projection` for two lanes, with an optional backtracking-free variant |
| `dkarma/solvers/rollout.py` | `step` (contact pass, problem, solve, joint/sphere update, projection, smooth residuals), `rollout` (`lax.scan`), `StepDiag`, `energies` (per-step energies, prefix log-weights, smooth worst step), `exact_gates` |
| `dkarma/solvers/exact.py` | `validate_translation` / `validate_rotation`: the reference primitive and rotation steps (`karma_jax.steps`) over the same commands, with fail codes and the pair the reference used |
| `dkarma/tests/refstates.py` | reference seed and BFS states of the Phase 0 designs from the optimizer's official result pickles, with the reference's scaled step parameters |
| `dkarma/tests/test_rolling_qp.py` (4), `test_projection.py` (4), `test_rollout.py` (18) | see §3 |

Not touched: `karma/`, `karma_jax/`, `optimize/`.

## 2. What was implemented

**Step problem.** Variables `z = [dq, dtheta]` (translation steps) or `z = dq`
(rotation steps, `dtheta` fixed to the command), rolling rows
`(J_surf,c dq - r_s [n_c]x dtheta)/r_s = dp/r_s` per contact lane, gap band on
`n_c . Jv_c dq`, joint and rotation boxes, objective `|rows|^2 + reg/r_s^2 |dq|^2 +
1e5 |dtheta - theta_c|^2`. For two lanes this is the reference's translation QP
and rotation QP element for element (test: `rtol 1e-12`). Lanes are masked, so
the same assembly serves three contacts in Phase 5.

**Soft solve.** `z = -P^-1 c` by Cholesky, `dq` clipped to the joint box. Every
infeasibility (gap band, slip, clearances, limits, friction cone) is measured
after the projection and enters the energies; nothing is gated.

**Hard solve.** The reference IPM (`karma_jax.qp.solve_box_qp`) under
`stop_gradient` gives the value. The derivative is that of the
equality-constrained QP on the active set identified from the IPM slacks
(tolerance `1e-7 (1 + |h|)` in fp64), computed as one fixed-shape KKT solve
where inactive rows carry `lambda_i = 0`. The returned `x` is
`x_qp + (x_kkt - stop_grad(x_kkt))`: exact value, piecewise-exact gradient. The
`kkt_gap = |x_kkt - x_qp|` diagnostic is ~1e-7 at the reference tolerance and
~1e-9 with a tight solve.

**Rollout.** `step` = FK, fixed-assignment contact pass, assemble, solve,
`q += dq` (clipped, violation recorded), `p += dp`, `R <- exp(dtheta) R`,
projection (10 damped Gauss-Newton iterations, backtracking optional),
`R <- exp(dtheta_proj) R`, then the residuals: max tangential slip (m), spin
error, gap before and after projection, pre-clip joint-limit violation, soft and
exact object and self clearance, friction-cone margin, rotation efficiency, QP
success, projection success. `rollout` scans `H` steps. `energies` maps each
step to `e_k = sum_i w_i (excess_i / tolerance_i)^2` (slip over the gate, gap
over `eps_g`, clearance under `eps_col`, limit violation over the step cap, cone
margin, a fixed penalty for QP failure, `(1 - efficiency)^2`), the prefix
log-weights `log W_k = -sum_{j<=k} e_j / sigma` (soft truncation), the smooth
worst step, and the total.

**Exact validator.** The commands of a smooth rollout, chunked into primitives of
`n_sub` sub-steps, are fed to the reference's `attempt_primitive_step`
(translation) or `rotation_step` (rotation) in a `lax.scan`; the outcome holds
per-primitive `ok`, fail codes, states and the contact pair the reference
chose. Trajectories are credited only up to the first failed primitive.

## 3. Tests and results

All CPU fp64. Reference states come from the optimizer's official results for the
designs T (reference backend) and baseline (reference backend), evaluated with
the reference's `L_ref`-scaled parameters so the states are valid grasps.

| Test | What it checks | Result |
| --- | --- | --- |
| assembly equality | `P, c, A, l, u, M, b, theta_c` of the translation problem and `P, c, A, l, u, b` of the rotation problem equal `karma_jax.rolling` on 4 recorded states × 2 designs | `rtol 1e-12` |
| hard solve vs reference step | `dq, dtheta, slip` equal `solve_rolling_step` / `solve_rotation_step` (same IPM) | `1e-12`; `kkt_gap < 1e-6` on the compared states |
| KKT gradient vs finite differences | `d(w . x*)/d(q, p)` at states whose active set is unchanged under the FD perturbation (checked), tight IPM for the FD | rel 1e-3 (observed ≤ 2e-4) |
| soft solve gradient | vs FD | rel 1e-5 |
| projection equality | after a 0.02 rad perturbation off the manifold: `ok`, `q` (1e-11), `dtheta` (1e-10) equal `project_onto_contacts` on 5 states × 2 designs | pass |
| projection gradient (no backtracking) | vs FD | rel 1e-5 |
| scan = loop, jit = eager | soft and hard | 1e-13 / 1e-11 |
| hard step = reference sub-step | one step equals `attempt_primitive_step(n_sub=1)` state (`q` 1e-10, `p` 1e-15, `R` 1e-10) where the reference keeps the pair; the smooth step's own exact gates pass there | pass |
| exact validator vs smooth gates | six seed-frame primitives from the seed (8 sub-steps each): the reference verdict equals the conjunction of the smooth step's exact gates whenever the reference keeps the pair | all compared primitives agree (≥ 2 per design) |
| rotation steps vs reference | three 0.15 rad tilts: `q, R` equal the reference rotation step while it succeeds and keeps the pair | 1e-10 |
| rollout gradients w.r.t. morphology | 4-step rollout: `d(final q . w)/d phi` and `d(total energy)/d phi` vs FD; soft mode rel 1e-4; hard mode ≥ 80 % of coordinates within 1e-3 of scale (one-sided at kinks) | pass (T, baseline) |
| vmap over 6 commands, prefix weights | batched = single; `log W_k` non-increasing; worst finite | pass |

Totals: `test_rolling_qp` 4, `test_projection` 4, `test_rollout` 18, all passing;
the two FD-heavy gradient tests take about 11 minutes on CPU. The full
`dkarma/tests` suite is 139 passed, 6 skipped.

## 4. Known limitations and numerical issues

- **IPM solution accuracy limits finite differences, not gradients.** At the
  reference tolerance (`tol 1e-11` on residuals) the QP solution carries ~2e-7
  error in `x` because the `1e5` rotation prior conditions the Newton system;
  FD through it with `h = 1e-6` is noisy at the 1e-2 level. The KKT derivative
  is unaffected (autodiff of the QP path and of the KKT path agree to the last
  bit; FD with a tight solve agrees to 3 decimals). Gradient tests use the
  tight settings; the rollout keeps the reference settings for value parity.
- **Active-set kinks.** The hard solve's gradient is one-sided where a
  constraint enters or leaves the active set; the rollout gradient test
  tolerates up to 20 % of coordinates straddling such a kink over 4 steps.
  Continuation (SPEC §8.4) should start in soft mode.
- **Fixed assignment vs the reference's pair re-selection.** The exact path
  re-selects the nearest link with hysteresis; the smooth path keeps the seed
  assignment. Where the reference switches links the two legitimately diverge;
  the validator reports the pair so the metric can treat such trajectories by
  candidate assignments (SPEC §3.3 mode 2) in Phase 3 if needed.
- **Projection non-smoothness.** The three-candidate backtracking is a discrete
  choice; `proj_backtrack=False` gives a smooth map at the cost of occasional
  slower convergence. The `converged` freeze is a `where`, so a trajectory that
  converges in fewer iterations has a flat (zero) derivative from the frozen
  iterations, which is the correct derivative of the fixed-point map only to
  solver tolerance.
- **Energies are surrogates.** Their scale (`sigma`, the weights) is a
  continuation knob, not a physical quantity; exact feasibility comes only from
  `dkarma.solvers.exact`.
- **Cost.** A hard step is one IPM solve (≈ 30 Newton iterations on a 11×11
  system) plus 10 projection iterations with 3 FK evaluations each; on CPU a
  40-step rollout takes a few hundred milliseconds eager and tens of
  milliseconds jitted. GPU batching over seeds and commands is Phase 3.

## 5. Minimal usage

```python
import jax.numpy as jnp
from dkarma.core import build_hand_model, load_task_spec
from dkarma.core.contacts import ContactAssignment
from dkarma.solvers.rollout import GraspState, RollConfig, EnergyWeights, rollout, energies
from dkarma.solvers.exact import validate_translation

task = load_task_spec(); sp = task.step_params()
hm = build_hand_model(topo, params)                       # Phase 1
asg = ContactAssignment(caps=jnp.asarray(topo.seed_caps(), jnp.int32), valid=jnp.array([True, True]))
state0 = GraspState(q_seed, p_seed, jnp.eye(3))
dps = jnp.broadcast_to(task.rolling.step_size_m * jnp.array([1.0, 0, 0]), (40, 3)); dths = jnp.zeros((40, 3))
states, diags = rollout(hm, state0, dps, dths, asg, sp, RollConfig(mode="hard"))
E = energies(diags, sp, EnergyWeights())                  # E.prefix_logw are the per-state credits
exact = validate_translation(hm, state0, dps, asg, sp, n_sub=task.n_sub)   # exact.ok per 10 mm primitive
```

## 6. Comparison with Phase 1

Phase 1 differentiated static geometry. Phase 2 differentiates motion: a
40-step rolling trajectory under the reference's own QP and projection, with
exact parity to the reference sub-step where the contact pair is unchanged, and
with validated gradients through the morphology in both soft and hard modes.
The exact validator ties every smooth trajectory back to the reference's
verdict.

## 7. Remaining work before Phase 3

None blocking. Phase 3 adds seeds in physical units, the command library, the
smooth coverage objective on top of `energies`, and the surrogate-validity
experiments; its foundations (orientation grids, coverage kernels, command
library, the physical rolling tolerances in `task_spec.yaml`) are already in
`dkarma/metrics/` with tests.
