# Phase 4 report: quasi-static force and torque feasibility, Act-KaRMA (2026-10-11)

Phase 4 of `SPEC.md` §11. Deliverable: grasp matrix and torque map, the exact convex
force allocation, the smooth feasibility margin inside the rollout, wrench capacity,
the actuator spec loader, and Act-KaRMA as a configuration of the Phase 3 evaluator.
Exit criterion: physical sanity tests pass and the exact and smooth evaluators agree on
feasibility away from the margin. Both hold (§3), with one modelling change the tests
forced (§2.1).

## 1. Files and purpose

| File | Purpose |
| --- | --- |
| `dkarma/core/actuation.py` | `ActuationSpec` (the `actuation` block of `task_spec.yaml`), `ActuationParams` (per-design torque ceilings, link masses, gravity, object wrench), `grasp_matrix`, `contact_torque_map`, `angular_jacobians`, `gravity_torque` |
| `dkarma/solvers/contact_forces.py` | `ForceProblem` (`[f, m]` equilibrium and torque maps), `solve_soft` (Levenberg-Marquardt on smooth residuals, differentiable), `solve_exact` (SLSQP, convex formulation, independent equality rows), `violation` (certificate), `wrench_directions`, `wrench_capacity` |
| `dkarma/solvers/rollout.py` | `RollConfig.actuation` / `force_weights` / `grasp_mode`; the step solves the force problem at the post-projection state and records `force_E`, equilibrium residual, torque usage, cone and torsion margins; `step_energy` adds `w.force * E_F`; `exact_gates` gains a `force` gate |
| `dkarma/metrics/act_karma.py` | `actuation_for`, `evaluate` (Diff-KaRMA evaluator with actuation on), `force_problem_at`, `soft_and_exact_at`, `wrench_capacity_at`, `summarize` |
| `dkarma/metrics/config.py` | `ACT_KARMA` / `FULL_KARMA` carry `RollConfig(actuation=True)` |
| `dkarma/tests/test_actuation.py` (9) | §3 |

## 2. What was implemented

**Model.** Contact lane `c` applies a force `f_c` on the object at the sphere surface
point and a torsional moment `m_c n_c` about the contact normal; `x = [f, m]`.
Equilibrium `G x + w_ext = 0` with `G = [I .. 0 ..; [r_c]x .. n_c ..]`, `r_c = -r_s n_c`;
motor torque `tau = Jc^T f + Jw^T n_c m_c + tau_grav` with `tau_grav = -sum J_com^T m g`
(links as point masses at the capsule midpoints: 28 g per non-terminal link, 10 g terminal,
from the spec); constraints `|f_t| <= mu f_n`, `f_n >= 0`, `|m_c| <= gamma f_n`,
`|tau_j| <= tau_bar_j` with `tau_bar = 0.108 N·m` on every joint (D13, 50 % of the
XL330-M077-T stall) and `w_ext` the 35 g sphere's weight along `-z` (D7, D17).

### 2.1 The modelling change: soft-finger torsion

The spec's pure point contact (no transmitted moment) makes **every** two-finger
pinch infeasible in the exact sense: the weight acts at the sphere centre, the line
through the two contact points misses the centre by `r_s sin(angle between the
normals)`, and the resulting micro-N·m moment about the contact line lies in the null
space of the rank-5 grasp matrix. The first exact solver run reported exactly this
(`1.2e-5 N·m` unresisted). A compliant fingertip transmits that moment through torsional
friction, so the model now includes it as the soft-finger limit `|m_c| <= mu a f_n` with
patch radius `a = 3 mm` (`ForceWeights.torsion_radius_m`; `0` recovers the point
contact). SPEC §5.1 is amended accordingly; the moment capacity about the pinch axis is
reported, not assumed.

**Smooth solve.** `E_F* = min_x |r(x)|^2` with residuals: scaled equilibrium
`(G x + w)/F0`, softplus hinges of `(|tau| - tau_bar)/tau_bar`, `(|f_t| - mu f_n)/F0`,
`-f_n/F0`, `(|m| - gamma f_n)/(gamma F0)`, with `|.|` smoothed; `F0 = max(|w|, 1 N)`.
Solved by 25 Levenberg-Marquardt iterations (Nielsen damping) from the minimum-norm
equilibrium plus an inward preload; differentiable by unrolling. (Damped Gauss-Newton
with a single halving, the first attempt, oscillated on the hinge-squared objective;
LM converges to the minimum BFGS finds.)

**Exact solve.** `min 1/2 |x|^2` subject to the same constraints, SLSQP with analytic
Jacobians on a convex smooth formulation (`mu f_n - sqrt(|f_t|^2 + eps^2) >= 0`, two-sided
linear torque rows). The equality rows are reduced to the independent ones by SVD;
the component of the load outside the range of `G` is reported as `unresisted` and
makes the state infeasible. SLSQP diverged on the dependent 6-row system (the first
attempt); on the reduced system it converges in ~0.1 s per state. `feasible` certifies
`violation <= 1e-5 F0` (10 µN on a 1 N scale).

**Wrench capacity.** `alpha*(d) = max alpha` with `G x + w_0 + alpha d = 0` and the
constraints, over the 12 signed axes plus deterministic random 6-D directions scaled
by `(m g, m g r_s)`; a direction with a component outside the range of `G` gets 0.

**Act-KaRMA.** `RollConfig(actuation=True)` makes every step solve the smooth force
problem at its post-projection state and add `E_F` to the step energy; a trajectory
that cannot be held earns no coverage from that step on. The evaluator, seeds, commands
and coverage are those of Phase 3; `act_karma.summarize` adds max torque usage, cone
and torsion margins, the equilibrium residual, the fraction of force-feasible steps.

## 3. Tests and results (CPU fp64, baseline design, three physical-unit seeds)

| Test | Result |
| --- | --- |
| spec values (servo, 0.215 / 0.108 N·m, masses) | pass |
| gravity torque, planar toy under in-plane gravity | closed form `sum m g x_com`, rtol 1e-10 |
| contact torque, planar toy, 1 N on the fingertip | `tau_1 = -x_c`, `tau_2 = -(x_c - l1)`, rtol 1e-9 |
| grasp matrix structure | internal pinch forces map to zero; moment = `r x f` |
| exact allocation certificate | `G x + w = 0` to 2e-5 N, cones and torque within 1e-5 relative; feasible on the baseline seed grasps under the primary ceiling |
| torque monotonicity | doubling `tau_bar` never lowers `alpha*(d)` (12 directions) |
| friction monotonicity | `mu, gamma` × 1.5 never lowers `alpha*(d)` |
| load behaviour | doubling the object mass never lowers the min-norm allocation's torque usage |
| soft vs exact, loads × 1, 3, 10, 30 | exact feasible ⇒ `E_F < 2e-3` (observed 1e-7..1e-5); violation > 10 % ⇒ `E_F > 2e-3` (observed 1.6e-2..0.4) |
| ideal-actuation limit | `tau_bar × 1e3, mu = 50`: `E_F < 1e-6`, equilibrium residual < 1e-5 N |
| force-energy gradient w.r.t. morphology (6× load) | finite, FD agreement 1e-2 relative on 6 random coordinates |
| Act-KaRMA evaluate | finite score, force diagnostics `[S, U, H, 4]`; ideal actuation ⇒ score ≥ actuated; 50× mass ⇒ score ≤ actuated; kinematic-only evaluation equals the ideal limit within 5 %; objective gradient finite |
| wrench capacity | non-negative, some direction > 0 on a feasible seed; pinch-axis moment capacity is 0 with `gamma = 0` and ≥ that with torsion |

`dkarma/tests/test_actuation.py`: 9 passed (90 s). Regression: `test_rollout` and
`test_diff_karma` (18 non-FD tests) pass after the step changes.

## 4. Known limitations and numerical issues

- **Torsional friction is a modelling assumption** (patch radius 3 mm); it is a
  configuration value, and the exact path reports the unresisted moment when it is
  zero. Hardware testing (SPEC §12 IV) decides the patch radius.
- **Smooth energy scale.** `E_F` is dimensionless and continuous but not a distance to
  feasibility: 20 % load violations give `E_F ≈ 1e-2`, feasible states `≤ 1e-5`. The
  energy weight `EnergyWeights.force` (default 1) sets how strongly it truncates credit.
- **Exact solver precision.** SLSQP attains ~1e-6 relative; the certificate tolerance is
  1e-5. Infeasibility is reported as a violation magnitude, not proven by a dual
  certificate.
- **Quasi-static only** (D5): no joint acceleration, impact, thermal or transmission
  effects; transmission efficiency 1.0 (direct drive).
- **Load direction.** The primary task has gravity along `-z` of the palm frame; the
  evaluator takes any `w_ext`, so other orientations and disturbance wrenches are load
  cases, not code changes.
- **Cost.** One smooth force solve per step adds ~25 LM iterations on a 8- or 12-variable
  problem with forward-mode Jacobians; on the GPU with 8 seeds × 10 commands × 40 steps
  this roughly doubles the evaluation time of Diff-KaRMA.

## 5. Minimal usage

```python
from dkarma.core import build_hand_model, load_task_spec
from dkarma.solvers.seed_projection import find_seeds, assignment
from dkarma.solvers.rollout import GraspState
from dkarma.metrics import act_karma as AK
from dkarma.metrics.config import ACT_KARMA

task = load_task_spec(); sp = task.step_params()
hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
seeds = find_seeds(hm, topo, sp)
d = AK.evaluate(hm, topo, seeds, sp, task, ACT_KARMA, act); print(AK.summarize(d))
soft, exact = AK.soft_and_exact_at(hm, GraspState(seeds.q[0], seeds.p[0], jnp.eye(3)), assignment(topo), sp, act)
alphas, dirs = AK.wrench_capacity_at(hm, GraspState(seeds.q[0], seeds.p[0], jnp.eye(3)), assignment(topo), sp, act)
```

## 6. Comparison with Phase 3

Phase 3 credited any kinematically feasible rolling trajectory. Phase 4 requires, at
every state, that the contacts can hold the loaded object within friction and the
XL330's torque ceiling, with the hand's own weight included, and reports how close to
the limits each trajectory runs. The exact solver is independent of the smooth one and
agrees with it on feasibility away from the margin.

## 7. Remaining work before Phase 5

None blocking. The masked-lane force problem already accepts three contacts.
