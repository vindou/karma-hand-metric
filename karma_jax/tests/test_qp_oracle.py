"""Oracle tests for the batched QP and the rolling/rotation step builders vs OSQP."""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
import osqp
from scipy import sparse

from karma_jax import qp as Q
from karma_jax import rolling as RL
from karma_jax.geometry import ContactKin

from .conftest import load_hand
from ._record import record

jax.config.update("jax_enable_x64", True)

RECORD_HANDS = ["leap", "inspire", "shadowhand"]
_recordings: dict[str, object] = {}


def recording(name):
    if name not in _recordings:
        flat, cfg, ctx = load_hand(name)
        _recordings[name] = record(cfg, ctx)
    return _recordings[name]


def _osqp_solve(P, c, A, l, u):
    s = osqp.OSQP()
    s.setup(sparse.csc_matrix(np.triu(P)), c, sparse.csc_matrix(A), l, u,
            eps_abs=1e-7, eps_rel=1e-7, max_iter=4000, verbose=False, polish=False)
    r = s.solve()
    return r.x, r.info.status_val in (1, 2), r.info.status


def _objective(P, c, x):
    return 0.5 * x @ P @ x + c @ x


def _osqp_tight(P, c, A, l, u):
    s = osqp.OSQP()
    s.setup(sparse.csc_matrix(np.triu(P)), c, sparse.csc_matrix(A), l, u,
            eps_abs=1e-13, eps_rel=1e-13, max_iter=400000, verbose=False, polishing=False)
    return s.solve().x


def _assert_at_least_as_optimal(P, c, A, l, u, x, x_ref):
    """Ours must be feasible and no worse than the reference's answer.

    OSQP's 1e-7 feasibility tolerance lets the reference's x sit slightly
    outside an active bound and thereby undercut the true optimum; when that
    happens the comparison is made against a tight OSQP solve instead.
    Returns True when the reference answer was infeasible.
    """
    obj = lambda v: 0.5 * v @ P @ v + c @ v  # noqa: E731
    assert np.all(A @ x >= l - 1e-7) and np.all(A @ x <= u + 1e-7)
    viol_ref = max(0.0, float(np.max(l - A @ x_ref)), float(np.max(A @ x_ref - u)))
    target = x_ref if viol_ref <= 1e-9 else _osqp_tight(P, c, A, l, u)
    assert obj(x) <= obj(target) + 1e-9 * (1 + abs(obj(target))), (obj(x) - obj(target), viol_ref)
    return viol_ref > 1e-9


def test_box_qp_random_vs_osqp():
    rng = np.random.RandomState(0)
    f = jax.jit(lambda P, c, A, l, u: Q.solve_box_qp(P, c, A, l, u, Q.default_settings(jnp.float64)))
    n_osqp_stall = 0
    for k in range(200):
        n = rng.randint(3, 14)
        M = rng.randn(6, n)
        P = M.T @ M + 1e-2 * np.eye(n)
        if rng.rand() < 0.5:   # rolling-like: big penalty block on the last 3 vars
            P[-3:, -3:] += 1e5 * np.eye(3)
        c = rng.randn(n) * rng.choice([0.1, 1.0, 100.0])
        A_gap = rng.randn(2, n)
        A = np.vstack([A_gap, np.eye(n)])
        mid = rng.randn(n + 2) * 0.1
        half = np.abs(rng.randn(n + 2)) * 0.2 + 1e-3
        l, u = mid - half, mid + half
        x_ref, ok_ref, status = _osqp_solve(P, c, A, l, u)
        res = f(jnp.asarray(P), jnp.asarray(c), jnp.asarray(A), jnp.asarray(l), jnp.asarray(u))
        x = np.asarray(res.x)
        if ok_ref:
            assert bool(res.success), (k, res.prim_res, res.dual_res)
            assert np.all(A @ x >= l - 1e-6) and np.all(A @ x <= u + 1e-6)
            # ours should be at least as optimal as OSQP's 1e-7 solution (up to its tolerance)
            assert _objective(P, c, x) <= _objective(P, c, x_ref) + 1e-5 * (1 + abs(_objective(P, c, x_ref)))
        elif bool(res.success):
            # OSQP (ADMM, 4000 iterations) can stall on these synthetic problems
            # with a 1e5 penalty block; accept our answer only if it is verifiably
            # feasible and stationary.
            assert np.all(A @ x >= l - 1e-6) and np.all(A @ x <= u + 1e-6)
            assert float(res.prim_res) < 1e-8 and float(res.dual_res) < 1e-8, (k, status)
            n_osqp_stall += 1
    print(f"OSQP stalled on {n_osqp_stall} synthetic problems where the IPM converged")


def test_box_qp_detects_infeasible():
    f = jax.jit(lambda P, c, A, l, u: Q.solve_box_qp(P, c, A, l, u, Q.default_settings(jnp.float64)))
    n = 5
    P = np.eye(n)
    c = np.zeros(n)
    A = np.vstack([np.ones((1, n)), np.eye(n)])
    l = np.concatenate([[10.0], -np.ones(n)])   # sum(x) >= 10 but |x_i| <= 1 -> infeasible
    u = np.concatenate([[11.0], np.ones(n)])
    res = f(jnp.asarray(P), jnp.asarray(c), jnp.asarray(A), jnp.asarray(l), jnp.asarray(u))
    assert not bool(res.success)


def _pad_ck(d, flat, M=None):
    """Reference ContactKinematics dict -> ContactKin with J_pad columns."""
    J = flat.J_pad

    def pad(Jm):
        out = np.zeros((3, J))
        out[:, : Jm.shape[1]] = Jm
        return jnp.asarray(out)

    return ContactKin(
        cap=jnp.asarray(flat.capsule_index(d["name"])), axis_point=jnp.asarray(d["axis_point"]),
        sphere_point=jnp.asarray(d["sphere_point"]), normal=jnp.asarray(d["normal"]),
        g=jnp.asarray(d["g"]), Jv=pad(d["Jv"]), Jw=pad(d["Jw"]),
        origin=jnp.asarray(d["origin"]), Jv_origin=pad(d["Jv_origin"]),
    )


def _params(cfg, slip):
    return RL.StepParams(
        sphere_radius=cfg.sphere_radius_m, link_radius=cfg.link_radius_m, eps_g=cfg.eps_g_m,
        max_joint_step=cfg.max_joint_step_rad, qp_regularization=cfg.qp_regularization,
        slip_threshold=np.inf if slip is None else slip, friction_coeff=cfg.friction_coeff,
        eps_col=cfg.eps_col_m,
    )


def _pad_active(v, M_pad, fill=0.0):
    out = np.full(M_pad, fill)
    out[: len(v)] = v
    return jnp.asarray(out)


@pytest.mark.parametrize("name", RECORD_HANDS)
def test_rolling_step_matches_reference(name):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    assert len(rec.rolling) > 5   # low-DOF hands (Inspire) record only a handful
    dm = flat.to_device(jnp.float64)
    settings = Q.default_settings(jnp.float64)
    f = jax.jit(lambda ct, ci, dp, qa, lo, hi, p: RL.solve_rolling_step(dm, ct, ci, dp, qa, lo, hi, p, settings))
    build = jax.jit(lambda ct, ci, dp, qa, lo, hi, p: RL.build_rolling_qp(dm, ct, ci, dp, qa, lo, hi, p))
    m = flat.n_active
    n_ok = n_fail = n_success_disagree = n_ref_infeasible = n_osqp_stall = 0
    dq_diff = []
    for r in rec.rolling:
        ref = r["result"]
        args = (_pad_ck(r["ck_t"], flat), _pad_ck(r["ck_i"], flat), jnp.asarray(r["delta_p"]),
                _pad_active(r["q_active"], flat.M_pad), _pad_active(r["q_lower"], flat.M_pad),
                _pad_active(r["q_upper"], flat.M_pad), _params(cfg, r["slip_threshold"]))
        out = f(*args)
        qp_ok_ref = not ref.reason.startswith("qp_")
        if not qp_ok_ref:
            if ref.reason == "qp_maximum iterations reached" and bool(out.qp_ok):
                # OSQP stalled on a solvable problem; ours must then be feasible.
                qp = build(*args)
                A, l, u = (np.asarray(v) for v in (qp.A, qp.l, qp.u))
                x = np.concatenate([np.asarray(out.dq_active), np.asarray(out.dtheta)])
                assert np.all(A @ x >= l - 1e-7) and np.all(A @ x <= u + 1e-7)
                n_osqp_stall += 1
                continue
            assert not bool(out.qp_ok), (ref.reason, float(out.roll_residual))
            n_fail += 1
            continue
        assert bool(out.qp_ok), (ref.reason, float(out.roll_residual))
        n_ok += 1
        # The reference's OSQP terminates on a relative criterion scaled by the
        # 1e5 rotation penalty, so its x is only loosely optimal (up to ~1e-2 in
        # flat directions). Compare optimality on the identical assembled QP
        # instead of element-wise: ours must be feasible and at least as good.
        qp = build(*args)
        P, c, A, l, u = (np.asarray(v) for v in (qp.P, qp.c, qp.A, qp.l, qp.u))
        x = np.concatenate([np.asarray(out.dq_active), np.asarray(out.dtheta)])
        x_ref = np.concatenate([ref.delta_q_active, np.zeros(flat.M_pad - m), ref.delta_theta])
        n_ref_infeasible += _assert_at_least_as_optimal(P, c, A, l, u, x, x_ref)
        dq_diff.append(np.abs(x[:m] - ref.delta_q_active).max())
        if bool(out.success) != ref.success:
            n_success_disagree += 1
    # Success (the slip gate) may flip on steps the reference solved sloppily.
    assert n_success_disagree <= 0.02 * n_ok + 1, (n_success_disagree, n_ok)
    print(f"{name}: {n_ok} solved, {n_fail} qp failures matched, OSQP stalls solved by IPM "
          f"{n_osqp_stall}, success disagreements {n_success_disagree}, reference answers "
          f"infeasible {n_ref_infeasible}, |dq - dq_ref| median {np.median(dq_diff):.1e} "
          f"max {np.max(dq_diff):.1e}")


@pytest.mark.parametrize("name", RECORD_HANDS)
def test_rotation_step_matches_reference(name):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    assert len(rec.rotation) > 3
    dm = flat.to_device(jnp.float64)
    settings = Q.default_settings(jnp.float64)
    f = jax.jit(lambda ct, ci, dt, qa, lo, hi, p: RL.solve_rotation_step(dm, ct, ci, dt, qa, lo, hi, p, settings))
    build = jax.jit(lambda ct, ci, dt, qa, lo, hi, p: RL.build_rotation_qp(dm, ct, ci, dt, qa, lo, hi, p))
    m = flat.n_active
    dq_diff, eff_diff = [], []
    for r in rec.rotation:
        ref = r["result"]
        args = (_pad_ck(r["ck_t"], flat), _pad_ck(r["ck_i"], flat), jnp.asarray(r["dtheta_target"]),
                _pad_active(r["q_active"], flat.M_pad), _pad_active(r["q_lower"], flat.M_pad),
                _pad_active(r["q_upper"], flat.M_pad), _params(cfg, None))
        out = f(*args)
        assert bool(out.success) == ref.success, ref.reason
        if not ref.success:
            continue
        # OSQP's relative tolerance leaves the reference's dq ~1e-3 loose in flat
        # directions here too; compare optimality on the identical QP instead.
        qp = build(*args)
        P, c, A, l, u = (np.asarray(v) for v in (qp.P, qp.c, qp.A, qp.l, qp.u))
        x = np.asarray(out.dq_active)
        x_ref = np.concatenate([ref.delta_q_active, np.zeros(flat.M_pad - m)])
        _assert_at_least_as_optimal(P, c, A, l, u, x, x_ref)
        # Quantities derived from M @ dq are insensitive to the flat directions.
        np.testing.assert_allclose(np.asarray(out.dtheta_achieved), ref.delta_theta_achieved, atol=2e-3)
        np.testing.assert_allclose(float(out.efficiency), ref.efficiency, atol=2e-3)
        np.testing.assert_allclose(float(out.tangential_slip_mm), ref.tangential_slip, atol=5e-3)
        np.testing.assert_allclose(float(out.spin_achieved), ref.spin_achieved, atol=2e-3)
        dq_diff.append(np.abs(x[:m] - ref.delta_q_active).max())
        eff_diff.append(abs(float(out.efficiency) - ref.efficiency))
    print(f"{name}: {len(dq_diff)} rotation steps, |dq - dq_ref| median {np.median(dq_diff):.1e} "
          f"max {np.max(dq_diff):.1e}, |eff - eff_ref| max {np.max(eff_diff):.1e}")
