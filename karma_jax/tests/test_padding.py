"""Padding-invariance tests for karma_jax (SPEC.md Phase 0).

Two defects were found on 2026-10-10 with the optimizer's two-finger hands
(``optimize/karma_jax_padding_repro.py``) and fixed on 2026-10-11:

1. ``J_pad == n_joints`` gave wrong forward kinematics under ``jit`` whenever
   the ``DeviceModel`` is closed over, which every production kernel does.
   Cause: XLA (jax 0.7.1, CPU and GPU) rewrites a ``set``-scatter that has
   constant indices and as many updates as the operand has elements into a
   plain gather, ignoring duplicate indices. ``apply_mimic`` scattered the
   eight padded mimic lanes (all indexing joint 0) into ``q``, so on an
   8-joint hand (``K_pad == J_pad == 8``) every joint received ``q[0]``.
2. ``M_pad > n_active`` corrupted the Gauss-Newton projection: ``set_active``
   scattered the padded lanes (``q_ids == 0``) into joint 0 together with the
   real lane-0 write, and XLA does not define which duplicate wins.

``apply_mimic``, ``set_active`` and ``fold_active`` are now gather-only and the
rolling QPs mask padded lanes out of the solver's scalars. These tests pin
that (a) the closed-over-constant call pattern is exact at exact padding and
(b) results do not depend on ``J_pad`` or ``M_pad``. They run on CPU
(``JAX_PLATFORMS=cpu``) in about a minute.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from karma.robot import (update_kinematics, fold_mimic_jacobian, capsule_endpoints_world,
                         active_position_indices, active_velocity_indices)
from karma.projection import _project_onto_contacts
from karma.orientation import compute_motion_primitives_frame

from karma_jax.model import load_flat_model
from karma_jax import kinematics as K
from karma_jax import projection as PJ
from karma_jax import rolling as RL
from karma_jax import qp as Q
from karma_jax import steps as S

from .conftest import ROOT, METRIC_CFG, random_configs, load_hand
from .test_qp_oracle import recording, _pad_ck, _pad_active, _params

jax.config.update("jax_enable_x64", True)

# The three optimizer designs frozen as Phase 0 fixtures: 8 joints, two roots,
# no mimics, K_pad == J_pad == 8 (the trigger of defect 1).
DESIGNS = {Path(p).parent.name: p
           for p in sorted(glob.glob(str(ROOT / "dkarma" / "fixtures" / "designs" / "*" / "robot_*.yaml")))}
MIMIC_HANDS = ["inspire", "svh"]   # two mimics share one source on each
PAD_HANDS = ["leap", "inspire"]


def _load(name, **pad):
    if name in DESIGNS:
        return load_flat_model(METRIC_CFG, DESIGNS[name], **pad)
    return load_flat_model(METRIC_CFG, ROOT / "robots" / f"robot_{name}.yaml", **pad)


def _pads(flat):
    """(J_pad, M_pad) variants: exact, the former +1 workaround, generous buckets."""
    nJ, nM = flat.n_joints, flat.n_active
    return [(nJ, nM), (nJ + 1, nM), (nJ + 5, nM + 3)]


def _random_pinch_states(cfg, ctx, n, seed=0):
    """Random joint configurations with the sphere centre between the two seed capsules."""
    rng = np.random.RandomState(seed)
    lo = np.array(ctx.model.lowerPositionLimit)
    hi = np.array(ctx.model.upperPositionLimit)
    cases = []
    for _ in range(n):
        q = lo + rng.rand(ctx.model.nq) * (hi - lo)
        update_kinematics(ctx, q.copy())
        _, b = capsule_endpoints_world(ctx, cfg.seed_thumb_link)
        _, d = capsule_endpoints_world(ctx, cfg.seed_index_link)
        cases.append((q, 0.5 * (b + d) + rng.randn(3) * 1e-3))
    return cases


def test_fixture_designs_present():
    assert set(DESIGNS) == {"baseline", "T", "A1"}, DESIGNS


# ── defect 1: constant-folded mimic scatter ──────────────────────────────────

@pytest.mark.parametrize("name", sorted(DESIGNS) + MIMIC_HANDS + ["leap"])
def test_apply_mimic_closed_over_jit_is_exact(name):
    flat, cfg, ctx = _load(name)
    nJ = flat.n_joints
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        f = jax.jit(lambda q: K.apply_mimic(dm, q))          # model closed over: constant-folded
        for q in random_configs(ctx, 6):
            qp = np.zeros(J_pad)
            qp[:nJ] = q
            expected = flat.A_mimic @ q + flat.b_mimic           # dense form, numpy fp64
            ours = np.asarray(f(jnp.asarray(qp)))
            np.testing.assert_allclose(ours[:nJ], expected, rtol=0, atol=1e-15)
            assert np.all(ours[nJ:] == 0.0)


@pytest.mark.parametrize("name", sorted(DESIGNS) + PAD_HANDS)
def test_fk_closed_over_jit_padding_invariant(name):
    flat, cfg, ctx = _load(name)
    nJ, nF = flat.n_joints, flat.n_frames
    per_pad = []
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        f = jax.jit(lambda q: K.forward_kinematics(dm, q))
        res = []
        for q in random_configs(ctx, 6, seed=7):
            qp = np.zeros(J_pad)
            qp[:nJ] = q
            fk = f(jnp.asarray(qp))
            update_kinematics(ctx, q.copy())
            P_ref = np.stack([np.array(ctx.data.oMf[k].translation) for k in range(nF)])
            O_ref = np.stack([np.array(ctx.data.oMi[k].translation) for k in range(1, ctx.model.njoints)])
            np.testing.assert_allclose(np.asarray(fk.p_frame)[:nF], P_ref, rtol=0, atol=1e-10)
            np.testing.assert_allclose(np.asarray(fk.o_joint)[:nJ], O_ref, rtol=0, atol=1e-10)
            res.append(tuple(np.asarray(a)[:n] for a, n in
                             ((fk.o_joint, nJ), (fk.R_joint, nJ), (fk.w_axis, nJ), (fk.p_frame, nF), (fk.R_frame, nF))))
        per_pad.append(res)
    for res in per_pad[1:]:
        for a, b in zip(per_pad[0], res):
            for x, y in zip(a, b):
                assert np.array_equal(x, y)                    # bit-identical across padding


# ── defect 2: active-lane scatter ────────────────────────────────────────────

@pytest.mark.parametrize("name", ["inspire", "T"])
def test_set_active_ignores_padded_lanes(name):
    flat, cfg, ctx = _load(name)
    nJ, nM = flat.n_joints, flat.n_active
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        f = jax.jit(lambda q, qa: K.set_active(dm, q, qa))
        g = jax.jit(lambda q: K.get_active(dm, q))
        rng = np.random.RandomState(1)
        q = np.zeros(J_pad)
        q[:nJ] = rng.randn(nJ)
        qa = rng.randn(M_pad)                                  # padded lanes hold garbage on purpose
        out = np.asarray(f(jnp.asarray(q), jnp.asarray(qa)))
        exp = q.copy()
        exp[flat.q_ids[:nM]] = qa[:nM]
        assert np.array_equal(out, exp)
        back = np.asarray(g(jnp.asarray(out)))
        assert np.array_equal(back[:nM], qa[:nM]) and np.all(back[nM:] == 0.0)


@pytest.mark.parametrize("name", MIMIC_HANDS + ["leap", "T"])
def test_fold_active_matches_reference_under_padding(name):
    flat, cfg, ctx = _load(name)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    nJ, nM = flat.n_joints, flat.n_active
    rng = np.random.RandomState(3)
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        f = jax.jit(lambda J: K.fold_active(dm, J))
        for _ in range(4):
            J = rng.randn(6, nJ)
            Jp = np.zeros((6, J_pad))
            Jp[:, :nJ] = J
            ref = fold_mimic_jacobian(J, v_ids, ctx)
            ours = np.asarray(f(jnp.asarray(Jp)))
            # same additions in the same order as the reference: bit-identical
            assert np.array_equal(ours[:, :nM], ref)
            assert np.all(ours[:, nM:] == 0.0)


@pytest.mark.parametrize("name", sorted(DESIGNS))
def test_projection_padding_invariant_and_matches_reference(name):
    flat, cfg, ctx = _load(name)
    nJ = flat.n_joints
    cases = _random_pinch_states(cfg, ctx, 100)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    r_s, r_l, eps_g = cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_g_m
    per_pad = []
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        pair = jnp.asarray([flat_p.seed_thumb_cap, flat_p.seed_index_cap])
        f = jax.jit(lambda q, p: PJ.project_onto_contacts(dm, q, p, pair, r_s, r_l, eps_g))
        outs = []
        for q, p in cases:
            qp = np.zeros(J_pad)
            qp[:nJ] = q
            o = f(jnp.asarray(qp), jnp.asarray(p))
            outs.append((np.asarray(o.q)[:nJ], np.asarray(o.dtheta), bool(o.ok)))
        per_pad.append(outs)
    # exact padding vs the reference (this is the repro's second check)
    mism, worst = 0, 0.0
    for (q, p), (qo, _, ok) in zip(cases, per_pad[0]):
        qr, _, okr, _, _ = _project_onto_contacts(ctx, q.copy(), p, (cfg.seed_thumb_link, cfg.seed_index_link),
                                                  q_ids, v_ids, r_s, r_l, eps_g)
        if ok != bool(okr):
            mism += 1
        elif okr:
            worst = max(worst, float(np.abs(qo - qr).max()))
    assert mism == 0, mism
    assert worst < 1e-9, worst
    # padded vs exact: the padded lanes decouple; what remains is the rounding
    # of the dense Gauss-Newton solve on a larger (block-diagonal) matrix.
    for outs in per_pad[1:]:
        for (q0, t0, ok0), (q1, t1, ok1) in zip(per_pad[0], outs):
            assert ok0 == ok1
            np.testing.assert_allclose(q1, q0, rtol=1e-11, atol=1e-12)
            np.testing.assert_allclose(t1, t0, rtol=1e-11, atol=1e-12)
    print(f"{name}: {sum(ok for _, _, ok in per_pad[0])}/{len(cases)} projections ok, "
          f"|q - q_ref| max {worst:.1e}")


# ── QP steps and the full primitive step across padding ──────────────────────

@pytest.mark.parametrize("name", PAD_HANDS)
def test_rolling_and_rotation_steps_padding_invariant(name):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    nM = flat.n_active
    settings = Q.default_settings(jnp.float64)
    per_pad = []
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        roll = jax.jit(lambda ct, ci, dp, qa, lo, hi, p: RL.solve_rolling_step(dm, ct, ci, dp, qa, lo, hi, p, settings))
        rot = jax.jit(lambda ct, ci, dt, qa, lo, hi, p: RL.solve_rotation_step(dm, ct, ci, dt, qa, lo, hi, p, settings))
        outs = []
        for r in rec.rolling:
            o = roll(_pad_ck(r["ck_t"], flat_p), _pad_ck(r["ck_i"], flat_p), jnp.asarray(r["delta_p"]),
                     _pad_active(r["q_active"], M_pad), _pad_active(r["q_lower"], M_pad),
                     _pad_active(r["q_upper"], M_pad), _params(cfg, r["slip_threshold"]))
            outs.append(((bool(o.success), bool(o.qp_ok), bool(o.slip_ok)),
                         np.concatenate([np.asarray(o.dq_active)[:nM], np.asarray(o.dtheta)])))
        for r in rec.rotation:
            o = rot(_pad_ck(r["ck_t"], flat_p), _pad_ck(r["ck_i"], flat_p), jnp.asarray(r["dtheta_target"]),
                    _pad_active(r["q_active"], M_pad), _pad_active(r["q_lower"], M_pad),
                    _pad_active(r["q_upper"], M_pad), _params(cfg, None))
            outs.append(((bool(o.success),),
                         np.concatenate([np.asarray(o.dq_active)[:nM], np.asarray(o.dtheta_achieved),
                                         np.atleast_1d(np.asarray(o.efficiency))])))
        per_pad.append(outs)
    worst = 0.0
    for outs in per_pad[1:]:
        for (f0, x0), (f1, x1) in zip(per_pad[0], outs):
            assert f0 == f1
            worst = max(worst, float(np.abs(x1 - x0).max()))
    # The padded lanes are masked out of the IPM; what remains is the summation
    # order of the dense Cholesky on a larger matrix.
    assert worst < 1e-9, worst
    print(f"{name}: {len(rec.rolling)} rolling + {len(rec.rotation)} rotation steps, "
          f"max |x_padded - x_exact| = {worst:.1e}")


def test_primitive_step_padding_invariant():
    name = "leap"
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    nJ = flat.n_joints
    slip = cfg.rotation_slip_threshold_mm / 1000.0 if cfg.rotation_slip_threshold_mm is not None else None
    params = _params(cfg, slip)
    n_sub = max(1, int(np.ceil(cfg.voxel_size_m / cfg.step_size_m)))
    prims = compute_motion_primitives_frame(rec.R_seed, cfg.voxel_size_m)
    nodes = list(rec.discovered.values())[:48]
    per_pad = []
    for J_pad, M_pad in _pads(flat):
        flat_p, _, _ = _load(name, J_pad=J_pad, M_pad=M_pad)
        dm = flat_p.to_device(jnp.float64)
        step = S.make_batched_step(dm, n_sub, params, Q.default_settings(jnp.float64))
        names = flat_p.capsule_names
        qs, ps, Rs, pairs, dps = [], [], [], [], []
        for k, node in enumerate(nodes):
            q = np.zeros(J_pad)
            q[:nJ] = node.q
            qs.append(q); ps.append(node.p); Rs.append(node.R_sphere)
            pairs.append([names.index(node.contact_pair[0]), names.index(node.contact_pair[1])])
            dps.append(prims[k % len(prims)])
        state = S.GraspState(jnp.asarray(np.stack(qs)), jnp.asarray(np.stack(ps)), jnp.asarray(np.stack(Rs)),
                             jnp.asarray(np.array(pairs, np.int32)))
        out = step(state, jnp.asarray(np.stack(dps)))
        per_pad.append((np.asarray(out.ok), np.asarray(out.code), np.asarray(out.state.q)[:, :nJ],
                        np.asarray(out.state.p), np.asarray(out.state.R), np.asarray(out.state.pair)))
    ok0, code0, q0, p0, R0, pair0 = per_pad[0]
    worst = 0.0
    for ok, code, q, p, R, pair in per_pad[1:]:
        assert np.array_equal(ok, ok0) and np.array_equal(code, code0) and np.array_equal(pair, pair0)
        assert np.array_equal(p, p0)
        worst = max(worst, float(np.abs(q[ok0] - q0[ok0]).max()), float(np.abs(R[ok0] - R0[ok0]).max()))
    assert worst < 1e-9, worst
    print(f"{name}: {len(nodes)} primitive steps ({int(ok0.sum())} ok), max |state_padded - state_exact| = {worst:.1e}")
