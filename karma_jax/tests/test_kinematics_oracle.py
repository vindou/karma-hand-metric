"""Oracle tests: karma_jax kinematics/geometry vs Pinocchio and the reference karma.*.

Run from the repo root with the JAX env:
    ../.venv/bin/python -m pytest karma_jax/tests -q
"""
from __future__ import annotations

import numpy as np
import pinocchio as pin
import pytest

import jax
import jax.numpy as jnp

from karma.robot import update_kinematics, capsule_endpoints_world, fold_mimic_jacobian
from karma.contacts import contact_kinematics as ref_contact_kinematics
from karma.contacts import contact_gap_only, select_contact_pair as ref_select_pair
from karma import collisions as ref_col
from karma.grasp import check_antipodal_friction as ref_grasp

from karma_jax import kinematics as K
from karma_jax import geometry as G

from .conftest import random_configs

jax.config.update("jax_enable_x64", True)

N_CFG = 12


def _dev(flat, dtype):
    return flat.to_device(dtype)


def _tol(dtype):
    return dict(rtol=0, atol=1e-10) if dtype == jnp.float64 else dict(rtol=0, atol=2e-6)


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
def test_forward_kinematics_matches_pinocchio(hand, dtype):
    flat, cfg, ctx = hand
    dm = _dev(flat, dtype)
    fk_fn = jax.jit(K.forward_kinematics)
    for q in random_configs(ctx, N_CFG):
        update_kinematics(ctx, q)  # applies mimics in place; our FK applies them internally
        fk = fk_fn(dm, jnp.asarray(q, dtype))
        P_ref = np.stack([np.array(ctx.data.oMf[f].translation) for f in range(ctx.model.nframes)])
        R_ref = np.stack([np.array(ctx.data.oMf[f].rotation) for f in range(ctx.model.nframes)])
        np.testing.assert_allclose(np.asarray(fk.p_frame)[: flat.n_frames], P_ref, **_tol(dtype))
        np.testing.assert_allclose(np.asarray(fk.R_frame)[: flat.n_frames], R_ref, **_tol(dtype))
        # joint origins/rotations vs oMi
        O_ref = np.stack([np.array(ctx.data.oMi[j].translation) for j in range(1, ctx.model.njoints)])
        np.testing.assert_allclose(np.asarray(fk.o_joint)[: flat.n_joints], O_ref, **_tol(dtype))


def test_frame_jacobians_match_pinocchio(hand):
    flat, cfg, ctx = hand
    dm = _dev(flat, jnp.float64)
    fk_fn = jax.jit(K.forward_kinematics)
    jac_fn = jax.jit(K.frame_jacobian)
    frames = sorted({int(c) for c in flat.cap_frame[: flat.n_capsules]}
                    | {int(c) for c in flat.cap_distal[: flat.n_capsules] if c >= 0})
    for q in random_configs(ctx, N_CFG, seed=1):
        update_kinematics(ctx, q)
        fk = fk_fn(dm, jnp.asarray(q))
        for f in frames:
            Jref = pin.getFrameJacobian(ctx.model, ctx.data, f, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
            Jv, Jw = jac_fn(dm, fk, f)
            np.testing.assert_allclose(np.asarray(Jv)[:, : flat.n_joints], Jref[:3], atol=1e-10)
            np.testing.assert_allclose(np.asarray(Jw)[:, : flat.n_joints], Jref[3:], atol=1e-10)


def test_fold_active_matches_reference(hand):
    flat, cfg, ctx = hand
    dm = _dev(flat, jnp.float64)
    v_ids = flat.q_ids[: flat.n_active]
    rng = np.random.RandomState(3)
    J = rng.randn(3, flat.n_joints)
    ref = fold_mimic_jacobian(J, v_ids, ctx)
    ours = np.asarray(K.fold_active(dm, jnp.asarray(J)))
    np.testing.assert_allclose(ours[:, : flat.n_active], ref, atol=1e-12)


def test_capsule_endpoints_match_reference(hand):
    flat, cfg, ctx = hand
    dm = _dev(flat, jnp.float64)
    fk_fn = jax.jit(K.forward_kinematics)
    for q in random_configs(ctx, N_CFG, seed=2):
        update_kinematics(ctx, q)
        a, b = G.capsule_endpoints(dm, fk_fn(dm, jnp.asarray(q)))
        for c, name in enumerate(flat.capsule_names):
            ar, br = capsule_endpoints_world(ctx, name)
            np.testing.assert_allclose(np.asarray(a[c]), ar, atol=1e-10)
            np.testing.assert_allclose(np.asarray(b[c]), br, atol=1e-10)


def _sphere_centres(ctx, flat, q, rng, n=6):
    """Centres near random capsules so that interior and endpoint regimes both occur."""
    update_kinematics(ctx, q)
    out = []
    for _ in range(n):
        name = flat.capsule_names[rng.randint(flat.n_capsules)]
        a, b = capsule_endpoints_world(ctx, name)
        t = rng.uniform(-0.3, 1.3)
        p = a + t * (b - a) + rng.randn(3) * 0.02
        out.append(p)
    return out


def test_capsule_pass_gaps_and_contacts_match_reference(hand):
    flat, cfg, ctx = hand
    dm = _dev(flat, jnp.float64)
    r_s, r_l = cfg.sphere_radius_m, cfg.link_radius_m
    fk_fn = jax.jit(K.forward_kinematics)
    pass_fn = jax.jit(G.capsule_pass)
    ck_fn = jax.jit(G.contact_kinematics)
    rng = np.random.RandomState(4)
    for q in random_configs(ctx, 6, seed=5):
        for centre in _sphere_centres(ctx, flat, q, rng):
            update_kinematics(ctx, q)
            fk = fk_fn(dm, jnp.asarray(q))
            cp = pass_fn(dm, fk, jnp.asarray(centre), r_s, r_l)
            for c, name in enumerate(flat.capsule_names):
                g_ref = contact_gap_only(ctx, centre, name, r_s, r_l)
                np.testing.assert_allclose(float(cp.gap[c]), g_ref, atol=1e-10)
                ck_ref = ref_contact_kinematics(ctx, q, centre, name, r_s, r_l)
                ck = ck_fn(dm, fk, cp, c, jnp.asarray(centre), r_s)
                nJ = flat.n_joints
                np.testing.assert_allclose(np.asarray(ck.axis_point), ck_ref.axis_point_world, atol=1e-10)
                np.testing.assert_allclose(np.asarray(ck.sphere_point), ck_ref.sphere_point_world, atol=1e-10)
                np.testing.assert_allclose(np.asarray(ck.normal), ck_ref.normal_world, atol=1e-10)
                np.testing.assert_allclose(float(ck.g), ck_ref.g_m, atol=1e-10)
                np.testing.assert_allclose(np.asarray(ck.Jv)[:, :nJ], ck_ref.Jv_world, atol=1e-9)
                np.testing.assert_allclose(np.asarray(ck.Jw)[:, :nJ], ck_ref.Jw_world, atol=1e-10)
                np.testing.assert_allclose(np.asarray(ck.origin), ck_ref.origin_world, atol=1e-10)
                np.testing.assert_allclose(np.asarray(ck.Jv_origin)[:, :nJ], ck_ref.Jv_origin_world, atol=1e-10)


def test_pair_selection_and_checks_match_reference(hand):
    flat, cfg, ctx = hand
    dm = _dev(flat, jnp.float64)
    r_s, r_l, eps_g, eps_col = cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_g_m, cfg.eps_col_m
    fk_fn = jax.jit(K.forward_kinematics)
    pass_fn = jax.jit(G.capsule_pass)
    sel_fn = jax.jit(G.select_contact_pair)
    col_fn = jax.jit(G.check_collisions)
    self_fn = jax.jit(G.check_self_collisions)
    rng = np.random.RandomState(6)
    names = flat.capsule_names
    for q in random_configs(ctx, 8, seed=7):
        for centre in _sphere_centres(ctx, flat, q, rng, n=4):
            update_kinematics(ctx, q)
            fk = fk_fn(dm, jnp.asarray(q))
            cp = pass_fn(dm, fk, jnp.asarray(centre), r_s, r_l)
            # pair selection with and without hysteresis, from a random current pair
            cur = (cfg.thumb_contact_links[rng.randint(len(cfg.thumb_contact_links))],
                   cfg.index_contact_links[rng.randint(len(cfg.index_contact_links))])
            cur_idx = jnp.asarray([names.index(cur[0]), names.index(cur[1])])
            for hyst in (0.0, 0.5 * eps_g, 5.0 * eps_g):
                ref_pair = ref_select_pair(ctx, q, centre, cfg.thumb_contact_links,
                                           cfg.index_contact_links, r_s, r_l,
                                           current_pair=cur, hysteresis_m=hyst)
                ours = sel_fn(dm, cp, cur_idx, hyst)
                assert (names[int(ours[0])], names[int(ours[1])]) == ref_pair
            # collision checks with the reference-selected pair
            pair = ref_select_pair(ctx, q, centre, cfg.thumb_contact_links,
                                   cfg.index_contact_links, r_s, r_l)
            pair_idx = jnp.asarray([names.index(pair[0]), names.index(pair[1])])
            ok_ref, mc_ref = ref_col.check_collisions(ctx, q, centre, pair, r_s, r_l, eps_col)
            ok, mc = col_fn(dm, cp, pair_idx, eps_col)
            assert bool(ok) == ok_ref
            # The reference skips capsules rejected by an AABB prefilter, so its
            # min_clearance is only exact when some capsule is within eps_col
            # (the only regime where the value is consumed, by the seed-IK
            # penalty). Otherwise ours is the true minimum, which is <= theirs.
            if mc_ref < eps_col:
                np.testing.assert_allclose(float(mc), mc_ref, atol=1e-10)
            else:
                assert float(mc) >= eps_col - 1e-10
                assert float(mc) <= mc_ref + 1e-10
            sok_ref, smc_ref = ref_col.check_self_collisions(
                ctx, cfg.thumb_contact_links, cfg.index_contact_links, r_l, eps_col)
            sok, smc = self_fn(dm, cp, r_l, eps_col)
            assert bool(sok) == sok_ref
            np.testing.assert_allclose(float(smc), smc_ref, atol=1e-10)


def test_segment_segment_dist_matches_reference():
    rng = np.random.RandomState(8)
    f = jax.jit(G.segment_segment_dist)
    cases = []
    for _ in range(300):
        a0, b0, a1, b1 = rng.randn(4, 3) * 0.05
        cases.append((a0, b0, a1, b1))
    # degenerate and parallel cases
    cases.append((np.zeros(3), np.zeros(3), np.array([1.0, 0, 0]), np.array([2.0, 0, 0])))
    cases.append((np.zeros(3), np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([1.0, 1.0, 0])))
    cases.append((np.zeros(3), np.array([1.0, 0, 0]), np.array([0.5, 1.0, 0]), np.array([0.5, 1.0, 0])))
    cases.append((np.zeros(3), np.zeros(3), np.ones(3), np.ones(3)))
    for a0, b0, a1, b1 in cases:
        ref = ref_col._segment_segment_dist(a0, b0, a1, b1)
        ours = float(f(jnp.asarray(a0), jnp.asarray(b0), jnp.asarray(a1), jnp.asarray(b1)))
        np.testing.assert_allclose(ours, ref, atol=1e-12)


def test_grasp_check_matches_reference():
    rng = np.random.RandomState(9)
    f = jax.jit(G.check_antipodal_friction)
    for _ in range(200):
        n0 = rng.randn(3)
        n1 = -n0 + rng.randn(3) * rng.choice([0.05, 0.5, 2.0])
        mu = rng.choice([0.3, 0.6, 1.0])
        ref = ref_grasp(np.array([n0, n1]), mu)
        ok, cos_val = f(jnp.asarray(n0), jnp.asarray(n1), mu)
        assert bool(ok) == ref.feasible
        np.testing.assert_allclose(float(cos_val), ref.cos_angle, atol=1e-12)
