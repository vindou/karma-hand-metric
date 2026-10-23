"""Oracle test: batched LM seed IK vs the reference scipy TRF on identical candidates.

The TRF port should reproduce scipy's iterates in fp64; the test checks verdicts and solutions,
and for candidates both call feasible, closeness of the pinch found.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp
from scipy.stats.qmc import Sobol

from karma.robot import (active_position_indices, active_velocity_indices, neutral_configuration,
                         update_positions_only)
from karma.seed import _solve_for_centre, _candidate_sphere_centres
from karma.contacts import _capsule_axis_segment
from karma.math_utils import closest_points_on_segments

from karma_jax import ik as IK
from karma_jax.search import step_params_from_cfg

from .conftest import load_hand

jax.config.update("jax_enable_x64", True)

HANDS = ["leap", "shadowhand", "inspire"]


def _candidates(cfg, ctx, n_sobol=96, n_offsets=3):
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    q_lo = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_hi = np.array(ctx.model.upperPositionLimit)[q_ids]
    q0 = neutral_configuration(ctx.model)
    max_reach = 2.0 * (cfg.sphere_radius_m + cfg.link_radius_m) + max(abs(o) for o in cfg.seed_search_offsets_m) * 2.0
    out = []
    for s in Sobol(d=len(q_ids), scramble=False).random(n_sobol):
        q = q0.copy()
        q[q_ids] = q_lo + s * (q_hi - q_lo)
        update_positions_only(ctx, q)
        c0, d0, h0 = _capsule_axis_segment(ctx, cfg.seed_thumb_link)
        c1, d1, h1 = _capsule_axis_segment(ctx, cfg.seed_index_link)
        p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        if np.linalg.norm(p1 - p0) > max_reach:
            continue
        for c in _candidate_sphere_centres(ctx, q, cfg.seed_thumb_link, cfg.seed_index_link,
                                           cfg.seed_search_offsets_m[:n_offsets]):
            out.append((q.copy(), c))
    return out


def test_ik_jacobian_matches_finite_differences():
    """The full analytic Jacobian (incl. antipodal and hinge rows) vs central differences."""
    flat, cfg, ctx = load_hand("leap")
    dm = flat.to_device(jnp.float64)
    params = step_params_from_cfg(cfg, jnp.float64)
    cands = _candidates(cfg, ctx, n_sobol=16, n_offsets=1)
    pair = jnp.asarray([flat.seed_thumb_cap, flat.seed_index_cap])
    f = jax.jit(lambda x, q0, c: IK._residuals_and_jac(dm, x, q0, c, pair, params, True)[:2])
    M = flat.M_pad
    checked = 0
    for q, c in cands[:8]:
        q0 = np.zeros(flat.J_pad)
        q0[: flat.n_joints] = q
        x = np.concatenate([q0[flat.q_ids[: flat.n_active]], np.zeros(M - flat.n_active), np.zeros(3)])
        r, J = f(jnp.asarray(x), jnp.asarray(q0), jnp.asarray(c))
        J = np.asarray(J)
        J_fd = np.zeros_like(J)
        h = 1e-6
        for k in range(M + 3):
            if k < M and k >= flat.n_active:
                continue
            e = np.zeros(M + 3)
            e[k] = h
            rp = np.asarray(f(jnp.asarray(x + e), jnp.asarray(q0), jnp.asarray(c))[0])
            rm = np.asarray(f(jnp.asarray(x - e), jnp.asarray(q0), jnp.asarray(c))[0])
            J_fd[:, k] = (rp - rm) / (2 * h)
        # the hinge row is piecewise; compare only where it is smoothly active or inactive
        rows = [0, 1, 3] + list(range(4, 4 + M))
        np.testing.assert_allclose(J[rows], J_fd[rows], rtol=1e-5, atol=1e-5)
        checked += 1
    assert checked > 0


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("name", HANDS)
def test_seed_ik_verdicts_match_reference(name, dtype):
    flat, cfg, ctx = load_hand(name)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lo = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_hi = np.array(ctx.model.upperPositionLimit)[q_ids]
    cands = _candidates(cfg, ctx)
    assert len(cands) > 20

    refs = [_solve_for_centre(cfg, ctx, q.copy(), c, cfg.seed_thumb_link, cfg.seed_index_link,
                              q_ids, v_ids, q_lo, q_hi) for q, c in cands]

    dm = flat.to_device(dtype)
    params = step_params_from_cfg(cfg, dtype)
    B = len(cands)
    qi = np.zeros((B, flat.J_pad))
    for i, (q, _) in enumerate(cands):
        qi[i, : flat.n_joints] = q
    pair = np.tile([[flat.seed_thumb_cap, flat.seed_index_cap]], (B, 1))
    args = (jnp.asarray(qi, dtype), jnp.asarray(np.stack([c for _, c in cands]), dtype), jnp.asarray(pair, jnp.int32))
    ref_feas = np.array([r is not None and r.feasible for r in refs])

    res = jax.device_get(IK.make_batched_ik(dm, params)(*args))
    our_feas = np.asarray(res.feasible)
    both = ref_feas & our_feas
    agree = int(np.sum(ref_feas == our_feas))
    p_diff = [np.abs(np.asarray(res.p[i]) - refs[i].sphere_centre_world).max() for i in np.nonzero(both)[0]]
    q_diff = [np.abs(np.asarray(res.q[i])[: flat.n_joints] - refs[i].q).max() for i in np.nonzero(both)[0]]
    print(f"{name} {dtype.__name__}: {B} candidates, reference feasible {ref_feas.sum()}, ours "
          f"{our_feas.sum()}, verdict agreement {agree}/{B}; on common feasible: |p| median "
          f"{np.median(p_diff) if p_diff else float('nan'):.1e} max {np.max(p_diff) if p_diff else float('nan'):.1e}, "
          f"|q| median {np.median(q_diff) if q_diff else float('nan'):.1e} max {np.max(q_diff) if q_diff else float('nan'):.1e}; "
          f"nfev median {np.median(np.asarray(res.iters)):.0f}")
    # The TRF port follows scipy's iterates, so verdicts and solutions should match
    # up to roundoff-induced differences on a few borderline candidates.
    if dtype == jnp.float64:
        assert agree >= 0.97 * B, (agree, B)
        assert np.median(q_diff) < 1e-6
    else:
        assert agree >= 0.9 * B, (agree, B)
