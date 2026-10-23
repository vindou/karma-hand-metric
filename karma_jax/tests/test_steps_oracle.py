"""Step-level oracle: batched translation primitive step vs the reference kernel.

The reference's QP answers are only loosely optimal (see test_qp_oracle), so
trajectories can diverge over the 8 sub-steps; the comparison is therefore
statistical on step outcomes plus exact where determinism allows (the sphere
centre), with a report of the disagreements.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from karma.robot import update_kinematics
from karma.contacts import contact_kinematics as ref_ck, snapshot_ck
from karma.seed_selection import _attempt_primitive_step as ref_step
from karma.orientation import compute_motion_primitives_frame

from karma_jax import steps as S
from karma_jax import rolling as RL
from karma_jax import qp as Q

from .conftest import load_hand
from .test_qp_oracle import recording, _params

jax.config.update("jax_enable_x64", True)

STEP_HANDS = ["leap", "shadowhand"]


def _state_from_node(flat, node):
    names = flat.capsule_names
    q = np.zeros(flat.J_pad)
    q[: flat.n_joints] = node.q
    return S.GraspState(jnp.asarray(q), jnp.asarray(node.p), jnp.asarray(node.R_sphere),
                        jnp.asarray([names.index(node.contact_pair[0]), names.index(node.contact_pair[1])]))


@pytest.mark.parametrize("name", STEP_HANDS)
def test_primitive_step_matches_reference(name):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    dm = flat.to_device(jnp.float64)
    slip = cfg.rotation_slip_threshold_mm / 1000.0 if cfg.rotation_slip_threshold_mm is not None else None
    params = _params(cfg, slip)
    n_sub = max(1, int(np.ceil(cfg.voxel_size_m / cfg.step_size_m)))
    step = S.make_batched_step(dm, n_sub, params, Q.default_settings(jnp.float64))
    prims = compute_motion_primitives_frame(rec.R_seed, cfg.voxel_size_m)

    from karma.robot import active_position_indices, active_velocity_indices
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lo = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_hi = np.array(ctx.model.upperPositionLimit)[q_ids]

    nodes = list(rec.discovered.values())
    states, dps, refs = [], [], []
    for node in nodes:
        update_kinematics(ctx, node.q)
        ckt = snapshot_ck(ref_ck(ctx, node.q, node.p, node.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m))
        cki = snapshot_ck(ref_ck(ctx, node.q, node.p, node.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m))
        for k, dp in enumerate(prims):
            if node.arrival_prim >= 0 and k == (node.arrival_prim ^ 1):
                continue
            out = ref_step(ctx, cfg, dp, n_sub, node.q, node.p, node.R_sphere, node.contact_pair,
                           ckt, cki, q_ids, v_ids, q_lo, q_hi)
            states.append(_state_from_node(flat, node))
            dps.append(jnp.asarray(dp))
            refs.append(out)

    batch = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *states)
    res = step(batch, jnp.stack(dps))
    ok = np.asarray(res.ok)
    codes = np.asarray(res.code)

    n = len(refs)
    agree = sum(int(ok[i]) == int(refs[i].success) for i in range(n))
    reason_counts = {}
    p_err, q_err, R_err, pair_mismatch = [], [], [], 0
    nJ = flat.n_joints
    names = flat.capsule_names
    for i, ref in enumerate(refs):
        ours = S.STEP_FAIL_NAMES[int(codes[i])]
        theirs = "ok" if ref.success else ref.fail_reason
        reason_counts[(theirs, ours)] = reason_counts.get((theirs, ours), 0) + 1
        if ref.success and ok[i]:
            p_err.append(np.abs(np.asarray(res.state.p[i]) - ref.p).max())
            q_err.append(np.abs(np.asarray(res.state.q[i])[:nJ] - ref.q).max())
            R_err.append(np.abs(np.asarray(res.state.R[i]) - ref.R_sphere).max())
            pair = (names[int(res.state.pair[i, 0])], names[int(res.state.pair[i, 1])])
            pair_mismatch += pair != ref.contact_pair
    print(f"\n{name}: {n} attempts, outcome agreement {agree}/{n}; (reference, ours) reason pairs: "
          f"{dict(sorted(reason_counts.items(), key=lambda kv: -kv[1]))}")
    if p_err:
        print(f"  on common successes: |p| max {np.max(p_err):.1e}, |q| median {np.median(q_err):.1e} "
              f"max {np.max(q_err):.1e}, |R| max {np.max(R_err):.1e}, pair mismatches {pair_mismatch}")
        assert np.max(p_err) < 1e-12          # p is p_start + delta_p, deterministic
    assert agree >= 0.9 * n, (agree, n)
