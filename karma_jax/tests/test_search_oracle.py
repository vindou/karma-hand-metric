"""Search-level oracle: batched BFS and rotation exploration vs the reference, same seed.

Trajectories diverge slightly because the reference's QP answers are loosely
optimal, so voxel sets are compared with a tolerance and reported.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from karma.seed_selection import _run_trial_bfs
from karma.orientation import compute_motion_primitives_frame, healpix_npix_lines
from karma import metric as km
from karma.robot import update_kinematics, active_position_indices, active_velocity_indices
from karma.contacts import contact_kinematics
from karma.orientation import axis_to_healpix_bin, contact_line_axis_body

from karma_jax import search as SR

from .conftest import load_hand
from .test_qp_oracle import recording

jax.config.update("jax_enable_x64", True)

HANDS = ["leap", "shadowhand"]


def _seed_spec(flat, rec):
    names = flat.capsule_names
    s = rec.seed
    return SR.SeedSpec(q=np.asarray(s.q, float), p=np.asarray(s.sphere_centre_world, float),
                       pair=(names.index(s.contact_pair[0]), names.index(s.contact_pair[1])),
                       R_seed=np.asarray(rec.R_seed, float))


@pytest.mark.parametrize("dtype", [jnp.float64, jnp.float32])
@pytest.mark.parametrize("name", HANDS)
def test_translation_bfs_matches_reference(name, dtype):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    budget = 120
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lo = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_hi = np.array(ctx.model.upperPositionLimit)[q_ids]
    prims = compute_motion_primitives_frame(rec.R_seed, cfg.voxel_size_m)
    stats, disc = _run_trial_bfs(cfg, ctx, rec.seed, rec.R_seed, trial_budget=budget, q_ids=q_ids,
                                 v_ids=v_ids, q_lower=q_lo, q_upper=q_hi, primitives=prims, return_configs=True)

    search = SR.TranslationSearch(flat, cfg, dtype=dtype)
    (res,) = search.run([_seed_spec(flat, rec)], budget)

    ref_vox = set(disc.keys())
    our_vox = set(res.voxel_to_node.keys())
    sym = ref_vox ^ our_vox
    print(f"\n{name} {dtype.__name__}: reference {len(ref_vox)} voxels, ours {len(our_vox)}, symmetric "
          f"difference {len(sym)}; ref stats attempts={stats.attempts} fails={stats.fail_moves} "
          f"revisits={stats.revisits} boundary={stats.unique_boundary} frontier={stats.frontier_size}; "
          f"ours attempts={res.attempts} fails={res.fail_moves} revisits={res.revisits} "
          f"boundary={res.unique_boundary} frontier={res.frontier_size} fail_reasons={res.fail_reasons}")
    # lattice positions must agree exactly on common voxels
    for vox in ref_vox & our_vox:
        np.testing.assert_allclose(res.nodes[res.voxel_to_node[vox]].p, disc[vox].p, atol=1e-12)
    assert len(sym) <= max(3, 0.05 * len(ref_vox)), sym


@pytest.mark.parametrize("name", HANDS)
def test_rotation_exploration_matches_reference(name):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    names = flat.capsule_names
    nodes = list(rec.discovered.items())[:6]
    starts, ref_bins = [], []
    for vox, node in nodes:
        update_kinematics(ctx, node.q)
        ckt = contact_kinematics(ctx, node.q, node.p, node.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m)
        cki = contact_kinematics(ctx, node.q, node.p, node.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m)
        R = km._reorthogonalize(node.R_sphere)
        ob = axis_to_healpix_bin(contact_line_axis_body(ckt.normal_world, cki.normal_world, R), cfg.healpix_nside)
        vc = km.VoxelConfig(vox, node.p.copy(), node.q.copy(), R, node.contact_pair, ob)
        _, oc, fr = km._explore_rotations_at_voxel(vc, cfg, None)
        ref_bins.append(set(oc.keys()))
        starts.append(SR.VoxelStart(np.asarray(node.q, float), np.asarray(node.p, float), R,
                                    (names.index(node.contact_pair[0]), names.index(node.contact_pair[1]))))
    search = SR.RotationSearch(flat, cfg, dtype=jnp.float64)
    ours = search.run(starts)
    total = healpix_npix_lines(cfg.healpix_nside)
    report = []
    for (vox, _), rb, ob in zip(nodes, ref_bins, ours):
        report.append((len(rb), len(ob.bins), len(rb ^ ob.bins)))
    print(f"\n{name}: (ref bins, our bins, symmetric diff) per voxel: {report}")
    ref_cov = np.mean([r[0] for r in report]) / total
    our_cov = np.mean([r[1] for r in report]) / total
    assert abs(ref_cov - our_cov) < 0.03, (ref_cov, our_cov)
