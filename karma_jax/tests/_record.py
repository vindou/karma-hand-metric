"""Record real kernel inputs/outputs from a reference BFS run, for oracle tests.

Runs a short trial BFS and a few Phase-2 voxels on a hand with the reference
implementation, wrapping ``solve_rolling_step`` / ``solve_rotation_step`` /
``_project_onto_contacts`` so every call's arguments and results are kept.
No data files are committed; recording a hand takes a second or two.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from karma import seed_selection as ss
from karma import metric as km
from karma import rolling_qp as rq
from karma import projection as pj
from karma.contacts import snapshot_ck
from karma.robot import update_kinematics
from karma.seed import _solve_for_centre, _candidate_sphere_centres
from karma.contacts import contact_kinematics, _capsule_axis_segment
from karma.math_utils import closest_points_on_segments
from karma.orientation import compute_motion_primitives_frame
from scipy.stats.qmc import Sobol


@dataclass
class Recording:
    rolling: list = field(default_factory=list)     # dicts: inputs + RollingStepResult
    rotation: list = field(default_factory=list)    # dicts: inputs + RotationStepResult
    projection: list = field(default_factory=list)  # dicts: inputs + outputs
    seed: object = None
    R_seed: np.ndarray | None = None
    discovered: dict | None = None


def _ck_to_dict(ck):
    ck = snapshot_ck(ck)
    return dict(name=ck.geom_name, axis_point=ck.axis_point_world, sphere_point=ck.sphere_point_world,
                normal=ck.normal_world, g=ck.g_m, Jv=ck.Jv_world, Jw=ck.Jw_world,
                origin=ck.origin_world, Jv_origin=ck.Jv_origin_world)


def find_seed(cfg, ctx, n_sobol: int = 64):
    """Cheapest reference seed: a few Sobol samples, first offset only, best gap."""
    from karma.robot import (active_position_indices, active_velocity_indices,
                             neutral_configuration, update_positions_only)
    q_ids = active_position_indices(ctx.model, cfg.finger_joint_names)
    v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
    q_lo = np.array(ctx.model.lowerPositionLimit)[q_ids]
    q_hi = np.array(ctx.model.upperPositionLimit)[q_ids]
    q0 = neutral_configuration(ctx.model)
    seeds = []
    for s in Sobol(d=len(q_ids), scramble=False).random(n_sobol):
        q = q0.copy()
        q[q_ids] = q_lo + s * (q_hi - q_lo)
        update_positions_only(ctx, q)
        c0, d0, h0 = _capsule_axis_segment(ctx, cfg.seed_thumb_link)
        c1, d1, h1 = _capsule_axis_segment(ctx, cfg.seed_index_link)
        p0, p1 = closest_points_on_segments(c0, d0, h0, c1, d1, h1)
        if np.linalg.norm(p1 - p0) > 2 * (cfg.sphere_radius_m + cfg.link_radius_m) + 0.02:
            continue
        for c in _candidate_sphere_centres(ctx, q, cfg.seed_thumb_link, cfg.seed_index_link,
                                           cfg.seed_search_offsets_m[:1]):
            r = _solve_for_centre(cfg, ctx, q.copy(), c, cfg.seed_thumb_link, cfg.seed_index_link,
                                  q_ids, v_ids, q_lo, q_hi)
            if r is not None and r.feasible:
                seeds.append(r)
    if not seeds:
        raise RuntimeError("no feasible seed found for recording")
    seeds.sort(key=lambda r: r.max_gap)
    return ss._polish_one_seed(cfg, ctx, seeds[0], q_ids, v_ids, q_lo, q_hi), q_ids, v_ids, q_lo, q_hi


def record(cfg, ctx, bfs_budget: int = 40, n_rot_voxels: int = 3) -> Recording:
    rec = Recording()
    seed, q_ids, v_ids, q_lo, q_hi = find_seed(cfg, ctx)
    rec.seed = seed

    orig_roll = rq.solve_rolling_step
    orig_rot = rq.solve_rotation_step
    orig_proj = pj._project_onto_contacts

    def wrap_roll(contact_kin, delta_p_world, q_active, q_lower, q_upper, active_v_ids, *args, **kw):
        inp = dict(ck_t=_ck_to_dict(contact_kin[0]), ck_i=_ck_to_dict(contact_kin[1]),
                   delta_p=np.array(delta_p_world, float), q_active=np.array(q_active, float),
                   q_lower=np.array(q_lower, float), q_upper=np.array(q_upper, float),
                   slip_threshold=kw.get("tangential_slip_threshold_m"))
        out = orig_roll(contact_kin, delta_p_world, q_active, q_lower, q_upper, active_v_ids, *args, **kw)
        rec.rolling.append(dict(inp, result=copy.deepcopy(out)))
        return out

    def wrap_rot(contact_kin, delta_theta_target, q_active, q_lower, q_upper, active_v_ids, *args, **kw):
        inp = dict(ck_t=_ck_to_dict(contact_kin[0]), ck_i=_ck_to_dict(contact_kin[1]),
                   dtheta_target=np.array(delta_theta_target, float), q_active=np.array(q_active, float),
                   q_lower=np.array(q_lower, float), q_upper=np.array(q_upper, float))
        out = orig_rot(contact_kin, delta_theta_target, q_active, q_lower, q_upper, active_v_ids, *args, **kw)
        rec.rotation.append(dict(inp, result=copy.deepcopy(out)))
        return out

    def wrap_proj(ctx_, q, sphere_centre, contact_pair, *args, **kw):
        inp = dict(q=np.array(q, float), p=np.array(sphere_centre, float), pair=tuple(contact_pair))
        out = orig_proj(ctx_, q, sphere_centre, contact_pair, *args, **kw)
        q_proj, dtheta, ok, ck0, ck1 = out
        rec.projection.append(dict(inp, q_proj=np.array(q_proj, float), dtheta=np.array(dtheta, float),
                                   ok=bool(ok), g0=float(ck0.g_m), g1=float(ck1.g_m)))
        return out

    # Patch every module-level name the reference resolves at call time.
    ss.solve_rolling_step = wrap_roll
    km.solve_rotation_step = wrap_rot
    ss._project_onto_contacts = wrap_proj
    km._project_onto_contacts = wrap_proj
    try:
        R_seed = ss._compute_trial_frame(cfg, ctx, seed, v_ids)
        prims = compute_motion_primitives_frame(R_seed, cfg.voxel_size_m)
        stats, disc = ss._run_trial_bfs(cfg, ctx, seed, R_seed, trial_budget=bfs_budget,
                                        q_ids=q_ids, v_ids=v_ids, q_lower=q_lo, q_upper=q_hi,
                                        primitives=prims, return_configs=True)
        rec.R_seed = R_seed
        rec.discovered = disc
        from karma.orientation import axis_to_healpix_bin, contact_line_axis_body
        for vox, node in list(disc.items())[:n_rot_voxels]:
            update_kinematics(ctx, node.q)
            ckt = contact_kinematics(ctx, node.q, node.p, node.contact_pair[0], cfg.sphere_radius_m, cfg.link_radius_m)
            cki = contact_kinematics(ctx, node.q, node.p, node.contact_pair[1], cfg.sphere_radius_m, cfg.link_radius_m)
            ob = axis_to_healpix_bin(contact_line_axis_body(ckt.normal_world, cki.normal_world, node.R_sphere), cfg.healpix_nside)
            vc = km.VoxelConfig(vox, node.p.copy(), node.q.copy(), km._reorthogonalize(node.R_sphere), node.contact_pair, ob)
            km._explore_rotations_at_voxel(vc, cfg, None)
    finally:
        ss.solve_rolling_step = orig_roll
        km.solve_rotation_step = orig_rot
        ss._project_onto_contacts = orig_proj
        km._project_onto_contacts = orig_proj
    return rec
