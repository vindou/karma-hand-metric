"""Phase 9.2 (SPEC §8.1 D18): the hard admissibility check, the smooth base-separation penalty and the
all-finger self-collision gate of the karma_jax kernels (``build_flat_model(self_collision_all=True)``)."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import Topology, anthropomorphic, build_hand_model, encode, decode, fixed_from_params, Bounds
from dkarma.core.admissibility import admissibility, MIN_BASE_SEPARATION_M
from dkarma.export import design_from_dict, write_design
from dkarma.optimization.objectives import Constraints, D18_CONSTRAINTS, reach_penalty
from dkarma.fixtures import METRIC_CFG
from .conftest import load_design

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
ARM_A = ROOT / "results" / "fabrication_candidates" / "armA" / "armA.design.json"
ARM_B = ROOT / "results" / "fabrication_candidates" / "armB" / "optimized_morphology.design.json"


def test_admissibility_of_start_and_arms():
    topo = Topology.three_finger(4)
    a = admissibility(anthropomorphic(topo), topo)
    assert a["admissible"] and abs(a["min_base_separation_m"] - 0.030) < 1e-9 and a["max_reach_m"] < 0.25
    assert abs(MIN_BASE_SEPARATION_M - 0.0241) < 1e-12
    if ARM_A.exists():
        t, p, _ = design_from_dict(json.loads(ARM_A.read_text()))
        r = admissibility(p, t)
        assert not r["admissible"] and r["min_base_separation_m"] < 0.0181 and r["violations"]
    if ARM_B.exists():
        t, p, _ = design_from_dict(json.loads(ARM_B.read_text()))
        assert admissibility(p, t)["admissible"]


def test_base_separation_penalty_is_zero_when_admissible_and_active_when_not():
    topo = Topology.three_finger(4)
    params = anthropomorphic(topo)
    assert float(reach_penalty(params, topo, D18_CONSTRAINTS)) < 1e-6
    close = params._replace(base_xyz=jnp.asarray(params.base_xyz).at[1].set(jnp.asarray(params.base_xyz)[0] + jnp.array([0.0, 0.015, 0.0])))
    pen = float(reach_penalty(close, topo, D18_CONSTRAINTS))
    assert pen > 1e-4
    bounds = Bounds(base_xyz_max_m=0.10)
    phi = encode(close, topo, bounds)
    g = jax.grad(lambda ph: reach_penalty(decode(ph, topo, bounds, fixed_from_params(close)), topo, D18_CONSTRAINTS))(phi)
    gf = jax.flatten_util.ravel_pytree(g)[0]
    assert bool(jnp.all(jnp.isfinite(gf))) and float(jnp.linalg.norm(gf)) > 0
    # without the D18 field the Phase 7 penalty is unchanged
    assert float(reach_penalty(close, topo, Constraints())) < 1e-9


def _flat(topo, params, tmp_path, name, self_collision_all):
    from karma.config import load_config
    from karma.robot import load_robot
    from karma_jax.model import build_flat_model
    _, yaml_path, _ = write_design(topo, params, tmp_path / name, name)
    cfg = load_config(METRIC_CFG, yaml_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    return build_flat_model(ctx, cfg, self_collision_all=self_collision_all), cfg


def test_all_finger_self_collision_gate(tmp_path):
    from karma_jax.kinematics import forward_kinematics
    from karma_jax.geometry import capsule_pass, check_self_collisions, check_self_collisions_pair, check_self_collisions_all
    from karma_jax.search import step_params_from_cfg
    topo = Topology.three_finger(4)
    for name, params, expect_overlap in (("start", anthropomorphic(topo), False),) + \
            (((("armA", design_from_dict(json.loads(ARM_A.read_text()))[1], True),)) if ARM_A.exists() else ()):
        flat, cfg = _flat(topo, params, tmp_path, name, True)
        groups = flat.cap_group[flat.cap_valid]
        assert len(set(groups.tolist())) == 3, groups                  # three fingers, three groups
        for fname in topo.fingers:                                      # every capsule of a finger shares its group
            caps = [i for i, n in enumerate(flat.capsule_names) if n.startswith(fname + "_")]
            assert len({int(flat.cap_group[c]) for c in caps}) == 1
        flat0, _ = _flat(topo, params, tmp_path, name + "_pair", False)
        assert flat0.cap_group is None
        dm = flat.to_device(jnp.float64)
        sp = step_params_from_cfg(cfg, jnp.float64)
        q = jnp.zeros(dm.parent.shape[0])
        cp = capsule_pass(dm, forward_kinematics(dm, q), jnp.array([0.0, 0.0, 0.5]), sp.sphere_radius, sp.link_radius)
        ok_all, clr_all = check_self_collisions_all(dm, cp, sp.link_radius, sp.eps_col)
        ok_pair, clr_pair = check_self_collisions_pair(dm, cp, sp.link_radius, sp.eps_col)
        ok_auto, clr_auto = check_self_collisions(dm, cp, sp.link_radius, sp.eps_col)
        assert float(clr_auto) == float(clr_all)                        # the model carries the gate
        assert float(clr_all) <= float(clr_pair) + 1e-12                # the all-finger minimum is never larger
        # oracle: dkarma's own all-finger self clearance on the same design and configuration
        from dkarma.core import geometry as dgeo
        from dkarma.core.contacts import self_clearance
        hm = build_hand_model(topo, params)
        q_d = jnp.zeros(hm.dm.q_ids.shape[0])
        cp_d = dgeo.capsule_pass(hm, forward_kinematics(hm.dm, q_d), jnp.array([0.0, 0.0, 0.5]), sp.sphere_radius)
        # both are (segment distance - radii); karma_jax uses the config's L_ref-scaled uniform radius (published task),
        # dkarma the physical 11.8 mm capsules: compare the segment distances
        np.testing.assert_allclose(float(clr_all) + 2.0 * float(sp.link_radius), float(self_clearance(hm, cp_d)) + 2.0 * 0.0118, atol=1e-9)
        if expect_overlap:
            assert bool(ok_pair) and not bool(ok_all) and float(clr_all) < -0.005   # arm A: index/middle overlap, pair check blind to it


def test_all_finger_gate_equals_pair_gate_on_two_finger_hands(tmp_path):
    from karma_jax.kinematics import forward_kinematics
    from karma_jax.geometry import capsule_pass, check_self_collisions_pair, check_self_collisions_all
    from karma_jax.search import step_params_from_cfg
    topo, params, _ = load_design("baseline")
    flat, cfg = _flat(topo, params, tmp_path, "baseline", True)
    dm = flat.to_device(jnp.float64)
    sp = step_params_from_cfg(cfg, jnp.float64)
    rng = np.random.RandomState(0)
    lo, hi = np.asarray(dm.q_lower), np.asarray(dm.q_upper)
    for _ in range(5):
        q = jnp.asarray(lo + rng.rand(lo.shape[0]) * (hi - lo))
        cp = capsule_pass(dm, forward_kinematics(dm, q), jnp.array([0.0, 0.0, 0.5]), sp.sphere_radius, sp.link_radius)
        _, a = check_self_collisions_all(dm, cp, sp.link_radius, sp.eps_col)
        _, b = check_self_collisions_pair(dm, cp, sp.link_radius, sp.eps_col)
        assert abs(float(a) - float(b)) < 1e-12
