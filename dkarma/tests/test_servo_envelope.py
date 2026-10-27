"""Phase 11.1 (SPEC §8.1 D19, §11.4): per-capsule radii and the servo envelope.

The D19 box (34 mm floor on links 0-2, 16.40 mm servo links, 11.8 mm fingertips, 33.30 mm bases), its hard admissibility
check and smooth penalty; the collision envelope with mixed radii (object and finger-finger clearances, and the
reference two-finger kernels' gates re-applied per capsule), where a configuration that clears 11.8 mm capsules but not
16.40 mm ones is rejected; and the MuJoCo scene and the URDF carrying each capsule's own radius."""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import Topology, anthropomorphic, capsule_radii, build_hand_model, decode, encode, fixed_from_params, params_to_fingers, load_task_spec
from dkarma.core import geometry as geo
from dkarma.core.admissibility import admissibility, D19_ENVELOPE, R_SERVO_M, MIN_BASE_SEPARATION_M, D19_MIN_BASE_SEPARATION_M
from dkarma.core.contacts import object_clearance, self_clearance, mixed_radii, reference_gates_per_capsule
from dkarma.core.kinematics import capsule_endpoints
from dkarma.solvers.seed_projection import assignment
from dkarma.optimization.objectives import D19_CONSTRAINTS, reach_penalty
from dkarma.experiments.continuation_study import nominal_d19, BOUNDS, BOUNDS_D19
from karma_jax.kinematics import forward_kinematics

jax.config.update("jax_enable_x64", True)

ROOT = Path(__file__).resolve().parents[2]
TASK = load_task_spec()
SP = TASK.step_params()
TOPO = Topology.three_finger(4)
P10_SELECTED = ROOT / "results" / "fabrication_candidates" / "selected_full_d18_start31_sobol41_s0.25" / "optimized_morphology.design.json"


def _uniform(params):
    return params._replace(cap_radius=jnp.full_like(jnp.asarray(params.cap_radius), 0.0118))


# ── the D19 box ──────────────────────────────────────────────────────────────

def test_d19_length_floors():
    lo0, hi0 = BOUNDS.len_bounds(TOPO)
    lo, hi = BOUNDS_D19.len_bounds(TOPO)
    np.testing.assert_array_equal(hi, hi0)
    np.testing.assert_array_equal(lo[:, 3], lo0[:, 3])                      # the fingertip keeps 12 mm
    assert np.all(lo[:, 0:3] == 0.034)                                      # links 0-2: the 34 mm case
    params0 = nominal_d19(TOPO)
    fixed = fixed_from_params(params0)
    phi = encode(params0, TOPO, BOUNDS_D19)
    np.testing.assert_allclose(np.asarray(decode(phi, TOPO, BOUNDS_D19, fixed).link_len), np.asarray(params0.link_len), atol=1e-9)
    x0, unravel = jax.flatten_util.ravel_pytree(phi)
    rng = np.random.RandomState(0)
    for scale in (0.5, 5.0, 40.0):                                          # far outside the interior, both signs
        for _ in range(20):
            ll = np.asarray(decode(unravel(x0 + scale * jnp.asarray(rng.randn(x0.shape[0]))), TOPO, BOUNDS_D19, fixed).link_len)
            assert np.all(ll[:, 0:3] >= 0.034) and np.all(ll[:, 3] >= 0.012) and np.all(ll <= 0.090)
            assert np.all(admissibility(decode(unravel(x0), TOPO, BOUNDS_D19, fixed), TOPO, envelope=D19_ENVELOPE)["envelope"]["ok"])


def test_n11_and_d19_admissibility():
    n11 = nominal_d19(TOPO)
    np.testing.assert_allclose(np.asarray(n11.link_len) * 1e3, [[40, 40, 40, 23], [40, 40, 40, 23], [40, 45, 40, 25]], atol=1e-9)
    np.testing.assert_array_equal(np.asarray(n11.cap_radius), capsule_radii(TOPO, 0.0164, 0.0118))
    assert np.asarray(n11.base_xyz)[TOPO.finger_index("middle"), 1] == 0.040
    a = admissibility(n11, TOPO, envelope=D19_ENVELOPE)
    assert a["admissible"] and a["min_sep_required_m"] == D19_MIN_BASE_SEPARATION_M == 0.0333 and abs(a["min_base_separation_m"] - 0.040) < 1e-12
    # 30 mm pitch: admissible under D18's 24.1 mm, not under 33.30 mm
    close = nominal_d19(TOPO, pitch_m=0.030)
    assert admissibility(close, TOPO, min_sep_m=MIN_BASE_SEPARATION_M)["admissible"]
    b = admissibility(close, TOPO, envelope=D19_ENVELOPE)
    assert not b["admissible"] and "33.3 mm" in b["violations"][0]
    # the envelope: a 30 mm link 0 or link 1, an 11.8 mm servo link, a 16.40 mm fingertip
    for k in (0, 1):
        short = n11._replace(link_len=jnp.asarray(n11.link_len).at[0, k].set(0.030))
        assert any(f"link {k} length 30.0 mm < 34 mm" in v for v in admissibility(short, TOPO, envelope=D19_ENVELOPE)["violations"])
    thin = n11._replace(cap_radius=jnp.asarray(n11.cap_radius).at[1].set(0.0118))
    assert any("link 1 radius" in v for v in admissibility(thin, TOPO, envelope=D19_ENVELOPE)["violations"])
    fat = n11._replace(cap_radius=jnp.asarray(n11.cap_radius).at[3].set(R_SERVO_M))
    assert any("fingertip radius" in v for v in admissibility(fat, TOPO, envelope=D19_ENVELOPE)["violations"])
    # the Phase 10 start: D18-admissible (its own radii give 24.1 mm), outside the D19 envelope (12 mm link 0, 25 mm link 2, 11.8 mm servo links)
    start = anthropomorphic(TOPO)
    a18 = admissibility(start, TOPO)
    assert a18["admissible"] and a18["min_sep_required_m"] == MIN_BASE_SEPARATION_M
    assert not admissibility(start, TOPO, envelope=D19_ENVELOPE)["admissible"]
    if P10_SELECTED.exists():
        from dkarma.export import design_from_dict
        t, p, _ = design_from_dict(json.loads(P10_SELECTED.read_text()))
        r = admissibility(p, t, envelope=D19_ENVELOPE)
        assert not r["admissible"] and r["envelope"]["min_servo_link_len_m"] < 0.034
    # the smooth penalty: zero at N11, active at a 30 mm pitch
    assert float(reach_penalty(n11, TOPO, D19_CONSTRAINTS)) < 1e-6           # the softplus tail at a 6.7 mm margin
    assert float(reach_penalty(close, TOPO, D19_CONSTRAINTS)) > 1e-5


# ── the collision envelope with mixed radii ──────────────────────────────────

def _sphere_above(hm, cap, height):
    """Sphere centre ``height`` above the middle of capsule ``cap``'s axis (palm +z) at q = 0."""
    fk = forward_kinematics(hm.dm, jnp.zeros(hm.dm.q_ids.shape[0]))
    a, b = capsule_endpoints(hm, fk)
    return fk, 0.5 * (a[cap] + b[cap]) + jnp.array([0.0, 0.0, height])


def test_mixed_radius_gaps_and_clearances():
    n11 = nominal_d19(TOPO)
    hm, hu = build_hand_model(TOPO, n11), build_hand_model(TOPO, _uniform(n11))
    assert mixed_radii(hm, SP.link_radius) and not mixed_radii(hu, SP.link_radius)
    q = jnp.asarray(np.random.RandomState(1).uniform(np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper)))
    p = jnp.array([0.02, 0.01, 0.06])
    fk = forward_kinematics(hm.dm, q)
    cp, cu = geo.capsule_pass(hm, fk, p, SP.sphere_radius), geo.capsule_pass(hu, fk, p, SP.sphere_radius)
    np.testing.assert_array_equal(np.asarray(cp.a), np.asarray(cu.a))      # the radius does not move the axes
    np.testing.assert_allclose(np.asarray(cu.gap - cp.gap), np.asarray(n11.cap_radius) - 0.0118, atol=1e-15)
    clr, m = geo.capsule_pair_clearance(hm, cp)
    clu, _ = geo.capsule_pair_clearance(hu, cu)
    r = np.asarray(n11.cap_radius)
    np.testing.assert_allclose(np.asarray(clu - clr)[np.asarray(m)], (r[:, None] + r[None, :] - 0.0236)[np.asarray(m)], atol=1e-15)


def test_object_collision_that_clears_11_8_but_not_16_4_is_rejected():
    n11 = nominal_d19(TOPO)
    hm, hu = build_hand_model(TOPO, n11), build_hand_model(TOPO, _uniform(n11))
    cap = TOPO.joint_id("index", 1)                       # a servo link: 14 mm from the sphere surface to its axis
    fk, p = _sphere_above(hm, cap, SP.sphere_radius + 0.014)
    asg = assignment(TOPO, 3, (TOPO.thumb_finger, TOPO.index_finger, TOPO.finger_index("middle")))
    cp, cu = geo.capsule_pass(hm, fk, p, SP.sphere_radius), geo.capsule_pass(hu, fk, p, SP.sphere_radius)
    assert int(jnp.argmin(cp.gap)) == cap
    oc_u, oc = float(object_clearance(hu, cu, asg)), float(object_clearance(hm, cp, asg))
    assert oc_u >= float(SP.eps_col) and abs(oc_u - 0.0022) < 1e-9          # 14 - 11.8 mm
    assert oc < float(SP.eps_col) and abs(oc + 0.0024) < 1e-9               # 14 - 16.4 mm: rejected
    # the rollout's exact gate and the three-contact seed verification use the same quantity
    from dkarma.solvers.seed_projection import _verify_multi
    assert float(_verify_multi(hm, jnp.zeros(hm.dm.q_ids.shape[0]), p, asg, SP)[1]) == oc
    # the reference two-finger kernels see 11.8 mm everywhere; the per-capsule re-check rejects
    from karma_jax.geometry import capsule_pass as kj_pass, check_collisions
    pair = jnp.asarray(TOPO.seed_caps(), jnp.int32)
    ok_kernel, _ = check_collisions(hm.dm, kj_pass(hm.dm, fk, p, SP.sphere_radius, SP.link_radius), pair, SP.eps_col)
    g, c, s = reference_gates_per_capsule(hm, fk, p, SP.sphere_radius, pair, SP.eps_g, SP.eps_col)
    assert bool(ok_kernel) and not bool(c)
    gu, cu_, su = reference_gates_per_capsule(hu, fk, p, SP.sphere_radius, pair, SP.eps_g, SP.eps_col)
    assert bool(cu_) and bool(c) == bool(ok_kernel & c)


def test_finger_clearance_that_clears_11_8_but_not_16_4_is_rejected():
    close = nominal_d19(TOPO, pitch_m=0.030)              # index and middle parallel, 30 mm apart at q = 0
    hm, hu = build_hand_model(TOPO, close), build_hand_model(TOPO, _uniform(close))
    fk = forward_kinematics(hm.dm, jnp.zeros(hm.dm.q_ids.shape[0]))
    p = jnp.array([0.0, 0.0, 0.5])
    cp, cu = geo.capsule_pass(hm, fk, p, SP.sphere_radius), geo.capsule_pass(hu, fk, p, SP.sphere_radius)
    fi, fm = TOPO.index_finger, TOPO.finger_index("middle")
    pair_min = lambda h, c: float(geo.hard_min_clearance(*geo.finger_pair_clearance(h, c, fi, fm)))   # noqa: E731
    sc_u, sc = pair_min(hu, cu), pair_min(hm, cp)
    assert sc_u >= float(SP.eps_col) and abs(sc_u - 0.0064) < 1e-9          # 30 - 2 x 11.8 mm
    assert sc < float(SP.eps_col) and abs(sc + 0.0028) < 1e-9               # 30 - 2 x 16.4 mm: rejected
    assert float(self_clearance(hm, cp)) <= sc                               # the all-finger gate includes this pair
    # the reference kernels' self-collision scope is the contact pair's fingers: make it index-middle
    t2 = Topology(fingers=TOPO.fingers, n_joints=TOPO.n_joints, pair=("middle", "index"))
    hm2, hu2 = build_hand_model(t2, close), build_hand_model(t2, _uniform(close))
    from karma_jax.geometry import capsule_pass as kj_pass, check_self_collisions
    ok_kernel, _ = check_self_collisions(hm2.dm, kj_pass(hm2.dm, fk, p, SP.sphere_radius, SP.link_radius), SP.link_radius, SP.eps_col)
    pair = jnp.asarray(t2.seed_caps(), jnp.int32)
    _, _, s = reference_gates_per_capsule(hm2, fk, p, SP.sphere_radius, pair, SP.eps_g, SP.eps_col)
    _, _, su = reference_gates_per_capsule(hu2, fk, p, SP.sphere_radius, pair, SP.eps_g, SP.eps_col)
    assert bool(ok_kernel) and bool(su) and not bool(s)


def test_exact_validator_envelope_codes():
    """A primitive the kernel accepts at a state that collides only with the 16.40 mm envelope fails with the kernel's
    collision code (translation 4, rotation 6)."""
    from karma_jax.steps import GraspState as RefGraspState
    from dkarma.solvers.exact import _envelope, _STEP_CODES, _ROT_CODES
    n11 = nominal_d19(TOPO)
    hm, hu = build_hand_model(TOPO, n11), build_hand_model(TOPO, _uniform(n11))
    _, p = _sphere_above(hm, TOPO.joint_id("index", 1), SP.sphere_radius + 0.014)
    st = RefGraspState(jnp.zeros(hm.dm.q_ids.shape[0]), p, jnp.eye(3), jnp.asarray(TOPO.seed_caps(), jnp.int32))
    # the seed pair is not in contact in this state: the gap gate comes first in the reference order (codes 3 / 5) ...
    for codes, want in ((_STEP_CODES, 3), (_ROT_CODES, 5)):
        ok, code = _envelope(hm, jnp.asarray(True), jnp.asarray(0, jnp.int32), st, SP, codes)
        assert not bool(ok) and int(code) == want
    # ... so widen eps_g to reach the collision gate, which only the 16.40 mm envelope fails
    sp_wide = SP._replace(eps_g=jnp.asarray(1.0))
    for codes, want in ((_STEP_CODES, 4), (_ROT_CODES, 6)):
        ok, code = _envelope(hm, jnp.asarray(True), jnp.asarray(0, jnp.int32), st, sp_wide, codes)
        assert not bool(ok) and int(code) == want
        ok, code = _envelope(hu, jnp.asarray(True), jnp.asarray(0, jnp.int32), st, sp_wide, codes)
        assert int(code) != want            # 11.8 mm capsules clear the sphere (N11's long thumb crosses the index at q = 0: self-collision)
        ok, code = _envelope(hm, jnp.asarray(False), jnp.asarray(2, jnp.int32), st, sp_wide, codes)   # an earlier failure keeps its code
        assert not bool(ok) and int(code) == 2


# ── MuJoCo scene and URDF ────────────────────────────────────────────────────

def _scene_design(topo, params):
    d = dict(fingers=params_to_fingers(topo, params), cap_radius=np.asarray(params.cap_radius).tolist(),
             topology=dict(fingers=list(topo.fingers), n_joints=list(topo.n_joints), pair=list(topo.pair)))
    return json.loads(json.dumps(d, default=float))


def test_mujoco_scene_matches_dkarma_capsules():
    mujoco = pytest.importorskip("mujoco")
    from dkarma.experiments.simulation_replay import mjcf_from_design
    from dkarma.experiments.compliant_pad import load_pads
    n11 = nominal_d19(TOPO)
    hm = build_hand_model(TOPO, n11)
    m = mujoco.MjModel.from_xml_string(mjcf_from_design(_scene_design(TOPO, n11), TASK, 0.108))
    d = mujoco.MjData(m)
    rng = np.random.RandomState(3)
    for _ in range(3):
        q = rng.uniform(np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper))
        for j, name in enumerate(TOPO.joint_names):
            d.qpos[m.jnt_qposadr[m.joint(name).id]] = q[j]
        mujoco.mj_forward(m, d)
        a, b = (np.asarray(x) for x in capsule_endpoints(hm, forward_kinematics(hm.dm, jnp.asarray(q))))
        for c, link in enumerate(TOPO.link_names):
            g = m.geom(link.replace("_l", "_c"))
            assert abs(m.geom_size[g.id][0] - float(n11.cap_radius[c])) < 1e-12
            ax = d.geom_xmat[g.id].reshape(3, 3)[:, 2] * m.geom_size[g.id][1]
            ends = sorted([d.geom_xpos[g.id] - ax, d.geom_xpos[g.id] + ax], key=lambda e: np.linalg.norm(e - a[c]))
            np.testing.assert_allclose(ends[0], a[c], atol=1e-9)
            np.testing.assert_allclose(ends[1], b[c], atol=1e-9)
    # the compliant pads are fitted to the 11.8 mm fingertip: allowed on N11, refused on a fingertip of another radius
    pad = load_pads()["20A_2mm"]
    mjcf_from_design(_scene_design(TOPO, n11), TASK, 0.108, pad=pad)
    with pytest.raises(ValueError):
        mjcf_from_design(_scene_design(TOPO, n11._replace(cap_radius=jnp.asarray(n11.cap_radius).at[3].set(R_SERVO_M))), TASK, 0.108, pad=pad)


def test_rigid_scene_of_a_phase10_design_unchanged():
    """Uniform radii: the scene string is the one Phase 10 wrote (``workspace/``, present on the machine that ran it)."""
    xml_p = ROOT / "workspace/phase10/10.2/sim/default/rigid/full_d18_start31.xml"
    des_p = ROOT / "workspace/phase10/10.2/sim/default/full_d18_start31_design.json"
    if not (xml_p.exists() and des_p.exists()):
        pytest.skip("Phase 10 workspace not present")
    from dkarma.experiments.simulation_replay import mjcf_from_design
    assert mjcf_from_design(json.loads(des_p.read_text()), TASK, 0.108) == xml_p.read_text()


def test_urdf_carries_capsules_for_mixed_radii(tmp_path):
    from dkarma.export import write_design, design_from_dict
    from dkarma.export.urdf_export import to_urdf, design_to_dict
    n11 = nominal_d19(TOPO)
    assert "<visual>" not in to_urdf(TOPO, anthropomorphic(TOPO))           # uniform radii: as before
    u = to_urdf(TOPO, n11)
    assert u.count('<cylinder radius="0.016400"') == 9 and u.count('<sphere radius="0.016400"') == 18 and u.count('<sphere radius="0.011800"') == 6
    t2, p2, _ = design_from_dict(design_to_dict(TOPO, n11))
    np.testing.assert_array_equal(np.asarray(p2.cap_radius), np.asarray(n11.cap_radius))
    # the reference reads the same joint tree and fingertips with and without the capsule visuals
    from karma.config import load_config
    from karma.robot import load_robot
    from dkarma.fixtures import METRIC_CFG
    caps = []
    for flag in (True, False):
        _, yaml_path, _ = write_design(TOPO, n11, tmp_path / str(flag), "n11")
        (tmp_path / str(flag) / "n11.urdf").write_text(to_urdf(TOPO, n11, "n11", capsules=flag))
        cfg = load_config(METRIC_CFG, yaml_path)
        ctx = load_robot(cfg.urdf_path, cfg)
        caps.append({k: (v.link_frame_id, v.distal_frame_id, None if v.tip_offset_xyz_m is None else np.asarray(v.tip_offset_xyz_m).tolist())
                     for k, v in ctx.link_capsules.items()})
        caps[-1]["_nq"] = ctx.model.nq
    assert caps[0] == caps[1]
