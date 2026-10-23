"""Phase 10.1: the compliant silicone-fingertip model of the MuJoCo replay scene (``compliant_pad`` and
``simulation_replay.mjcf_from_design``), and the Phase 10 outcome functions (``compliant_contact_study``,
``powered_comparison``, ``robust_grasps``).

The scene checks are the calibration-scene measurements of the Phase 10 pre-registration: the rigid scene is the
8.6/9.3 one string for string; the pad changes only the terminal capsules' contact with the sphere (geometry and
inertia identical); the static indentation follows the pad law independently of the link's inertia; the friction
creep stays below the rigid contact's; and the frozen parameter file is reproduced by the fit."""
import json
from pathlib import Path

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from dkarma.core import load_task_spec, Topology, anthropomorphic, params_to_fingers  # noqa: E402
from dkarma.experiments.simulation_replay import mjcf_from_design, _mjcf_rigid  # noqa: E402
from dkarma.experiments.compliant_pad import load_pads, make_pad, target_indentation, CENTRAL, GRID  # noqa: E402
from dkarma.experiments.compliant_contact_study import outcomes, verdict  # noqa: E402
from dkarma.experiments.powered_comparison import endpoint, per_seed_counts  # noqa: E402
from dkarma.experiments.robust_grasps import subset_measures, seed_score  # noqa: E402

TASK = load_task_spec()
PADS = load_pads()
GEOM_FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_type", "body_pos", "body_quat", "body_mass", "body_inertia", "jnt_axis", "jnt_range")


def _start_design():
    t = Topology.three_finger(4); p = anthropomorphic(t)
    d = dict(fingers=params_to_fingers(t, p), cap_radius=np.asarray(p.cap_radius).tolist(),
             topology=dict(fingers=list(t.fingers), n_joints=list(t.n_joints), pair=list(t.pair)))
    return json.loads(json.dumps(d, default=float))          # the export's round trip (simulation_replay.export_all)


def test_rigid_scene_unchanged():
    d = _start_design()
    xml = mjcf_from_design(d, TASK, 0.108)
    assert xml == _mjcf_rigid(d, TASK, 0.108)
    assert "<pair" not in xml
    ref = Path("workspace/phase9/9.3/start.xml")                 # the Phase 9.3 scene file of the same design
    if ref.exists():
        assert xml == ref.read_text()


@pytest.mark.parametrize("label,variant", [(CENTRAL, "patch"), ("10A_1mm", "patch"), ("40A_3mm", "patch"), (CENTRAL, "rigid_friction")])
def test_pad_only_changes_terminal_contacts(label, variant):
    d = _start_design()
    mr = mujoco.MjModel.from_xml_string(mjcf_from_design(d, TASK, 0.108))
    mp = mujoco.MjModel.from_xml_string(mjcf_from_design(d, TASK, 0.108, pad=PADS[label], pad_variant=variant))
    for f in GEOM_FIELDS:
        assert np.array_equal(getattr(mr, f), getattr(mp, f)), f
    assert mr.npair == 0 and mp.npair == 3
    assert sorted(mp.geom(int(g)).name for g in mp.pair_geom1) == ["index_c3", "middle_c3", "thumb_c3"]
    assert all(mp.geom(int(g)).name == "sphere_geom" for g in mp.pair_geom2)
    assert mp.pair_dim.tolist() == ([6, 6, 6] if variant == "patch" else [4, 4, 4])
    if variant == "patch":
        np.testing.assert_allclose(mp.pair_friction[:, 2], PADS[label].torsion_m, rtol=1e-5)
        np.testing.assert_allclose(mp.pair_friction[:, 3], PADS[label].rolling_m, rtol=1e-5)
    else:
        np.testing.assert_allclose(mp.pair_friction[:, 2], 0.6 * 0.003, rtol=1e-9)


def _calibration_scene(pair_attrs, link_joint=False):
    """A capsule of the fingertip radius (fixed, or on a stiff slide joint of 10 g so its inverse weight differs) and the
    35 g sphere on three slide joints (no rotation), gravity off."""
    lj = '<joint name="lz" type="slide" axis="0 0 1" stiffness="1e5" damping="5"/>' if link_joint else ""
    pair = f'<contact><pair geom1="pad" geom2="ball" {pair_attrs}/></contact>' if pair_attrs else ""
    return f'''<mujoco><option timestep="5e-4" gravity="0 0 0" integrator="implicitfast" cone="elliptic"/>
<default><geom condim="4" solref="0.002 1" solimp="0.95 0.99 0.001"/></default>
<worldbody><body name="link">{lj}<geom name="pad" type="sphere" size="0.0118" mass="0.01" friction="0.6 0.0018 0.0001"/></body>
 <body name="sph" pos="0 0 0.0268"><joint type="slide" axis="1 0 0"/><joint type="slide" axis="0 1 0"/><joint type="slide" axis="0 0 1"/>
 <geom name="ball" type="sphere" size="0.015" mass="0.035" friction="0.6 0.0018 0.0001"/></body></worldbody>{pair}</mujoco>'''


def _pair_A(xml):
    m = mujoco.MjModel.from_xml_string(xml)
    return float(m.body_invweight0[m.body("sph").id][0] + m.body_invweight0[m.body("link").id][0])


def _settle(m, d, force_xyz, steps=6000):
    mujoco.mj_resetData(m, d)
    sid = m.body("sph").id
    for _ in range(steps):
        d.xfrc_applied[sid, :3] = force_xyz
        mujoco.mj_step(m, d)


@pytest.mark.parametrize("link_joint", [False, True])
def test_pad_static_law(link_joint):
    """Static indentation under 0.05 / 0.26 / 1 N equals the pad law within 6 % (fit error <= 12 % over 0.02-2 N),
    for a fixed link and for a jointed one (different inverse weight: the s = s_u A scaling)."""
    pad = PADS[CENTRAL]
    A = _pair_A(_calibration_scene("", link_joint))
    m = mujoco.MjModel.from_xml_string(_calibration_scene(pad.pair_attrs(A), link_joint)); d = mujoco.MjData(m)
    assert (A > 40.0) == link_joint
    for F in (0.05, 0.26, 1.0):
        _settle(m, d, [0, 0, -F])
        pen = -d.contact[0].dist
        assert d.ncon == 1
        assert abs(pen / target_indentation(F, pad.E_pa, pad.thickness_m) - 1.0) < 0.06, (F, pen)


def test_pad_creep_below_rigid():
    """0.1 N tangential under 0.3 N normal, inside the friction cone: the pad's creep (soft-constraint drift) stays
    below the rigid 9.3 contact's."""
    creep = {}
    for lab in ("rigid", CENTRAL):
        attrs = "" if lab == "rigid" else PADS[lab].pair_attrs(_pair_A(_calibration_scene("")))
        m = mujoco.MjModel.from_xml_string(_calibration_scene(attrs)); d = mujoco.MjData(m)
        _settle(m, d, [0.1, 0, -0.3], steps=2000)
        creep[lab] = abs(d.qvel[0])
    assert creep[CENTRAL] < creep["rigid"] < 3e-4


def test_frozen_parameters_reproduced():
    rec = json.loads(Path("dkarma/configs/compliant_pads.json").read_text())
    assert rec["central"] == CENTRAL and len(rec["grid"]) == len(GRID) == 9
    p = make_pad(20.0, 0.002).as_dict()
    for k, v in rec["pads"][CENTRAL].items():
        if isinstance(v, float):
            assert v == pytest.approx(p[k], rel=1e-6, abs=1e-12), k
    assert rec["pads"][f"{CENTRAL}_zeta0.3"]["zeta"] == 0.3


def _row(seed, n_exact, flags, settle=0.001, library="three_contact", duty=1.0, n_prim=4):
    """A replay row with per-primitive pose/retention flags (``flags``: list of (pose_ok, retain_ok))."""
    info = [dict(pose_ok=p, retain_ok=r, ok_rot=p and r, drop=False, contact_duty=[duty, duty, duty]) for p, r in flags]
    ff = lambda key: next((k for k, i in enumerate(info) if not i[key]), len(info))   # noqa: E731
    return dict(seed=seed, library=library, kind="T", command=0, n_primitives=n_prim, n_exact=n_exact, settle_err_m=settle, info=info,
                n_sim_rot=ff("ok_rot"), n_pose=ff("pose_ok"), n_retain=ff("retain_ok"))


def test_outcomes_and_verdict():
    rows = [_row(0, 3, [(1, 1), (1, 0), (1, 1), (0, 0)]),             # holds: strict 1 of 3 valid, pose 3, retention (end) 1
            _row(1, 2, [(1, 1), (1, 1), (1, 1), (1, 1)], duty=0.4),   # holds: strict 2, duty retention 0 (lanes < 50 %)
            _row(2, 4, [(1, 1)] * 4, settle=0.01)]                    # does not hold: excluded
    o = outcomes(rows, "three_contact")
    assert o["seeds_hold"] == 2 and o["exactly_valid"] == 5
    assert o["counts"]["strict"] == 3 and o["counts"]["pose"] == 5 and o["counts"]["retain_end"] == 3
    assert o["counts"]["retain_duty"] == 3 and o["counts"]["pose_duty"] == 3
    assert o["fp_strict"] == pytest.approx(0.4)
    rigid = dict(a=0.5, b=0.2, c=0.3)
    assert verdict(dict(a=0.1, b=0.05, c=-0.2), rigid) == "resolved"
    assert verdict(dict(a=0.3, b=0.15, c=0.2), rigid) == "not resolved"
    assert verdict(dict(a=0.3, b=0.05, c=0.2), rigid) == "partial"


def test_primary_endpoint():
    rows = [_row(0, 1, [(1, 1), (1, 1), (0, 1), (1, 1)]),   # 2 successes before the first failure, 1 exactly valid
            _row(0, 4, [(1, 1)] * 4),                        # second command of seed 0: 4
            _row(3, 0, [(1, 0), (1, 1), (1, 1), (1, 1)]),    # seed 3: 0 (retention fails first)
            _row(1, 4, [(1, 1)] * 4, library="two_finger")]  # other library: ignored
    assert per_seed_counts(rows) == {0: 6, 3: 0}
    assert endpoint(rows) == pytest.approx(6 / 12)                       # regardless of the exact verdict
    assert endpoint(rows, capped=True) == pytest.approx(5 / 12)          # the 9.2 measure
    assert endpoint(rows, seeds=[0], per="subset") == pytest.approx(6.0)
    assert endpoint(rows, seeds=[3]) == 0.0


def test_subset_measures():
    d = dict(n=3, alpha_min=[1.2, 0.4, 1.0], cells=[80, 8, 40], bins=[1382, 10, 461], q=[[0.0]] * 3)
    m = subset_measures(d)
    assert m["s_task"] == [0, 2] and m["n_task"] == 2 and m["frac_budget"] == pytest.approx(2 / 12)
    sc = seed_score(d["cells"], d["bins"])
    assert m["full_all"] == pytest.approx(sc.mean()) and m["full_task"] == pytest.approx(sc[[0, 2]].mean())
    assert subset_measures(dict(n=1, alpha_min=[0.5], cells=[1], bins=[1], q=[[0.0]]))["full_task"] == 0.0
