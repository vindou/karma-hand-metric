"""URDF round trip (SPEC §10.1): two-finger exports are byte-identical to the Phase 0 fixture
files written by optimize/hand_family.py; the JSON design record round-trips; three-finger
YAMLs list every tip."""
import json
from pathlib import Path

import jax
import numpy as np
import pytest
import yaml

from dkarma.core import Topology, random_params
from dkarma.export import (to_urdf, to_robot_yaml, write_design, design_to_dict, design_from_dict,
                           from_hand_spec, to_hand_spec)
from .conftest import DESIGNS, DESIGN_JSON

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name,stem", [("baseline", "baseline_hand"), ("T", "opt_T"), ("A1", "opt_A1")])
def test_two_finger_export_is_byte_identical_to_fixture_files(name, stem):
    spec = json.loads(DESIGN_JSON[name].read_text())
    topo, params, nm = from_hand_spec(spec)
    assert nm == stem
    assert to_urdf(topo, params, nm) == (DESIGNS / name / f"{stem}.urdf").read_text()
    assert to_robot_yaml(topo, params, f"{stem}.urdf", nm) == yaml.safe_load((DESIGNS / name / f"robot_{stem}.yaml").read_text())
    back = to_hand_spec(topo, params, nm)
    for key in ("thumb", "index"):
        for k in ("knuckle_xyz", "knuckle_rotvec", "axis_theta", "axis_phi", "link_len", "lim_lo", "lim_hi"):
            np.testing.assert_allclose(back[key][k], spec[key][k], atol=1e-12)


def test_design_json_roundtrip():
    topo = Topology(fingers=("index", "middle", "thumb"), n_joints=(4, 5, 3), pair=("thumb", "index"))
    params = random_params(topo, jax.random.PRNGKey(1))
    topo2, params2, name2 = design_from_dict(json.loads(json.dumps(design_to_dict(topo, params, "three"))))
    assert topo2 == topo and name2 == "three"
    mask = topo.joint_mask
    for k in params._fields:
        a, b = np.asarray(getattr(params, k)), np.asarray(getattr(params2, k))
        if a.ndim == 2 and a.shape[1] == topo.K:
            a, b = a[mask], b[mask]
        np.testing.assert_allclose(a, b, atol=1e-15)


def test_three_finger_yaml_lists_every_tip(tmp_path):
    topo = Topology.three_finger(4, names=("thumb", "middle", "index"))     # pair = (thumb, middle)
    params = random_params(topo, jax.random.PRNGKey(2))
    u, y, j = write_design(topo, params, tmp_path, "tri")
    cfg = yaml.safe_load(Path(y).read_text())
    assert cfg["fingers"]["thumb"]["active_joints"][0] == "thumb_j0"
    assert cfg["fingers"]["index"]["active_joints"][0] == "middle_j0"
    assert set(cfg["links"]) == {"thumb_l3", "index_l3", "middle_l3"}
    assert cfg["seed"] == {"thumb_link": "thumb_l3", "index_link": "middle_l3"}
    assert '<robot name="tri">' in Path(u).read_text()
