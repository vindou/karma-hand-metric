"""Shared fixtures for dkarma tests: the Phase 0 designs (baseline, T, A1) as
(Topology, HandParams, name), random three-finger designs, and the planar toy."""
from __future__ import annotations

import json
from pathlib import Path

import jax
import numpy as np
import pytest

from dkarma.core import Topology, HandParams, params_from_fingers, random_params
from dkarma.export import from_hand_spec

ROOT = Path(__file__).resolve().parents[2]
DESIGNS = ROOT / "dkarma" / "fixtures" / "designs"
DESIGN_JSON = {"baseline": DESIGNS / "baseline" / "baseline_hand.json",
               "T": DESIGNS / "T" / "opt_T.json",
               "A1": DESIGNS / "A1" / "opt_A1.json"}


def load_design(name: str):
    """(topo, params, name): a two-finger fixture design or a random three-finger hand."""
    if name in DESIGN_JSON:
        return from_hand_spec(json.loads(DESIGN_JSON[name].read_text()))
    if name == "rand3x4":
        topo = Topology.three_finger(4)
        return topo, random_params(topo, jax.random.PRNGKey(7)), name
    if name == "rand345":
        topo = Topology(fingers=("index", "middle", "thumb"), n_joints=(4, 5, 3), pair=("thumb", "index"))
        return topo, random_params(topo, jax.random.PRNGKey(11)), name
    raise KeyError(name)


DESIGN_NAMES = ["baseline", "T", "A1", "rand3x4", "rand345"]


@pytest.fixture(params=DESIGN_NAMES)
def design(request):
    return load_design(request.param)


@pytest.fixture(autouse=True, scope="module")
def _clear_jax_caches_between_modules():
    """Host-memory hygiene (Phase 9): compiled programs accumulate over the suite (43 GB host); every
    module starts from empty JAX caches. Values are unaffected."""
    yield
    jax.clear_caches()


def planar_toy(l1=0.05, l2=0.04, r=0.01):
    """Two-link planar finger (both axes +z, base at the origin) plus a far-away one-joint
    dummy finger (the topology needs a pair). Returns (topo, params, r); the planar finger
    is flat finger 0 ("finger"), capsules 0 and 1."""
    topo = Topology(fingers=("finger", "other"), n_joints=(2, 1), pair=("finger", "other"))
    params = params_from_fingers(topo, {
        "finger": dict(base_xyz=[0, 0, 0], base_rotvec=[0, 0, 0], axis_theta=[np.pi / 2, np.pi / 2],
                       axis_phi=[np.pi / 2, np.pi / 2], link_len=[l1, l2], lim_lo=[1.0, 1.0], lim_hi=[1.0, 1.0]),
        "other": dict(base_xyz=[0, 0.5, 0], base_rotvec=[0, 0, 0], axis_theta=[np.pi / 2], axis_phi=[np.pi / 2],
                      link_len=[0.03], lim_lo=[1.0], lim_hi=[1.0]),
    }, cap_radius=r)
    return topo, params, r


def random_q(topo: Topology, params: HandParams, n: int, seed: int = 0) -> np.ndarray:
    """Configurations uniformly inside the joint limits, ``[n, J]`` in flat joint order."""
    rng = np.random.RandomState(seed)
    lo = -np.asarray(params.lim_lo)[topo.joint_mask]
    hi = np.asarray(params.lim_hi)[topo.joint_mask]
    return lo + rng.rand(n, topo.J) * (hi - lo)
