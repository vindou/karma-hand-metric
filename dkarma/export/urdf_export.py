"""Export a design as a KaRMA-readable URDF + robot YAML, and load it back through the
reference (Pinocchio) path for oracle tests and exact scoring (SPEC.md §9.1 ``export/``).

The writer generalizes ``optimize/hand_family.py:HandSpec`` to N fingers: for a
two-finger (thumb, index) design the files are byte-identical to the Phase 0
fixture files (``dkarma/tests/test_export.py``). Fingers are written pair-first
(thumb-like, index-like, then the rest); Pinocchio re-sorts siblings by name, so
the file order does not affect the flat model.

The reference reads only the joint tree (origins, axes, limits), the inertial
CoM (tip direction; placed at half the link along +x) and the robot YAML
(active joints, contact links, seed links, ``tip_length_m`` of every terminal
link). ``effort``/``velocity`` are placeholders no evaluator reads (CLAUDE.md
rule 3).
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from ..core.topology import Topology
from ..core.morphology import (HandParams, params_from_fingers, params_to_fingers, from_hand_spec as _from_hand_spec,
                               to_hand_spec_dict, DEFAULT_CAPSULE_RADIUS_M)


def _axis(theta: float, phi: float) -> np.ndarray:
    a = np.array([math.cos(theta), math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi)])
    return a / np.linalg.norm(a)


def _rpy(rotvec) -> np.ndarray:
    # URDF rpy = extrinsic x-y-z = scipy lowercase 'xyz' (as hand_family).
    return Rotation.from_rotvec(np.asarray(rotvec, dtype=float)).as_euler("xyz")


def _write_order(topo: Topology) -> list[str]:
    return list(topo.pair) + [f for f in topo.fingers if f not in topo.pair]


# ── URDF ─────────────────────────────────────────────────────────────────────

def to_urdf(topo: Topology, params: HandParams, name: str = "opt_hand") -> str:
    fd = params_to_fingers(topo, params)
    lines = ['<?xml version="1.0"?>', f'<robot name="{name}">', '  <link name="palm"/>']
    for fname in _write_order(topo):
        f = fd[fname]
        parent = "palm"
        for k in range(len(f["axis_theta"])):
            xyz = f["base_xyz"] if k == 0 else [f["link_len"][k - 1], 0.0, 0.0]
            rpy = _rpy(f["base_rotvec"]) if k == 0 else [0.0, 0.0, 0.0]
            ax = _axis(f["axis_theta"][k], f["axis_phi"][k])
            ln = f"{fname}_l{k}"
            lines += [
                f'  <joint name="{fname}_j{k}" type="revolute">',
                f'    <origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>',
                f'    <parent link="{parent}"/>',
                f'    <child link="{ln}"/>',
                f'    <axis xyz="{ax[0]:.9f} {ax[1]:.9f} {ax[2]:.9f}"/>',
                f'    <limit lower="{-f["lim_lo"][k]:.6f}" upper="{f["lim_hi"][k]:.6f}" effort="1" velocity="1"/>',
                '  </joint>',
                f'  <link name="{ln}">',
                '    <inertial>',
                f'      <origin xyz="{0.5 * f["link_len"][k]:.9f} 0 0" rpy="0 0 0"/>',
                '      <mass value="0.01"/>',
                '      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>',
                '    </inertial>',
                '  </link>',
            ]
            parent = ln
    lines.append('</robot>')
    return "\n".join(lines) + "\n"


# ── robot YAML ───────────────────────────────────────────────────────────────

def to_robot_yaml(topo: Topology, params: HandParams, urdf_filename: str, name: str = "opt_hand") -> dict:
    """Per-hand KaRMA config: ``topo.pair`` are thumb/index for the two-finger reference; the
    other fingers stay in the URDF as collision obstacles and get their ``tip_length_m``."""
    fd = params_to_fingers(topo, params)

    def n_of(fname):
        return len(fd[fname]["axis_theta"])

    def links_of(fname):
        return [f"{fname}_l{k}" for k in reversed(range(n_of(fname)))]

    def joints_of(fname):
        return [f"{fname}_j{k}" for k in range(n_of(fname))]

    def tip_link(fname):
        return f"{fname}_l{n_of(fname) - 1}"

    t, i = topo.pair
    links = {tip_link(t): {"tip_length_m": float(fd[t]["link_len"][-1])},
             tip_link(i): {"tip_length_m": float(fd[i]["link_len"][-1])}}
    for fname in topo.fingers:
        if fname not in topo.pair:
            links[tip_link(fname)] = {"tip_length_m": float(fd[fname]["link_len"][-1])}
    return {
        "robot_name": name,
        "urdf_path": urdf_filename,
        "base_link": "palm",
        "fingers": {"thumb": {"active_joints": joints_of(t), "contact_links": links_of(t)},
                    "index": {"active_joints": joints_of(i), "contact_links": links_of(i)}},
        "seed": {"thumb_link": tip_link(t), "index_link": tip_link(i)},
        "links": links,
    }


# ── JSON design record ───────────────────────────────────────────────────────

def design_to_dict(topo: Topology, params: HandParams, name: str = "opt_hand") -> dict:
    return {"name": name, "fingers": list(topo.fingers), "n_joints": list(topo.n_joints), "pair": list(topo.pair),
            "cap_radius_m": np.asarray(params.cap_radius, dtype=float).tolist(),
            "finger_params": params_to_fingers(topo, params)}


def design_from_dict(d: dict) -> tuple[Topology, HandParams, str]:
    topo = Topology(fingers=tuple(d["fingers"]), n_joints=tuple(int(n) for n in d["n_joints"]), pair=tuple(d["pair"]))
    r = d.get("cap_radius_m", DEFAULT_CAPSULE_RADIUS_M)
    return topo, params_from_fingers(topo, d["finger_params"], np.asarray(r)), d.get("name", "opt_hand")


def write_design(topo: Topology, params: HandParams, dirpath, name: str = "opt_hand") -> tuple[Path, Path, Path]:
    """Write ``<name>.urdf``, ``robot_<name>.yaml`` and ``<name>.design.json`` into ``dirpath``."""
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    urdf_path = dirpath / f"{name}.urdf"
    yaml_path = dirpath / f"robot_{name}.yaml"
    json_path = dirpath / f"{name}.design.json"
    urdf_path.write_text(to_urdf(topo, params, name))
    with open(yaml_path, "w") as fh:
        yaml.safe_dump(to_robot_yaml(topo, params, urdf_path.name, name), fh, sort_keys=False)
    json_path.write_text(json.dumps(design_to_dict(topo, params, name), indent=2))
    return urdf_path, yaml_path, json_path


# ── HandSpec interop ─────────────────────────────────────────────────────────

def from_hand_spec(spec, cap_radius=DEFAULT_CAPSULE_RADIUS_M) -> tuple[Topology, HandParams, str]:
    """HandSpec object or JSON (str/dict) -> (Topology, HandParams, name)."""
    if isinstance(spec, str):
        spec = json.loads(spec)
    topo, params = _from_hand_spec(spec, cap_radius)
    name = spec.get("name", "opt_hand") if isinstance(spec, dict) else getattr(spec, "name", "opt_hand")
    return topo, params, name


def to_hand_spec(topo: Topology, params: HandParams, name: str = "opt_hand") -> dict:
    return to_hand_spec_dict(topo, params, name)


# ── reference round trip ─────────────────────────────────────────────────────

def load_reference(topo: Topology, params: HandParams, workdir, name: str = "opt_hand", metric_cfg=None):
    """Write the design, load it through ``karma`` (URDF -> Pinocchio -> capsules) and
    ``karma_jax.model.build_flat_model``. Returns ``(cfg, ctx, flat)``."""
    from karma.config import load_config
    from karma.robot import load_robot
    from karma_jax.model import build_flat_model
    from ..fixtures import METRIC_CFG

    _, yaml_path, _ = write_design(topo, params, workdir, name)
    cfg = load_config(metric_cfg or METRIC_CFG, yaml_path)
    cfg = dataclasses.replace(cfg, out_dir=Path(workdir))
    ctx = load_robot(cfg.urdf_path, cfg)
    return cfg, ctx, build_flat_model(ctx, cfg)


__all__ = ["to_urdf", "to_robot_yaml", "write_design", "design_to_dict", "design_from_dict", "from_hand_spec",
           "to_hand_spec", "load_reference"]
