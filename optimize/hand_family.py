"""Parametric two-finger (thumb + index) serial-chain hand family.

A design is a :class:`HandSpec`. :class:`HandFamily` maps a flat real vector
(the genotype the optimizer searches over) to a HandSpec, and a HandSpec writes
itself out as a KaRMA-readable URDF + robot YAML pair.

Conventions
-----------
* The palm is the base link and sits at the world origin.
* The INDEX knuckle is pinned to the palm frame (identity pose). KaRMA is
  exactly invariant to rigid transforms of the whole hand, so a free index
  knuckle pose would be 6 redundant parameters. The THUMB knuckle pose
  (xyz + rotation vector) is free.
* Each finger is a serial chain of N revolute joints. At the neutral pose
  (q = 0) the finger is straight along its own +x axis. Joint i sits at
  distance link_len[i-1] along the previous link, with zero rotation; the
  joint axis is a unit vector in the link frame given by a polar angle theta
  (measured from +x) and an azimuth phi about +x, so theta = pi/2 is a
  flexion/abduction-type axis and theta -> 0 degenerates to a twist axis.
* link_len[-1] is the tip length (last joint origin -> physical fingertip),
  which KaRMA reads from the robot YAML as tip_length_m.
* The index wire (sum of link_len) is pinned to ``index_wire_m``. KaRMA is
  exactly scale-invariant, so overall size is a flat direction; pinning it
  removes that redundancy. The thumb's lengths are absolute.
* Every link of both fingers is declared a contact link, so contact
  switching is allowed on every phalanx and thumb-index self-collision is
  checked over the whole of both fingers (the strictest reading).
* Joint limits are [-lim_lo, +lim_hi] with both sides >= 15 deg, so q = 0 is
  always inside the limits (KaRMA samples around the neutral pose) and the
  total range is >= 30 deg, which keeps L_ref's "advance the knuckle past a
  low-ROM joint" rule from being gamed by a token proximal joint.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation


# ── Design representation ────────────────────────────────────────────────────

@dataclass
class FingerSpec:
    name: str
    knuckle_xyz: list[float]       # (3,) metres, in the palm frame
    knuckle_rotvec: list[float]    # (3,) rotation vector, radians
    axis_theta: list[float]        # (N,) polar angle of joint axis from link +x
    axis_phi: list[float]          # (N,) azimuth of joint axis about link +x
    link_len: list[float]          # (N,) metres; link_len[-1] is the tip length
    lim_lo: list[float]            # (N,) radians, lower limit is -lim_lo (>= 0)
    lim_hi: list[float]            # (N,) radians, upper limit is +lim_hi (>= 0)

    @property
    def n_joints(self) -> int:
        return len(self.axis_theta)

    @property
    def wire_m(self) -> float:
        return float(sum(self.link_len))

    def axis(self, i: int) -> np.ndarray:
        th, ph = self.axis_theta[i], self.axis_phi[i]
        a = np.array([math.cos(th), math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph)])
        return a / np.linalg.norm(a)

    def joint_name(self, i: int) -> str:
        return f"{self.name}_j{i}"

    def link_name(self, i: int) -> str:
        return f"{self.name}_l{i}"

    def knuckle_rpy(self) -> np.ndarray:
        # URDF rpy is R = Rz(yaw) Ry(pitch) Rx(roll) = extrinsic x-y-z, which is
        # scipy's lowercase 'xyz'.
        return Rotation.from_rotvec(np.asarray(self.knuckle_rotvec, dtype=float)).as_euler("xyz")


@dataclass
class HandSpec:
    thumb: FingerSpec
    index: FingerSpec
    name: str = "opt_hand"

    # ── serialization ──
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "HandSpec":
        d = json.loads(s)
        return HandSpec(thumb=FingerSpec(**d["thumb"]), index=FingerSpec(**d["index"]), name=d.get("name", "opt_hand"))

    # ── URDF ──
    def to_urdf(self) -> str:
        lines = ['<?xml version="1.0"?>', f'<robot name="{self.name}">', '  <link name="palm"/>']
        for f in (self.thumb, self.index):
            parent = "palm"
            for i in range(f.n_joints):
                if i == 0:
                    xyz = f.knuckle_xyz
                    rpy = f.knuckle_rpy()
                else:
                    xyz = [f.link_len[i - 1], 0.0, 0.0]
                    rpy = [0.0, 0.0, 0.0]
                ax = f.axis(i)
                ln = f.link_name(i)
                lines += [
                    f'  <joint name="{f.joint_name(i)}" type="revolute">',
                    f'    <origin xyz="{xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}" rpy="{rpy[0]:.9f} {rpy[1]:.9f} {rpy[2]:.9f}"/>',
                    f'    <parent link="{parent}"/>',
                    f'    <child link="{ln}"/>',
                    f'    <axis xyz="{ax[0]:.9f} {ax[1]:.9f} {ax[2]:.9f}"/>',
                    f'    <limit lower="{-f.lim_lo[i]:.6f}" upper="{f.lim_hi[i]:.6f}" effort="1" velocity="1"/>',
                    '  </joint>',
                    f'  <link name="{ln}">',
                    '    <inertial>',
                    # CoM half-way along the link so KaRMA's tip-direction
                    # auto-detection (2x CoM) agrees with the chain direction (+x).
                    f'      <origin xyz="{0.5 * f.link_len[i]:.9f} 0 0" rpy="0 0 0"/>',
                    '      <mass value="0.01"/>',
                    '      <inertia ixx="1e-7" ixy="0" ixz="0" iyy="1e-7" iyz="0" izz="1e-7"/>',
                    '    </inertial>',
                    '  </link>',
                ]
                parent = ln
        lines.append('</robot>')
        return "\n".join(lines) + "\n"

    # ── robot YAML (the KaRMA per-hand config) ──
    def to_robot_yaml(self, urdf_filename: str) -> dict:
        def links_of(f: FingerSpec) -> list[str]:
            # distal first, like the shipped configs
            return [f.link_name(i) for i in reversed(range(f.n_joints))]

        return {
            "robot_name": self.name,
            "urdf_path": urdf_filename,
            "base_link": "palm",
            "fingers": {
                "thumb": {
                    "active_joints": [self.thumb.joint_name(i) for i in range(self.thumb.n_joints)],
                    "contact_links": links_of(self.thumb),
                },
                "index": {
                    "active_joints": [self.index.joint_name(i) for i in range(self.index.n_joints)],
                    "contact_links": links_of(self.index),
                },
            },
            "seed": {
                "thumb_link": self.thumb.link_name(self.thumb.n_joints - 1),
                "index_link": self.index.link_name(self.index.n_joints - 1),
            },
            "links": {
                self.thumb.link_name(self.thumb.n_joints - 1): {"tip_length_m": float(self.thumb.link_len[-1])},
                self.index.link_name(self.index.n_joints - 1): {"tip_length_m": float(self.index.link_len[-1])},
            },
        }

    def write(self, dirpath: str | Path) -> tuple[Path, Path]:
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        urdf_path = dirpath / f"{self.name}.urdf"
        yaml_path = dirpath / f"robot_{self.name}.yaml"
        urdf_path.write_text(self.to_urdf())
        with open(yaml_path, "w") as fh:
            yaml.safe_dump(self.to_robot_yaml(urdf_path.name), fh, sort_keys=False)
        (dirpath / f"{self.name}.json").write_text(self.to_json())
        return urdf_path, yaml_path

    # ── human-readable summary ──
    def describe(self) -> str:
        out = []
        for f in (self.thumb, self.index):
            out.append(f"{f.name}: knuckle xyz = {np.round(np.array(f.knuckle_xyz) * 1e3, 1).tolist()} mm, "
                       f"rotvec = {np.round(f.knuckle_rotvec, 3).tolist()} rad, wire = {f.wire_m * 1e3:.1f} mm")
            for i in range(f.n_joints):
                ax = f.axis(i)
                kind = "twist" if abs(ax[0]) > 0.9 else ("flex/abd" if abs(ax[0]) < 0.3 else "oblique")
                seg = "tip" if i == f.n_joints - 1 else "link"
                out.append(f"   j{i}: axis = ({ax[0]:+.2f},{ax[1]:+.2f},{ax[2]:+.2f}) [{kind}], "
                           f"limits = [{-math.degrees(f.lim_lo[i]):.0f}, {math.degrees(f.lim_hi[i]):.0f}] deg, "
                           f"{seg} = {f.link_len[i] * 1e3:.1f} mm")
        return "\n".join(out)


# ── Genotype <-> HandSpec ────────────────────────────────────────────────────

@dataclass
class FamilyConfig:
    n_thumb: int = 4
    n_index: int = 4
    index_wire_m: float = 0.100
    thumb_xyz_max_m: float = 0.080
    link_len_min_m: float = 0.012
    link_len_max_m: float = 0.080
    index_w_min: float = 0.10
    index_w_max: float = 1.00
    lim_min_rad: float = math.radians(15.0)
    lim_max_rad: float = math.radians(75.0)


class HandFamily:
    """Flat-vector genotype for a HandSpec with bounds and per-coordinate scales.

    Layout: [thumb xyz(3), thumb rotvec(3), thumb joints N_t x (theta, phi, len, lo, hi),
             index joints N_i x (theta, phi, w, lo, hi)]
    Angles (rotvec, theta, phi) are unbounded (periodic); everything else is bounded.
    """

    def __init__(self, fc: FamilyConfig | None = None):
        self.fc = fc or FamilyConfig()
        fc = self.fc
        names, lower, upper, stds = [], [], [], []

        def add(n, lo, hi, sd):
            names.append(n); lower.append(lo); upper.append(hi); stds.append(sd)

        for c in "xyz":
            add(f"thumb_knuckle_{c}", -fc.thumb_xyz_max_m, fc.thumb_xyz_max_m, 0.02)
        for c in "xyz":
            add(f"thumb_rotvec_{c}", None, None, 0.4)
        for k in range(fc.n_thumb):
            add(f"thumb_j{k}_theta", None, None, 0.4)
            add(f"thumb_j{k}_phi", None, None, 0.6)
            add(f"thumb_j{k}_len", fc.link_len_min_m, fc.link_len_max_m, 0.012)
            add(f"thumb_j{k}_lo", fc.lim_min_rad, fc.lim_max_rad, 0.25)
            add(f"thumb_j{k}_hi", fc.lim_min_rad, fc.lim_max_rad, 0.25)
        for k in range(fc.n_index):
            add(f"index_j{k}_theta", None, None, 0.4)
            add(f"index_j{k}_phi", None, None, 0.6)
            add(f"index_j{k}_w", fc.index_w_min, fc.index_w_max, 0.15)
            add(f"index_j{k}_lo", fc.lim_min_rad, fc.lim_max_rad, 0.25)
            add(f"index_j{k}_hi", fc.lim_min_rad, fc.lim_max_rad, 0.25)

        self.names = names
        self.lower = lower
        self.upper = upper
        self.stds = np.array(stds, dtype=float)
        self.dim = len(names)

    # bounds in the format `cma` expects (None = unbounded)
    def cma_bounds(self) -> list[list]:
        return [list(self.lower), list(self.upper)]

    def clip(self, x: np.ndarray) -> np.ndarray:
        x = np.array(x, dtype=float).copy()
        for i, (lo, hi) in enumerate(zip(self.lower, self.upper)):
            if lo is not None:
                x[i] = min(max(x[i], lo), hi)
        return x

    def decode(self, x: np.ndarray) -> HandSpec:
        x = self.clip(x)
        fc = self.fc
        i = 0
        xyz = x[i:i + 3].tolist(); i += 3
        rv = x[i:i + 3].tolist(); i += 3
        t_th, t_ph, t_len, t_lo, t_hi = [], [], [], [], []
        for _ in range(fc.n_thumb):
            t_th.append(float(x[i])); t_ph.append(float(x[i + 1])); t_len.append(float(x[i + 2]))
            t_lo.append(float(x[i + 3])); t_hi.append(float(x[i + 4])); i += 5
        i_th, i_ph, i_w, i_lo, i_hi = [], [], [], [], []
        for _ in range(fc.n_index):
            i_th.append(float(x[i])); i_ph.append(float(x[i + 1])); i_w.append(float(x[i + 2]))
            i_lo.append(float(x[i + 3])); i_hi.append(float(x[i + 4])); i += 5
        w = np.array(i_w); i_len = (fc.index_wire_m * w / w.sum()).tolist()
        thumb = FingerSpec("thumb", xyz, rv, t_th, t_ph, t_len, t_lo, t_hi)
        index = FingerSpec("index", [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], i_th, i_ph, i_len, i_lo, i_hi)
        return HandSpec(thumb=thumb, index=index)

    def encode(self, hand: HandSpec) -> np.ndarray:
        """Inverse of decode (index lengths become weights proportional to length)."""
        x = []
        x += list(hand.thumb.knuckle_xyz) + list(hand.thumb.knuckle_rotvec)
        for k in range(hand.thumb.n_joints):
            x += [hand.thumb.axis_theta[k], hand.thumb.axis_phi[k], hand.thumb.link_len[k],
                  hand.thumb.lim_lo[k], hand.thumb.lim_hi[k]]
        wire = sum(hand.index.link_len)
        for k in range(hand.index.n_joints):
            x += [hand.index.axis_theta[k], hand.index.axis_phi[k], hand.index.link_len[k] / wire,
                  hand.index.lim_lo[k], hand.index.lim_hi[k]]
        return np.array(x, dtype=float)

    # ── a hand-drawn anthropomorphic starting point ──
    def anthropomorphic_x0(self) -> np.ndarray:
        """Index along +x in the palm plane (xy), curling toward -z; thumb set
        proximal/ulnar and rotated into opposition. Only meant to be a feasible
        starting basin; the optimizer is free to leave it."""
        fc = self.fc
        d = math.radians
        half_pi = math.pi / 2
        # thumb frame: point along +x+y (yaw 60 deg), then roll into opposition
        R = Rotation.from_euler("z", 60, degrees=True) * Rotation.from_euler("x", -70, degrees=True)
        rv = R.as_rotvec().tolist()
        x = [-0.015, -0.040, -0.020] + rv
        # thumb joints: CMC abduction (about local z), then three flexions (about local y)
        thumb_j = [
            (half_pi, half_pi, 0.015, d(45), d(45)),
            (half_pi, 0.0,     0.045, d(15), d(60)),
            (half_pi, 0.0,     0.030, d(15), d(60)),
            (half_pi, 0.0,     0.025, d(15), d(75)),
        ]
        for k in range(fc.n_thumb):
            x += list(thumb_j[min(k, len(thumb_j) - 1)])
        # index joints: MCP abduction (about z), then flexions (about y)
        index_j = [
            (half_pi, half_pi, 0.12, d(20), d(20)),
            (half_pi, 0.0,     0.40, d(15), d(75)),
            (half_pi, 0.0,     0.25, d(15), d(75)),
            (half_pi, 0.0,     0.23, d(15), d(75)),
        ]
        for k in range(fc.n_index):
            x += list(index_j[min(k, len(index_j) - 1)])
        return np.array(x, dtype=float)


def design_hash(x: np.ndarray) -> str:
    import hashlib
    return hashlib.sha1(np.asarray(x, dtype=float).round(9).tobytes()).hexdigest()[:10]
