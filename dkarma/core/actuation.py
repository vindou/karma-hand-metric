"""Actuator and load model for Act-KaRMA (SPEC.md §2.1, §5.1): joint-side torque ceilings,
link masses for the hand's own gravity torque, the grasp matrix and the contact-force
to joint-torque map.

Conventions (SI, palm frame = world frame, gravity from the task spec):

* ``f_c`` is the force the finger applies **on the object** at contact ``c``; object
  equilibrium is ``G f + w_ext = 0`` with ``G = [I ... I; [r_1]x ... [r_Nc]x]`` and
  ``r_c`` the contact point relative to the sphere centre (``r_c = -r_s n_c``).
* The finger feels ``-f_c``. Static equilibrium of the finger chain,
  ``tau + sum_links J_com^T (m g) + sum_c J_c^T (-f_c) = 0``, gives the motor torque
  ``tau = J_c^T f + tau_grav`` with ``tau_grav = -sum_links J_com^T m g`` (the torque that
  holds the links up). ``J_c`` is the surface-point Jacobian of the contact.
* Torque ceilings are per joint and identical across designs (D3, D13). URDF ``effort``
  values are never read (CLAUDE.md rule 3).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import yaml

from karma_jax.kinematics import FKResult, frame_jacobian, skew
from karma_jax.linalg import mm

from .morphology import HandModel
from .object_model import DEFAULT_SPEC


@dataclass(frozen=True)
class ActuationSpec:
    """The ``actuation`` block of ``task_spec.yaml``."""
    servo: str = "XL330-M077-T"
    stall_torque_nm: float = 0.215
    torque_limit_nm: float = 0.108
    torque_limit_basis: str = "0.5 x stall at 5 V, intermittent duty"
    transmission_efficiency_assumed: float = 1.0
    include_link_gravity: bool = True
    link_mass_kg: float = 0.028
    terminal_link_mass_kg: float = 0.010
    ablations: dict = field(default_factory=lambda: {"continuous": 0.043, "stall": 0.215})

    def with_torque(self, tau_bar_nm: float) -> "ActuationSpec":
        return ActuationSpec(**{**self.__dict__, "torque_limit_nm": float(tau_bar_nm)})


def load_actuation_spec(path: str | Path = DEFAULT_SPEC) -> ActuationSpec:
    raw = yaml.safe_load(Path(path).read_text())["actuation"]
    return ActuationSpec(servo=str(raw.get("servo", "XL330-M077-T")), stall_torque_nm=float(raw["stall_torque_nm"]),
                         torque_limit_nm=float(raw["torque_limit_nm"]),
                         torque_limit_basis=str(raw.get("torque_limit_basis", "")),
                         transmission_efficiency_assumed=float(raw.get("transmission_efficiency_assumed", 1.0)),
                         include_link_gravity=bool(raw.get("include_link_gravity", True)),
                         link_mass_kg=float(raw.get("link_mass_kg", 0.028)),
                         terminal_link_mass_kg=float(raw.get("terminal_link_mass_kg", 0.010)))


class ActuationParams(NamedTuple):
    """Device-side actuation data (pytree)."""
    tau_bar: object        # [J] N·m per joint
    link_mass: object      # [C] kg per link (capsule)
    gravity: object        # [3] m/s^2 in the palm frame
    w_ext: object          # [6] external wrench on the object (force, moment), N and N·m
    mu: object             # friction coefficient


def actuation_params(hm: HandModel, spec: ActuationSpec, gravity_m_s2, object_mass_kg: float, mu: float,
                     dtype=None) -> ActuationParams:
    """Per-design actuation arrays: uniform torque ceiling, link masses by position in the finger,
    object weight as the external wrench (force only; gravity acts at the sphere centre)."""
    dtype = dtype or hm.dm.t_joint.dtype
    J = hm.dm.q_ids.shape[0]
    C = hm.cap_radius.shape[0]
    tau_bar = jnp.full((J,), spec.torque_limit_nm * spec.transmission_efficiency_assumed, dtype)
    is_last = hm.dm.cap_distal < 0
    mass = jnp.where(is_last, spec.terminal_link_mass_kg, spec.link_mass_kg).astype(dtype)
    if not spec.include_link_gravity:
        mass = jnp.zeros((C,), dtype)
    g = jnp.asarray(gravity_m_s2, dtype)
    w_ext = jnp.concatenate([object_mass_kg * g, jnp.zeros((3,), dtype)])
    return ActuationParams(tau_bar=tau_bar, link_mass=mass, gravity=g, w_ext=w_ext, mu=jnp.asarray(mu, dtype))


def grasp_matrix(p_contacts, p_centre, valid):
    """``G [6, 3Cn]`` for contact points ``p_contacts [Cn,3]`` on the object about ``p_centre``;
    invalid lanes contribute zero columns."""
    r = (p_contacts - p_centre[None, :]) * valid[:, None]
    top = jnp.concatenate([jnp.eye(3, dtype=r.dtype) * v for v in valid.astype(r.dtype)], axis=1)  # [3,3Cn]
    bottom = jnp.concatenate([skew(r[c]) for c in range(r.shape[0])], axis=1)                       # [3,3Cn]
    return jnp.concatenate([top, bottom], axis=0)


def contact_torque_map(J_surf, valid):
    """``J_c^T`` stacked: ``[J, 3Cn]`` mapping contact forces to joint torques."""
    Cn, _, J = J_surf.shape
    cols = [J_surf[c].T * valid[c].astype(J_surf.dtype) for c in range(Cn)]
    return jnp.concatenate(cols, axis=1)


def angular_jacobians(J_w, valid):
    """``Jw [Cn, 3, J]`` with invalid lanes zeroed (input from ``ContactSet.Jw``, full-q columns)."""
    return J_w * valid.astype(J_w.dtype)[:, None, None]


def gravity_torque(hm: HandModel, fkr: FKResult, act: ActuationParams):
    """``tau_grav [J] = -sum_links J_com^T (m g)``: motor torque holding the links against gravity.
    Each link's mass sits at the midpoint of its capsule axis."""
    from .kinematics import capsule_endpoints
    a, b = capsule_endpoints(hm, fkr)
    com = 0.5 * (a + b)                                             # [C,3]
    C = com.shape[0]

    def one(c):
        f = hm.dm.cap_frame[c]
        Jv, Jw = frame_jacobian(hm.dm, fkr, f)
        Jcom = Jv - mm(skew(com[c] - fkr.p_frame[f]), Jw)           # [3,J]
        return -(Jcom.T @ (act.link_mass[c] * act.gravity))
    return jnp.sum(jax.vmap(one)(jnp.arange(C)), axis=0)


__all__ = ["ActuationSpec", "load_actuation_spec", "ActuationParams", "actuation_params", "grasp_matrix",
           "contact_torque_map", "angular_jacobians", "gravity_torque"]
