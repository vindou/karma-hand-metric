"""Kinematics of a :class:`~dkarma.core.morphology.HandModel`.

Thin wrappers over ``karma_jax.kinematics`` (the oracle-tested FK and analytic
frame Jacobians) that take the model as a *traced argument*, so derivatives flow
through the morphology as well as through ``q`` (CLAUDE.md rule 5). ``q`` is the
full ``[J]`` configuration (all joints active in this family); Jacobians are
``[3, J]``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from karma_jax.kinematics import FKResult, forward_kinematics, frame_jacobian as _frame_jacobian, skew
from karma_jax.geometry import capsule_endpoints as _capsule_endpoints
from karma_jax.linalg import mm

from .morphology import HandModel, HandParams, build_hand_model
from .topology import Topology


def fk(hm: HandModel, q) -> FKResult:
    """Forward kinematics of one configuration."""
    return forward_kinematics(hm.dm, q)


def joint_origins_axes(hm: HandModel, q):
    """World joint origins ``[J,3]`` and unit axes ``[J,3]``."""
    r = fk(hm, q)
    return r.o_joint, r.w_axis


def capsule_endpoints(hm: HandModel, fkr: FKResult):
    """World axis endpoints ``(a, b)`` of every capsule, ``[C,3]`` each."""
    return _capsule_endpoints(hm.dm, fkr)


def frame_jacobian(hm: HandModel, fkr: FKResult, f):
    """``(Jv, Jw)`` of frame ``f`` (LOCAL_WORLD_ALIGNED), each ``[3, J]``."""
    return _frame_jacobian(hm.dm, fkr, f)


def point_jacobian(hm: HandModel, fkr: FKResult, f, offset_local):
    """``(Jv, Jw)`` of the point at ``offset_local`` (in frame ``f``) rigidly attached to frame ``f``:
    ``Jv = Jv_f - skew(R_f offset) Jw_f``."""
    Jv, Jw = _frame_jacobian(hm.dm, fkr, f)
    lever = fkr.R_frame[f] @ offset_local
    return Jv - mm(skew(lever), Jw), Jw


def point_jacobian_world(hm: HandModel, fkr: FKResult, f, p_world):
    """As :func:`point_jacobian` for a point given in world coordinates."""
    Jv, Jw = _frame_jacobian(hm.dm, fkr, f)
    return Jv - mm(skew(p_world - fkr.p_frame[f]), Jw), Jw


def capsule_end_jacobians(hm: HandModel, fkr: FKResult, c):
    """Linear Jacobians ``(J_a, J_b)`` of capsule ``c``'s axis endpoints."""
    f = hm.dm.cap_frame[c]
    a, b = capsule_endpoints(hm, fkr)
    return point_jacobian_world(hm, fkr, f, a[c])[0], point_jacobian_world(hm, fkr, f, b[c])[0]


def fk_from_params(topo: Topology, params: HandParams, q) -> FKResult:
    """``(params, q) -> FK`` as one differentiable map."""
    return fk(build_hand_model(topo, params), q)


fk_configs = jax.vmap(fk, in_axes=(None, 0))     # one design, many configurations ``[B, J]``
fk_designs = jax.vmap(fk, in_axes=(0, 0))        # a batch of models (all leaves stacked) and ``[B, J]``


def fk_designs_shared(hm_batched: HandModel, qs, template: HandModel):
    """Batch of models whose integer leaves are shared (see ``design_in_axes``)."""
    from .morphology import design_in_axes
    return jax.vmap(fk, in_axes=(design_in_axes(template), 0))(hm_batched, qs)


__all__ = ["FKResult", "fk", "joint_origins_axes", "capsule_endpoints", "frame_jacobian", "point_jacobian",
           "point_jacobian_world", "capsule_end_jacobians", "fk_from_params", "fk_configs", "fk_designs",
           "fk_designs_shared"]
