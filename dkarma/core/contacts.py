"""Fixed-assignment contact pass and clearances (SPEC.md §3.3, contact mode 1).

A :class:`ContactAssignment` names the capsules that are in contact for a whole
rollout (``[Cn]`` lanes, padded with ``valid = False``); there is no argmin
switching inside differentiable code. :func:`contact_pass` evaluates the
reference contact kinematics (``karma.contacts.contact_kinematics``, through
``karma_jax.geometry.contact_kinematics``) for every assigned capsule and adds
the capsule-surface point and its surface Jacobian with the capsule's own
radius. The clearances are the quantities of the reference's collision gates,
``karma.collisions.check_collisions`` (sphere vs every capsule not in contact)
and ``check_self_collisions`` (capsules of different fingers), as exact
minima (``beta=None``) or as soft minima for gradients.

Mode 2 (a fixed set of candidate assignments aggregated with a smooth
maximum) is served by :func:`smooth_max_over_assignments`.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np
import jax
import jax.numpy as jnp

from karma_jax import geometry as G
from karma_jax.kinematics import fold_active, skew
from karma_jax.linalg import mm

from .morphology import HandModel
from .topology import Topology
from . import geometry as geo

__all__ = ["ContactAssignment", "ContactSet", "from_seed", "contact_pass", "object_clearance",
           "self_clearance", "smooth_max_over_assignments", "antipodal_margin", "assignment_mask", "mixed_radii",
           "reference_gates_per_capsule"]


class ContactAssignment(NamedTuple):
    """Capsules in contact. ``caps`` int32 ``[Cn]`` (0 on padded lanes), ``valid`` bool ``[Cn]``."""
    caps: object
    valid: object


class ContactSet(NamedTuple):
    """Contact kinematics of the assigned capsules; every field is zero on invalid lanes."""
    gap: object          # [Cn]      signed gap d - r_s - r_l (m)
    normal: object       # [Cn,3]    unit vector from the capsule axis point to the sphere centre
    p_capsule: object    # [Cn,3]    contact point on the capsule surface = axis_point + r_l n
    p_sphere: object     # [Cn,3]    contact point on the sphere surface = p - r_s n
    axis_point: object   # [Cn,3]    closest point on the capsule axis
    Jv: object           # [Cn,3,J]  Jacobian of the closest axis point (reference closest-point formula)
    Jw: object           # [Cn,3,J]  link angular-velocity Jacobian
    origin: object       # [Cn,3]    link frame origin
    Jv_origin: object    # [Cn,3,J]  Jacobian of the link frame origin
    J_surf: object       # [Cn,3,J]  Jv_origin - skew(p_capsule - origin) @ Jw: velocity of the material point under the contact


def from_seed(topo: Topology, n_lanes: int = 2) -> ContactAssignment:
    """The reference's seed pair: distal capsules of ``topo.pair`` (thumb-like first), padded to ``n_lanes``."""
    if n_lanes < 2:
        raise ValueError("a seed assignment has two contacts")
    caps = np.zeros(n_lanes, np.int32)
    valid = np.zeros(n_lanes, bool)
    caps[:2] = topo.seed_caps()
    valid[:2] = True
    return ContactAssignment(caps=jnp.asarray(caps), valid=jnp.asarray(valid))


def assignment_mask(C: int, assignment: ContactAssignment):
    """bool ``[C]``: capsule ``c`` is one of the (valid) assigned contacts."""
    idx = jnp.arange(C)
    hit = (idx[:, None] == assignment.caps[None, :]) & assignment.valid[None, :]
    return jnp.any(hit, axis=1)


def _surface_jacobian(hm: HandModel, ck: G.ContactKin, r_l):
    """As ``karma_jax.rolling._surface_jacobian`` with the capsule's own radius ``r_l``."""
    p_capsule = ck.axis_point + r_l * ck.normal
    r = p_capsule - ck.origin
    Jv_o = fold_active(hm.dm, ck.Jv_origin)
    Jw = fold_active(hm.dm, ck.Jw)
    return p_capsule, Jv_o - mm(skew(r), Jw)


def contact_pass(hm: HandModel, fkr, p, r_s, assignment: ContactAssignment) -> ContactSet:
    """Contact kinematics of every assigned capsule for the sphere (radius ``r_s``) at ``p`` [3].

    Invalid lanes evaluate capsule 0 (always a real capsule, so nothing is NaN)
    and are then zeroed, so their gradients are exactly zero.
    """
    cp = geo.capsule_pass(hm, fkr, p, r_s)

    def one(c, valid):
        ci = jnp.where(valid, c, 0)
        ck = G.contact_kinematics(hm.dm, fkr, cp, ci, p, r_s)
        r_l = hm.cap_radius[ci]
        p_cap, J_surf = _surface_jacobian(hm, ck, r_l)
        out = ContactSet(gap=ck.g, normal=ck.normal, p_capsule=p_cap, p_sphere=ck.sphere_point,
                         axis_point=ck.axis_point, Jv=ck.Jv, Jw=ck.Jw, origin=ck.origin,
                         Jv_origin=ck.Jv_origin, J_surf=J_surf)
        return jax.tree_util.tree_map(lambda x: jnp.where(valid, x, jnp.zeros_like(x)), out)

    return jax.vmap(one)(assignment.caps, assignment.valid)


def object_clearance(hm: HandModel, cp: geo.CapsulePass, assignment: ContactAssignment, beta=None):
    """Minimum gap of the sphere over the capsules NOT in the assignment (the reference's
    ``check_collisions`` quantity); soft minimum with sharpness ``beta`` [1/m], exact when ``beta`` is None."""
    C = cp.gap.shape[0]
    mask = hm.dm.cap_valid & ~assignment_mask(C, assignment)
    if beta is None:
        return geo.hard_min(cp.gap, mask)
    return geo.softmin(cp.gap, beta, mask)


def self_clearance(hm: HandModel, cp: geo.CapsulePass, beta=None):
    """Minimum capsule-capsule clearance over pairs of different fingers (the reference's
    ``check_self_collisions`` quantity); soft with ``beta`` [1/m], exact when None."""
    clearance, mask = geo.capsule_pair_clearance(hm, cp)
    if beta is None:
        return geo.hard_min_clearance(clearance, mask)
    return geo.softmin(clearance, beta, mask)


def mixed_radii(hm: HandModel, link_radius) -> bool:
    """Host check: some capsule's radius differs from the task's contact radius ``link_radius`` (D19's servo links).
    The reference two-finger kernels (``karma_jax``) apply one radius to every capsule; with mixed radii their
    collision gates are re-applied per capsule (:func:`reference_gates_per_capsule`)."""
    return not np.allclose(np.asarray(hm.cap_radius, float), float(link_radius), rtol=0.0, atol=1e-12)


def reference_gates_per_capsule(hm: HandModel, fkr, p, r_s, pair, eps_g, eps_col):
    """The gates of the reference two-finger kernels (``karma_jax.steps._validate``, the pinch-IK feasibility and the
    seed polish), with each capsule's own radius (SPEC §11.4 item 11.1): ``(gap_ok, collision_ok, self_ok)`` for the
    contact ``pair`` [2]. Gap: both pair capsules within ``eps_g``. Collision: the sphere clears every other capsule by
    ``eps_col``. Self-collision: the kernels' own scope (thumb vs index capsules of ``hm.dm``, or every pair of fingers
    when the model carries ``cap_group``), clearance ``dist - r_i - r_j >= eps_col``."""
    cp = geo.capsule_pass(hm, fkr, p, r_s)
    gap_ok = jnp.maximum(jnp.abs(cp.gap[pair[0]]), jnp.abs(cp.gap[pair[1]])) <= eps_g
    C = cp.gap.shape[0]
    idx = jnp.arange(C)
    mask = hm.dm.cap_valid & (idx != pair[0]) & (idx != pair[1])
    col_ok = jnp.min(jnp.where(mask, cp.gap, jnp.inf)) >= eps_col
    if getattr(hm.dm, "cap_group", None) is not None:
        clr, m = geo.capsule_pair_clearance(hm, cp)
    else:
        tc, ic = hm.dm.thumb_caps, hm.dm.index_caps
        tv, iv = tc >= 0, ic >= 0
        ti, ii = jnp.where(tv, tc, 0), jnp.where(iv, ic, 0)
        d = geo.segment_segment_dist(cp.a[ti][:, None, :], cp.b[ti][:, None, :], cp.a[ii][None, :, :], cp.b[ii][None, :, :])
        clr, m = d - hm.cap_radius[ti][:, None] - hm.cap_radius[ii][None, :], tv[:, None] & iv[None, :]
    self_ok = jnp.min(jnp.where(m, clr, jnp.inf)) >= eps_col
    return gap_ok, col_ok, self_ok


def smooth_max_over_assignments(values, beta, mask=None):
    """Log-sum-exp maximum over candidate assignments (SPEC.md §3.3 mode 2); ``beta`` in ``1/values``."""
    return geo.softmax(values, beta, mask)


def antipodal_margin(normals, mu):
    """``dot(-n0, n1) - cos(2 atan mu)`` for two contact normals ``normals`` [2,3]: the smooth
    version of the reference friction-cone gate (``check_antipodal_friction``), feasible when >= 0.
    Normals are unit-normalized with the guarded norm (zero-length normals give the degenerate
    value ``-cos(2 atan mu)``, as the reference reports ``cos_val = 0``, infeasible)."""
    n0, n1 = normals[0], normals[1]
    d0, s0 = geo.safe_norm(n0)
    d1, s1 = geo.safe_norm(n1)
    degenerate = s0 | s1
    u0 = n0 / jnp.where(degenerate, 1.0, d0)
    u1 = n1 / jnp.where(degenerate, 1.0, d1)
    cos_val = jnp.where(degenerate, 0.0, jnp.dot(-u0, u1))
    return cos_val - jnp.cos(2.0 * jnp.arctan(mu))
