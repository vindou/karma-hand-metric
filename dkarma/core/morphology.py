"""Morphology parameters and the pure-JAX hand model constructor (SPEC.md §3.1, §3.2, §9.2).

A design becomes the ``karma_jax.model.DeviceModel`` arrays that every existing
kernel (FK, Jacobians, capsule pass, QP, projection, step gates) takes as its
first argument, **without** a URDF or Pinocchio in the loop. The constructor is
plain ``jax.numpy`` over a static :class:`~dkarma.core.topology.Topology`, so
``jax.grad``/``jacfwd`` flow from any downstream quantity into the morphology,
``jax.jit`` traces the model as an argument (never a closed-over constant,
CLAUDE.md rule 5), and ``jax.vmap`` over a batch of designs of one topology is
an outer axis.

Family (the N-finger generalization of ``optimize/hand_family.py``):

* palm = base link at the world origin; ``n_fingers`` independent serial chains
  of revolute joints, nothing tied across fingers (SPEC D14);
* joint 0 of a finger sits at ``base_xyz`` with orientation ``exp(base_rotvec)``
  in the palm frame; joint ``k > 0`` sits at ``[link_len[k-1], 0, 0]`` in the
  frame of joint ``k-1`` with zero rotation (finger straight along +x at q = 0);
* the axis of joint ``k`` is ``(cos th, sin th cos ph, sin th sin ph)`` in its
  own frame;
* ``link_len[n-1]`` is the tip length. The reference builds the terminal
  capsule axis with length ``max(tip - r_l, 0)`` along the auto-detected tip
  direction (+x for the exported URDF, CoM at half the link); reproduced here;
* limits ``[-lim_lo, +lim_hi]`` are fixed in the principal experiments (D3) and
  live in :class:`FixedParams`, not in the optimizer vector;
* one capsule per link with its own radius ``cap_radius[c]`` (uniform 11.8 mm by
  default); the palm has no geometry.

Index spaces, frames and capsule tables come from :class:`Topology` and
replicate Pinocchio on the exported URDF (``0 universe, 1 root_joint, 2 palm``,
then joint/body frame pairs), so the model built here and
``karma_jax.model.build_flat_model`` of the same design share indices bit for
bit (``dkarma/tests/test_kinematics.py``).

Parameter layers:

* :class:`HandParams`: the *physical* design (pytree, padded ``[F, K]``);
  consumed by :func:`build_hand_model` and the URDF exporter.
* :class:`Phi`: the *unconstrained* optimizer coordinates. :func:`decode` maps
  ``Phi -> HandParams`` given :class:`Bounds` (sigmoid boxes on lengths and base
  positions; rotation vectors and axis angles raw) and :class:`FixedParams`
  (limits, radii, pinned finger bases); :func:`encode` is the inverse on the
  interior of the boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from karma_jax.model import DeviceModel

from .topology import Topology, MIMIC_K_PAD

DEFAULT_CAPSULE_RADIUS_M = 0.0118   # karma_config.yaml link_radius_m (nominal)


# ── Parameters ───────────────────────────────────────────────────────────────

class HandParams(NamedTuple):
    """Physical design of one hand (pytree). ``[F, K]`` slots beyond a finger's joints are ignored."""
    base_xyz: object      # [F,3] m, palm frame
    base_rotvec: object   # [F,3] rad
    axis_theta: object    # [F,K] rad, polar angle of the joint axis from link +x
    axis_phi: object      # [F,K] rad, azimuth about link +x
    link_len: object      # [F,K] m; slot n_f-1 is the tip length
    lim_lo: object        # [F,K] rad >= 0 (lower limit is -lim_lo)
    lim_hi: object        # [F,K] rad >= 0
    cap_radius: object    # [C]   m, one per capsule (flat capsule order)


class Phi(NamedTuple):
    """Unconstrained optimizer coordinates (pytree)."""
    base_xyz_u: object    # [F,3]
    base_rotvec: object   # [F,3]
    axis_theta: object    # [F,K]
    axis_phi: object      # [F,K]
    link_len_u: object    # [F,K]


@dataclass(frozen=True)
class Bounds:
    """Admissible design box (SPEC §8.1). ``l_min`` only avoids degenerate links."""
    link_len_min_m: float = 0.012
    link_len_max_m: float = 0.080
    tip_len_min_m: float = 0.012
    tip_len_max_m: float = 0.080
    base_xyz_max_m: float = 0.080
    base_xyz_centre_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def len_bounds(self, topo: Topology):
        """(lo, hi) ``[F, K]``: tip slots use the tip bounds, others the link bounds."""
        last = np.zeros((topo.n_fingers, topo.K), dtype=bool)
        for f, n in enumerate(topo.n_joints):
            last[f, n - 1] = True
        return (np.where(last, self.tip_len_min_m, self.link_len_min_m),
                np.where(last, self.tip_len_max_m, self.link_len_max_m))


class FixedParams(NamedTuple):
    """Quantities held fixed during optimization (pytree)."""
    lim_lo: object           # [F,K]
    lim_hi: object           # [F,K]
    cap_radius: object       # [C]
    pin_base_xyz: object     # [F,3]  base pose used for pinned fingers
    pin_base_rotvec: object  # [F,3]
    pin_mask: object         # bool [F]  (rigid-invariance gauge for kinematic-only objectives)


def _logit(p):
    p = jnp.clip(p, 1e-9, 1.0 - 1e-9)
    return jnp.log(p) - jnp.log1p(-p)


def decode(phi: Phi, topo: Topology, bounds: Bounds, fixed: FixedParams) -> HandParams:
    """``Phi -> HandParams`` (pure JAX, differentiable)."""
    dt = jnp.asarray(phi.link_len_u).dtype
    lo, hi = bounds.len_bounds(topo)
    lo, hi = jnp.asarray(lo, dt), jnp.asarray(hi, dt)
    link_len = lo + (hi - lo) * jax.nn.sigmoid(phi.link_len_u)
    c = jnp.asarray(bounds.base_xyz_centre_m, dt)
    base_xyz = c + bounds.base_xyz_max_m * (2.0 * jax.nn.sigmoid(phi.base_xyz_u) - 1.0)
    pin = jnp.asarray(fixed.pin_mask)[:, None]
    return HandParams(
        base_xyz=jnp.where(pin, fixed.pin_base_xyz, base_xyz),
        base_rotvec=jnp.where(pin, fixed.pin_base_rotvec, phi.base_rotvec),
        axis_theta=phi.axis_theta, axis_phi=phi.axis_phi, link_len=link_len,
        lim_lo=jnp.asarray(fixed.lim_lo, dt), lim_hi=jnp.asarray(fixed.lim_hi, dt),
        cap_radius=jnp.asarray(fixed.cap_radius, dt),
    )


def encode(params: HandParams, topo: Topology, bounds: Bounds) -> Phi:
    """Inverse of :func:`decode` on the interior of the boxes (clipped at the edges)."""
    ll = jnp.asarray(params.link_len)
    lo, hi = bounds.len_bounds(topo)
    lo, hi = jnp.asarray(lo, ll.dtype), jnp.asarray(hi, ll.dtype)
    c = jnp.asarray(bounds.base_xyz_centre_m, ll.dtype)
    return Phi(base_xyz_u=_logit(((jnp.asarray(params.base_xyz) - c) / bounds.base_xyz_max_m + 1.0) / 2.0),
               base_rotvec=jnp.asarray(params.base_rotvec), axis_theta=jnp.asarray(params.axis_theta),
               axis_phi=jnp.asarray(params.axis_phi), link_len_u=_logit((ll - lo) / (hi - lo)))


def fixed_from_params(params: HandParams, pin_fingers: tuple[int, ...] = ()) -> FixedParams:
    """Trace-safe (usable under ``jit``/``vmap``): ``pin_fingers`` is static."""
    F = jnp.shape(params.base_xyz)[0]
    pin = np.zeros(F, dtype=bool)
    for f in pin_fingers:
        pin[f] = True
    return FixedParams(lim_lo=jnp.asarray(params.lim_lo), lim_hi=jnp.asarray(params.lim_hi),
                       cap_radius=jnp.asarray(params.cap_radius), pin_base_xyz=jnp.asarray(params.base_xyz),
                       pin_base_rotvec=jnp.asarray(params.base_rotvec), pin_mask=jnp.asarray(pin))


# ── Rotations ────────────────────────────────────────────────────────────────

def rotvec_to_matrix(v):
    """so(3) -> SO(3) exponential, finite value and gradient at ``v = 0``."""
    ssq = jnp.sum(v * v, axis=-1)
    small = ssq < 1e-14
    th = jnp.sqrt(jnp.where(small, 1.0, ssq))
    a = jnp.where(small, 1.0 - ssq / 6.0, jnp.sin(th) / th)
    b = jnp.where(small, 0.5 - ssq / 24.0, (1.0 - jnp.cos(th)) / jnp.where(small, 1.0, ssq))
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    o = jnp.zeros_like(x)
    K = jnp.stack([jnp.stack([o, -z, y], -1), jnp.stack([z, o, -x], -1), jnp.stack([-y, x, o], -1)], -2)
    return jnp.eye(3, dtype=v.dtype) + a[..., None, None] * K + b[..., None, None] * (K @ K)


def axis_from_angles(theta, phi):
    return jnp.stack([jnp.cos(theta), jnp.sin(theta) * jnp.cos(phi), jnp.sin(theta) * jnp.sin(phi)], axis=-1)


# ── Model ────────────────────────────────────────────────────────────────────

class HandModel(NamedTuple):
    """``karma_jax`` DeviceModel plus the per-design extras ``dkarma`` needs (pytree)."""
    dm: DeviceModel
    cap_radius: object      # [C] m
    tip_len: object         # [C] m (0 for non-terminal capsules)
    finger_caps: object     # int [F, L] capsules per finger, distal first, -1 padded
    cap_finger: object      # int [C]
    cap_pos: object         # int [C] position of the link in its finger
    joint_mask: object      # bool [F, K]
    flat_of_slot: object    # int [F, K] flat joint of slot (f, k); 0 on padding slots


def build_hand_model(topo: Topology, params: HandParams, dtype=None) -> HandModel:
    """``HandParams`` -> :class:`HandModel`; jit/vmap/grad-able in ``params`` (``topo`` static)."""
    J, C, Ff = topo.J, topo.C, topo.F_frames
    F, K = topo.n_fingers, topo.K
    if dtype is None:
        dtype = jnp.asarray(params.link_len).dtype
    P = jax.tree_util.tree_map(lambda a: jnp.asarray(a, dtype), params)

    # slot tables (static)
    jf = jnp.asarray(topo.joint_finger)                 # [J]
    jk = jnp.asarray(topo.joint_index_in_finger)        # [J]
    first = jk == 0
    last = jnp.asarray(topo.is_terminal)

    # placement of every joint in its parent frame, gathered from the slot arrays
    R_base = rotvec_to_matrix(P.base_rotvec)            # [F,3,3]
    prev_len = jnp.concatenate([jnp.zeros((F, 1), dtype), P.link_len[:, :-1]], axis=1)   # [F,K]
    R_joint = jnp.where(first[:, None, None], R_base[jf], jnp.eye(3, dtype=dtype)[None])
    t_chain = jnp.stack([prev_len[jf, jk], jnp.zeros(J, dtype), jnp.zeros(J, dtype)], -1)
    t_joint = jnp.where(first[:, None], P.base_xyz[jf], t_chain)
    axis = axis_from_angles(P.axis_theta[jf, jk], P.axis_phi[jf, jk])
    q_lower = -P.lim_lo[jf, jk]
    q_upper = P.lim_hi[jf, jk]

    # capsules: terminal axis ends at max(tip - r, 0) along +x of the body frame
    cap_r = jnp.broadcast_to(P.cap_radius, (C,))
    own_len = P.link_len[jf, jk]
    tip_len = jnp.where(last, own_len, 0.0)
    tip_axis = jnp.where(last, jnp.maximum(tip_len - cap_r, 0.0), 0.0)
    cap_tip = jnp.stack([tip_axis, jnp.zeros(C, dtype), jnp.zeros(C, dtype)], -1)

    ones_J = jnp.ones(J, dtype=bool)
    eye_F = jnp.broadcast_to(jnp.eye(3, dtype=dtype), (Ff, 3, 3))
    arangeJ = jnp.arange(J, dtype=jnp.int32)
    finger_caps = jnp.asarray(topo.finger_caps)
    dm = DeviceModel(
        parent=jnp.asarray(topo.parent), R_joint=R_joint, t_joint=t_joint, axis=axis,
        joint_valid=ones_J, q_lower=q_lower, q_upper=q_upper,
        frame_parent=jnp.asarray(topo.frame_parent), R_frame=eye_F, t_frame=jnp.zeros((Ff, 3), dtype),
        frame_valid=jnp.ones(Ff, dtype=bool), anc=jnp.asarray(topo.anc),
        cap_frame=jnp.asarray(topo.cap_frame), cap_distal=jnp.asarray(topo.cap_distal),
        cap_tip=cap_tip, cap_valid=ones_J,
        q_ids=arangeJ, active_valid=ones_J,
        A_mimic=jnp.eye(J, dtype=dtype), b_mimic=jnp.zeros(J, dtype),
        mimic_target=jnp.zeros(MIMIC_K_PAD, jnp.int32), mimic_source=jnp.zeros(MIMIC_K_PAD, jnp.int32),
        mimic_mult=jnp.zeros(MIMIC_K_PAD, dtype), mimic_offset=jnp.zeros(MIMIC_K_PAD, dtype),
        mimic_valid=jnp.zeros(MIMIC_K_PAD, bool),
        jm_source=arangeJ, jm_mult=jnp.ones(J, dtype), jm_offset=jnp.zeros(J, dtype),
        is_active=ones_J, active_lane=arangeJ,
        thumb_caps=finger_caps[topo.thumb_finger], index_caps=finger_caps[topo.index_finger],
    )
    flat_of_slot = np.zeros((F, K), np.int32)
    for f, n in enumerate(topo.n_joints):
        flat_of_slot[f, :n] = np.arange(topo.first_joint[f], topo.first_joint[f] + n)
    return HandModel(dm=dm, cap_radius=cap_r, tip_len=tip_len, finger_caps=finger_caps,
                     cap_finger=jf, cap_pos=jk, joint_mask=jnp.asarray(topo.joint_mask),
                     flat_of_slot=jnp.asarray(flat_of_slot))


def build_hand_model_from_phi(topo: Topology, phi: Phi, bounds: Bounds, fixed: FixedParams, **kw) -> HandModel:
    return build_hand_model(topo, decode(phi, topo, bounds, fixed), **kw)


def design_in_axes(template: HandModel):
    """``in_axes`` tree for ``jax.vmap`` over designs whose float leaves are stacked and whose
    integer/boolean leaves are shared (0 for floats, None otherwise)."""
    return jax.tree_util.tree_map(
        lambda a: 0 if jnp.issubdtype(jnp.asarray(a).dtype, jnp.floating) else None, template)


# ── Constructors ─────────────────────────────────────────────────────────────

def params_from_fingers(topo: Topology, fingers: dict, cap_radius=DEFAULT_CAPSULE_RADIUS_M,
                        dtype=np.float64) -> HandParams:
    """``fingers``: name -> dict(base_xyz, base_rotvec, axis_theta, axis_phi, link_len, lim_lo, lim_hi)."""
    F, K, C = topo.n_fingers, topo.K, topo.C
    base_xyz = np.zeros((F, 3), dtype)
    base_rotvec = np.zeros((F, 3), dtype)
    per = {k: np.zeros((F, K), dtype) for k in ("axis_theta", "axis_phi", "link_len", "lim_lo", "lim_hi")}
    # benign padding values (unused by the kernels)
    per["link_len"][:] = 0.02
    per["axis_theta"][:] = math.pi / 2
    per["lim_lo"][:] = 1.0
    per["lim_hi"][:] = 1.0
    for f, name in enumerate(topo.fingers):
        d = fingers[name]
        n = topo.n_joints[f]
        base_xyz[f] = d["base_xyz"]
        base_rotvec[f] = d["base_rotvec"]
        for k in per:
            v = np.asarray(d[k], dtype)
            if v.shape != (n,):
                raise ValueError(f"finger {name!r}: {k} has shape {v.shape}, expected ({n},)")
            per[k][f, :n] = v
    r = np.asarray(cap_radius, dtype)
    r = np.broadcast_to(r, (C,)).copy() if r.ndim == 0 else r
    if r.shape != (C,):
        raise ValueError(f"cap_radius must be scalar or [{C}]")
    return HandParams(base_xyz=jnp.asarray(base_xyz), base_rotvec=jnp.asarray(base_rotvec), cap_radius=jnp.asarray(r),
                      **{k: jnp.asarray(v) for k, v in per.items()})


def params_to_fingers(topo: Topology, params: HandParams) -> dict:
    P = jax.tree_util.tree_map(lambda a: np.asarray(a, dtype=float), params)
    out = {}
    for f, name in enumerate(topo.fingers):
        n = topo.n_joints[f]
        out[name] = dict(base_xyz=P.base_xyz[f].tolist(), base_rotvec=P.base_rotvec[f].tolist(),
                         axis_theta=P.axis_theta[f, :n].tolist(), axis_phi=P.axis_phi[f, :n].tolist(),
                         link_len=P.link_len[f, :n].tolist(), lim_lo=P.lim_lo[f, :n].tolist(),
                         lim_hi=P.lim_hi[f, :n].tolist())
    return out


def from_hand_spec(spec, cap_radius=DEFAULT_CAPSULE_RADIUS_M) -> tuple[Topology, HandParams]:
    """``optimize/hand_family.py`` HandSpec (object or its JSON dict) -> (Topology, HandParams).
    Finger order is alphabetical (Pinocchio's), ``pair = (thumb, index)``."""
    if isinstance(spec, dict):
        fs = {k: spec[k] for k in ("thumb", "index")}
        get = lambda f, k: f[k]                                   # noqa: E731
        names = {k: fs[k].get("name", k) for k in fs}
    else:
        fs = {"thumb": spec.thumb, "index": spec.index}
        get = lambda f, k: getattr(f, k)                           # noqa: E731
        names = {k: fs[k].name for k in fs}
    order = sorted(fs, key=lambda k: names[k])
    topo = Topology(fingers=tuple(names[k] for k in order),
                    n_joints=tuple(len(get(fs[k], "axis_theta")) for k in order),
                    pair=(names["thumb"], names["index"]))
    fingers = {names[k]: dict(base_xyz=list(get(fs[k], "knuckle_xyz")), base_rotvec=list(get(fs[k], "knuckle_rotvec")),
                              axis_theta=list(get(fs[k], "axis_theta")), axis_phi=list(get(fs[k], "axis_phi")),
                              link_len=list(get(fs[k], "link_len")), lim_lo=list(get(fs[k], "lim_lo")),
                              lim_hi=list(get(fs[k], "lim_hi"))) for k in fs}
    return topo, params_from_fingers(topo, fingers, cap_radius)


def to_hand_spec_dict(topo: Topology, params: HandParams, name: str = "opt_hand") -> dict:
    if topo.n_fingers != 2:
        raise ValueError("HandSpec holds exactly two fingers")
    fd = params_to_fingers(topo, params)
    out = {}
    for role, fname in zip(("thumb", "index"), topo.pair):
        d = fd[fname]
        out[role] = dict(name=fname, knuckle_xyz=d["base_xyz"], knuckle_rotvec=d["base_rotvec"],
                         axis_theta=d["axis_theta"], axis_phi=d["axis_phi"], link_len=d["link_len"],
                         lim_lo=d["lim_lo"], lim_hi=d["lim_hi"])
    out["name"] = name
    return out


def random_params(topo: Topology, key, bounds: Bounds | None = None, lim_deg=(15.0, 75.0),
                  base_xyz_max_m: float = 0.06, cap_radius=DEFAULT_CAPSULE_RADIUS_M) -> HandParams:
    """A random design inside the bounds (deterministic in ``key``)."""
    bounds = bounds or Bounds()
    F, K, C = topo.n_fingers, topo.K, topo.C
    k1, k2, k3, k4, k5, k6, k7 = jax.random.split(key, 7)
    lo, hi = bounds.len_bounds(topo)
    link_len = lo + (hi - lo) * (0.2 + 0.6 * jax.random.uniform(k1, (F, K)))
    return HandParams(
        base_xyz=jax.random.uniform(k2, (F, 3), minval=-base_xyz_max_m, maxval=base_xyz_max_m),
        base_rotvec=jax.random.normal(k3, (F, 3)) * 0.8,
        axis_theta=jnp.pi / 2 + 0.6 * jax.random.normal(k4, (F, K)),
        axis_phi=jax.random.uniform(k5, (F, K), minval=-jnp.pi, maxval=jnp.pi),
        link_len=link_len,
        lim_lo=jnp.radians(jax.random.uniform(k6, (F, K), minval=lim_deg[0], maxval=lim_deg[1])),
        lim_hi=jnp.radians(jax.random.uniform(k7, (F, K), minval=lim_deg[0], maxval=lim_deg[1])),
        cap_radius=jnp.full((C,), cap_radius),
    )


def anthropomorphic(topo: Topology, cap_radius=DEFAULT_CAPSULE_RADIUS_M) -> HandParams:
    """Hand-drawn feasible start. The thumb-like finger (``topo.pair[0]``) is set
    proximal/ulnar and rolled into opposition; every other finger lies along +x in the
    palm plane (abduction, then flexions), spaced 22 mm apart in y. For the two-finger
    family it reproduces ``optimize/hand_family.py:anthropomorphic_x0`` exactly."""
    from scipy.spatial.transform import Rotation
    d = math.radians
    hp = math.pi / 2
    fingers = {}
    others = 0
    for f, name in enumerate(topo.fingers):
        n = topo.n_joints[f]
        if name == topo.pair[0]:
            R = Rotation.from_euler("z", 60, degrees=True) * Rotation.from_euler("x", -70, degrees=True)
            rows = [(hp, hp, 0.015, d(45), d(45)), (hp, 0.0, 0.045, d(15), d(60)),
                    (hp, 0.0, 0.030, d(15), d(60)), (hp, 0.0, 0.025, d(15), d(75))]
            base_xyz, rv = [-0.015, -0.040, -0.020], R.as_rotvec().tolist()
        else:
            w = np.array([0.12, 0.40, 0.25, 0.23])
            lens = (0.100 * w / w.sum()).tolist()
            rows = [(hp, hp, lens[0], d(20), d(20)), (hp, 0.0, lens[1], d(15), d(75)),
                    (hp, 0.0, lens[2], d(15), d(75)), (hp, 0.0, lens[3], d(15), d(75))]
            base_xyz, rv = [0.0, 0.030 * others, 0.0], [0.0, 0.0, 0.0]     # 30 mm pitch: 6.4 mm clearance between 11.8 mm capsules
            others += 1
        rows = [rows[min(k, len(rows) - 1)] for k in range(n)]
        fingers[name] = dict(base_xyz=base_xyz, base_rotvec=rv,
                             axis_theta=[r[0] for r in rows], axis_phi=[r[1] for r in rows],
                             link_len=[r[2] for r in rows], lim_lo=[r[3] for r in rows], lim_hi=[r[4] for r in rows])
    return params_from_fingers(topo, fingers, cap_radius)


__all__ = ["DEFAULT_CAPSULE_RADIUS_M", "HandParams", "Phi", "Bounds", "FixedParams", "decode", "encode",
           "fixed_from_params", "rotvec_to_matrix", "axis_from_angles", "HandModel", "build_hand_model",
           "build_hand_model_from_phi", "design_in_axes", "params_from_fingers", "params_to_fingers",
           "from_hand_spec", "to_hand_spec_dict", "random_params", "anthropomorphic"]
