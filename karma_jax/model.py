"""Flat, padded array model of a hand, extracted from the reference RobotContext.

:func:`build_flat_model` reads a :class:`karma.robot.RobotContext` (so URDF
parsing, base canonicalization, grid snapping, procedural capsules and mimic
resolution are exactly the reference's) and produces a :class:`FlatModel` of
numpy arrays. :meth:`FlatModel.to_device` turns it into a :class:`DeviceModel`
(a NamedTuple of ``jax`` arrays, hence a pytree) that every kernel in this
package takes as its first argument.

Conventions, chosen to match Pinocchio so that ``q`` vectors and frame ids stay
interchangeable with the reference and its saved results:

* Joint ``j`` of the flat model is Pinocchio joint ``j + 1`` (the universe is
  dropped). ``parent[j] == -1`` means the universe. Pinocchio orders joints so
  that ``parent[j] < j``; the kernels rely on that.
* Every joint is a revolute about a unit ``axis[j]`` expressed in the joint
  frame; the joint frame at ``q`` is ``parent_frame * M_joint[j] * Rot(axis, q)``.
  All 16 shipped hands use only revolute joints (RX/RY/RZ/unaligned).
* ``nq == nv`` and ``idx_q == idx_v`` for every joint (1-DoF revolutes), so one
  index space serves positions and velocities.
* Frame ``f`` is Pinocchio frame ``f`` (frame 0 is the universe) with
  ``frame_parent[f]`` the flat joint index (-1 for the universe) and
  ``M_frame[f]`` the placement in the parent joint frame.
* Capsule ``c`` is ``RobotContext.link_capsules`` entry ``c`` in insertion
  order (the reference iterates that dict in the same order for collisions).

Mimic joints are a dense affine map on the full configuration,
``q_consistent = A_mimic @ q + b_mimic`` with ``A_mimic`` the identity except for
each mimic target row, which holds ``multiplier`` at the source column. The
same ``A_mimic`` folds Jacobians (``J_eff = J @ A_mimic``), which is exactly
:func:`karma.robot.fold_mimic_jacobian`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# ``jax`` is imported lazily in ``to_device`` so this module stays importable
# in the reference (CPU-only, Python 3.10) environment for oracle tests.


# ── Flat model (host side, numpy) ─────────────────────────────────────────────

@dataclass(frozen=True)
class FlatModel:
    """Padded numpy arrays describing one hand. See the module docstring."""

    # Sizes (actual, before padding)
    n_joints: int
    n_frames: int
    n_capsules: int
    n_active: int

    # Joints  [J_pad]
    parent: np.ndarray            # int32, -1 = universe; padding rows: -1
    R_joint: np.ndarray           # [J_pad,3,3] placement rotation in parent joint frame
    t_joint: np.ndarray           # [J_pad,3]   placement translation
    axis: np.ndarray              # [J_pad,3]   unit revolute axis in joint frame
    joint_valid: np.ndarray       # bool [J_pad]
    q_lower: np.ndarray           # [J_pad]
    q_upper: np.ndarray           # [J_pad]

    # Frames  [F_pad]
    frame_parent: np.ndarray      # int32 flat joint index, -1 = universe
    R_frame: np.ndarray           # [F_pad,3,3]
    t_frame: np.ndarray           # [F_pad,3]
    frame_valid: np.ndarray       # bool [F_pad]
    anc: np.ndarray               # bool [F_pad,J_pad]: joint j moves frame f

    # Capsules  [C_pad]
    cap_frame: np.ndarray         # int32 link frame id (0 for padding)
    cap_distal: np.ndarray        # int32 distal frame id, -1 = terminal (tip offset)
    cap_tip: np.ndarray           # [C_pad,3] tip offset in link frame (0 if distal)
    cap_valid: np.ndarray         # bool [C_pad]

    # Active joints and mimic map
    q_ids: np.ndarray             # int32 [M_pad] sorted active q indices (0 for padding)
    active_valid: np.ndarray      # bool [M_pad]
    A_mimic: np.ndarray           # [J_pad,J_pad] (dense form, for tests)
    b_mimic: np.ndarray           # [J_pad]
    mimic_target: np.ndarray      # int32 [K_pad] target q index (0 for padding)
    mimic_source: np.ndarray      # int32 [K_pad]
    mimic_mult: np.ndarray        # [K_pad]
    mimic_offset: np.ndarray      # [K_pad]
    mimic_valid: np.ndarray       # bool [K_pad]

    # Per-joint gather tables (derived from the active set and the mimic list).
    # The kernels use these instead of scatters: a ``set``-scatter with constant
    # indices and as many updates as the operand has elements is miscompiled by
    # XLA (jax 0.7.1, CPU and GPU) into a plain gather that ignores duplicate
    # indices, which padded lanes always produce. See kinematics.apply_mimic.
    jm_source: np.ndarray         # int32 [J_pad] mimic source q index (j itself if not a mimic)
    jm_mult: np.ndarray           # [J_pad] mimic multiplier (1.0 if not a mimic)
    jm_offset: np.ndarray         # [J_pad] mimic offset (0.0 if not a mimic)
    is_active: np.ndarray         # bool [J_pad] joint j is an active joint
    active_lane: np.ndarray       # int32 [J_pad] lane m with q_ids[m] == j (0 if not active)

    # Contact-link capsule indices (padded with -1)
    thumb_caps: np.ndarray        # int32 [L_pad]
    index_caps: np.ndarray        # int32 [L_pad]
    seed_thumb_cap: int
    seed_index_cap: int

    # Names, for debugging and for mapping results back to the reference types
    joint_names: tuple[str, ...] = field(default_factory=tuple)
    frame_names: tuple[str, ...] = field(default_factory=tuple)
    capsule_names: tuple[str, ...] = field(default_factory=tuple)

    # Phase 9.2 (SPEC §8.1 D18): finger group of every capsule (the flat index of the joint
    # directly below the root on its kinematic chain), or None. When set, ``check_self_collisions``
    # tests every pair of capsules of different groups instead of thumb vs index only.
    cap_group: np.ndarray | None = None

    # ── convenience ──
    @property
    def J_pad(self) -> int:
        return int(self.parent.shape[0])

    @property
    def F_pad(self) -> int:
        return int(self.frame_parent.shape[0])

    @property
    def C_pad(self) -> int:
        return int(self.cap_frame.shape[0])

    @property
    def M_pad(self) -> int:
        return int(self.q_ids.shape[0])

    @property
    def L_pad(self) -> int:
        return int(self.thumb_caps.shape[0])

    def capsule_index(self, link_name: str) -> int:
        return self.capsule_names.index(link_name)

    def to_device(self, dtype=None) -> "DeviceModel":
        """Upload to the default JAX device as a :class:`DeviceModel` pytree."""
        import jax.numpy as jnp

        if dtype is None:
            dtype = jnp.float32
        f = lambda a: jnp.asarray(a, dtype=dtype)      # noqa: E731
        i = lambda a: jnp.asarray(a, dtype=jnp.int32)  # noqa: E731
        b = lambda a: jnp.asarray(a, dtype=bool)       # noqa: E731
        return DeviceModel(
            parent=i(self.parent), R_joint=f(self.R_joint), t_joint=f(self.t_joint),
            axis=f(self.axis), joint_valid=b(self.joint_valid),
            q_lower=f(self.q_lower), q_upper=f(self.q_upper),
            frame_parent=i(self.frame_parent), R_frame=f(self.R_frame),
            t_frame=f(self.t_frame), frame_valid=b(self.frame_valid), anc=b(self.anc),
            cap_frame=i(self.cap_frame), cap_distal=i(self.cap_distal),
            cap_tip=f(self.cap_tip), cap_valid=b(self.cap_valid),
            q_ids=i(self.q_ids), active_valid=b(self.active_valid),
            A_mimic=f(self.A_mimic), b_mimic=f(self.b_mimic),
            mimic_target=i(self.mimic_target), mimic_source=i(self.mimic_source),
            mimic_mult=f(self.mimic_mult), mimic_offset=f(self.mimic_offset), mimic_valid=b(self.mimic_valid),
            jm_source=i(self.jm_source), jm_mult=f(self.jm_mult), jm_offset=f(self.jm_offset),
            is_active=b(self.is_active), active_lane=i(self.active_lane),
            thumb_caps=i(self.thumb_caps), index_caps=i(self.index_caps),
            cap_group=(None if self.cap_group is None else i(self.cap_group)),
        )


class DeviceModel(NamedTuple):
    """Device-resident arrays of a :class:`FlatModel` (a pytree)."""
    parent: object
    R_joint: object
    t_joint: object
    axis: object
    joint_valid: object
    q_lower: object
    q_upper: object
    frame_parent: object
    R_frame: object
    t_frame: object
    frame_valid: object
    anc: object
    cap_frame: object
    cap_distal: object
    cap_tip: object
    cap_valid: object
    q_ids: object
    active_valid: object
    A_mimic: object
    b_mimic: object
    mimic_target: object
    mimic_source: object
    mimic_mult: object
    mimic_offset: object
    mimic_valid: object
    jm_source: object
    jm_mult: object
    jm_offset: object
    is_active: object
    active_lane: object
    thumb_caps: object
    index_caps: object
    cap_group: object = None      # int32 [C_pad] finger group per capsule, or None (pair-only self-collision)


# ── Extraction from the reference RobotContext ───────────────────────────────

def _joint_axis_local(model, data, j: int) -> np.ndarray:
    """Revolute axis of Pinocchio joint ``j`` in its own frame.

    Read off the joint's motion subspace: with ``v = e_j`` the local spatial
    velocity of joint ``j`` is ``[0, 0, 0, axis]``. This works uniformly for
    RX/RY/RZ and unaligned revolutes without touching the joint variant types.
    """
    import pinocchio as pin

    q = pin.neutral(model)
    v = np.zeros(model.nv)
    v[model.joints[j].idx_v] = 1.0
    pin.forwardKinematics(model, data, q, v)
    mv = data.v[j]
    lin = np.asarray(mv.linear, dtype=float)
    ang = np.asarray(mv.angular, dtype=float)
    if np.linalg.norm(lin) > 1e-12 or abs(np.linalg.norm(ang) - 1.0) > 1e-9:
        raise NotImplementedError(
            f"Joint {model.names[j]!r} is not a unit revolute (linear={lin}, angular={ang}); "
            "only revolute joints are supported"
        )
    return ang


def _frame_parent_joint(frame) -> int:
    """Pinocchio 2.x exposes ``frame.parent``; 3.x renamed it ``parentJoint``."""
    pj = getattr(frame, "parentJoint", None)
    if pj is None:
        pj = frame.parent
    return int(pj)


def _pad_rows(a: np.ndarray, n: int, fill) -> np.ndarray:
    if a.shape[0] == n:
        return a
    out = np.full((n,) + a.shape[1:], fill, dtype=a.dtype)
    out[: a.shape[0]] = a
    return out


def build_flat_model(
    ctx,
    cfg,
    *,
    J_pad: int | None = None,
    F_pad: int | None = None,
    C_pad: int | None = None,
    M_pad: int | None = None,
    L_pad: int = 4,
    self_collision_all: bool = False,
) -> FlatModel:  # L_pad: minimum contact-link padding (grown to fit)
    """Flatten a reference :class:`karma.robot.RobotContext` into a :class:`FlatModel`.

    ``self_collision_all`` (Phase 9.2, SPEC §8.1 D18): fill ``cap_group`` so that the
    self-collision gate of every kernel tests all pairs of capsules on different kinematic
    chains below the root (all fingers), not only thumb vs index. Off by default: the default
    model reproduces the reference bit for bit.

    ``cfg`` is the :class:`karma.config.MetricConfig` the context was built
    from (for active joint names and contact links). Padding sizes default to
    the exact sizes, which are correct; pass bucket sizes to share one compiled
    program across hands. Every kernel ignores padded joints and padded active
    lanes (``karma_jax/tests/test_padding.py`` checks that results do not
    depend on ``J_pad`` or ``M_pad``).
    """
    from karma.robot import active_position_indices, active_velocity_indices

    model = ctx.model
    data = model.createData()
    nJ = model.njoints - 1
    nF = model.nframes

    if model.nq != model.nv:
        raise NotImplementedError("nq != nv: only 1-DoF revolute joints are supported")
    for j in range(1, model.njoints):
        jm = model.joints[j]
        if jm.nq != 1 or jm.nv != 1 or jm.idx_q != jm.idx_v:
            raise NotImplementedError(f"Joint {model.names[j]!r} is not a 1-DoF joint")
        if jm.idx_q != j - 1:
            # Pinocchio assigns idx_q in joint order for 1-DoF chains; the flat
            # model assumes q index == flat joint index.
            raise NotImplementedError(
                f"Joint {model.names[j]!r} has idx_q={jm.idx_q}, expected {j - 1}"
            )

    # ── joints ──
    parent = np.empty(nJ, dtype=np.int32)
    R_joint = np.empty((nJ, 3, 3))
    t_joint = np.empty((nJ, 3))
    axis = np.empty((nJ, 3))
    for j in range(1, model.njoints):
        parent[j - 1] = int(model.parents[j]) - 1
        pl = model.jointPlacements[j]
        R_joint[j - 1] = np.asarray(pl.rotation, dtype=float)
        t_joint[j - 1] = np.asarray(pl.translation, dtype=float)
        axis[j - 1] = _joint_axis_local(model, data, j)
    if np.any(parent >= np.arange(nJ)):
        raise ValueError("Pinocchio joint order is not parent-first; cannot flatten")
    q_lower = np.asarray(model.lowerPositionLimit, dtype=float).copy()
    q_upper = np.asarray(model.upperPositionLimit, dtype=float).copy()

    # ── frames ──
    frame_parent = np.empty(nF, dtype=np.int32)
    R_frame = np.empty((nF, 3, 3))
    t_frame = np.empty((nF, 3))
    for f in range(nF):
        fr = model.frames[f]
        frame_parent[f] = _frame_parent_joint(fr) - 1
        R_frame[f] = np.asarray(fr.placement.rotation, dtype=float)
        t_frame[f] = np.asarray(fr.placement.translation, dtype=float)

    # ancestor matrix: joint j moves frame f iff j is the frame's parent joint
    # or an ancestor of it
    anc = np.zeros((nF, nJ), dtype=bool)
    for f in range(nF):
        j = int(frame_parent[f])
        while j >= 0:
            anc[f, j] = True
            j = int(parent[j])

    # ── capsules ──
    cap_names = tuple(ctx.link_capsules.keys())
    nC = len(cap_names)
    cap_frame = np.zeros(nC, dtype=np.int32)
    cap_distal = np.full(nC, -1, dtype=np.int32)
    cap_tip = np.zeros((nC, 3))
    for c, name in enumerate(cap_names):
        cap = ctx.link_capsules[name]
        cap_frame[c] = int(cap.link_frame_id)
        if cap.distal_frame_id is not None:
            cap_distal[c] = int(cap.distal_frame_id)
        elif cap.tip_offset_xyz_m is not None:
            cap_tip[c] = np.asarray(cap.tip_offset_xyz_m, dtype=float)
        # else: zero-length capsule (start == end); cap_tip stays 0

    # ── active joints and mimic ──
    q_ids = active_position_indices(model, cfg.finger_joint_names).astype(np.int32)
    v_ids = active_velocity_indices(model, cfg.finger_joint_names).astype(np.int32)
    if not np.array_equal(q_ids, v_ids):
        raise NotImplementedError("active q and v indices differ")
    nM = len(q_ids)

    targets = [m.target_q for m in ctx.mimic_qs]
    if len(set(targets)) != len(targets):
        raise NotImplementedError("a joint is the target of more than one mimic coupling")
    if set(targets) & {m.source_q for m in ctx.mimic_qs}:
        # The reference applies mimics in list order, so a chain could resolve
        # there; the kernels gather from the un-coupled q and would not.
        raise NotImplementedError("chained mimic couplings (a source that is itself a mimic target)")
    if set(targets) & set(int(i) for i in q_ids):
        raise ValueError("a mimic target joint is listed as an active joint")
    A_mimic = np.eye(nJ)
    b_mimic = np.zeros(nJ)
    for m in ctx.mimic_qs:
        A_mimic[m.target_q, :] = 0.0
        A_mimic[m.target_q, m.source_q] = m.multiplier
        b_mimic[m.target_q] = m.offset
    K_pad = max(8, len(ctx.mimic_qs))
    mimic_target = np.zeros(K_pad, np.int32)
    mimic_source = np.zeros(K_pad, np.int32)
    mimic_mult = np.zeros(K_pad)
    mimic_offset = np.zeros(K_pad)
    mimic_valid = np.zeros(K_pad, bool)
    for k, m in enumerate(ctx.mimic_qs):
        mimic_target[k], mimic_source[k] = m.target_q, m.source_q
        mimic_mult[k], mimic_offset[k], mimic_valid[k] = m.multiplier, m.offset, True

    # L_pad is a minimum; hands with more contact links per finger (the optimizer's
    # 5+5 family) grow it, so the default never raises.
    L_pad = max(L_pad, len(cfg.thumb_contact_links), len(cfg.index_contact_links))

    def _cap_list(names: list[str]) -> np.ndarray:
        out = np.full(L_pad, -1, dtype=np.int32)
        if len(names) > L_pad:
            raise ValueError(f"More than L_pad={L_pad} contact links: {names}")
        for k, n in enumerate(names):
            out[k] = cap_names.index(n)
        return out

    thumb_caps = _cap_list(cfg.thumb_contact_links)
    index_caps = _cap_list(cfg.index_contact_links)

    # ── padding ──
    J_pad = nJ if J_pad is None else J_pad
    F_pad = nF if F_pad is None else F_pad
    C_pad = nC if C_pad is None else C_pad
    M_pad = nM if M_pad is None else M_pad
    for name, have, want in (("J_pad", nJ, J_pad), ("F_pad", nF, F_pad),
                             ("C_pad", nC, C_pad), ("M_pad", nM, M_pad)):
        if want < have:
            raise ValueError(f"{name}={want} smaller than required {have}")

    eye3 = np.broadcast_to(np.eye(3), (1, 3, 3))
    A_pad = np.eye(J_pad)
    A_pad[:nJ, :nJ] = A_mimic
    anc_pad = np.zeros((F_pad, J_pad), dtype=bool)
    anc_pad[:nF, :nJ] = anc

    # per-joint gather tables (padding joints: identity coupling, not active)
    jm_source = np.arange(J_pad, dtype=np.int32)
    jm_mult = np.ones(J_pad)
    jm_offset = np.zeros(J_pad)
    for m in ctx.mimic_qs:
        jm_source[m.target_q] = m.source_q
        jm_mult[m.target_q] = m.multiplier
        jm_offset[m.target_q] = m.offset
    is_active = np.zeros(J_pad, dtype=bool)
    active_lane = np.zeros(J_pad, dtype=np.int32)
    for lane, j in enumerate(q_ids):
        is_active[j] = True
        active_lane[j] = lane

    cap_group = None
    if self_collision_all:
        cap_group = np.full(C_pad, -1, dtype=np.int32)
        roots = [j for j in range(nJ) if int(parent[j]) < 0]
        # one joint without a parent that every other joint descends from = a shared root (palm) joint:
        # the groups are its children; otherwise the parentless joints are the finger roots themselves
        shared_root = roots[0] if len(roots) == 1 and nJ > 1 else None
        for c in range(nC):
            j = int(frame_parent[cap_frame[c]])
            while j >= 0 and int(parent[j]) >= 0 and int(parent[j]) != shared_root:
                j = int(parent[j])
            cap_group[c] = j                          # -1 = a capsule on the universe frame; shared_root = on the palm link
    return FlatModel(
        n_joints=nJ, n_frames=nF, n_capsules=nC, n_active=nM,
        parent=_pad_rows(parent, J_pad, -1),
        R_joint=np.concatenate([R_joint, np.repeat(eye3, J_pad - nJ, axis=0)], axis=0),
        t_joint=_pad_rows(t_joint, J_pad, 0.0),
        axis=_pad_rows(axis, J_pad, 0.0) + np.concatenate(
            [np.zeros((nJ, 3)), np.tile([0.0, 0.0, 1.0], (J_pad - nJ, 1))], axis=0),
        joint_valid=_pad_rows(np.ones(nJ, dtype=bool), J_pad, False),
        q_lower=_pad_rows(q_lower, J_pad, 0.0),
        q_upper=_pad_rows(q_upper, J_pad, 0.0),
        frame_parent=_pad_rows(frame_parent, F_pad, -1),
        R_frame=np.concatenate([R_frame, np.repeat(eye3, F_pad - nF, axis=0)], axis=0),
        t_frame=_pad_rows(t_frame, F_pad, 0.0),
        frame_valid=_pad_rows(np.ones(nF, dtype=bool), F_pad, False),
        anc=anc_pad,
        cap_frame=_pad_rows(cap_frame, C_pad, 0),
        cap_distal=_pad_rows(cap_distal, C_pad, -1),
        cap_tip=_pad_rows(cap_tip, C_pad, 0.0),
        cap_valid=_pad_rows(np.ones(nC, dtype=bool), C_pad, False),
        q_ids=_pad_rows(q_ids, M_pad, 0),
        active_valid=_pad_rows(np.ones(nM, dtype=bool), M_pad, False),
        A_mimic=A_pad,
        b_mimic=_pad_rows(b_mimic, J_pad, 0.0),
        mimic_target=mimic_target, mimic_source=mimic_source, mimic_mult=mimic_mult,
        mimic_offset=mimic_offset, mimic_valid=mimic_valid,
        jm_source=jm_source, jm_mult=jm_mult, jm_offset=jm_offset,
        is_active=is_active, active_lane=active_lane,
        thumb_caps=thumb_caps,
        index_caps=index_caps,
        seed_thumb_cap=cap_names.index(cfg.seed_thumb_link),
        seed_index_cap=cap_names.index(cfg.seed_index_link),
        joint_names=tuple(str(model.names[j]) for j in range(1, model.njoints)),
        frame_names=tuple(str(model.frames[f].name) for f in range(nF)),
        capsule_names=cap_names,
        cap_group=cap_group,
    )


def load_flat_model(metric_config_path, robot_config_path, **pad) -> tuple[FlatModel, object, object]:
    """Convenience: load config + reference robot and flatten. Returns (flat, cfg, ctx)."""
    from karma.config import load_config
    from karma.robot import load_robot

    cfg = load_config(metric_config_path, robot_config_path)
    ctx = load_robot(cfg.urdf_path, cfg)
    return build_flat_model(ctx, cfg, **pad), cfg, ctx
