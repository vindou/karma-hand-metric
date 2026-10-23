"""Static hand topology: fingers, joint counts, names and the flat ordering.

The flat (array) ordering is finger-major in the order of ``fingers``; within a
finger joints go proximal to distal. Pinocchio's URDF parser visits sibling
joints in alphabetical order, so a topology whose ``fingers`` are sorted by name
produces exactly the joint, frame and capsule indexing of
``karma_jax.model.build_flat_model`` on the exported URDF
(``Topology.two_finger`` / ``three_finger`` / ``from_hand_spec`` sort for you).

Frame layout (replicates Pinocchio on the exported URDF, all placements identity):
``0 universe, 1 root_joint, 2 palm`` (parents -1), then for flat joint ``j``
the joint frame ``3 + 2j`` and the body (link) frame ``4 + 2j``, both hanging
off joint ``j``. One capsule per link, capsule ``c`` = link of joint ``c``.

Nothing here is traced; it is plain Python/numpy used at trace time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np

MIMIC_K_PAD = 8          # karma_jax pads the (empty) mimic table to 8 lanes


@dataclass(frozen=True)
class Topology:
    fingers: tuple[str, ...]        # flat order
    n_joints: tuple[int, ...]       # per finger, same order
    pair: tuple[str, str] = ("thumb", "index")   # (thumb-like, index-like) fingers for the two-finger kernels

    def __post_init__(self):
        if len(self.fingers) != len(self.n_joints):
            raise ValueError("fingers and n_joints differ in length")
        if len(set(self.fingers)) != len(self.fingers):
            raise ValueError("duplicate finger names")
        if any(n < 1 for n in self.n_joints):
            raise ValueError("every finger needs at least one joint")
        for name in self.pair:
            if name not in self.fingers:
                raise ValueError(f"pair finger {name!r} not in {self.fingers}")
        if self.pair[0] == self.pair[1]:
            raise ValueError("pair fingers must differ")

    # ── constructors ──
    @staticmethod
    def two_finger(n_thumb: int = 4, n_index: int = 4) -> "Topology":
        """The optimizer's family (``optimize/hand_family.py``): alphabetical order index, thumb."""
        return Topology(fingers=("index", "thumb"), n_joints=(n_index, n_thumb))

    @staticmethod
    def three_finger(n: int = 4, names: tuple[str, str, str] = ("thumb", "index", "middle")) -> "Topology":
        """SPEC D2: thumb + two fingers, ``n`` joints each; sorted so the flat order matches Pinocchio."""
        order = sorted(names)
        return Topology(fingers=tuple(order), n_joints=(n,) * 3, pair=(names[0], names[1]))

    @staticmethod
    def from_hand_spec(spec) -> "Topology":
        fs = sorted([spec.thumb, spec.index], key=lambda f: f.name)
        return Topology(fingers=tuple(f.name for f in fs), n_joints=tuple(f.n_joints for f in fs),
                        pair=(spec.thumb.name, spec.index.name))

    # ── sizes ──
    @property
    def n_fingers(self) -> int:
        return len(self.fingers)

    @property
    def J(self) -> int:
        return int(sum(self.n_joints))

    @property
    def K(self) -> int:
        """Max joints per finger; ``phi`` arrays are ``[F, K]`` with ``joint_mask`` for shorter fingers."""
        return int(max(self.n_joints))

    @property
    def C(self) -> int:
        return self.J

    @property
    def F_frames(self) -> int:
        return 3 + 2 * self.J

    @property
    def L(self) -> int:
        """Contact-link padding per finger (max joints)."""
        return self.K

    # ── indexing ──
    @cached_property
    def first_joint(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.n_joints)[:-1]]).astype(np.int32)

    @cached_property
    def joint_mask(self) -> np.ndarray:
        """bool [F, K]: joint k of finger f exists."""
        m = np.zeros((self.n_fingers, self.K), dtype=bool)
        for f, n in enumerate(self.n_joints):
            m[f, :n] = True
        return m

    @cached_property
    def joint_finger(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_fingers, dtype=np.int32), self.n_joints)

    @cached_property
    def joint_index_in_finger(self) -> np.ndarray:
        return np.concatenate([np.arange(n, dtype=np.int32) for n in self.n_joints])

    @cached_property
    def parent(self) -> np.ndarray:
        """int32 [J]: previous joint of the finger, -1 at the knuckle."""
        p = np.arange(self.J, dtype=np.int32) - 1
        p[self.first_joint] = -1
        return p

    @cached_property
    def is_terminal(self) -> np.ndarray:
        t = np.zeros(self.J, dtype=bool)
        t[self.first_joint + np.asarray(self.n_joints) - 1] = True
        return t

    def joint_id(self, finger: str | int, k: int) -> int:
        f = self.fingers.index(finger) if isinstance(finger, str) else int(finger)
        if not 0 <= k < self.n_joints[f]:
            raise IndexError(f"finger {self.fingers[f]!r} has {self.n_joints[f]} joints")
        return int(self.first_joint[f]) + k

    # ── names (hand_family conventions: <finger>_j<k>, <finger>_l<k>) ──
    @cached_property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(f"{self.fingers[f]}_j{k}" for f, n in enumerate(self.n_joints) for k in range(n))

    @cached_property
    def link_names(self) -> tuple[str, ...]:
        return tuple(f"{self.fingers[f]}_l{k}" for f, n in enumerate(self.n_joints) for k in range(n))

    @property
    def capsule_names(self) -> tuple[str, ...]:
        return self.link_names

    @cached_property
    def frame_names(self) -> tuple[str, ...]:
        names = ["universe", "root_joint", "palm"]
        for j, l in zip(self.joint_names, self.link_names):
            names += [j, l]
        return tuple(names)

    def joint_frame(self, j: int) -> int:
        return 3 + 2 * j

    def body_frame(self, j: int) -> int:
        return 4 + 2 * j

    # ── frames and capsules (static tables) ──
    @cached_property
    def frame_parent(self) -> np.ndarray:
        fp = np.full(self.F_frames, -1, dtype=np.int32)
        for j in range(self.J):
            fp[self.joint_frame(j)] = j
            fp[self.body_frame(j)] = j
        return fp

    @cached_property
    def anc(self) -> np.ndarray:
        """bool [F_frames, J]: joint j moves frame f."""
        a = np.zeros((self.F_frames, self.J), dtype=bool)
        for f in range(self.F_frames):
            j = int(self.frame_parent[f])
            while j >= 0:
                a[f, j] = True
                j = int(self.parent[j])
        return a

    @cached_property
    def cap_frame(self) -> np.ndarray:
        return np.array([self.body_frame(c) for c in range(self.C)], dtype=np.int32)

    @cached_property
    def cap_distal(self) -> np.ndarray:
        d = np.full(self.C, -1, dtype=np.int32)
        for c in range(self.C):
            if not self.is_terminal[c]:
                d[c] = self.body_frame(c + 1)
        return d

    @cached_property
    def finger_caps(self) -> np.ndarray:
        """int32 [F, L]: capsules of each finger, distal first (the contact-link order), -1 padded."""
        out = np.full((self.n_fingers, self.L), -1, dtype=np.int32)
        for f, n in enumerate(self.n_joints):
            j0 = int(self.first_joint[f])
            out[f, :n] = np.arange(j0 + n - 1, j0 - 1, -1)
        return out

    def finger_index(self, name: str) -> int:
        return self.fingers.index(name)

    @property
    def thumb_finger(self) -> int:
        return self.finger_index(self.pair[0])

    @property
    def index_finger(self) -> int:
        return self.finger_index(self.pair[1])

    def seed_caps(self) -> tuple[int, int]:
        """Distal capsules of the pair fingers (the reference's seed links)."""
        return int(self.finger_caps[self.thumb_finger, 0]), int(self.finger_caps[self.index_finger, 0])
