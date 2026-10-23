"""Fixed-size object-motion command libraries (SPEC.md §4.1).

A command is a sequence of per-step object displacements ``dp [H, 3]`` and extra
rotations ``dtheta [H, 3]`` in the world frame. The library of a seed is built
from its manipulability frame ``R_seed`` (the reference's grid frame) and the
pinch axis:

* translations: ``+-e_k`` of the seed frame, ``n_cells`` task cells of ``n_sub``
  sub-steps each (``6`` commands of ``H_T = n_cells * n_sub`` steps);
* rotations: tilts of ``rotation_step`` rad per step about ``+-perp_1, +-perp_2``,
  the two axes perpendicular to the pinch axis, ``n_rot_steps`` steps
  (``4`` commands of ``H_R`` steps; the reference's Phase-2 primitives);
* optional diagonals: the 12 ``(+-e_i +- e_j)/sqrt 2`` translations.

Shapes are static per library configuration so ``vmap`` over commands works.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class CommandConfig:
    n_cells: int = 5              # cells per translation command (5 x 10 mm = 50 mm)
    n_sub: int = 8                # sub-steps per cell
    step_m: float = 0.00125       # sub-step length
    n_rot_steps: int = 12         # rotation steps per command
    rotation_step_rad: float = 0.15
    diagonals: bool = False

    @property
    def H_T(self) -> int:
        return self.n_cells * self.n_sub

    @property
    def H_R(self) -> int:
        return self.n_rot_steps

    @property
    def n_translation(self) -> int:
        return 6 + (12 if self.diagonals else 0)

    @property
    def n_rotation(self) -> int:
        return 4


class CommandLibrary(NamedTuple):
    dp_T: object        # [U_T, H_T, 3]
    dth_T: object       # [U_T, H_T, 3] zeros
    dp_R: object        # [U_R, H_R, 3] zeros
    dth_R: object       # [U_R, H_R, 3]
    dirs_T: object      # [U_T, 3] unit directions (world)
    axes_R: object      # [U_R, 3] unit tilt axes (world)


def perpendicular_axes(pinch_axis):
    """Two unit vectors orthogonal to ``pinch_axis`` and to each other (the reference's
    ``compute_rotation_primitives`` construction: cross with the least-aligned world axis)."""
    a = pinch_axis / jnp.linalg.norm(pinch_axis)
    e = jnp.eye(3, dtype=a.dtype)
    k = jnp.argmin(jnp.abs(a))
    p1 = jnp.cross(a, e[k])
    p1 = p1 / jnp.linalg.norm(p1)
    p2 = jnp.cross(a, p1)
    return p1, p2 / jnp.linalg.norm(p2)


def build_library(R_seed, pinch_axis, cfg: CommandConfig) -> CommandLibrary:
    """Commands for one seed (pure JAX, differentiable in ``R_seed`` and ``pinch_axis``)."""
    dtype = jnp.asarray(R_seed).dtype
    e = R_seed.T                                   # rows = frame axes
    dirs = [e[0], -e[0], e[1], -e[1], e[2], -e[2]]
    if cfg.diagonals:
        for i in range(3):
            for j in range(i + 1, 3):
                for si in (1.0, -1.0):
                    for sj in (1.0, -1.0):
                        dirs.append((si * e[i] + sj * e[j]) / jnp.sqrt(2.0))
    dirs_T = jnp.stack(dirs)                                            # [U_T,3]
    dp_T = jnp.broadcast_to((cfg.step_m * dirs_T)[:, None, :], (dirs_T.shape[0], cfg.H_T, 3))
    dth_T = jnp.zeros_like(dp_T)
    p1, p2 = perpendicular_axes(pinch_axis)
    axes_R = jnp.stack([p1, -p1, p2, -p2])                              # [4,3]
    dth_R = jnp.broadcast_to((cfg.rotation_step_rad * axes_R)[:, None, :], (4, cfg.H_R, 3))
    dp_R = jnp.zeros_like(dth_R)
    return CommandLibrary(dp_T, dth_T, dp_R, dth_R, dirs_T, axes_R)


def seed_frame(J_obj):
    """Differentiable analogue of the reference's ``compute_seed_frame``: eigenvectors of
    ``J J^T`` in descending order with the reference's sign convention and ``det = +1``.
    (Eigenvectors are differentiable away from repeated eigenvalues.)"""
    M = J_obj @ J_obj.T
    lam, V = jnp.linalg.eigh(M)
    V = V[:, ::-1]
    sign = jnp.sign(V[jnp.argmax(jnp.abs(V), axis=0), jnp.arange(3)])
    sign = jnp.where(sign == 0, 1.0, sign)
    V = V * sign[None, :]
    det = jnp.linalg.det(V)
    return V.at[:, 2].multiply(jnp.where(det < 0, -1.0, 1.0))


def seed_frame_np(J_obj: np.ndarray) -> np.ndarray:
    """The reference's own implementation (host numpy), for validation."""
    from karma.math_utils import compute_seed_frame
    return compute_seed_frame(np.asarray(J_obj, float))


__all__ = ["CommandConfig", "CommandLibrary", "perpendicular_axes", "build_library", "seed_frame", "seed_frame_np"]
