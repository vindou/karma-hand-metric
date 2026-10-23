"""The fixed physical object, task volume and orientation targets (SPEC.md §2, §3.4).

``E = (r_s, m_s, mu, g, w_ext, X_task)`` is identical for every design: a 15 mm,
35 g sphere at mu 0.6 under gravity along -z of the palm frame (D7, D17), a
200 mm cube of sphere-centre positions on a 10 mm grid centred on the nominal
grasp (D8), 228 antipodal pinch-axis bins / 576 smooth SO(3) targets / 4608
exact SO(3) bins (D16). Values live in ``dkarma/configs/task_spec.yaml``; the
``actuation`` block of that file is read by Phase 4, not here.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

DEFAULT_SPEC = Path(__file__).resolve().parents[1] / "configs" / "task_spec.yaml"


@dataclass(frozen=True)
class ObjectModel:
    radius_m: float = 0.015
    mass_kg: float = 0.035
    friction_coefficient: float = 0.6
    gravity_m_s2: tuple = (0.0, 0.0, -9.81)      # palm frame

    @property
    def weight_n(self) -> np.ndarray:
        """Gravity wrench force on the object (N)."""
        return self.mass_kg * np.asarray(self.gravity_m_s2, dtype=float)


@dataclass(frozen=True)
class TaskVolume:
    cube_side_m: float = 0.200
    grid_m: float = 0.010
    centre: str = "nominal_grasp_sphere_centre"

    @property
    def n_per_axis(self) -> int:
        return int(round(self.cube_side_m / self.grid_m))

    @property
    def n_cells(self) -> int:
        return self.n_per_axis ** 3

    def cell_centres(self, centre_xyz) -> np.ndarray:
        """``[n_cells, 3]`` cell centres of the cube about ``centre_xyz`` (fp64 numpy)."""
        n = self.n_per_axis
        ax = (np.arange(n) - (n - 1) / 2.0) * self.grid_m
        g = np.stack(np.meshgrid(ax, ax, ax, indexing="ij"), axis=-1).reshape(-1, 3)
        return g + np.asarray(centre_xyz, dtype=float)


@dataclass(frozen=True)
class OrientationTargets:
    pinch_axis_bins: int = 228
    so3_targets_smooth: int = 576
    so3_bins_exact: int = 4608
    sigma_r_rad: float = 0.15


@dataclass(frozen=True)
class RollingSpec:
    """Physical tolerances of the rolling evaluator (the reference's nominal values, unscaled)."""
    link_radius_m: float = 0.0118
    eps_g_m: float = 0.00035
    eps_col_m: float = 0.0005
    step_size_m: float = 0.00125
    slip_gate_m: float = 0.0005
    max_joint_step_deg: float = 10.0
    qp_regularization: float = 1.0e-6
    rotation_step_rad: float = 0.15
    rotation_efficiency_min: float = 0.3

    @property
    def max_joint_step_rad(self) -> float:
        return float(np.radians(self.max_joint_step_deg))


@dataclass(frozen=True)
class TaskSpec:
    obj: ObjectModel = field(default_factory=ObjectModel)
    task: TaskVolume = field(default_factory=TaskVolume)
    orientation: OrientationTargets = field(default_factory=OrientationTargets)
    rolling: RollingSpec = field(default_factory=RollingSpec)

    @property
    def n_sub(self) -> int:
        """Rolling sub-steps per task cell (8 for the defaults)."""
        return int(round(self.task.grid_m / self.rolling.step_size_m))

    def step_params(self, dtype=None):
        """``karma_jax.rolling.StepParams`` of this task in physical units (no L_ref scaling)."""
        import jax.numpy as jnp
        from karma_jax.rolling import StepParams
        dtype = dtype or jnp.float64
        f = lambda v: jnp.asarray(v, dtype)  # noqa: E731
        r = self.rolling
        return StepParams(sphere_radius=f(self.obj.radius_m), link_radius=f(r.link_radius_m), eps_g=f(r.eps_g_m),
                          max_joint_step=f(r.max_joint_step_rad), qp_regularization=f(r.qp_regularization),
                          slip_threshold=f(r.slip_gate_m), friction_coeff=f(self.obj.friction_coefficient),
                          eps_col=f(r.eps_col_m))


def load_task_spec(path: str | Path = DEFAULT_SPEC) -> TaskSpec:
    raw = yaml.safe_load(Path(path).read_text())
    o, t, r = raw["object"], raw["task"], raw["orientation"]
    rl = raw.get("rolling", {})
    return TaskSpec(
        rolling=RollingSpec(**{k: (float(v) if k != "max_joint_step_deg" else float(v)) for k, v in rl.items()}),
        obj=ObjectModel(radius_m=float(o["radius_m"]), mass_kg=float(o["mass_kg"]),
                        friction_coefficient=float(o["friction_coefficient"]),
                        gravity_m_s2=tuple(float(x) for x in o["gravity_m_s2"])),
        task=TaskVolume(cube_side_m=float(t["cube_side_m"]), grid_m=float(t["grid_m"]), centre=str(t["centre"])),
        orientation=OrientationTargets(pinch_axis_bins=int(r["pinch_axis_bins"]),
                                       so3_targets_smooth=int(r["so3_targets_smooth"]),
                                       so3_bins_exact=int(r["so3_bins_exact"]),
                                       sigma_r_rad=float(r.get("sigma_r_rad", 0.15))),
    )
