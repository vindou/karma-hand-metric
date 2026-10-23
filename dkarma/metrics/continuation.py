"""Continuation schedule for the smooth objective (SPEC.md §8.4): broad kernels and mild
penalties first, tightening toward the final resolution; the last stage is strict and is
always followed by exact verification.

Phase 8.2 options (``docs/phase8_report.md`` §8.2): the end values of the three widths can be
set explicitly (a gentler final ``sigma_energy`` keeps the last stage less strict than the
base configuration) and ``skip_strict`` lays the geometric path over ``n_stages + 1`` points
and drops the strict end point, so the run finishes one geometric step short of it."""
from __future__ import annotations

from dataclasses import replace

from ..solvers.rollout import EnergyWeights, RollConfig
from .config import MetricConfig, SmoothConfig


def schedule(base: MetricConfig, n_stages: int = 4, sigma_p_start_m: float = 0.015, sigma_r_start_rad: float = 0.25,
             sigma_energy_start: float = 8.0, soft_first: bool = True, sigma_energy_end: float | None = None,
             sigma_p_end_m: float | None = None, sigma_r_end_rad: float | None = None,
             skip_strict: bool = False) -> list[MetricConfig]:
    """Geometric interpolation from the start values to the end values (``base``'s by default)
    over ``n_stages``. Stage 0 may use the soft solver (no active-set kinks); later stages use
    ``base.roll.mode``."""
    p_end = base.smooth.sigma_p_m if sigma_p_end_m is None else sigma_p_end_m
    r_end = base.smooth.sigma_r_rad if sigma_r_end_rad is None else sigma_r_end_rad
    e_end = base.energy.sigma if sigma_energy_end is None else sigma_energy_end
    n_pts = n_stages + (1 if skip_strict else 0)
    out = []
    for i in range(n_stages):
        t = i / max(n_pts - 1, 1)
        g = lambda a, b: a * (b / a) ** t   # noqa: E731
        sm = replace(base.smooth, sigma_p_m=g(sigma_p_start_m, p_end), sigma_r_rad=g(sigma_r_start_rad, r_end))
        en = base.energy._replace(sigma=g(sigma_energy_start, e_end))
        roll = base.roll._replace(mode="soft") if (soft_first and i == 0) else base.roll
        out.append(replace(base, smooth=sm, energy=en, roll=roll))
    return out


def describe(cfg: MetricConfig) -> dict:
    """The continuation knobs of one stage (for logs and reports)."""
    return dict(mode=cfg.roll.mode, sigma_p_m=cfg.smooth.sigma_p_m, sigma_r_rad=cfg.smooth.sigma_r_rad,
                sigma_energy=cfg.energy.sigma)


__all__ = ["schedule", "describe"]
