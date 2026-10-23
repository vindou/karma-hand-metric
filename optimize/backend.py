"""Backend adapter: run the proxy and the official metric on either the CPU
reference (``karma``) or the GPU port (``karma_jax``) through one interface.

Padding: ``build_flat_model`` is called with its defaults (exact sizes). The
two ``karma_jax`` padding defects found 2026-10-10 (a joint table padded to
exactly n_joints broke the jitted FK of these two-finger designs; padded
active-joint lanes corrupted the projection) were fixed at the root on
2026-10-11 (Phase 0, ``docs/phase0_report.md``); the former ``J_pad =
n_joints + 1`` workaround is gone. Results are bitwise unchanged by that
switch, so runs before and after it stay comparable.
"""
from __future__ import annotations

import dataclasses
import os
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
METRIC_CFG = REPO / "karma_config.yaml"


def default_backend() -> str:
    try:
        import karma_jax  # noqa: F401
        return "jax"
    except ImportError:
        return "reference"


def gpu_env() -> None:
    """Call before spawning several JAX worker processes that share one GPU."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


class Model:
    """(cfg, reference ctx, flat model or None) for one hand."""

    def __init__(self, metric_cfg, robot_yaml, backend: str, out_dir: Path | None = None):
        from karma.config import load_config
        from karma.robot import load_robot
        self.backend = backend
        self.cfg = load_config(metric_cfg, robot_yaml)
        if out_dir is not None:
            self.cfg = dataclasses.replace(self.cfg, out_dir=Path(out_dir))
        self.ctx = load_robot(self.cfg.urdf_path, self.cfg)
        self.flat = None
        if backend == "jax":
            gpu_env()
            from karma_jax.model import build_flat_model
            self.flat = build_flat_model(self.ctx, self.cfg)

    # ── proxy ──
    def proxy(self, fe):
        """``fe`` is optimize.fast_eval.FastEvalConfig; returns an object with the FastEvalResult fields."""
        if self.backend == "jax":
            from karma_jax import fast
            return fast.evaluate(self.cfg, self.flat, fast.FastConfig(**dataclasses.asdict(fe)))
        from optimize.fast_eval import evaluate_loaded
        return evaluate_loaded(self.cfg, self.ctx, fe)

    # ── official ──
    def official(self, precision: str = "fp32", rotation_workers: int | None = None):
        if self.backend == "jax":
            from karma_jax.metric import compute_metric
            return compute_metric(self.cfg, self.flat, precision=precision)
        from karma.metric import compute_metric
        return compute_metric(self.cfg, self.ctx, n_rotation_workers=rotation_workers)
