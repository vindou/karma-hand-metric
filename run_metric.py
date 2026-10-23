#!/usr/bin/env python3
"""CLI entry point: compute the fine-manipulation metric and write results."""
from __future__ import annotations

import argparse
import logging
import time

from karma.config import load_config
from karma.metric import compute_metric
from karma.robot import load_robot
from karma.results import write_outputs


def default_backend() -> str:
    """JAX (CUDA) when importable, otherwise the reference implementation."""
    try:
        import karma_jax  # noqa: F401
        return "jax"
    except ImportError:
        return "reference"


def run_backend(cfg, ctx, backend: str, precision: str, rotation_workers: int | None = None):
    """Compute the metric with the selected backend. Returns a MetricResult."""
    if backend == "jax":
        import jax
        from karma_jax.model import build_flat_model
        from karma_jax.metric import compute_metric as compute_metric_jax
        logging.getLogger(__name__).info(
            "Backend: JAX on %s, precision %s", jax.devices()[0].platform, precision)
        return compute_metric_jax(cfg, build_flat_model(ctx, cfg), precision=precision)
    logging.getLogger(__name__).info("Backend: reference (numpy/Pinocchio/OSQP)")
    return compute_metric(cfg, ctx, n_rotation_workers=rotation_workers)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute the KaRMA fine-manipulation metric for one robot hand."
    )
    parser.add_argument(
        "--config", default="robots/robot_shadowhand.yaml",
        help="Robot config YAML, e.g. robots/robot_leap.yaml",
    )
    parser.add_argument(
        "--metric-config",
        default="karma_config.yaml",
        help="Global KaRMA config (shared across robots)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument(
        "--rotation-workers", type=int, default=None,
        help="Number of parallel workers for rotation exploration (reference backend only)",
    )
    parser.add_argument(
        "--backend", choices=["jax", "reference"], default=None,
        help="'jax' (batched kernels on the default JAX device; default when karma_jax imports) "
             "or 'reference' (the paper's numpy/Pinocchio/OSQP implementation)",
    )
    parser.add_argument(
        "--precision", choices=["fp32", "fp64"], default="fp32",
        help="Search precision for the JAX backend (the seed IK always runs in fp64)",
    )
    args = parser.parse_args()
    backend = args.backend or default_backend()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    cfg = load_config(args.metric_config, args.config)
    ctx = load_robot(cfg.urdf_path, cfg)

    rotation_workers = args.rotation_workers if args.rotation_workers is not None else cfg.rotation_workers

    start_time = time.perf_counter()

    result = run_backend(cfg, ctx, backend, args.precision, rotation_workers)

    elapsed = time.perf_counter() - start_time
    out_path = write_outputs(cfg, result)

    karma_s = (result.seed_sensitivity or {}).get("karma_s")

    print(f"\n{'='*54}")
    print("  KaRMA metric")
    print(f"{'='*54}")
    print(f"  Voxels reached:              {result.n_voxels_reached}")
    print(f"  States reached:              {result.n_states_reached}")
    print(f"  KaRMA-T (translational):     {result.translational_score:.6f}")
    print(f"  KaRMA-R (rotational):        {result.global_rotational_score:.4f}")
    if karma_s is not None:
        print(f"  KaRMA-S (seed sensitivity):  {karma_s:.4f}")
    print(f"  Translational volume:        {result.translational_volume_m3:.6e} m^3")
    print(f"  L_ref:                       {result.l_ref_m:.6f} m")
    print(f"  Voxel size (scaled):         {cfg.voxel_size_m*1e3:.3f} mm")
    print(f"  Backend:                     {backend} ({args.precision if backend == 'jax' else 'fp64'})")
    print(f"  Elapsed time:                {elapsed:.2f}s")
    print(f"  Output written to:           {out_path}")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    main()
