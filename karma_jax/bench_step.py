"""Phase B benchmark gate: throughput of the batched translation step.

    ../.venv/bin/python -m karma_jax.bench_step --hand leap --batch 1024 8192 32768 --dtype float32 float64

Builds a batch of real BFS states (recorded from a short reference run,
tiled), compiles the batched 8-sub-step primitive step, and reports time per
batched call and sub-steps per second. Use ``JAX_PLATFORMS=cpu`` to measure
the CPU backend.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import jax
import jax.numpy as jnp

from karma.orientation import compute_motion_primitives_frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="leap")
    ap.add_argument("--batch", type=int, nargs="+", default=[1024, 8192, 32768])
    ap.add_argument("--dtype", nargs="+", default=["float32", "float64"])
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()

    from karma_jax.tests.conftest import load_hand
    from karma_jax.tests._record import record
    from karma_jax.tests.test_qp_oracle import _params
    from karma_jax import steps as S, qp as Q

    flat, cfg, ctx = load_hand(a.hand)
    rec = record(cfg, ctx, bfs_budget=60, n_rot_voxels=0)
    nodes = list(rec.discovered.values())
    prims = compute_motion_primitives_frame(rec.R_seed, cfg.voxel_size_m)
    n_sub = max(1, int(np.ceil(cfg.voxel_size_m / cfg.step_size_m)))
    slip = cfg.rotation_slip_threshold_mm / 1000.0 if cfg.rotation_slip_threshold_mm is not None else None
    names = flat.capsule_names
    print(f"device={jax.devices()[0]} hand={a.hand} n_states={len(nodes)} n_sub={n_sub} "
          f"J_pad={flat.J_pad} F_pad={flat.F_pad} C_pad={flat.C_pad} M_pad={flat.M_pad}")

    for dt_name in a.dtype:
        dtype = getattr(jnp, dt_name)
        if dtype == jnp.float64:
            jax.config.update("jax_enable_x64", True)
        dm = flat.to_device(dtype)
        params = jax.tree_util.tree_map(lambda v: jnp.asarray(v, dtype), _params(cfg, slip))
        step = S.make_batched_step(dm, n_sub, params, Q.default_settings(dtype))
        for B in a.batch:
            idx = np.arange(B) % len(nodes)
            q = np.zeros((B, flat.J_pad))
            for i, k in enumerate(idx):
                q[i, : flat.n_joints] = nodes[k].q
            st = S.GraspState(
                jnp.asarray(q, dtype),
                jnp.asarray(np.stack([nodes[k].p for k in idx]), dtype),
                jnp.asarray(np.stack([nodes[k].R_sphere for k in idx]), dtype),
                jnp.asarray(np.stack([[names.index(nodes[k].contact_pair[0]),
                                       names.index(nodes[k].contact_pair[1])] for k in idx]), jnp.int32),
            )
            dp = jnp.asarray(np.stack([prims[i % 6] for i in range(B)]), dtype)
            t0 = time.perf_counter()
            out = step(st, dp)
            jax.block_until_ready(out)
            t_compile = time.perf_counter() - t0
            times = []
            for _ in range(a.repeats):
                t0 = time.perf_counter()
                out = step(st, dp)
                jax.block_until_ready(out)
                times.append(time.perf_counter() - t0)
            t = float(np.median(times))
            ok_frac = float(np.mean(np.asarray(out.ok)))
            print(f"{dt_name:8s} B={B:6d}  compile+first {t_compile:6.1f}s  per call {t*1e3:8.1f} ms  "
                  f"{B * n_sub / t / 1e3:8.1f}k sub-steps/s  {B / t:8.0f} attempts/s  ok={ok_frac:.2f}")
        if dtype == jnp.float64:
            jax.config.update("jax_enable_x64", False)


if __name__ == "__main__":
    main()
