"""Small dense products as fused multiply-reduce instead of ``dot_general``.

Under ``vmap`` every per-lane ``a @ b`` becomes a batched cuBLAS GEMM, which
XLA runs in TF32 (10-bit mantissa) by default on NVIDIA GPUs. Forcing
``jax_default_matmul_precision="highest"`` restores fp32 accuracy but costs
~2x. For the tiny shapes in these kernels (at most ~28 x 13 x 13) a
broadcast-multiply-sum is exact fp32, fuses with neighbouring elementwise
work, and avoids the library call entirely.
"""
from __future__ import annotations

import jax.numpy as jnp


def mm(a, b):
    """``a @ b`` for ``[..., m, k] @ [..., k, n]``."""
    return jnp.sum(a[..., :, :, None] * b[..., None, :, :], axis=-2)


def mv(a, v):
    """``a @ v`` for ``[..., m, k] @ [..., k]``."""
    return jnp.sum(a * v[..., None, :], axis=-1)


def vm(v, a):
    """``v @ a`` for ``[..., k] @ [..., k, n]``."""
    return jnp.sum(v[..., :, None] * a, axis=-2)


def mtm(a):
    """``a.T @ a`` for ``[..., m, k]`` -> ``[..., k, k]``."""
    return jnp.sum(a[..., :, :, None] * a[..., :, None, :], axis=-3)
