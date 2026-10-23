"""KaRMA on JAX: batched kernels for the KaRMA metric.

The reference implementation lives in :mod:`karma` and stays the oracle. This
package re-implements the per-step compute (forward kinematics, contact
geometry, the rolling QP, manifold projection, the search) as ``jit``-compiled
functions over a leading batch axis, following ``docs/jax_rewrite_plan.md``.

Model loading, canonicalization and grid snapping still go through
:func:`karma.robot.load_robot`; :mod:`karma_jax.model` flattens the resulting
``RobotContext`` into padded arrays the kernels consume.
"""
from __future__ import annotations

__version__ = "0.1.0"

# Precision is controlled by array dtypes (fp32 by default for the search
# kernels, fp64 for the seed IK); enabling x64 only makes fp64 arrays possible.
import os as _os
import jax as _jax
_jax.config.update("jax_enable_x64", True)

# XLA on NVIDIA GPUs runs fp32 matmuls (cuBLAS GEMMs) in TF32 by default. The
# kernels avoid dense products with contractions wider than 3 (see linalg.py and
# kinematics.apply_mimic), so no global precision override is needed; set
# KARMA_JAX_MATMUL_PRECISION=highest to force true fp32 GEMMs anyway (~2x slower).
if _os.environ.get("KARMA_JAX_MATMUL_PRECISION"):
    _jax.config.update("jax_default_matmul_precision", _os.environ["KARMA_JAX_MATMUL_PRECISION"])

# Persistent compilation cache: every padded batch shape compiles once per
# machine, not once per process (a compile is ~1.5 to 10 s).
_cache_dir = _os.environ.get("KARMA_JAX_CACHE_DIR", _os.path.expanduser("~/.cache/karma_jax_xla"))
try:
    _os.makedirs(_cache_dir, exist_ok=True)
    _jax.config.update("jax_compilation_cache_dir", _cache_dir)
    _jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    _jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
except Exception:  # pragma: no cover - cache is an optimization only
    pass
