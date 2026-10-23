"""The compacting chunked IK must reproduce the plain vmapped TRF."""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from karma_jax import ik as IK
from karma_jax.search import step_params_from_cfg

from .conftest import load_hand
from .test_ik_oracle import _candidates


def test_chunked_ik_matches_plain():
    flat, cfg, ctx = load_hand("leap")
    dm = flat.to_device(jnp.float64)
    params = step_params_from_cfg(cfg, jnp.float64)
    cands = _candidates(cfg, ctx, n_sobol=64, n_offsets=1)
    B = len(cands)
    qi = np.zeros((B, flat.J_pad))
    for i, (q, _) in enumerate(cands):
        qi[i, : flat.n_joints] = q
    pair = np.tile([[flat.seed_thumb_cap, flat.seed_index_cap]], (B, 1)).astype(np.int32)
    args = (jnp.asarray(qi), jnp.asarray(np.stack([c for _, c in cands])), jnp.asarray(pair))
    plain = jax.device_get(IK.make_batched_ik(dm, params)(*args))
    chunked = jax.device_get(IK.ChunkedIK(dm, params, chunk=8)(*args))
    assert np.array_equal(np.asarray(plain.feasible), np.asarray(chunked.feasible))
    np.testing.assert_allclose(np.asarray(chunked.q), np.asarray(plain.q), atol=1e-6)
    np.testing.assert_allclose(np.asarray(chunked.p), np.asarray(plain.p), atol=1e-6)
