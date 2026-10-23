"""Oracle test: batched Gauss-Newton projection vs the reference on recorded calls."""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

from karma_jax import projection as PJ

from .conftest import load_hand
from .test_qp_oracle import recording, RECORD_HANDS

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("name", RECORD_HANDS)
def test_projection_matches_reference(name):
    flat, cfg, ctx = load_hand(name)
    rec = recording(name)
    if not rec.projection:
        pytest.skip("no projection calls recorded (every step of this hand dies at the slip gate)")
    dm = flat.to_device(jnp.float64)
    r_s, r_l, eps_g = cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_g_m
    f = jax.jit(lambda q, p, pair: PJ.project_onto_contacts(dm, q, p, pair, r_s, r_l, eps_g))
    names = flat.capsule_names
    nJ = flat.n_joints
    q_err, th_err = [], []
    n_ok_mismatch = 0
    for r in rec.projection:
        pair = jnp.asarray([names.index(r["pair"][0]), names.index(r["pair"][1])])
        out = f(jnp.asarray(r["q"]), jnp.asarray(r["p"]), pair)
        q_ours = np.asarray(out.q)[:nJ]
        if bool(out.ok) != r["ok"]:
            n_ok_mismatch += 1
            continue
        q_err.append(np.abs(q_ours - r["q_proj"]).max())
        th_err.append(np.abs(np.asarray(out.dtheta) - r["dtheta"]).max())
        np.testing.assert_allclose(float(out.ck_t.g), r["g0"], atol=1e-7)
        np.testing.assert_allclose(float(out.ck_i.g), r["g1"], atol=1e-7)
    # Exact-arithmetic equivalence; the only differences are linear-solve rounding.
    assert n_ok_mismatch == 0, n_ok_mismatch
    assert np.max(q_err) < 1e-7, np.max(q_err)
    assert np.max(th_err) < 1e-6, np.max(th_err)
    print(f"{name}: {len(q_err)} projections, |q - q_ref| max {np.max(q_err):.1e}, "
          f"|dtheta - ref| max {np.max(th_err):.1e}")
