"""Parameterization round trips, bounds, pinning, jit with a traced model, and vmap over designs."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import (Topology, Bounds, build_hand_model, decode, encode, fixed_from_params, random_params,
                         anthropomorphic, design_in_axes, rotvec_to_matrix)
from dkarma.core.kinematics import fk, capsule_endpoints, fk_designs_shared
from .conftest import load_design, random_q

jax.config.update("jax_enable_x64", True)


def test_topology_tables():
    t = Topology(fingers=("index", "thumb"), n_joints=(4, 3), pair=("thumb", "index"))
    assert t.J == 7 and t.K == 4 and t.F_frames == 17
    assert t.parent.tolist() == [-1, 0, 1, 2, -1, 4, 5]
    assert t.cap_distal.tolist() == [t.body_frame(1), t.body_frame(2), t.body_frame(3), -1, t.body_frame(5), t.body_frame(6), -1]
    assert t.finger_caps[0].tolist() == [3, 2, 1, 0] and t.finger_caps[1].tolist() == [6, 5, 4, -1]
    assert t.thumb_finger == 1 and t.index_finger == 0 and t.seed_caps() == (6, 3)
    assert t.anc[t.body_frame(6)].tolist() == [False] * 4 + [True] * 3
    assert t.anc[2].tolist() == [False] * 7           # palm frame moves with nothing


def test_rotvec_matrix_matches_scipy_and_is_safe_at_zero():
    from scipy.spatial.transform import Rotation
    v = jnp.array([0.3, -1.1, 0.7])
    np.testing.assert_allclose(np.asarray(rotvec_to_matrix(v)), Rotation.from_rotvec(np.array(v)).as_matrix(), atol=1e-14)
    np.testing.assert_allclose(np.asarray(rotvec_to_matrix(jnp.zeros(3))), np.eye(3), atol=1e-15)
    g = jax.jacfwd(rotvec_to_matrix)(jnp.zeros(3))
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(np.asarray(g[:, :, 2]), [[0, -1, 0], [1, 0, 0], [0, 0, 0]], atol=1e-12)


BOUNDS = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)


@pytest.mark.parametrize("name", ["baseline", "A1", "rand3x4"])
def test_encode_decode_roundtrip(name):
    topo, params, _ = load_design(name)
    fixed = fixed_from_params(params)
    back = decode(encode(params, topo, BOUNDS), topo, BOUNDS, fixed)
    mask = topo.joint_mask
    for k in ("axis_theta", "axis_phi", "link_len", "lim_lo", "lim_hi"):
        np.testing.assert_allclose(np.asarray(getattr(back, k))[mask], np.asarray(getattr(params, k))[mask], atol=1e-10)
    np.testing.assert_allclose(np.asarray(back.base_xyz), np.asarray(params.base_xyz), atol=1e-10)
    np.testing.assert_allclose(np.asarray(back.base_rotvec), np.asarray(params.base_rotvec), atol=1e-12)
    np.testing.assert_array_equal(np.asarray(back.cap_radius), np.asarray(params.cap_radius))


def test_decode_respects_bounds_and_pinning():
    topo = Topology.three_finger(4)
    params = random_params(topo, jax.random.PRNGKey(3))
    bounds = Bounds()
    fixed = fixed_from_params(params, pin_fingers=(1,))
    phi = encode(params, topo, bounds)
    wild = jax.tree_util.tree_map(lambda a: a * 50.0, phi)
    p = decode(wild, topo, bounds, fixed)
    lo, hi = bounds.len_bounds(topo)
    assert np.all(np.asarray(p.link_len) >= lo - 1e-12) and np.all(np.asarray(p.link_len) <= hi + 1e-12)
    assert np.all(np.abs(np.asarray(p.base_xyz)) <= bounds.base_xyz_max_m + 1e-12)
    np.testing.assert_array_equal(np.asarray(p.base_xyz[1]), np.asarray(params.base_xyz[1]))
    np.testing.assert_array_equal(np.asarray(p.base_rotvec[1]), np.asarray(params.base_rotvec[1]))
    np.testing.assert_array_equal(np.asarray(p.lim_lo), np.asarray(params.lim_lo))
    g = jax.grad(lambda ph: jnp.sum(decode(ph, topo, bounds, fixed).base_xyz ** 2))(phi)
    assert float(jnp.abs(g.base_xyz_u[1]).max()) == 0.0 and float(jnp.abs(g.base_xyz_u[0]).max()) > 0.0


def test_anthropomorphic_matches_hand_family():
    from optimize.hand_family import HandFamily
    from dkarma.export import to_hand_spec
    topo = Topology.two_finger(4, 4)
    spec = to_hand_spec(topo, anthropomorphic(topo))
    fam = HandFamily()
    ref = fam.decode(fam.anthropomorphic_x0())
    for key in ("thumb", "index"):
        for k in ("knuckle_xyz", "knuckle_rotvec", "axis_theta", "axis_phi", "link_len", "lim_lo", "lim_hi"):
            np.testing.assert_allclose(spec[key][k], getattr(getattr(ref, key), k), atol=1e-12, err_msg=f"{key}.{k}")


def test_jit_with_traced_model_and_vmap_over_designs():
    topo = Topology.three_finger(4)
    designs = [random_params(topo, k) for k in jax.random.split(jax.random.PRNGKey(0), 5)]
    q = jnp.asarray(random_q(topo, designs[0], 1, seed=1)[0])

    def tips(params, q):
        m = build_hand_model(topo, params)
        return capsule_endpoints(m, fk(m, q))[1]

    single = np.stack([np.asarray(tips(p, q)) for p in designs])
    batched = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *designs)
    np.testing.assert_allclose(np.asarray(jax.jit(jax.vmap(tips, in_axes=(0, None)))(batched, q)), single, atol=1e-15)
    models = [build_hand_model(topo, p) for p in designs]
    shared = jax.tree_util.tree_map(
        lambda *a: jnp.stack(a) if jnp.issubdtype(jnp.asarray(a[0]).dtype, jnp.floating) else a[0], *models)
    out = fk_designs_shared(shared, jnp.broadcast_to(q, (5, topo.J)), models[0])
    b = jax.vmap(capsule_endpoints, in_axes=(design_in_axes(models[0]), 0))(shared, out)[1]
    np.testing.assert_allclose(np.asarray(b), single, atol=1e-15)
