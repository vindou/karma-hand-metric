"""Orientation grids (228 pinch-axis bins, Hopf SO(3) grids) and the smooth coverage kernels."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.metrics.orientation import (pinch_axis_bin_centres, pinch_axis_bins, hopf_grid, hopf_rotations,
                                        nearest_neighbour_angles, rotation_angle, PINCH_N_BINS, so3_bin_index)
from dkarma.metrics.smooth_coverage import (translation_coverage, pinch_axis_coverage, so3_coverage, occupancy,
                                            soft_seed_aggregate, region_orientation_coverage, pinch_axis_log_kernel)
from dkarma.metrics.commands import CommandConfig, build_library, seed_frame, seed_frame_np, perpendicular_axes

jax.config.update("jax_enable_x64", True)


def test_pinch_axis_bins_are_the_reference_bins():
    c = pinch_axis_bin_centres()
    assert c.shape == (PINCH_N_BINS, 3) == (228, 3)
    # every centre maps to its own index under the reference binning, and -centre too
    # the equator ring's centres lie exactly on z = 0, where ang2pix's ring choice is a coin toss;
    # nudge every centre a hair north before asking the reference which bin it is
    cn = c + np.array([0.0, 0.0, 1e-9]); cn /= np.linalg.norm(cn, axis=1, keepdims=True)
    assert np.array_equal(pinch_axis_bins(cn), np.arange(228))
    assert np.array_equal(pinch_axis_bins(-cn), np.arange(228))
    # random lines: the reference index is one of the 228, and the kernel's argmax agrees
    rng = np.random.RandomState(0)
    a = rng.randn(500, 3); a /= np.linalg.norm(a, axis=1, keepdims=True)
    idx = pinch_axis_bins(a)
    assert idx.min() >= 0 and idx.max() < 228
    k = np.asarray(pinch_axis_log_kernel(jnp.asarray(a), jnp.asarray(c), 0.09))
    agree = np.mean(np.argmax(k, axis=1) == idx)
    assert agree > 0.85        # nearest centre vs HEALPix pixel membership differ only near pixel edges


@pytest.mark.parametrize("n,spacing_deg", [(72, 60.0), (576, 30.0), (4608, 15.0)])
def test_hopf_grid_uniformity(n, spacing_deg):
    q = hopf_grid(n)
    assert q.shape == (n, 4)
    np.testing.assert_allclose(np.linalg.norm(q, axis=1), 1.0, atol=1e-12)
    R = hopf_rotations(n)
    np.testing.assert_allclose(np.einsum("nij,nkj->nik", R, R), np.broadcast_to(np.eye(3), (n, 3, 3)), atol=1e-12)
    nn = nearest_neighbour_angles(R)
    # uniformity: nearest-neighbour spacing within 15 % of the mean (measured: 72 and 576 within
    # 10 %, 4608 within 12 % because HEALPix pixel shapes vary between the cap and the belt)
    assert np.all(np.abs(nn - nn.mean()) <= 0.15 * nn.mean()), (nn.min(), nn.mean(), nn.max())
    # resolution of the level (documented spacing), loose factor
    assert 0.6 * spacing_deg <= np.degrees(nn.mean()) <= 1.3 * spacing_deg
    # Haar uniformity: the fraction of grid points within angle t of a random rotation
    # matches (t - sin t)/pi for a few t (loosely; this is a grid, not a sample)
    rng = np.random.RandomState(1)
    from scipy.spatial.transform import Rotation
    Rr = Rotation.random(50, random_state=rng).as_matrix()
    for t in (1.0, 1.5):
        frac = np.mean([np.mean(rotation_angle(Rr[i][None], R) <= t) for i in range(50)])
        assert abs(frac - (t - np.sin(t)) / np.pi) < 0.03


def test_so3_bin_index_identity():
    R = hopf_rotations(72)
    assert np.array_equal(so3_bin_index(R, 72), np.arange(72))


def test_occupancy_properties():
    logk = jnp.log(jnp.array([[0.5, 0.0], [0.5, 1.0]]) + 1e-300)
    logw = jnp.array([0.0, 0.0])
    o = occupancy(logk, logw)
    np.testing.assert_allclose(np.asarray(o), [0.75, 1.0 - 1e-7], atol=1e-6)
    # monotone in weights, and zero weight contributes nothing
    o2 = occupancy(logk, jnp.array([0.0, -jnp.inf]))
    np.testing.assert_allclose(np.asarray(o2), [0.5, 0.0], atol=1e-12)
    g = jax.grad(lambda lw: jnp.sum(occupancy(logk, lw)))(logw)
    assert np.all(np.asarray(g) >= 0)


def test_translation_coverage_limits():
    centres = jnp.asarray(np.stack(np.meshgrid(*[np.arange(4) * 0.01] * 3, indexing="ij"), -1).reshape(-1, 3))
    p = centres[:5]
    full = translation_coverage(p, jnp.zeros(5), centres, 1e-3)
    assert abs(float(full.fraction) - 5 / 64) < 1e-3                   # narrow kernel: one cell per state
    none = translation_coverage(p, jnp.full(5, -1e9), centres, 1e-3)
    assert float(none.fraction) == 0.0
    wide = translation_coverage(p, jnp.zeros(5), centres, 1.0)
    assert float(wide.fraction) > 0.99


def test_orientation_coverages_and_regions():
    c = jnp.asarray(pinch_axis_bin_centres())
    axes = c[:10]
    cov = pinch_axis_coverage(axes, jnp.zeros(10), c, 0.05)
    assert 10 / 228 - 0.01 < float(cov.fraction) < 10 / 228 + 0.02
    T = jnp.asarray(hopf_rotations(576))
    cov3 = so3_coverage(T[:20], jnp.zeros(20), T, 0.15)
    assert 20 / 576 < float(cov3.fraction) < 60 / 576
    p = jnp.asarray(np.random.RandomState(0).randn(10, 3) * 0.02)
    regions = p[:2]
    rc = region_orientation_coverage(p, jnp.zeros(10), regions, 0.005, pinch_axis_log_kernel(axes, c, 0.05))
    assert rc.shape == (2,) and bool(jnp.all(rc > 0)) and bool(jnp.all(rc <= cov.fraction + 1e-9))


def test_soft_seed_aggregate():
    s = jnp.array([0.1, 0.5, 0.9])
    assert abs(float(soft_seed_aggregate(s, 0.0, 0.01)) - 0.5) < 1e-12
    assert 0.1 <= float(soft_seed_aggregate(s, 1.0, 1e-3)) <= 0.1 + 1e-3 * np.log(3) + 1e-12   # min <= softmin <= min + tau log n


def test_command_library_and_seed_frame():
    cfg = CommandConfig(n_cells=2, n_sub=4, n_rot_steps=3)
    R = jnp.eye(3)
    lib = build_library(R, jnp.array([0.0, 0.0, 1.0]), cfg)
    assert lib.dp_T.shape == (6, 8, 3) and lib.dth_R.shape == (4, 3, 3)
    np.testing.assert_allclose(np.asarray(lib.dp_T[0].sum(0)), [8 * cfg.step_m, 0, 0], atol=1e-15)
    p1, p2 = perpendicular_axes(jnp.array([0.3, 0.4, 0.5]))
    for v in (p1, p2):
        assert abs(float(jnp.dot(v, jnp.array([0.3, 0.4, 0.5])))) < 1e-12
    assert abs(float(jnp.dot(p1, p2))) < 1e-12
    J = jnp.asarray(np.random.RandomState(2).randn(3, 8) * [[1.0], [0.5], [0.2]])
    Rs = np.asarray(seed_frame(J))
    np.testing.assert_allclose(Rs, seed_frame_np(np.asarray(J)), atol=1e-10)
    assert abs(np.linalg.det(Rs) - 1.0) < 1e-10
