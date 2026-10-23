"""Autodiff through build_hand_model + FK + contact geometry vs central finite differences
(SPEC §10.1 "Gradient check"; Phase 1 exit criterion: grad of a contact gap w.r.t. phi)."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import Bounds, build_hand_model, decode, encode, fixed_from_params
from dkarma.core.kinematics import fk, capsule_endpoints
from dkarma.core.geometry import capsule_pass, contact_kinematics, softmin, capsule_pair_clearance
from dkarma.core.contacts import ContactAssignment, contact_pass, object_clearance, self_clearance, antipodal_margin
from .conftest import load_design, random_q, planar_toy

jax.config.update("jax_enable_x64", True)

BOUNDS = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
R_S = 0.015


def _fd(fun, x, h=1e-6):
    x = np.asarray(x, dtype=float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (float(fun(x + e)) - float(fun(x - e))) / (2 * h)
    return g


def _check(fun, x0, rel=1e-5, floor=1e-9):
    g_ad = np.asarray(jax.grad(fun)(jnp.asarray(x0)))
    g_fd = _fd(fun, x0)
    assert np.all(np.isfinite(g_ad)), "non-finite autodiff gradient"
    err = np.abs(g_ad - g_fd)
    tol = rel * np.maximum(np.abs(g_fd), np.abs(g_ad)) + floor
    assert np.all(err <= tol), f"max rel err {np.max(err / (np.abs(g_fd) + floor)):.2e}"
    assert np.linalg.norm(g_ad) > 0.0


def _setup(name):
    topo, params, _ = load_design(name)
    fixed = fixed_from_params(params)
    flat0, unravel = jax.flatten_util.ravel_pytree(encode(params, topo, BOUNDS))
    q = jnp.asarray(random_q(topo, params, 1, seed=5)[0])
    build = lambda x: build_hand_model(topo, decode(unravel(x), topo, BOUNDS, fixed))   # noqa: E731
    return topo, flat0, q, build


def _sphere_near(build, flat0, q, c, offset):
    """Sphere centre off the interior of capsule ``c`` (generic: not on the axis, not at an end)."""
    m = build(flat0)
    a, b = capsule_endpoints(m, fk(m, q))
    seg = b[c] - a[c]
    n = jnp.cross(seg, jnp.array([0.3, 0.5, 0.8]))
    return a[c] + 0.6 * seg + offset * n / jnp.linalg.norm(n)


@pytest.mark.parametrize("name", ["baseline", "T", "rand3x4"])
def test_tip_position_gradient(name):
    topo, flat0, q, build = _setup(name)
    w = jnp.asarray(np.random.RandomState(0).randn(topo.C, 3))
    _check(lambda x: jnp.sum(w * capsule_endpoints(build(x), fk(build(x), q))[1]), flat0)


@pytest.mark.parametrize("name", ["baseline", "T", "A1", "rand3x4", "rand345"])
def test_contact_gap_gradient_wrt_phi_and_q(name):
    """The Phase 1 exit criterion."""
    topo, flat0, q, build = _setup(name)
    c = topo.seed_caps()[0]
    centre = _sphere_near(build, flat0, q, c, 0.03)
    _check(lambda x: capsule_pass(build(x), fk(build(x), q), centre, R_S).gap[c], flat0)
    m0 = build(flat0)
    _check(lambda qq: capsule_pass(m0, fk(m0, qq), centre, R_S).gap[c], np.asarray(q))


@pytest.mark.parametrize("name", ["A1", "rand3x4"])
def test_smooth_clearances_gradient(name):
    topo, flat0, q, build = _setup(name)
    c = topo.seed_caps()[0]
    centre = _sphere_near(build, flat0, q, c, 0.02)
    asg = ContactAssignment(caps=jnp.asarray(topo.seed_caps()), valid=jnp.array([True, True]))

    def obj(x):
        m = build(x)
        return object_clearance(m, capsule_pass(m, fk(m, q), centre, R_S), asg, beta=500.0)
    _check(obj, flat0)

    def selfc(x):
        m = build(x)
        return self_clearance(m, capsule_pass(m, fk(m, q), centre, R_S), beta=500.0)
    _check(selfc, flat0)


@pytest.mark.parametrize("name", ["T", "rand3x4"])
def test_contact_pass_gradients(name):
    """Fixed-assignment contact pass: gap, normal and surface Jacobian are differentiable in phi."""
    topo, flat0, q, build = _setup(name)
    ct, ci = topo.seed_caps()
    centre = _sphere_near(build, flat0, q, ct, 0.03)
    asg = ContactAssignment(caps=jnp.asarray([ct, ci]), valid=jnp.array([True, True]))
    w = jnp.asarray(np.random.RandomState(1).randn(3))

    def f(x):
        m = build(x)
        cs = contact_pass(m, fk(m, q), centre, R_S, asg)
        return jnp.sum(cs.gap) + jnp.dot(w, cs.normal[0]) + jnp.sum(cs.J_surf[1] ** 2) * 1e-2 + antipodal_margin(cs.normal, 0.6)
    _check(f, flat0)


def test_contact_jacobian_is_derivative_of_axis_point():
    topo, params, _ = load_design("T")
    m = build_hand_model(topo, params)
    q = jnp.asarray(random_q(topo, params, 1, seed=6)[0])
    c = topo.seed_caps()[1]
    a, b = capsule_endpoints(m, fk(m, q))
    seg = b[c] - a[c]
    n = jnp.cross(seg, jnp.array([0.3, 0.5, 0.8]))
    centre = a[c] + 0.5 * seg + 0.03 * n / jnp.linalg.norm(n)
    axis_pt = lambda qq: capsule_pass(m, fk(m, qq), centre, R_S).axis_pt[c]   # noqa: E731
    f = fk(m, q)
    ck = contact_kinematics(m, f, capsule_pass(m, f, centre, R_S), c, centre, R_S)
    np.testing.assert_allclose(np.asarray(ck.Jv), np.asarray(jax.jacfwd(axis_pt)(q)), atol=1e-10)


def test_toy_gap_gradient_fd():
    topo, params, r = planar_toy()
    centre = jnp.array([0.07, 0.03, 0.0])

    def g(x):
        p = params._replace(link_len=params.link_len.at[0, :].set(x[:2]))
        m = build_hand_model(topo, p)
        return capsule_pass(m, fk(m, jnp.concatenate([x[2:], jnp.zeros(1)])), centre, R_S).gap[1]
    _check(g, np.array([0.05, 0.04, 0.3, -0.4]))


def test_softmin_bounds_and_gradient():
    x = jnp.array([0.02, 0.011, 0.05, 0.3])
    mask = jnp.array([True, True, True, False])
    s = softmin(x, 1000.0, mask)
    assert float(s) <= 0.011 and float(s) > 0.011 - np.log(3) / 1000.0
    g = jax.grad(lambda y: softmin(y, 1000.0, mask))(x)
    assert float(g[3]) == 0.0 and float(jnp.sum(g)) == pytest.approx(1.0, abs=1e-12)


def test_exp_so3_safe_matches_reference_and_is_finite_at_zero():
    """Phase 9.1: the rollout's exponential map equals ``karma_jax.steps.exp_so3`` in value and
    derivative away from zero and has a finite (zero) derivative at an exactly zero rotation vector,
    where the reference form's gradient is NaN (a failed rotation QP produces that state)."""
    import jax
    import jax.numpy as jnp
    from karma_jax.steps import exp_so3
    from dkarma.core.geometry import exp_so3_safe
    for w in (jnp.array([0.3, -0.2, 0.1]), jnp.array([1e-7, 0.0, 0.0]), jnp.array([1e-12, 0.0, 0.0])):
        np.testing.assert_allclose(np.asarray(exp_so3_safe(w)), np.asarray(exp_so3(w)), atol=1e-15)
        ja = jax.jacfwd(exp_so3_safe)(w); jb = jax.jacfwd(exp_so3)(w)
        np.testing.assert_allclose(np.asarray(ja), np.asarray(jb), atol=1e-12)
    z = jnp.zeros(3)
    np.testing.assert_allclose(np.asarray(exp_so3_safe(z)), np.eye(3))
    g_ref = jax.grad(lambda w: jnp.sum(exp_so3(w)))(z)
    g_safe = jax.grad(lambda w: jnp.sum(exp_so3_safe(w)))(z)
    assert not np.all(np.isfinite(np.asarray(g_ref))) and np.all(np.isfinite(np.asarray(g_safe)))
