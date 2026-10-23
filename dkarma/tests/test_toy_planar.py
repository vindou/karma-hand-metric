"""Two-link planar finger with closed-form FK and derivatives (CLAUDE.md rule 7: the
analytic toy comes before spatial hands)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import build_hand_model
from dkarma.core.kinematics import fk, capsule_endpoints, frame_jacobian
from dkarma.core.geometry import capsule_pass
from .conftest import planar_toy

jax.config.update("jax_enable_x64", True)

Q_OTHER = 0.0   # the dummy finger's joint


def analytic(l1, l2, r, q1, q2):
    j2 = np.array([l1 * np.cos(q1), l1 * np.sin(q1), 0.0])
    tip = j2 + (l2 - r) * np.array([np.cos(q1 + q2), np.sin(q1 + q2), 0.0])
    return j2, tip


@pytest.mark.parametrize("q", [(0.0, 0.0), (0.3, -0.7), (-1.2, 0.9)])
def test_fk_matches_closed_form(q):
    topo, params, r = planar_toy()
    m = build_hand_model(topo, params)
    fkr = fk(m, jnp.array([q[0], q[1], Q_OTHER]))
    a, b = capsule_endpoints(m, fkr)
    j2, tip = analytic(0.05, 0.04, r, *q)
    np.testing.assert_allclose(np.asarray(fkr.o_joint[1]), j2, atol=1e-15)
    np.testing.assert_allclose(np.asarray(a[0]), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(b[0]), j2, atol=1e-15)      # link 1 ends at joint 2
    np.testing.assert_allclose(np.asarray(a[1]), j2, atol=1e-15)
    np.testing.assert_allclose(np.asarray(b[1]), tip, atol=1e-15)     # terminal: tip - r along the link
    np.testing.assert_allclose(np.asarray(fkr.w_axis[:2]), [[0, 0, 1], [0, 0, 1]], atol=1e-15)


def test_jacobian_matches_closed_form():
    topo, params, r = planar_toy()
    m = build_hand_model(topo, params)
    q1, q2 = 0.4, -0.5
    fkr = fk(m, jnp.array([q1, q2, Q_OTHER]))
    Jv, Jw = frame_jacobian(m, fkr, topo.body_frame(1))      # joint-2 body frame origin
    l1 = 0.05
    np.testing.assert_allclose(np.asarray(Jv[:, 0]), [-l1 * np.sin(q1), l1 * np.cos(q1), 0.0], atol=1e-15)
    np.testing.assert_allclose(np.asarray(Jv[:, 1]), 0.0, atol=1e-15)
    np.testing.assert_allclose(np.asarray(Jw[:, :2]), [[0, 0], [0, 0], [1, 1]], atol=1e-15)
    np.testing.assert_allclose(np.asarray(Jv[:, 2]), 0.0, atol=1e-15)   # the other finger does not move it


def test_morphology_gradients_match_closed_form():
    topo, params, r = planar_toy()
    q = jnp.array([0.4, -0.5, Q_OTHER])

    def tip(link_len, qq):
        p = params._replace(link_len=params.link_len.at[0, :].set(link_len))
        m = build_hand_model(topo, p)
        return capsule_endpoints(m, fk(m, qq))[1][1]

    g_len = jax.jacfwd(tip, argnums=0)(jnp.array([0.05, 0.04]), q)
    g_q = jax.jacfwd(tip, argnums=1)(jnp.array([0.05, 0.04]), q)
    q1, q2 = 0.4, -0.5
    c1, s1, c12, s12 = np.cos(q1), np.sin(q1), np.cos(q1 + q2), np.sin(q1 + q2)
    np.testing.assert_allclose(np.asarray(g_len[:, 0]), [c1, s1, 0.0], atol=1e-14)
    np.testing.assert_allclose(np.asarray(g_len[:, 1]), [c12, s12, 0.0], atol=1e-14)
    l1, l2r = 0.05, 0.04 - r
    np.testing.assert_allclose(np.asarray(g_q[:, 0]), [-l1 * s1 - l2r * s12, l1 * c1 + l2r * c12, 0.0], atol=1e-14)
    np.testing.assert_allclose(np.asarray(g_q[:, 1]), [-l2r * s12, l2r * c12, 0.0], atol=1e-14)
    np.testing.assert_allclose(np.asarray(g_q[:, 2]), 0.0, atol=1e-14)


def test_radius_gradient_shortens_terminal_axis():
    topo, params, r = planar_toy()
    q = jnp.array([0.0, 0.0, Q_OTHER])

    def tip_x(rad):
        m = build_hand_model(topo, params._replace(cap_radius=params.cap_radius.at[1].set(rad)))
        return capsule_endpoints(m, fk(m, q))[1][1][0]
    assert float(jax.grad(tip_x)(jnp.asarray(r))) == pytest.approx(-1.0, abs=1e-14)


def test_gap_gradient_closed_form():
    """Sphere on the link axis beyond the fingertip: g = |c - tip| - r_s - r, dg/dl1 = dg/dl2 = -1."""
    topo, params, r = planar_toy()
    q = jnp.array([0.0, 0.0, Q_OTHER])
    r_s = 0.015
    centre = jnp.array([0.05 + 0.04 - r + 0.03, 0.0, 0.0])

    def gap(link_len):
        p = params._replace(link_len=params.link_len.at[0, :].set(link_len))
        m = build_hand_model(topo, p)
        return capsule_pass(m, fk(m, q), centre, r_s).gap[1]

    assert float(gap(jnp.array([0.05, 0.04]))) == pytest.approx(0.03 - r_s - r, abs=1e-15)
    np.testing.assert_allclose(np.asarray(jax.grad(gap)(jnp.array([0.05, 0.04]))), [-1.0, -1.0], atol=1e-12)
