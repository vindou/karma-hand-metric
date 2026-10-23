"""Phase 1 tests for ``dkarma.core.kinematics`` (SPEC.md §10.1: FK consistency, Jacobian finite
differences, jit/vmap agreement, determinism, gradients through ``phi``).

fp64, CPU (``JAX_PLATFORMS=cpu``), fast. The shared helpers at the top (fixture loading, random
pinch states, finite differences) are also used by ``test_contacts.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from dkarma import fixtures as FX                                    # noqa: E402
from dkarma.core import morphology as M                               # noqa: E402
from dkarma.core import kinematics as KIN                             # noqa: E402
from dkarma.core.topology import Topology                             # noqa: E402
from karma_jax import kinematics as K                                 # noqa: E402
from karma_jax import geometry as G                                   # noqa: E402

FIXTURES = ("baseline", "T", "A1")
MODELS = FIXTURES + ("three_finger",)
URDF_TOL = 2e-9          # the family URDF writes 9 decimals; FK of phi vs the reference flat model
FD_H = 1e-6


# ── shared helpers ───────────────────────────────────────────────────────────

_cache: dict = {}


def load_design(name):
    """Fixture design: ``dict(flat, cfg, ctx, dm_ref, spec, topo, phi, hm)`` (cached per session)."""
    if name not in _cache:
        from optimize.hand_family import HandSpec
        flat, cfg, ctx = FX.load_flat(name)
        spec = HandSpec.from_json((FX.ROOT / FX.load_scores()["fixtures"][name]["design_json"]).read_text())
        topo, phi = M.from_hand_spec(spec, cap_radius=cfg.link_radius_m)
        phi = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), phi)
        hm = M.build_hand_model(topo, phi)
        _cache[name] = dict(flat=flat, cfg=cfg, ctx=ctx, dm_ref=flat.to_device(jnp.float64),
                            spec=spec, topo=topo, phi=phi, hm=hm)
    return _cache[name]


def load_model(name):
    """``(topo, phi, hm)`` for a fixture design or the three-finger anthropomorphic start."""
    if name == "three_finger":
        if name not in _cache:
            topo = Topology.three_finger(4)
            phi = jax.tree_util.tree_map(lambda a: jnp.asarray(a, jnp.float64), M.anthropomorphic(topo))
            _cache[name] = dict(topo=topo, phi=phi, hm=M.build_hand_model(topo, phi))
        d = _cache[name]
    else:
        d = load_design(name)
    return d["topo"], d["phi"], d["hm"]


def random_q(hm, n, rng):
    """``[n, J]`` configurations uniform in the joint limits (numpy fp64)."""
    lo, hi = np.asarray(hm.dm.q_lower), np.asarray(hm.dm.q_upper)
    return lo + rng.rand(n, lo.shape[0]) * (hi - lo)


def random_states(topo, hm, n, seed=0, spread_m=2e-3):
    """``[(q, p)]``: ``q`` uniform in limits, ``p`` at the midpoint of the two seed capsules' distal
    ends (the reference's ``_random_pinch_states``) plus ``N(0, spread)``; the second half of the
    samples instead centres ``p`` between the capsule mid-points, so closest points fall both
    at the tip end and in the interior of the segments."""
    rng = np.random.RandomState(seed)
    ct, ci = topo.seed_caps()
    out = []
    for i, q in enumerate(random_q(hm, n, rng)):
        a, b = KIN.capsule_endpoints(hm, KIN.fk(hm, jnp.asarray(q)))
        a, b = np.asarray(a), np.asarray(b)
        if i % 2 == 0:
            p = 0.5 * (b[ct] + b[ci])
        else:
            p = 0.25 * (a[ct] + b[ct] + a[ci] + b[ci])
        out.append((q, p + rng.randn(3) * spread_m))
    return out


def fd_grad(f, x, h=FD_H):
    """Central finite-difference gradient of scalar ``f`` at array ``x`` (fp64 numpy)."""
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    flat = x.reshape(-1)
    for i in range(flat.size):
        xp, xm = flat.copy(), flat.copy()
        xp[i] += h
        xm[i] -= h
        g.reshape(-1)[i] = (float(f(xp.reshape(x.shape))) - float(f(xm.reshape(x.shape)))) / (2 * h)
    return g


def fd_grad_tree(f, tree, h=FD_H):
    """Central FD of scalar ``f(tree)`` w.r.t. every leaf of a pytree."""
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    leaves = [np.asarray(l, dtype=np.float64) for l in leaves]
    grads = []
    for k, leaf in enumerate(leaves):
        def fk_(x, k=k):
            ls = list(leaves)
            ls[k] = x
            return f(jax.tree_util.tree_unflatten(treedef, ls))
        grads.append(fd_grad(fk_, leaf, h))
    return jax.tree_util.tree_unflatten(treedef, grads)


def assert_tree_close(ad, fd, rel, what=""):
    """``max|ad - fd| <= rel * max|fd|`` over the whole tree (a relative test against the
    gradient's scale, so exactly-zero leaves such as the joint limits count too)."""
    ad_l = [np.asarray(x, dtype=np.float64) for x in jax.tree_util.tree_leaves(ad)]
    fd_l = [np.asarray(x, dtype=np.float64) for x in jax.tree_util.tree_leaves(fd)]
    assert all(np.all(np.isfinite(x)) for x in ad_l), f"{what}: non-finite gradient"
    scale = max(np.abs(x).max() for x in fd_l)
    assert scale > 0, f"{what}: zero finite-difference gradient"
    err = max(np.abs(a - b).max() for a, b in zip(ad_l, fd_l))
    assert err <= rel * scale, f"{what}: FD mismatch {err:.3e} vs scale {scale:.3e} (rel {err / scale:.2e})"


def tip_scalar(topo):
    """``f(phi, q, w) = w . b_tip`` of the thumb-like distal capsule: a scalar fingertip coordinate."""
    ct, _ = topo.seed_caps()

    def f(phi, q, w):
        hm = M.build_hand_model(topo, phi)
        _, b = KIN.capsule_endpoints(hm, KIN.fk(hm, q))
        return jnp.dot(w, b[ct])
    return f


# ── FK and Jacobians against the reference flat model ────────────────────────

@pytest.mark.parametrize("name", FIXTURES)
def test_seed_capsules_match_reference(name):
    d = load_design(name)
    flat, cfg, topo = d["flat"], d["cfg"], d["topo"]
    assert topo.seed_caps() == (flat.capsule_index(cfg.seed_thumb_link), flat.capsule_index(cfg.seed_index_link))
    assert tuple(flat.joint_names) == topo.joint_names
    assert tuple(flat.capsule_names) == topo.capsule_names


@pytest.mark.parametrize("name", FIXTURES)
def test_fk_and_jacobians_match_reference_flat_model(name):
    d = load_design(name)
    hm, dm_ref, topo = d["hm"], d["dm_ref"], d["topo"]
    rng = np.random.RandomState(1)
    for q in random_q(hm, 6, rng):
        q = jnp.asarray(q)
        r = KIN.fk(hm, q)
        rr = K.forward_kinematics(dm_ref, q)
        for a, b in zip(r, rr):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0, atol=URDF_TOL)
        o, w = KIN.joint_origins_axes(hm, q)
        np.testing.assert_allclose(np.asarray(o), np.asarray(rr.o_joint), rtol=0, atol=URDF_TOL)
        np.testing.assert_allclose(np.asarray(w), np.asarray(rr.w_axis), rtol=0, atol=URDF_TOL)
        for f in range(topo.F_frames):
            Jv, Jw = KIN.frame_jacobian(hm, r, f)
            Jv_r, Jw_r = K.frame_jacobian(dm_ref, rr, f)
            np.testing.assert_allclose(np.asarray(Jv), np.asarray(Jv_r), rtol=0, atol=URDF_TOL)
            np.testing.assert_allclose(np.asarray(Jw), np.asarray(Jw_r), rtol=0, atol=URDF_TOL)
        a, b = KIN.capsule_endpoints(hm, r)
        a_r, b_r = G.capsule_endpoints(dm_ref, rr)
        np.testing.assert_allclose(np.asarray(a), np.asarray(a_r), rtol=0, atol=URDF_TOL)
        np.testing.assert_allclose(np.asarray(b), np.asarray(b_r), rtol=0, atol=URDF_TOL)


@pytest.mark.parametrize("name", FIXTURES)
def test_wrappers_are_bit_identical_on_the_same_device_model(name):
    d = load_design(name)
    hm_ref = d["hm"]._replace(dm=d["dm_ref"])
    rng = np.random.RandomState(2)
    for q in random_q(hm_ref, 3, rng):
        q = jnp.asarray(q)
        r = KIN.fk(hm_ref, q)
        rr = K.forward_kinematics(d["dm_ref"], q)
        for a, b in zip(r, rr):
            assert np.array_equal(np.asarray(a), np.asarray(b))
        for f in (2, 5, d["topo"].F_frames - 1):
            for a, b in zip(KIN.frame_jacobian(hm_ref, r, f), K.frame_jacobian(d["dm_ref"], rr, f)):
                assert np.array_equal(np.asarray(a), np.asarray(b))
        for a, b in zip(KIN.capsule_endpoints(hm_ref, r), G.capsule_endpoints(d["dm_ref"], rr)):
            assert np.array_equal(np.asarray(a), np.asarray(b))


# ── point Jacobian vs finite differences in q ────────────────────────────────

@pytest.mark.parametrize("name", MODELS)
def test_point_jacobian_finite_differences(name):
    topo, phi, hm = load_model(name)
    rng = np.random.RandomState(3)
    frames = [topo.body_frame(j) for j in (0, topo.J // 2, topo.J - 1)] + [topo.joint_frame(1)]
    point = jax.jit(lambda q, f, off: (lambda r: r.p_frame[f] + r.R_frame[f] @ off)(KIN.fk(hm, q)))
    rot = jax.jit(lambda q, f: KIN.fk(hm, q).R_frame[f])
    jac = jax.jit(lambda q, f, off: KIN.point_jacobian(hm, KIN.fk(hm, q), f, off))
    for q in random_q(hm, 3, rng):
        for f in frames:
            off = jnp.asarray(rng.randn(3) * 0.02)
            Jv, Jw = jac(jnp.asarray(q), f, off)
            Jv_fd = np.stack([fd_grad(lambda x: point(jnp.asarray(x), f, off)[i], q) for i in range(3)])
            scale = np.abs(Jv_fd).max()
            assert np.abs(np.asarray(Jv) - Jv_fd).max() <= 1e-7 * scale
            # angular part: skew(Jw[:, j]) == dR/dq_j R^T
            R = np.asarray(rot(jnp.asarray(q), f))
            for j in range(topo.J):
                qp, qm = q.copy(), q.copy()
                qp[j] += FD_H
                qm[j] -= FD_H
                dR = (np.asarray(rot(jnp.asarray(qp), f)) - np.asarray(rot(jnp.asarray(qm), f))) / (2 * FD_H)
                S = dR @ R.T
                w_fd = np.array([S[2, 1], S[0, 2], S[1, 0]])
                np.testing.assert_allclose(np.asarray(Jw)[:, j], w_fd, rtol=0, atol=1e-7)
            # the point Jacobian reduces to the frame Jacobian at zero offset
            Jv0, _ = jac(jnp.asarray(q), f, jnp.zeros(3))
            Jv_f, _ = KIN.frame_jacobian(hm, KIN.fk(hm, jnp.asarray(q)), f)
            np.testing.assert_allclose(np.asarray(Jv0), np.asarray(Jv_f), rtol=0, atol=1e-15)


# ── jit / eager / vmap over designs, determinism ─────────────────────────────

def _perturbed_phis(phi, n, rng, scale=0.05):
    """``n`` morphologies stacked on a leading axis: ``phi`` times ``(1 + scale * N(0,1))`` per entry."""
    return jax.tree_util.tree_map(
        lambda a: jnp.stack([a * (1.0 + scale * rng.randn(*a.shape)) for _ in range(n)]), phi)


@pytest.mark.parametrize("name", ("baseline", "three_finger"))
def test_jit_eager_and_vmap_over_designs_agree(name):
    topo, phi, hm = load_model(name)
    rng = np.random.RandomState(4)
    if name == "baseline":      # the three fixture designs share one topology: a real design batch
        phis = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[load_design(n)["phi"] for n in FIXTURES])
    else:
        phis = _perturbed_phis(phi, 3, rng)
    B = 3
    qs = jnp.asarray(np.stack([random_q(hm, 1, rng)[0] for _ in range(B)]))

    build_fk = lambda ph, q: KIN.fk(M.build_hand_model(topo, ph), q)      # noqa: E731
    eager = [build_fk(jax.tree_util.tree_map(lambda a: a[i], phis), qs[i]) for i in range(B)]
    jitted = jax.jit(build_fk)
    batched_hm = jax.vmap(lambda ph: M.build_hand_model(topo, ph))(phis)
    assert batched_hm.dm.parent.shape == (B, topo.J) and batched_hm.cap_radius.shape == (B, topo.C)
    vm = KIN.fk_designs(batched_hm, qs)
    vm_jit = jax.jit(KIN.fk_designs)(batched_hm, qs)
    for i in range(B):
        ji = jitted(jax.tree_util.tree_map(lambda a: a[i], phis), qs[i])
        for e, j, v, vj in zip(eager[i], ji, vm, vm_jit):
            np.testing.assert_allclose(np.asarray(j), np.asarray(e), rtol=0, atol=1e-13)
            np.testing.assert_allclose(np.asarray(v[i]), np.asarray(e), rtol=0, atol=1e-13)
            np.testing.assert_allclose(np.asarray(vj[i]), np.asarray(e), rtol=0, atol=1e-13)
    # one design, many configurations
    vc = KIN.fk_configs(hm, qs)
    for i in range(B):
        for v, e in zip(vc, KIN.fk(hm, qs[i])):
            np.testing.assert_allclose(np.asarray(v[i]), np.asarray(e), rtol=0, atol=1e-13)


@pytest.mark.parametrize("name", ("T", "three_finger"))
def test_determinism(name):
    topo, phi, hm = load_model(name)
    q = jnp.asarray(random_q(hm, 1, np.random.RandomState(5))[0])
    f = jax.jit(lambda ph, q: KIN.fk(M.build_hand_model(topo, ph), q))
    g = jax.jit(jax.grad(tip_scalar(topo)))
    w = jnp.asarray([0.3, -0.5, 0.8])
    for fn, args in ((KIN.fk, (hm, q)), (f, (phi, q)), (g, (phi, q, w))):
        r1 = fn(*args)
        r2 = fn(*args)
        for a, b in zip(jax.tree_util.tree_leaves(r1), jax.tree_util.tree_leaves(r2)):
            assert np.array_equal(np.asarray(a), np.asarray(b))


# ── gradients of a fingertip position through every leaf of phi ──────────────

@pytest.mark.parametrize("name", MODELS)
def test_fingertip_gradient_wrt_phi_matches_finite_differences(name):
    topo, phi, hm = load_model(name)
    rng = np.random.RandomState(6)
    f = tip_scalar(topo)
    f_jit = jax.jit(f)
    grad_phi = jax.jit(jax.grad(f, argnums=0))
    grad_q = jax.jit(jax.grad(f, argnums=1))
    for q in random_q(hm, 2, rng):
        q = jnp.asarray(q)
        w = jnp.asarray(rng.randn(3))
        g_phi = grad_phi(phi, q, w)
        assert jax.tree_util.tree_structure(g_phi) == jax.tree_util.tree_structure(phi)
        fd_phi = fd_grad_tree(lambda ph: f_jit(jax.tree_util.tree_map(jnp.asarray, ph), q, w), phi)
        assert_tree_close(g_phi, fd_phi, rel=1e-6, what=f"{name} d/dphi")
        # limits do not move the tip; the radius only shortens the terminal capsule axis
        assert np.all(np.asarray(g_phi.lim_lo) == 0.0) and np.all(np.asarray(g_phi.lim_hi) == 0.0)
        fd_q = fd_grad(lambda x: f_jit(phi, jnp.asarray(x), w), np.asarray(q))
        assert_tree_close(grad_q(phi, q, w), fd_q, rel=1e-6, what=f"{name} d/dq")


def test_axes_stay_unit_after_parameter_update():
    topo, phi, hm = load_model("three_finger")
    rng = np.random.RandomState(7)
    phi2 = jax.tree_util.tree_map(lambda a: a + 0.3 * rng.randn(*a.shape), phi)
    hm2 = M.build_hand_model(topo, phi2)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(hm2.dm.axis), axis=-1), 1.0, rtol=0, atol=1e-15)
    w = KIN.fk(hm2, jnp.asarray(random_q(hm2, 1, rng)[0])).w_axis
    np.testing.assert_allclose(np.linalg.norm(np.asarray(w), axis=-1), 1.0, rtol=0, atol=1e-14)
