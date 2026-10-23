"""FK consistency (SPEC §10.1; tolerance 1e-8: the URDF carries 9 decimals): the pure-JAX model built from ``HandParams`` must match
Pinocchio on the exported URDF (placements, FK, capsule geometry, Jacobians), by index
(the topology replicates Pinocchio's ordering) on the Phase 0 designs and random
three-finger hands."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from dkarma.core import build_hand_model
from dkarma.core.kinematics import fk, capsule_endpoints, frame_jacobian
from dkarma.export import load_reference
from karma_jax.kinematics import forward_kinematics as ref_fk, frame_jacobian as ref_jac
from karma_jax.geometry import capsule_endpoints as ref_caps
from .conftest import random_q

jax.config.update("jax_enable_x64", True)


@pytest.fixture
def ref(design, tmp_path):
    topo, params, name = design
    cfg, ctx, flat = load_reference(topo, params, tmp_path, name=name)
    m = build_hand_model(topo, params._replace(cap_radius=jnp.full((topo.C,), cfg.link_radius_m)))
    return topo, params, name, cfg, ctx, flat, m


def test_index_spaces_match_pinocchio(ref):
    topo, params, name, cfg, ctx, flat, m = ref
    assert tuple(flat.joint_names) == topo.joint_names
    assert tuple(flat.frame_names) == topo.frame_names
    assert tuple(flat.capsule_names) == topo.capsule_names
    assert flat.parent.tolist() == topo.parent.tolist()
    assert flat.frame_parent.tolist() == topo.frame_parent.tolist()
    assert flat.cap_frame.tolist() == topo.cap_frame.tolist()
    assert flat.cap_distal.tolist() == topo.cap_distal.tolist()
    strip = lambda a: [int(c) for c in np.asarray(a).tolist() if c >= 0]   # padding lengths may differ
    assert strip(flat.thumb_caps) == strip(m.dm.thumb_caps)
    assert strip(flat.index_caps) == strip(m.dm.index_caps)
    assert (flat.seed_thumb_cap, flat.seed_index_cap) == topo.seed_caps()


def test_joint_placements_match_pinocchio(ref):
    topo, params, name, cfg, ctx, flat, m = ref
    np.testing.assert_allclose(np.asarray(m.dm.t_joint), flat.t_joint, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.dm.R_joint), flat.R_joint, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.dm.axis), flat.axis, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.dm.q_lower), flat.q_lower, atol=1e-6)   # URDF has 6 decimals
    np.testing.assert_allclose(np.asarray(m.dm.q_upper), flat.q_upper, atol=1e-6)
    np.testing.assert_allclose(np.asarray(m.dm.cap_tip), flat.cap_tip, atol=1e-8)
    np.testing.assert_allclose(np.asarray(m.dm.R_frame), flat.R_frame, atol=1e-15)
    np.testing.assert_allclose(np.asarray(m.dm.t_frame), flat.t_frame, atol=1e-15)


def test_fk_and_capsules_match_pinocchio(ref):
    import pinocchio as pin
    topo, params, name, cfg, ctx, flat, m = ref
    dmr = flat.to_device(jnp.float64)
    model, data = ctx.model, ctx.model.createData()
    for qi in random_q(topo, params, 4, seed=3):
        q = jnp.asarray(qi)
        f, fr = fk(m, q), ref_fk(dmr, q)
        for a, b in zip(f, fr):
            np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=1e-8)
        pin.forwardKinematics(model, data, qi)
        for j in range(topo.J):
            np.testing.assert_allclose(np.asarray(f.o_joint[j]), np.asarray(data.oMi[j + 1].translation), atol=1e-8)
            np.testing.assert_allclose(np.asarray(f.R_joint[j]), np.asarray(data.oMi[j + 1].rotation), atol=1e-8)
        for ours, theirs in zip(capsule_endpoints(m, f), ref_caps(dmr, fr)):
            np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), atol=1e-8)


def test_frame_jacobians_match(ref):
    topo, params, name, cfg, ctx, flat, m = ref
    dmr = flat.to_device(jnp.float64)
    for qi in random_q(topo, params, 2, seed=4):
        f, fr = fk(m, jnp.asarray(qi)), ref_fk(dmr, jnp.asarray(qi))
        for fr_id in range(topo.F_frames):
            for ours, theirs in zip(frame_jacobian(m, f, fr_id), ref_jac(dmr, fr, fr_id)):
                np.testing.assert_allclose(np.asarray(ours), np.asarray(theirs), atol=1e-8)


def test_reference_loader_accepts_design(ref):
    topo, params, name, cfg, ctx, flat, m = ref
    assert cfg.l_ref_m > 0.05
