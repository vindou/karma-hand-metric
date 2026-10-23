"""The masked projection equals ``karma_jax.projection.project_onto_contacts`` for two lanes
(after a perturbation that takes the state off the manifold), and is differentiable."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from karma_jax.projection import project_onto_contacts
from dkarma.solvers.projection import project
from . import refstates

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(not all(refstates.available(n) for n in ("T", "A1")),
                                reason="optimizer result pickles not present")


@pytest.fixture(params=["T", "A1"])
def ref(request, tmp_path_factory):
    return refstates.load(request.param, tmp_path_factory.mktemp("ref"), n_nodes=5, seed=3)


def test_matches_reference_projection(ref):
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    rng = np.random.RandomState(0)
    n_ok = 0
    for st, cp_pair in ref["nodes"]:
        asg_n = asg._replace(caps=jnp.asarray(cp_pair, jnp.int32))
        q = st.q + jnp.asarray(rng.randn(st.q.shape[0]) * 0.02)      # off the manifold
        pr_ref = project_onto_contacts(hm.dm, q, st.p, jnp.asarray(cp_pair), sp.sphere_radius, sp.link_radius, sp.eps_g)
        pr = project(hm, q, st.p, asg_n, sp)
        assert bool(pr.ok) == bool(pr_ref.ok)
        np.testing.assert_allclose(np.asarray(pr.q), np.asarray(pr_ref.q), atol=1e-11)
        np.testing.assert_allclose(np.asarray(pr.dtheta), np.asarray(pr_ref.dtheta), atol=1e-10)
        n_ok += bool(pr.ok)
    assert n_ok >= 1


def test_projection_gradient_no_backtrack(ref):
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    st, cp_pair = ref["nodes"][0]
    asg_n = asg._replace(caps=jnp.asarray(cp_pair, jnp.int32))
    q0 = st.q + jnp.asarray(np.random.RandomState(1).randn(st.q.shape[0]) * 0.01)
    w = jnp.asarray(np.random.RandomState(2).randn(st.q.shape[0]))

    def f(x):
        return jnp.dot(w, project(hm, x[:-3], x[-3:], asg_n, sp, max_iters=4, backtrack=False).q)

    x0 = jnp.concatenate([q0, st.p])
    g_ad = np.asarray(jax.grad(f)(x0))
    h = 1e-6
    g_fd = np.array([(float(f(x0.at[i].add(h))) - float(f(x0.at[i].add(-h)))) / (2 * h) for i in range(x0.shape[0])])
    assert np.all(np.isfinite(g_ad))
    err = np.abs(g_ad - g_fd)
    assert np.all(err <= 1e-5 * np.maximum(np.abs(g_fd), np.abs(g_ad)) + 1e-9)
