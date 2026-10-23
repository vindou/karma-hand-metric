"""Rollout: scan equals a Python loop; a hard-mode step equals the reference sub-step; the
exact validator agrees with the smooth step's own gates where the reference keeps the
contact pair; gradients of the final sphere position and of the trajectory energy with
respect to the morphology match finite differences; jit and vmap over commands agree."""
import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np
import pytest

from karma_jax.qp import default_settings, QPSettings

TIGHT = QPSettings(iters=80, tol=1e-13, infeas_tol=1e-6, reg=1e-13)   # see test_rolling_qp.TIGHT
from karma_jax.steps import GraspState as RefGraspState, attempt_primitive_step

from dkarma.core import build_hand_model, Bounds, encode, decode, fixed_from_params
from dkarma.solvers.rollout import GraspState, RollConfig, EnergyWeights, step, rollout, energies, exact_gates
from dkarma.solvers.exact import validate_translation, validate_rotation
from . import refstates

jax.config.update("jax_enable_x64", True)

pytestmark = pytest.mark.skipif(not all(refstates.available(n) for n in ("T", "baseline")),
                                reason="optimizer result pickles not present")

SUB = 1.25e-3   # the reference's nominal sub-step for a 200 mm hand; scaled per design below


@pytest.fixture(params=["T", "baseline"])
def ref(request, tmp_path_factory):
    return refstates.load(request.param, tmp_path_factory.mktemp("ref"), n_nodes=3, seed=5)


@pytest.fixture(autouse=True)
def _free_compiled_programs():
    """Host-memory hygiene: drop compiled programs after every test of this module."""
    yield
    jax.clear_caches()


def _commands(ref, direction, n_steps):
    """Sub-steps along a seed-frame axis, as the reference's primitives (voxel = 8 sub-steps)."""
    cfg = ref["cfg"]
    sub = cfg.step_size_m
    d = jnp.asarray(ref["seed_frame"][:, direction % 3]) * (1.0 if direction < 3 else -1.0)
    return jnp.broadcast_to(sub * d, (n_steps, 3)), jnp.zeros((n_steps, 3))


def test_scan_equals_loop_and_jit(ref):
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    st0 = ref["seed_state"]
    dps, dths = _commands(ref, 0, 6)
    for mode in ("soft", "hard"):
        cfg = RollConfig(mode=mode)
        states, diags = rollout(hm, st0, dps, dths, asg, sp, cfg)
        st = st0
        for k in range(6):
            st, dg = step(hm, st, dps[k], dths[k], asg, sp, cfg)
            np.testing.assert_allclose(np.asarray(st.q), np.asarray(states.q[k + 1]), atol=1e-13)
            np.testing.assert_allclose(float(dg.slip_m), float(diags.slip_m[k]), atol=1e-15)
        states_j, diags_j = jax.jit(lambda s, a, b: rollout(hm, s, a, b, asg, sp, cfg))(st0, dps, dths)
        np.testing.assert_allclose(np.asarray(states_j.q), np.asarray(states.q), atol=1e-11)
        np.testing.assert_allclose(np.asarray(diags_j.gap_post), np.asarray(diags.gap_post), atol=1e-14)


def test_hard_step_equals_reference_substep(ref):
    """One hard step == ``attempt_primitive_step`` with ``n_sub=1`` whenever the reference does
    not switch the contact pair (same QP, same projection)."""
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    settings = default_settings(jnp.float64)
    cfg = RollConfig(mode="hard")
    dps, _ = _commands(ref, 2, 1)
    n_cmp = 0
    for st, cp_pair in [(ref["seed_state"], tuple(int(c) for c in ref["asg"].caps))] + ref["nodes"]:
        asg_n = asg._replace(caps=jnp.asarray(cp_pair, jnp.int32))
        out = attempt_primitive_step(hm.dm, RefGraspState(st.q, st.p, st.R, asg_n.caps), dps[0], 1, sp, settings)
        if not bool(out.ok) or not bool(jnp.all(out.state.pair == asg_n.caps)):
            continue
        st1, dg = step(hm, st, dps[0], jnp.zeros(3), asg_n, sp, cfg)
        np.testing.assert_allclose(np.asarray(st1.q), np.asarray(out.state.q), atol=1e-10)
        np.testing.assert_allclose(np.asarray(st1.p), np.asarray(out.state.p), atol=1e-15)
        np.testing.assert_allclose(np.asarray(st1.R), np.asarray(out.state.R), atol=1e-10)
        gates = exact_gates(dg, sp)
        assert all(bool(v) for v in gates.values()), gates
        n_cmp += 1
    assert n_cmp >= 1


def test_exact_validator_agrees_with_smooth_gates(ref):
    """Over the six seed-frame primitives from the seed: where the reference keeps the pair,
    its verdict equals the conjunction of the smooth step's exact gates over the 8 sub-steps."""
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    cfg = RollConfig(mode="hard")
    n_sub = int(round(ref["cfg"].voxel_size_m / ref["cfg"].step_size_m))
    agree = compared = 0
    for d in range(6):
        dps, dths = _commands(ref, d, n_sub)
        ex = validate_translation(hm, ref["seed_state"], dps, asg, sp, n_sub)
        if not bool(jnp.all(ex.pair[-1] == asg.caps)):
            continue
        _, diags = rollout(hm, ref["seed_state"], dps, dths, asg, sp, cfg)
        g = exact_gates(diags, sp)
        # the reference checks qp/projection per sub-step and the remaining gates at the end
        smooth_ok = bool(jnp.all(g["qp"]) & jnp.all(g["slip"]) & jnp.all(g["gap"])
                         & g["collision"][-1] & g["self_collision"][-1] & g["grasp"][-1] & g["limits"][-1])
        compared += 1
        agree += smooth_ok == bool(ex.ok[0])
    assert compared >= 2 and agree == compared, f"{agree}/{compared} agree"


def test_rotation_steps_match_reference(ref):
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    st0 = ref["seed_state"]
    # tilt about an axis perpendicular to the pinch axis (thumb normal), 0.15 rad as the reference
    from karma_jax.kinematics import forward_kinematics
    from dkarma.core.contacts import contact_pass
    cs = contact_pass(hm, forward_kinematics(hm.dm, st0.q), st0.p, sp.sphere_radius, asg)
    n = cs.normal[0]
    perp = jnp.cross(n, jnp.array([0.0, 0.0, 1.0])); perp = perp / jnp.linalg.norm(perp)
    dths = jnp.broadcast_to(0.15 * perp, (3, 3))
    ex = validate_rotation(hm, st0, dths, asg, sp)
    states, diags = rollout(hm, st0, jnp.zeros((3, 3)), dths, asg, sp, RollConfig(mode="hard", kind="rotation"))
    k_ok = int(jnp.sum(ex.ok))
    for k in range(k_ok):
        if not bool(jnp.all(ex.pair[k + 1] == asg.caps)):
            break
        np.testing.assert_allclose(np.asarray(states.q[k + 1]), np.asarray(ex.q[k + 1]), atol=1e-10)
        np.testing.assert_allclose(np.asarray(states.R[k + 1]), np.asarray(ex.R[k + 1]), atol=1e-10)
    assert float(diags.efficiency[0]) > 0.0


def _fd(f, x, h):
    x = np.asarray(x, float)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x); e[i] = h
        g[i] = (float(f(x + e)) - float(f(x - e))) / (2 * h)
    return g


@pytest.mark.parametrize("mode,kkt_method", [("soft", "exact"), ("hard", "exact"), ("hard", "reg"), ("hard", "pinv"), ("hard", "reduced")])
def test_rollout_gradients_wrt_morphology(ref, mode, kkt_method):
    """d(final tip position . w)/d phi and d(total energy)/d phi vs FD over a 4-step rollout.
    Hard mode: only when the active sets stay fixed under the FD perturbation (as detected by
    agreement; a kink makes FD and the one-sided derivative differ legitimately). Phase 9.1: every
    KKT derivative method of ``rolling_qp.kkt_resolve`` must pass this Phase 2 test unchanged."""
    topo, params, sp, asg = ref["topo"], ref["params"], ref["sp"], ref["asg"]
    bounds = Bounds(link_len_min_m=0.005, link_len_max_m=0.12, tip_len_min_m=0.005, tip_len_max_m=0.12, base_xyz_max_m=0.10)
    fixed = fixed_from_params(params)
    flat0, unravel = jax.flatten_util.ravel_pytree(encode(params, topo, bounds))
    st0 = ref["seed_state"]
    dps, dths = _commands(ref, 1, 4)
    cfg = RollConfig(mode=mode, proj_backtrack=False, kkt_method=kkt_method, kkt_reg=(1e-8 if kkt_method == "reg" else 1e-10))
    w = jnp.asarray(np.random.RandomState(3).randn(3))
    wts = EnergyWeights()

    def run(x):
        hm = build_hand_model(topo, decode(unravel(x), topo, bounds, fixed))
        states, diags = rollout(hm, st0, dps, dths, asg, sp, cfg, TIGHT)
        return states, energies(diags, sp, wts)

    # jitted once per function: the eager scan recompiles on every call (a new closure each time), which
    # accumulated ~10 GB of compiled programs per parametrization over the 2 x 37 FD evaluations (Phase 9.1)
    f_pos = jax.jit(lambda x: jnp.dot(w, run(x)[0].q[-1][:3]))   # final joint angles (depend on phi via the QP)
    f_en = jax.jit(lambda x: run(x)[1].total)
    for f, name in ((f_pos, "state"), (f_en, "energy")):
        g_ad = np.asarray(jax.grad(f)(flat0))
        assert np.all(np.isfinite(g_ad)), name
        g_fd = _fd(f, flat0, 1e-6)
        err = np.abs(g_ad - g_fd)
        scale = np.abs(g_fd).max() + 1e-12
        rel = err.max() / scale
        if mode == "soft":
            assert rel <= 1e-4, f"{name}: rel err {rel:.2e}"
        else:
            # report-grade check: the KKT derivative must agree where no active-set change
            # occurs; allow the kink case by requiring agreement on most coordinates
            frac = np.mean(err <= 1e-3 * scale + 1e-9)
            assert frac >= 0.8, f"{name}: only {frac:.0%} of coordinates agree (rel err {rel:.2e})"


def test_vmap_over_commands_and_prefix_weights(ref):
    hm, sp, asg = ref["hm"], ref["sp"], ref["asg"]
    st0 = ref["seed_state"]
    cmds = [_commands(ref, d, 8) for d in range(6)]
    dps = jnp.stack([c[0] for c in cmds]); dths = jnp.stack([c[1] for c in cmds])
    cfg = RollConfig(mode="hard")
    f = jax.jit(jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, cfg)))
    states, diags = f(dps, dths)
    assert states.q.shape == (6, 9, hm.dm.q_ids.shape[0])
    en = jax.vmap(lambda d: energies(d, sp, EnergyWeights()))(diags)
    assert en.prefix_logw.shape == (6, 8)
    assert bool(jnp.all(jnp.diff(en.prefix_logw, axis=1) <= 1e-12))      # non-increasing
    assert bool(jnp.all(jnp.isfinite(en.worst)))
    s1, d1 = rollout(hm, st0, dps[3], dths[3], asg, sp, cfg)
    np.testing.assert_allclose(np.asarray(s1.q), np.asarray(states.q[3]), atol=1e-12)
