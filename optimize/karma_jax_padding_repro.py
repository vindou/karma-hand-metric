"""Minimal reproductions of two karma_jax padding defects (found 2026-10-10 on the
two-finger designs of the optimization exercise; FIXED 2026-10-11 in Phase 0, see
docs/phase0_report.md and the regression tests in karma_jax/tests/test_padding.py).
Kept as a standalone check; on fixed code both sections print zero error.

    JAX_PLATFORMS=cpu python optimize/karma_jax_padding_repro.py

1. Joint table padded to exactly n_joints (the load_flat_model default) gave WRONG
   forward kinematics under jit when the DeviceModel is closed over (as every kernel
   in the package does), for an 8-joint two-root hand. Root cause: XLA rewrites a
   set-scatter with constant indices and as many updates as the operand has elements
   into a gather, so apply_mimic's eight padded mimic lanes (all indexing joint 0)
   overwrote every joint with q[0] whenever K_pad == J_pad. apply_mimic is now a
   gather over per-joint tables.
2. Active-joint table padded beyond n_active (M_pad > n_active) perturbed the
   Gauss-Newton projection: kinematics.set_active scattered the padded lanes (q_ids = 0)
   into joint 0 alongside the real lane-0 update, and XLA's duplicate-index scatter
   does not guarantee which write wins. set_active is now a gather as well, and the
   rolling QPs mask padded lanes out of the interior-point scalars.
"""
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import jax, jax.numpy as jnp
from karma_jax.model import load_flat_model
from karma_jax import kinematics as K, projection as PJ
from karma.robot import update_kinematics
from karma.projection import _project_onto_contacts
from karma.robot import active_position_indices, active_velocity_indices
from optimize.hand_family import HandFamily

fam = HandFamily(); hand = fam.decode(fam.anthropomorphic_x0())
tmp = Path(tempfile.mkdtemp()); _, yml = hand.write(tmp)
METRIC = REPO / "karma_config.yaml"

print("1) forward kinematics, jit with closed-over model, q = neutral except joint 0 = -0.3")
for J in [8, 9, 16]:
    flat, cfg, ctx = load_flat_model(METRIC, yml, J_pad=J)
    dm = flat.to_device(jnp.float64)
    q = np.zeros(J); q[0] = -0.3
    update_kinematics(ctx, q[:8].copy())
    ref = np.stack([np.array(ctx.data.oMi[j].translation) for j in range(1, ctx.model.njoints)])
    ours = np.asarray(jax.jit(lambda q1: K.forward_kinematics(dm, q1).o_joint)(jnp.asarray(q)))[:8]
    ours_nojit = np.asarray(K.forward_kinematics(dm, jnp.asarray(q)).o_joint)[:8]
    print(f"   J_pad={J:2d}: max |joint origin error| jit={np.abs(ours - ref).max():.2e}  no-jit={np.abs(ours_nojit - ref).max():.2e}")

print("2) projection onto the two-contact manifold vs karma.projection._project_onto_contacts")
flat0, cfg, ctx = load_flat_model(METRIC, yml, J_pad=9)
q_ids = active_position_indices(ctx.model, cfg.finger_joint_names); v_ids = active_velocity_indices(ctx.model, cfg.finger_joint_names)
rng = np.random.RandomState(0)
lo = np.array(ctx.model.lowerPositionLimit); hi = np.array(ctx.model.upperPositionLimit)
cases = []
for _ in range(200):
    q = lo + rng.rand(8) * (hi - lo)
    update_kinematics(ctx, q.copy())
    from karma.robot import capsule_endpoints_world
    a, b = capsule_endpoints_world(ctx, cfg.seed_thumb_link); c, d = capsule_endpoints_world(ctx, cfg.seed_index_link)
    p = 0.5 * (b + d) + rng.randn(3) * 1e-3
    cases.append((q.copy(), p))
for M in [8, 10]:
    flat, cfg, ctx = load_flat_model(METRIC, yml, J_pad=9, M_pad=M)
    dm = flat.to_device(jnp.float64); names = flat.capsule_names
    pair = jnp.asarray([names.index(cfg.seed_thumb_link), names.index(cfg.seed_index_link)])
    f = jax.jit(lambda q, p: PJ.project_onto_contacts(dm, q, p, pair, cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_g_m))
    worst = 0.0; mism = 0
    for q, p in cases:
        qr, _, okr, _, _ = _project_onto_contacts(ctx, q.copy(), p, (cfg.seed_thumb_link, cfg.seed_index_link), q_ids, v_ids,
                                                  cfg.sphere_radius_m, cfg.link_radius_m, cfg.eps_g_m)
        qq = np.zeros(9); qq[:8] = q
        out = f(jnp.asarray(qq), jnp.asarray(p))
        if bool(out.ok) != bool(okr): mism += 1
        elif okr: worst = max(worst, np.abs(np.asarray(out.q)[:8] - qr).max())
    print(f"   M_pad={M:2d} (n_active=8): ok-verdict mismatches {mism}/200, max |q - q_ref| on agreed-ok cases {worst:.1e}")
