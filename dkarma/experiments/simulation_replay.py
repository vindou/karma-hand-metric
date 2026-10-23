"""Phase 8.6 (SPEC.md §11.1): independent physical-task performance in simulation (MuJoCo).

Two stages, so the trajectory export is testable without the simulator:

``export``: for each design, every command of the exact trajectory library is replayed by the
hard-mode fixed-assignment rollout at sub-step resolution (8 sub-steps per 10 mm cell), giving
the joint path, the commanded sphere path and the exact verdict per primitive (the reference
kernels for two contacts, the fixed-assignment exact path for three). The number of exactly
valid primitives ``n_ok`` labels each trajectory: the prefix up to ``n_ok`` is "exactly valid",
the following primitive is "exactly rejected" (with the gate that rejected it).

``simulate`` (requires ``mujoco`` in the ``karma-jax`` environment; the owner confirms the install
first): the scene is built from the design parameters (capsule collision geometry of every
link, the 15 mm / 35 g sphere, Coulomb friction 0.6 with torsional coefficient ``mu * a`` for the
3 mm patch (0.0018 m) and a small rolling coefficient, gravity along -z of the palm frame,
position actuators with the XL330 ceiling as ``forcerange``). The controller tracks the joint
path with the position servos; per primitive the test is: the sphere stays in contact with the
assigned fingertips (no drop), slips less than the stated bound relative to the fingertips, and
its centre reaches the commanded position within the stated tolerance. Per design the report
gives the confusion matrix exact-valid vs simulation-success per primitive, and the
disagreements by gate.

    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay export --out workspace/phase8/8.6
    ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.simulation_replay simulate --out workspace/phase8/8.6
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import build_hand_model, load_task_spec, Topology, anthropomorphic, params_to_fingers  # noqa: E402
from dkarma.core.contacts import contact_pass  # noqa: E402
from dkarma.solvers.seed_projection import find_seeds, find_multi_seeds, SeedConfig, assignment  # noqa: E402
from dkarma.solvers.rollout import GraspState, rollout, exact_gates  # noqa: E402
from dkarma.solvers.exact import validate_translation, validate_rotation, fail_names  # noqa: E402
from dkarma.metrics.config import DIFF_KARMA, FULL_KARMA  # noqa: E402
from dkarma.metrics.commands import build_library, CommandConfig  # noqa: E402
from dkarma.metrics import act_karma as AK, multicontact as MC  # noqa: E402
from dkarma.export import design_from_dict  # noqa: E402
from dkarma.tests.conftest import load_design  # noqa: E402
from karma_jax.kinematics import forward_kinematics  # noqa: E402

def _print(*a):
    print(*a, flush=True)


ARM_A = "workspace/phase7/exp34/armA_design/armA.design.json"
ARM_B = "workspace/phase7/exp34/run_full/optimized_morphology/optimized_morphology.design.json"


DESIGN_DIR = None     # set by ``--design-dir``: <dir>/<name>.design.json for names outside the built-in list (Phase 9 candidates)


def load_designs(names):
    out = {}
    for n in names:
        if n not in ("baseline", "start", "armA", "armB"):
            p = Path(n) if n.endswith(".json") else Path(DESIGN_DIR or ".") / f"{n}.design.json"
            t, par, _ = design_from_dict(json.loads(p.read_text())); out[n] = (t, par); continue
        if n == "baseline":
            t, p, _ = load_design("baseline"); out[n] = (t, p)
        elif n == "start":
            t = Topology.three_finger(4); out[n] = (t, anthropomorphic(t))
        elif n == "armA":
            t, p, _ = design_from_dict(json.loads(Path(ARM_A).read_text())); out[n] = (t, p)
        elif n == "armB":
            t, p, _ = design_from_dict(json.loads(Path(ARM_B).read_text())); out[n] = (t, p)
    return out


# ── export ───────────────────────────────────────────────────────────────────

def squeeze_offsets(hm, q_path, p_path, asg, sp, delta_m=1e-3):
    """Joint-space offsets ``dq [H+1, J]`` that move every assigned fingertip ``delta_m`` into the
    sphere along its contact normal (least squares on the stacked surface Jacobians): the geometric
    squeeze the position servos are commanded with in the replay."""
    H1 = q_path.shape[0]
    out = np.zeros((H1, q_path.shape[1]))
    valid = np.asarray(asg.valid)
    for h in range(H1):
        cs = contact_pass(hm, forward_kinematics(hm.dm, jnp.asarray(q_path[h])), jnp.asarray(p_path[h]), sp.sphere_radius, asg)
        n = np.asarray(cs.normal)[valid]                       # [Cn, 3] outward from the capsule toward the sphere? (gap gradient)
        J = np.asarray(cs.J_surf)[valid]                       # [Cn, 3, J] surface-point Jacobian
        target = (delta_m * n).reshape(-1)                     # move the surface point along the normal toward the sphere centre
        out[h] = np.linalg.lstsq(J.reshape(-1, J.shape[-1]), target, rcond=None)[0]
    return out


def exact_force_path(hm, q_path, p_path, asg, sp, act, return_fn: bool = False):
    """Exact force allocation (SLSQP, the primary torque ceiling) at every state of a path:
    ``(feasible [H+1] bool, tau [H+1, J] N·m)`` and, with ``return_fn``, the allocation's contact
    normal forces ``fn [H+1, Cn]`` (N, the set-points of the Phase 9.3 force controller); where the
    allocation is infeasible the last feasible values are held (the controller still needs a command)."""
    from dkarma.solvers.contact_forces import solve_exact
    H1 = q_path.shape[0]
    Cn = int(asg.valid.shape[0])
    ok = np.zeros(H1, bool); tau = np.zeros((H1, q_path.shape[1])); fn = np.zeros((H1, Cn))
    last = np.zeros(q_path.shape[1]); last_fn = np.zeros(Cn)
    valid = np.asarray(asg.valid)
    for h in range(H1):
        pb = AK.force_problem_at(hm, GraspState(jnp.asarray(q_path[h]), jnp.asarray(p_path[h]), jnp.eye(3)), asg, sp, act)
        ex = solve_exact(pb, act.w_ext)
        ok[h] = bool(ex["feasible"])
        if ok[h]:
            last = np.asarray(ex["tau"], float)
            F = np.asarray(ex["x"], float)[: 3 * Cn].reshape(Cn, 3)
            last_fn = np.sum(F * np.asarray(pb.normals), axis=1) * valid
        tau[h] = last; fn[h] = last_fn
    return (ok, tau, fn) if return_fn else (ok, tau)


def lane_offsets(hm, q_path, p_path, asg, sp):
    """Per-lane joint-space directions ``dq [H+1, Cn, J]`` that move lane ``c``'s surface point one
    metre into the sphere along its contact normal (least squares on that lane's own surface
    Jacobian; lanes of different fingers share no joints). The Phase 9.3 force controller commands
    ``q_path + sum_c delta_c dq[:, c]`` with ``delta_c`` (m) regulated on the measured normal force."""
    H1, J = q_path.shape
    Cn = int(asg.valid.shape[0])
    out = np.zeros((H1, Cn, J))
    valid = np.asarray(asg.valid)
    for h in range(H1):
        cs = contact_pass(hm, forward_kinematics(hm.dm, jnp.asarray(q_path[h])), jnp.asarray(p_path[h]), sp.sphere_radius, asg)
        n = np.asarray(cs.normal); Js = np.asarray(cs.J_surf)
        for c in range(Cn):
            if valid[c]:
                out[h, c] = np.linalg.lstsq(Js[c], n[c], rcond=None)[0]
    return out


def export_two_finger(hm, topo, seeds, sp, task, cfg, act):
    """Every translation and rotation command of every seed: hard-mode joint path at sub-step
    resolution, the reference kernels' verdict per primitive, and the exact allocation's joint
    torques at every state (the servo offsets of the replay)."""
    asg = assignment(topo)
    n_sub = cfg.commands.n_sub
    roll_T = cfg.roll._replace(kind="translation", mode="hard")
    roll_R = cfg.roll._replace(kind="rotation", mode="hard")
    trajs = []
    for s in range(int(np.asarray(seeds.valid).sum())):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        n0 = contact_pass(hm, forward_kinematics(hm.dm, q), p, sp.sphere_radius, asg).normal[0]
        lib = build_library(frame, -n0, cfg.commands)
        st0 = GraspState(q, p, jnp.eye(3))
        sT, dT = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_T))(lib.dp_T, lib.dth_T)
        sR, dR = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_R))(lib.dp_R, lib.dth_R)
        exT = jax.vmap(lambda a: validate_translation(hm, st0, a, asg, sp, n_sub))(lib.dp_T)
        exR = jax.vmap(lambda a: validate_rotation(hm, st0, a, asg, sp))(lib.dth_R)
        okT, okR = np.asarray(exT.ok), np.asarray(exR.ok)
        cT, cR = np.asarray(exT.code), np.asarray(exR.code)
        for u in range(okT.shape[0]):
            n_ok = int(np.argmin(okT[u])) if not okT[u].all() else int(okT.shape[1])
            fail = fail_names("translation")[int(cT[u, n_ok])] if n_ok < okT.shape[1] else None
            q_u, p_u = np.asarray(sT.q[u]), np.asarray(sT.p[u])
            fok, tau, fn = exact_force_path(hm, q_u, p_u, asg, sp, act, return_fn=True)
            trajs.append(dict(seed=s, kind="T", command=u, n_primitives=int(okT.shape[1]), n_sub=n_sub, n_ok=n_ok, reject_gate=fail,
                              q=q_u, p=p_u, R=np.asarray(sT.R[u]), q_exact=np.asarray(exT.q[u]), tau=tau, force_ok=fok, fn_exact=fn,
                              dq_squeeze=squeeze_offsets(hm, q_u, p_u, asg, sp), dq_lane=lane_offsets(hm, q_u, p_u, asg, sp), contact_caps=np.asarray(asg.caps[:2])))
        for u in range(okR.shape[0]):
            n_ok = int(np.argmin(okR[u])) if not okR[u].all() else int(okR.shape[1])
            fail = fail_names("rotation")[int(cR[u, n_ok])] if n_ok < okR.shape[1] else None
            q_u, p_u = np.asarray(sR.q[u]), np.asarray(sR.p[u])
            fok, tau, fn = exact_force_path(hm, q_u, p_u, asg, sp, act, return_fn=True)
            trajs.append(dict(seed=s, kind="R", command=u, n_primitives=int(okR.shape[1]), n_sub=1, n_ok=n_ok, reject_gate=fail,
                              q=q_u, p=p_u, R=np.asarray(sR.R[u]), q_exact=np.asarray(exR.q[u]), tau=tau, force_ok=fok, fn_exact=fn,
                              dq_squeeze=squeeze_offsets(hm, q_u, p_u, asg, sp), dq_lane=lane_offsets(hm, q_u, p_u, asg, sp), contact_caps=np.asarray(asg.caps[:2])))
    return trajs


def export_three_contact(hm, topo, fingers, seeds, sp, task, act, cfg):
    """The fixed-assignment exact path (Full KaRMA: torque gates on) at sub-step resolution."""
    asg = assignment(topo, 3, fingers)
    n_sub = cfg.commands.n_sub
    trajs = []
    names = ("qp", "slip", "gap", "collision", "self_collision", "grasp", "limits", "force")
    for s in range(int(np.asarray(seeds.valid).sum())):
        q, p, frame = seeds.q[s], seeds.p[s], seeds.frame[s]
        st0 = GraspState(q, p, jnp.eye(3))
        dp_T, dth_T, dp_R, dth_R = MC.multi_library(frame, cfg.commands)
        roll_T = cfg.roll._replace(kind="translation", mode="hard", grasp_mode="equilibrium")
        roll_R = cfg.roll._replace(kind="rotation", mode="hard", grasp_mode="equilibrium")
        sT, dT = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_T, None, act))(dp_T, dth_T)
        sR, dR = jax.vmap(lambda a, b: rollout(hm, st0, a, b, asg, sp, roll_R, None, act))(dp_R, dth_R)
        gT = jax.vmap(lambda d: exact_gates(d, sp))(dT); gR = jax.vmap(lambda d: exact_gates(d, sp))(dR)
        seed_ok = MC._exact_equilibrium(hm, q, p, asg, sp, act, True)
        kin_names = ("qp", "slip", "gap", "collision", "self_collision", "limits")   # as the exact Full path: the smooth
        # "grasp" margin is not a gate; the exact SLSQP equilibrium at every sub-step is (force_ok)
        for kind, states, gates, H, sub in (("T", sT, gT, dp_T.shape[1], n_sub), ("R", sR, gR, dth_R.shape[1], 1)):
            K = H // sub
            for u in range(6):
                q_u, p_u = np.asarray(states.q[u]), np.asarray(states.p[u])
                fok, tau, fn = exact_force_path(hm, q_u, p_u, asg, sp, act, return_fn=True)   # exact force check of every sub-step
                n_ok, gate = 0, None
                if not seed_ok:
                    gate = "seed_force"
                else:
                    for k in range(K):
                        sl = slice(k * sub, (k + 1) * sub)
                        bad = [n for n in kin_names if not np.all(np.asarray(gates[n])[u, sl])]
                        if bad:
                            gate = bad[0]; break
                        if not np.all(fok[k * sub + 1:(k + 1) * sub + 1]):
                            gate = "force"; break
                        n_ok += 1
                trajs.append(dict(seed=s, kind=kind, command=u, n_primitives=K, n_sub=sub, n_ok=n_ok, reject_gate=gate,
                                  q=q_u, p=p_u, R=np.asarray(states.R[u]), q_exact=None, tau=tau, force_ok=fok, fn_exact=fn,
                                  dq_squeeze=squeeze_offsets(hm, q_u, p_u, asg, sp), dq_lane=lane_offsets(hm, q_u, p_u, asg, sp), contact_caps=np.asarray(asg.caps[:3])))
    return trajs


def export_all(out: Path, names, budget=6, log=_print, libraries=("two_finger", "three_contact"), seed_config: str = "default"):
    """``seed_config``: a seed-search configuration of ``admissible_baseline.SEED_CONFIGS`` (Phase 9.4); the
    default is the Phase 7/8 recipe used by every 8.6/9.x export."""
    from dkarma.experiments.admissible_baseline import SEED_CONFIGS
    ckw = SEED_CONFIGS[seed_config]
    task = load_task_spec(); sp = task.step_params()
    cfg2 = replace(DIFF_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10))
    cfg3 = replace(FULL_KARMA, commands=CommandConfig(n_cells=4, n_rot_steps=10))
    meta = {}
    for name, (topo, params) in load_designs(names).items():
        hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
        rec = dict(fingers=list(topo.fingers), n_joints=list(topo.n_joints), pair=list(topo.pair), libraries={}, seed_config=seed_config, budget=budget)
        trajs = []
        existing = load_trajectories(out, name) if (out / f"{name}_trajectories.npz").exists() else []
        if "two_finger" in libraries:
            s2 = find_seeds(hm, topo, sp, SeedConfig(budget=budget, **ckw))
            t2 = export_two_finger(hm, topo, s2, sp, task, cfg2, act) if int(s2.valid.sum()) else []
            rec["libraries"]["two_finger"] = dict(n_seeds=int(s2.valid.sum()), n_trajectories=len(t2), n_valid_primitives=sum(t["n_ok"] for t in t2))
            trajs += [dict(t, library="two_finger") for t in t2]
        else:
            trajs += [t for t in existing if t["library"] == "two_finger"]
        if topo.n_fingers >= 3 and "three_contact" in libraries:
            fingers = (topo.thumb_finger, topo.index_finger, topo.finger_index("middle"))
            s3 = find_multi_seeds(hm, topo, fingers, sp, act, SeedConfig(budget=budget, **ckw))
            t3 = export_three_contact(hm, topo, fingers, s3, sp, task, act, cfg3) if int(s3.valid.sum()) else []
            rec["libraries"]["three_contact"] = dict(n_seeds=int(s3.valid.sum()), n_trajectories=len(t3), n_valid_primitives=sum(t["n_ok"] for t in t3))
            trajs += [dict(t, library="three_contact") for t in t3]
        elif topo.n_fingers >= 3:
            trajs += [t for t in existing if t["library"] == "three_contact"]
        np.savez_compressed(out / f"{name}_trajectories.npz",
                            **{f"{i}_{k}": v for i, t in enumerate(trajs) for k, v in t.items() if isinstance(v, np.ndarray)},
                            index=json.dumps([{k: v for k, v in t.items() if not isinstance(v, np.ndarray) and v is not None} for t in trajs]))
        (out / f"{name}_design.json").write_text(json.dumps(dict(fingers=params_to_fingers(topo, params), cap_radius=np.asarray(params.cap_radius).tolist(),
                                                               topology=dict(fingers=list(topo.fingers), n_joints=list(topo.n_joints), pair=list(topo.pair))), indent=1, default=float))
        meta[name] = rec
        log(f"[{name}] {rec}")
    (out / "export_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def save_trajectories(out: Path, name, trajs):
    np.savez_compressed(out / f"{name}_trajectories.npz",
                        **{f"{i}_{k}": v for i, t in enumerate(trajs) for k, v in t.items() if isinstance(v, np.ndarray)},
                        index=json.dumps([{k: v for k, v in t.items() if not isinstance(v, np.ndarray) and v is not None} for t in trajs]))


def augment_archive(src: Path, out: Path, name, log=_print):
    """Phase 9.3: copy an 8.6 trajectory archive and add, per trajectory, the exact allocation's contact
    normal forces (``fn_exact``) and the per-lane squeeze directions (``dq_lane``), so the force-controlled
    replay runs on exactly the 8.6 trajectories (same seeds, joint paths and exact verdicts)."""
    import shutil
    task = load_task_spec(); sp = task.step_params()
    (topo, params), = load_designs([name]).values()
    hm = build_hand_model(topo, params); act = AK.actuation_for(hm, task)
    trajs = load_trajectories(src, name)
    for i, t in enumerate(trajs):
        n_lanes = len(t["contact_caps"])
        asg = assignment(topo) if n_lanes == 2 else assignment(topo, 3, (topo.thumb_finger, topo.index_finger, topo.finger_index("middle")))
        assert [int(c) for c in np.asarray(asg.caps[:n_lanes])] == [int(c) for c in t["contact_caps"]], "assignment differs from the archive"
        ok, tau, fn = exact_force_path(hm, t["q"], t["p"], asg, sp, act, return_fn=True)
        if t.get("force_ok") is not None and not np.array_equal(ok, np.asarray(t["force_ok"], bool)):
            log(f"  [{name}] trajectory {i}: exact force verdicts differ from the archive in {int(np.sum(ok != np.asarray(t['force_ok'], bool)))} states")
        t["fn_exact"] = fn; t["dq_lane"] = lane_offsets(hm, t["q"], t["p"], asg, sp)
        if i % 20 == 0:
            log(f"  [{name}] {i}/{len(trajs)}")
    out.mkdir(parents=True, exist_ok=True)
    save_trajectories(out, name, trajs)
    shutil.copy(src / f"{name}_design.json", out / f"{name}_design.json")
    log(f"[{name}] augmented {len(trajs)} trajectories from {src} -> {out}")


def load_trajectories(out: Path, name):
    z = np.load(out / f"{name}_trajectories.npz")
    index = json.loads(str(z["index"]))
    for i, t in enumerate(index):
        for k in ("q", "p", "R", "q_exact", "contact_caps", "tau", "force_ok", "dq_squeeze", "fn_exact", "dq_lane"):
            t[k] = z[f"{i}_{k}"] if f"{i}_{k}" in z.files else None
    return index


# ── MuJoCo scene ─────────────────────────────────────────────────────────────

def mjcf_from_design(design: dict, task, tau_bar: float, kp: float = 20.0, timestep: float = 5e-4, pad=None,
                     pad_variant: str = "patch") -> str:
    """MJCF of the design: palm fixed at the origin, hinge joints with the design's axes, capsule
    collision geometry of every link (radius ``cap_radius``), position actuators with
    ``forcerange = ±tau_bar``, the free sphere, gravity -z.

    ``pad`` (a ``compliant_pad.PadParams``, Phase 10.1): compliant silicone fingertips. Every terminal capsule
    keeps its radius; its contact with the sphere is an explicit ``<pair>`` with the pad's law (the dynamic
    contact of that geom pair is then not generated). ``pad_variant="rigid_friction"`` (diagnostic D1) keeps
    the pad's normal law with the rigid scene's friction (``condim 4``, torsional ``mu * 3 mm``). Without
    ``pad`` the scene is the 8.6/9.3 one, string for string. ``pad_variant="patch_friction_only"`` (post-hoc diagnostic
    D3) keeps the rigid normal law with the pad's patch friction."""
    xml = _mjcf_rigid(design, task, tau_bar, kp, timestep)
    if pad is None:
        return xml
    import mujoco
    m = mujoco.MjModel.from_xml_string(xml)
    sph = m.body("sphere").id
    mu = task.obj.friction_coefficient
    pairs = []
    for fname, f in design["fingers"].items():
        k = len(f["axis_theta"]) - 1
        A = float(m.body_invweight0[sph][0] + m.body_invweight0[m.body(f"{fname}_l{k}").id][0])   # the pair's diagApprox
        attrs = pad.pair_attrs(A)
        if pad_variant == "rigid_friction":
            attrs = attrs.replace('condim="6"', 'condim="4"')
            attrs = attrs[:attrs.index('friction="')] + f'friction="{mu} {mu} {mu * 0.003} 0.0001 0.0001" ' + attrs[attrs.index('solref='):]
        elif pad_variant == "patch_friction_only":
            # post-hoc diagnostic D3 (Phase 10.1): the rigid scene's normal law and friction regularization with the patch's
            # friction (condim 6, torsional and rolling coefficients of the pad)
            attrs = attrs[:attrs.index('solref=')] + 'solref="0.002 1" solimp="0.95 0.99 0.001"'
        elif pad_variant != "patch":
            raise ValueError(f"unknown pad variant {pad_variant}")
        pairs.append(f'    <pair name="pad_{fname}" geom1="{fname}_c{k}" geom2="sphere_geom" {attrs}/>')
    return xml.replace("  </contact>", "\n".join(pairs) + "\n  </contact>")


def _mjcf_rigid(design: dict, task, tau_bar: float, kp: float = 20.0, timestep: float = 5e-4) -> str:
    """The 8.6/9.3 scene (rigid capsules)."""
    from scipy.spatial.transform import Rotation
    fingers = design["fingers"]; r_l = design["cap_radius"][0]
    r_s, m_s, mu = task.obj.radius_m, task.obj.mass_kg, task.obj.friction_coefficient
    gamma = mu * 0.003
    fr = f"{mu} {gamma} 0.0001"
    lines = ['<mujoco model="dkarma_hand">', '  <compiler angle="radian"/>',          # joint ranges and axes in radians (MuJoCo defaults to degrees)
             f'  <option timestep="{timestep}" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic"/>',
             '  <default><geom condim="4" solref="0.002 1" solimp="0.95 0.99 0.001"/><joint damping="0.01" armature="1e-5"/></default>',
             '  <worldbody>', '    <body name="palm" pos="0 0 0">']
    actuators = []
    for fname, f in fingers.items():
        n = len(f["axis_theta"])
        quat0 = Rotation.from_rotvec(f["base_rotvec"]).as_quat()  # x y z w
        q0 = f"{quat0[3]} {quat0[0]} {quat0[1]} {quat0[2]}"
        indent = "      "
        open_bodies = 0
        for k in range(n):
            pos = f["base_xyz"] if k == 0 else [f["link_len"][k - 1], 0.0, 0.0]
            quat = q0 if k == 0 else "1 0 0 0"
            th, ph = f["axis_theta"][k], f["axis_phi"][k]
            ax = [np.cos(th), np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph)]
            L = f["link_len"][k] if k < n - 1 else max(f["link_len"][k] - r_l, 0.0)
            lines.append(f'{indent}<body name="{fname}_l{k}" pos="{pos[0]} {pos[1]} {pos[2]}" quat="{quat}">')
            lines.append(f'{indent}  <joint name="{fname}_j{k}" type="hinge" axis="{ax[0]} {ax[1]} {ax[2]}" range="{-f["lim_lo"][k]} {f["lim_hi"][k]}" limited="true"/>')
            mass = 0.028 if k < n - 1 else 0.010
            lines.append(f'{indent}  <geom name="{fname}_c{k}" type="capsule" fromto="0 0 0 {L} 0 0" size="{r_l}" mass="{mass}" friction="{fr}"/>')
            actuators.append(f'    <position name="{fname}_a{k}" joint="{fname}_j{k}" kp="{kp}" forcerange="{-tau_bar} {tau_bar}"/>')
            indent += "  "; open_bodies += 1
        for _ in range(open_bodies):
            indent = indent[:-2]; lines.append(f"{indent}</body>")
    # dkarma models self-collision between fingers only: exclude same-finger capsule pairs
    excl = []
    for fname, f in fingers.items():
        n = len(f["axis_theta"])
        for a in range(n):
            for b in range(a + 1, n):
                excl.append(f'    <exclude body1="{fname}_l{a}" body2="{fname}_l{b}"/>')
    lines += ['    </body>',
              f'    <body name="sphere" pos="0 0 0.1"><freejoint name="sphere_free"/><geom name="sphere_geom" type="sphere" size="{r_s}" mass="{m_s}" friction="{fr}"/></body>',
              '  </worldbody>', '  <contact>'] + excl + ['  </contact>', '  <actuator>'] + actuators + ['  </actuator>', '</mujoco>']
    return "\n".join(lines)


def _normal_forces(model, data, gids, sphere_gid, with_touch: bool = False):
    """Measured normal force (N) between the sphere and each assigned fingertip capsule; with ``with_touch``
    also whether any contact between them exists (the retention criterion's notion of contact)."""
    import mujoco
    f6 = np.zeros(6)
    out = np.zeros(len(gids))
    touch = np.zeros(len(gids), bool)
    for c in range(data.ncon):
        con = data.contact[c]
        for k, g in enumerate(gids):
            if (con.geom1 == g and con.geom2 == sphere_gid) or (con.geom2 == g and con.geom1 == sphere_gid):
                mujoco.mj_contactForce(model, data, c, f6)
                out[k] += abs(f6[0])
                touch[k] = True
    return (out, touch) if with_touch else out


def _admittance(delta, f_star, f_meas, dt, controller, k_i, leak, d_min, d_max):
    """One admittance update of the per-lane squeeze (m); returns ``(delta, err)`` with ``err`` the part of
    the force error the proportional term may act on."""
    err = f_star - f_meas
    if controller == "force_sym":
        return np.clip(delta + k_i * err * dt, -0.5e-3, d_max), err
    up = np.maximum(err, 0.0); down = np.maximum(-err, 0.0)
    return np.clip(delta + k_i * up * dt - leak * k_i * down * dt, d_min, d_max), up


def simulate_design(out: Path, name, task, tau_bar, log=print, settle_s=0.3, sub_s=0.05, pos_tol_m=0.004, kp: float = 20.0,
                    squeeze_mm: float = 0.5, rot_tol_rad: float = 0.2, drop_fingers=(), controller: str = "position",
                    force_scale: float = 1.0, k_i_m_per_Ns: float = 0.2, k_p_m_per_N: float = 0.0, delta_max_mm: float = 3.0,
                    leak: float = 0.1, f_min_n: float = 0.05, drop_tol_m: float = 0.02, stop_at_first_failure: bool | None = None,
                    pad=None, pad_variant: str = "patch", archive: Path | None = None, only=None, on_step=None):
    """Replay every exported trajectory in MuJoCo with one of two controllers.

    ``controller="position"`` (Phase 8.6): position servos (stiffness ``kp`` N·m/rad, force range
    ±tau_bar) commanded at ``q_path + squeeze_mm * dq_squeeze``: every assigned fingertip is commanded
    ``squeeze_mm`` millimetres into the sphere along its contact normal at every sub-step (the geometric
    grip; the servo's torque stays limited by the ceiling).

    ``controller="force"`` (Phase 9.3): the same position servos, with the squeeze of every lane regulated
    on the measured contact normal force (admittance control on the fingertip). ``f*_c`` is ``force_scale``
    times the exact allocation's normal force of that lane at the current sub-step (``fn_exact``, the
    minimum-norm SLSQP allocation under the primary torque ceiling) and ``f_c`` the normal force MuJoCo
    measures between the sphere and the lane's capsule. The exact allocation is a **lower bound** on the
    lane's force: ``d delta_c / dt = k_i max(f*_c - f_c, 0) - leak k_i max(f_c - f*_c, 0)`` with ``delta_c``
    in ``[squeeze_mm, delta_max_mm]`` (never below the 8.6 squeeze), started at ``squeeze_mm``; the command
    is ``q_path + sum_c delta_c dq_lane[:, c]``. A lane whose measured force falls below the allocation
    (contact slipping away) advances along its normal at up to ``k_i f*_c`` until the force is restored;
    one that presses harder leaks slowly back to the baseline squeeze. ``controller="force_sym"`` is the
    symmetric regulator (``d delta_c / dt = k_i (f*_c - f_c)`` plus ``k_p (f*_c - f_c)``, ``delta_c`` in
    ``[-0.5 mm, delta_max_mm]``) that was tried first: a two-finger pinch cannot realize two independent
    normal-force targets, so its integrators wind up against each other and the sphere is lost in the
    settle phase (0 of 5 baseline seeds hold at ``force_scale`` 1, 4 of 5 at 2 to 3; kept for the record).
    Tuning (on the two-finger baseline's settle phase and first primitives only, as the 8.6 squeeze):
    ``k_i = 0.2 m/(N s)``, ``leak = 0.1``, ``force_scale = 1``, ``k_p = 0``. Every assigned lane's set-point is
    at least ``f_min_n`` (0.05 N, about 15 % of the object's weight): the minimum-norm allocation leaves some
    three-contact lanes at ~0 N (22 % of the start's and 7 % of arm B's lanes at the seed states), which a zero
    floor would leave without restoring action; fixed from the allocations before any three-contact force replay.

    Per primitive three separated outcomes are recorded at the primitive end (SPEC §11.2 item 9.3):
    object-pose success (centre within ``pos_tol_m`` **and** orientation within ``rot_tol_rad`` of the command),
    contact retention (every assigned fingertip in contact), and drop / gross instability (centre more than
    ``drop_tol_m`` from the command, or no assigned fingertip in contact). A trajectory stops at a drop; with
    ``stop_at_first_failure`` (the 8.6 convention, default for the position controller) it also stops at the
    first primitive that fails position or contact retention. The 8.6 counts ``n_sim`` (position + contacts)
    and ``n_sim_rot`` (+ orientation) are the first-failure indices of those criteria and are therefore
    identical under both stopping rules; ``n_pose`` (position + orientation), ``n_retain`` (contacts) and
    ``n_drop`` are the separated first-failure indices. A seed whose sphere drifts more than ``pos_tol_m`` in the
    settle phase is recorded (``settle_err_m``) and counted separately as "seed does not hold".

    Phase 10.1: ``pad`` (``compliant_pad.PadParams``) replaces the terminal capsules' contact with the sphere by the
    compliant-pad law (``mjcf_from_design``); ``archive`` is the directory of the exported trajectories and design
    (default ``out``), so replays under several contact models write to their own directories.

    Observation only (``visualize_design``): ``only`` (a predicate on the exported trajectory record) replays a subset;
    ``on_step(model, data, t, k, h, a)`` is called after every physics step (``k = -1`` in the settle phase, else the
    primitive and sub-step ``h`` with the ramp fraction ``a``) and must not modify ``data``."""
    import mujoco
    archive = Path(archive) if archive is not None else out
    if stop_at_first_failure is None:
        stop_at_first_failure = controller == "position"
    design = json.loads((archive / f"{name}_design.json").read_text())
    if drop_fingers:                      # simulate the hand without these fingers (e.g. arm A as a two-finger hand)
        keep = [i for i, f in enumerate(design["topology"]["fingers"]) if f not in drop_fingers]
        design = dict(design, fingers={f: v for f, v in design["fingers"].items() if f not in drop_fingers},
                      topology=dict(fingers=[design["topology"]["fingers"][i] for i in keep], n_joints=[design["topology"]["n_joints"][i] for i in keep], pair=design["topology"]["pair"]))
    xml = mjcf_from_design(design, task, tau_bar, kp=kp, pad=pad, pad_variant=pad_variant)
    (out / f"{name}.xml").write_text(xml)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    # dkarma flat joint order = fingers in topology order, joints in order (matches Pinocchio on the export)
    order = [f"{fn}_j{k}" for fn, nj in zip(design["topology"]["fingers"], design["topology"]["n_joints"]) for k in range(nj)]
    full = json.loads((archive / f"{name}_design.json").read_text())["topology"]
    full_order = [f"{fn}_j{k}" for fn, nj in zip(full["fingers"], full["n_joints"]) for k in range(nj)]
    qcols = np.array([full_order.index(j) for j in order])          # columns of the exported q for the simulated joints
    full_caps = [f"{fn}_c{k}" for fn, nj in zip(full["fingers"], full["n_joints"]) for k in range(nj)]
    qadr = np.array([model.joint(j).qposadr[0] for j in order])
    sid = model.body("sphere").id
    sphere_gid = model.geom("sphere_geom").id
    sph_q = model.joint("sphere_free").qposadr[0]
    dt = model.opt.timestep
    rows = []
    for t in load_trajectories(archive, name):
        if only is not None and not only(t):
            continue
        caps = [full_caps[c] for c in t["contact_caps"]]
        if any(c.split("_c")[0] in drop_fingers for c in caps):
            continue                          # a library that uses a dropped finger is not simulated
        gids = [model.geom(c).id for c in caps]
        q, p = t["q"][:, qcols], t["p"]
        dq = (t["dq_squeeze"] if t.get("dq_squeeze") is not None else np.zeros_like(t["q"]))[:, qcols]
        q_cmd = q + squeeze_mm * dq
        Cn = len(gids)
        if controller in ("force", "force_sym"):
            if t.get("dq_lane") is None or t.get("fn_exact") is None:
                raise SystemExit("the force controller needs an export with dq_lane and fn_exact (re-run the export stage)")
            dq_lane = t["dq_lane"][:, :Cn][:, :, qcols]            # [H+1, Cn, J_sim]
            fn_star = np.maximum(force_scale * t["fn_exact"][:, :Cn], f_min_n)   # [H+1, Cn], every assigned lane >= f_min
            delta = np.full(Cn, squeeze_mm * 1e-3)
        aid = [model.actuator(j.replace('_j', '_a')).id for j in order]
        mujoco.mj_resetData(model, data)
        data.qpos[qadr] = q[0]; data.ctrl[:] = 0.0
        data.ctrl[aid] = q_cmd[0]
        data.qpos[sph_q:sph_q + 3] = p[0]; data.qpos[sph_q + 3:sph_q + 7] = [1, 0, 0, 0]
        mujoco.mj_forward(model, data)
        n_settle = int(settle_s / dt)
        for i in range(n_settle):
            if controller in ("force", "force_sym"):
                f_meas = _normal_forces(model, data, gids, sphere_gid)
                delta, err_f = _admittance(delta, fn_star[0], f_meas, dt, controller, k_i_m_per_Ns, leak, squeeze_mm * 1e-3, delta_max_mm * 1e-3)
                data.ctrl[aid] = q[0] + np.einsum("c,cj->j", delta + k_p_m_per_N * err_f, dq_lane[0])
            mujoco.mj_step(model, data)
            if on_step is not None:
                on_step(model, data, t, -1, 0, (i + 1) / n_settle)
        settle_err = float(np.linalg.norm(data.xpos[sid] - p[0]))
        settle_fn = _normal_forces(model, data, gids, sphere_gid).tolist()
        info = []
        n_steps = int(sub_s / dt)
        per_prim = t["n_sub"]
        dropped = False
        for k in range(t["n_primitives"]):
            touch_steps = np.zeros(Cn); f_sum = np.zeros(Cn); n_prim_steps = 0   # contact duty cycle per lane (instrumentation only)
            for h in range(k * per_prim + 1, (k + 1) * per_prim + 1):
                # the command is streamed: a linear ramp from the previous to the next sub-step command over
                # sub_s (a step command saturates the servo and kicks the fingertip into the sphere)
                for i in range(n_steps):
                    a = (i + 1) / n_steps
                    if controller in ("force", "force_sym"):
                        f_meas = _normal_forces(model, data, gids, sphere_gid)
                        fstar = (1 - a) * fn_star[h - 1] + a * fn_star[h]
                        delta, err_f = _admittance(delta, fstar, f_meas, dt, controller, k_i_m_per_Ns, leak, squeeze_mm * 1e-3, delta_max_mm * 1e-3)
                        base = (1 - a) * q[h - 1] + a * q[h]
                        dl = (1 - a) * dq_lane[h - 1] + a * dq_lane[h]
                        data.ctrl[aid] = base + np.einsum("c,cj->j", delta + k_p_m_per_N * err_f, dl)
                    else:
                        data.ctrl[aid] = q_cmd[h - 1] + a * (q_cmd[h] - q_cmd[h - 1])
                    mujoco.mj_step(model, data)
                    if on_step is not None:
                        on_step(model, data, t, k, h, a)
                    f_now, t_now = _normal_forces(model, data, gids, sphere_gid, with_touch=True)
                    touch_steps += t_now; f_sum += f_now; n_prim_steps += 1
            centre = data.xpos[sid].copy()
            err = float(np.linalg.norm(centre - p[(k + 1) * per_prim]))
            R_sim = data.xmat[sid].reshape(3, 3); R_cmd = t["R"][(k + 1) * per_prim]
            rot_err = float(np.arccos(np.clip((np.trace(R_sim.T @ R_cmd) - 1.0) / 2.0, -1.0, 1.0)))
            f_meas = _normal_forces(model, data, gids, sphere_gid)
            in_contact = [bool(f > 0.0) or any((data.contact[c].geom1 == g and data.contact[c].geom2 == sphere_gid) or
                                               (data.contact[c].geom2 == g and data.contact[c].geom1 == sphere_gid) for c in range(data.ncon))
                          for g, f in zip(gids, f_meas)]
            pos_ok = err <= pos_tol_m
            retain = all(in_contact)
            pose_ok = pos_ok and rot_err <= rot_tol_rad
            drop = err > drop_tol_m or not any(in_contact)
            rec = dict(err_m=err, rot_err_rad=rot_err, contacts=in_contact, pos_ok=pos_ok, retain_ok=retain, pose_ok=pose_ok, drop=drop,
                       ok=bool(pos_ok and retain), ok_rot=bool(pos_ok and retain and rot_err <= rot_tol_rad), fn_meas=f_meas.tolist(),
                       fn_star=(fn_star[(k + 1) * per_prim].tolist() if controller in ("force", "force_sym") else None),
                       delta_mm=((delta * 1e3).tolist() if controller in ("force", "force_sym") else None),
                       contact_duty=(touch_steps / max(n_prim_steps, 1)).tolist(), fn_mean=(f_sum / max(n_prim_steps, 1)).tolist())
            info.append(rec)
            if drop:
                dropped = True; break
            if stop_at_first_failure and not (pos_ok and retain):
                break

        def first_fail(flag):
            flags = [i[flag] for i in info]
            return int(np.argmin(flags)) if not all(flags) else len(flags)

        n_sim = first_fail("ok"); n_sim_rot = first_fail("ok_rot")
        n_pose = first_fail("pose_ok"); n_retain = first_fail("retain_ok")
        n_drop = int(np.argmax([i["drop"] for i in info])) if dropped else t["n_primitives"]
        fo = t.get("force_ok")
        rows.append(dict(seed=t["seed"], library=t["library"], kind=t["kind"], command=t["command"], n_primitives=t["n_primitives"],
                         n_exact=t["n_ok"], reject_gate=t.get("reject_gate"), n_sim=n_sim, n_sim_rot=n_sim_rot, n_pose=n_pose, n_retain=n_retain,
                         n_drop=n_drop, n_simulated=len(info), settle_err_m=settle_err, settle_fn=settle_fn,
                         force_ok_states=(int(fo.sum()) if fo is not None else None), controller=controller, info=info))
    return rows


def confusion(rows, key="n_sim"):
    """Per primitive: exact verdict (valid up to n_exact) vs simulation success (up to ``key``: ``n_sim`` =
    position and contacts, ``n_sim_rot`` = position, contacts and orientation)."""
    tp = fp = fn = tn = 0
    by_gate = {}
    for r in rows:
        for k in range(r["n_primitives"]):
            ex = k < r["n_exact"]; sim = k < r[key]
            if ex and sim: tp += 1
            elif ex and not sim: fn += 1
            elif not ex and sim: fp += 1
            else: tn += 1
            if k == r["n_exact"] and r["reject_gate"]:
                g = by_gate.setdefault(r["reject_gate"], dict(sim_success=0, sim_fail=0))
                g["sim_success" if sim else "sim_fail"] += 1
            if not ex:
                break           # only the first rejected primitive of a trajectory is informative
    return dict(exact_valid_sim_success=tp, exact_valid_sim_fail=fn, exact_rejected_sim_success=fp, exact_rejected_sim_fail=tn,
                frac_valid_succeed=(tp / (tp + fn) if tp + fn else None), frac_rejected_succeed=(fp / (fp + tn) if fp + tn else None),
                rejected_by_gate=by_gate)


def result_record(rows, settings):
    """The per-design record of a replay (``simulation<tag>.json``): confusion matrices, separated outcomes, rows, settings."""
    return dict(confusion=confusion(rows), confusion_with_orientation=confusion(rows, "n_sim_rot"),
                confusion_pose=confusion(rows, "n_pose"), confusion_retention=confusion(rows, "n_retain"), confusion_no_drop=confusion(rows, "n_drop"),
                separated=separated_outcomes(rows), rows=rows, settings=settings)


def separated_outcomes(rows, pos_tol_m=0.004):
    """Phase 9.3: per library, over the exactly valid primitives of seeds that hold (settle drift <= tolerance),
    the first-failure counts of the three separated criteria and the 8.6 strict criterion."""
    out = {}
    for lib in sorted({r["library"] for r in rows}):
        rs = [r for r in rows if r["library"] == lib]
        hold = [r for r in rs if r["settle_err_m"] <= pos_tol_m]
        seeds = sorted({r["seed"] for r in rs}); seeds_hold = sorted({r["seed"] for r in hold})
        n_valid = sum(r["n_exact"] for r in hold)
        cnt = lambda key: sum(min(r[key], r["n_exact"]) for r in hold)   # noqa: E731  exactly valid primitives that pass the criterion
        # primitives on which the three outcomes disagree: pose fine but a contact lost, etc.
        pose_not_retain = sum(1 for r in hold for k, i in enumerate(r["info"]) if k < r["n_exact"] and i["pose_ok"] and not i["retain_ok"])
        retain_not_pose = sum(1 for r in hold for k, i in enumerate(r["info"]) if k < r["n_exact"] and i["retain_ok"] and not i["pose_ok"])
        drops = sum(1 for r in hold if r["n_drop"] < r["n_primitives"])
        first_prim_fail = sum(1 for r in hold if r["n_exact"] > 0 and r["n_sim"] == 0)
        out[lib] = dict(n_seeds=len(seeds), seeds_hold=len(seeds_hold), n_trajectories_hold=len(hold), exact_valid_primitives=n_valid,
                        strict_position_contacts=cnt("n_sim"), strict_with_orientation=cnt("n_sim_rot"), pose_success=cnt("n_pose"),
                        contact_retention=cnt("n_retain"), no_drop=cnt("n_drop"),
                        frac=dict(strict=(cnt("n_sim") / n_valid if n_valid else None), strict_rot=(cnt("n_sim_rot") / n_valid if n_valid else None),
                                  pose=(cnt("n_pose") / n_valid if n_valid else None), retain=(cnt("n_retain") / n_valid if n_valid else None),
                                  no_drop=(cnt("n_drop") / n_valid if n_valid else None)),
                        pose_ok_but_contact_lost=pose_not_retain, contacts_kept_but_pose_failed=retain_not_pose, trajectories_dropped=drops,
                        trajectories_failing_first_primitive=first_prim_fail)
    return out


# Phase 9.3 decision rule, fixed before any force-controlled three-contact replay was run (docs/phase9_report.md §9.3):
# on each three-contact library, the strict success rate (exactly valid primitives of seeds that hold under that
# controller; position, orientation and every assigned contact) under the force controller minus that under the
# position servo: >= 0.30 on every three-contact design -> controller-induced; < 0.10 on every design -> model-induced;
# otherwise mixed (reported per design).
CONTROLLER_INDUCED_GAIN = 0.30
MODEL_INDUCED_GAIN = 0.10


def compare_controllers(out: Path, tags=("position", "force"), log=_print):
    """Tables of the 9.3 comparison: per design, library and controller the confusion matrices (exact verdict vs
    simulation, per primitive, 8.6 convention) and the three separated outcomes; the decision of the rule above."""
    res = {}
    for t in tags:                       # one file per controller, or one per controller and design (simulation_<t>__<design>.json)
        for p in [out / f"simulation_{t}.json"] + sorted(out.glob(f"simulation_{t}__*.json")):
            if p.exists():
                res.setdefault(t, {}).update(json.loads(p.read_text()))
    designs = sorted({d for r in res.values() for d in r})
    lines = ["# Phase 9.3: position servo (8.6) vs force-floor controller (9.3), same scene and trajectories as 8.6", "",
             "Fixed physical task. Success criteria at each primitive end: pose = centre within 4 mm and orientation within 0.2 rad of the command; "
             "retention = every assigned fingertip in contact; drop = centre > 20 mm from the command or no assigned fingertip in contact. "
             "Rates are over the exactly valid primitives of seeds that hold (settle drift <= 4 mm) under that controller; trajectories run to their end "
             "(stop only at a drop), so each criterion's first-failure index is measured independently.", "",
             "| design | library | controller | seeds hold | exactly valid primitives | strict (pose + all contacts) | pose | contact retention | no drop | pose ok, a contact lost | contacts kept, pose failed | exactly rejected first primitives succeeding (strict) |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    rec = dict(rows={}, decision={})
    for d in designs:
        for t in tags:
            r = res.get(t, {}).get(d)
            if r is None:
                continue
            for lib, v in r["separated"].items():
                rows = [x for x in r["rows"] if x["library"] == lib and x["settle_err_m"] <= 0.004]
                rej = [x for x in rows if x["n_exact"] < x["n_primitives"]]
                rej_ok = sum(1 for x in rej if x["n_sim_rot"] > x["n_exact"])
                fr = v["frac"]
                f = lambda z: "–" if z is None else f"{z:.2f}"   # noqa: E731
                lines.append(f"| {d} | {lib} | {t} | {v['seeds_hold']}/{v['n_seeds']} | {v['exact_valid_primitives']} | {v['strict_with_orientation']} ({f(fr['strict_rot'])}) | "
                             f"{v['pose_success']} ({f(fr['pose'])}) | {v['contact_retention']} ({f(fr['retain'])}) | {v['no_drop']} ({f(fr['no_drop'])}) | "
                             f"{v['pose_ok_but_contact_lost']} | {v['contacts_kept_but_pose_failed']} | {rej_ok}/{len(rej)} |")
                rec["rows"].setdefault(d, {}).setdefault(lib, {})[t] = dict(v, rejected_first_success=rej_ok, rejected_first=len(rej))
    lines.append("")
    gains = {}
    for d, libs in rec["rows"].items():
        v = libs.get("three_contact", {})
        if all(t in v for t in ("position", "force")) and v["position"]["frac"]["strict_rot"] is not None and v["force"]["frac"]["strict_rot"] is not None:
            gains[d] = v["force"]["frac"]["strict_rot"] - v["position"]["frac"]["strict_rot"]
    if gains:
        verdict = ("controller-induced" if all(g >= CONTROLLER_INDUCED_GAIN for g in gains.values()) else
                   "model-induced" if all(g < MODEL_INDUCED_GAIN for g in gains.values()) else "mixed")
        rec["decision"] = dict(gains=gains, verdict=verdict, rule=dict(controller_induced_gain=CONTROLLER_INDUCED_GAIN, model_induced_gain=MODEL_INDUCED_GAIN))
        lines += [f"Three-contact strict-success gain (force − position) per design: " + ", ".join(f"{d} {g:+.2f}" for d, g in gains.items()),
                  f"Decision rule (fixed before the force-controlled three-contact replays): >= {CONTROLLER_INDUCED_GAIN:.2f} on every design -> controller-induced; "
                  f"< {MODEL_INDUCED_GAIN:.2f} on every design -> model-induced; otherwise mixed. Verdict: **{verdict}**.", ""]
    # post hoc (not part of the decision rule): how the assigned contacts fail, from the per-primitive contact duty cycle
    lines += ["## Post hoc: contact duty cycle (not part of the decision rule)", "",
              "Lanes failing the retention criterion at an exactly valid primitive end: share of the primitive's simulation steps "
              "in which that lane touched the sphere, and its mean normal force; and the success rate when retention is read as "
              "'every assigned lane in contact during at least half of the primitive' instead of 'at the primitive end'.", "",
              "| design | library | controller | lanes failing retention | their duty: median, share = 0 | their mean normal force (N, median) | strict success | pose + duty >= 0.5 success |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for d in designs:
        for t in tags:
            r = res.get(t, {}).get(d)
            if r is None:
                continue
            for lib in sorted({x["library"] for x in r["rows"]}):
                rows = [x for x in r["rows"] if x["library"] == lib and x["settle_err_m"] <= 0.004]
                if not rows or "contact_duty" not in rows[0]["info"][0]:
                    continue
                duty, fmean = [], []
                for x in rows:
                    for i in x["info"][: x["n_exact"]]:
                        for li, ok in enumerate(i["contacts"]):
                            if not ok:
                                duty.append(i["contact_duty"][li]); fmean.append(i["fn_mean"][li])

                def ok_count(ok):
                    tot = 0
                    for x in rows:
                        k = next((k for k, i in enumerate(x["info"]) if not ok(i)), len(x["info"]))
                        tot += min(k, x["n_exact"])
                    return tot
                n_valid = sum(x["n_exact"] for x in rows)
                a = ok_count(lambda i: i["pose_ok"] and i["retain_ok"]); b = ok_count(lambda i: i["pose_ok"] and all(dd >= 0.5 for dd in i["contact_duty"]))
                du = np.asarray(duty)
                lines.append(f"| {d} | {lib} | {t} | {len(du)} | {(np.median(du) if du.size else float('nan')):.2f}, {(np.mean(du == 0) if du.size else float('nan')):.2f} | "
                             f"{(np.median(fmean) if fmean else float('nan')):.3f} | {a}/{n_valid} ({a / max(n_valid, 1):.2f}) | {b}/{n_valid} ({b / max(n_valid, 1):.2f}) |")
                rec.setdefault("post_hoc", {}).setdefault(d, {}).setdefault(lib, {})[t] = dict(n_lost=len(du), duty_median=(float(np.median(du)) if du.size else None),
                                                                                             duty_zero_share=(float(np.mean(du == 0)) if du.size else None),
                                                                                             fn_mean_median=(float(np.median(fmean)) if fmean else None),
                                                                                             strict=a, duty_success=b, n_valid=n_valid)
    lines.append("")
    # confusion matrices (8.6 convention: per primitive up to the first rejected one, all seeds)
    lines += ["## Confusion matrices (all seeds, including those that do not hold; per primitive up to the first exactly rejected one, 8.6 convention; "
              "criterion = pose + all assigned contacts at the primitive end)", "",
              "| design | library | controller | exactly valid: succeed / fail | exactly rejected (first primitive): succeed / fail |", "| --- | --- | --- | --- | --- |"]
    for d in designs:
        for t in tags:
            r = res.get(t, {}).get(d)
            if r is None:
                continue
            for lib in sorted({x["library"] for x in r["rows"]}):
                c = confusion([x for x in r["rows"] if x["library"] == lib], "n_sim_rot")
                rec.setdefault("confusion", {}).setdefault(d, {}).setdefault(lib, {})[t] = c
                lines.append(f"| {d} | {lib} | {t} | {c['exact_valid_sim_success']} / {c['exact_valid_sim_fail']} | {c['exact_rejected_sim_success']} / {c['exact_rejected_sim_fail']} |")
    (out / "controller_comparison.json").write_text(json.dumps(rec, indent=1, default=float))
    (out / "controller_comparison.md").write_text("\n".join(lines) + "\n")
    log("\n".join(lines))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["export", "simulate", "augment", "compare"])
    ap.add_argument("--src", default="workspace/phase8/8.6", help="augment: the 8.6 archive directory")
    ap.add_argument("--out", default="workspace/phase8/8.6")
    ap.add_argument("--designs", nargs="*", default=["baseline", "start", "armA", "armB"])
    ap.add_argument("--budget", type=int, default=6)
    ap.add_argument("--kp", type=float, default=20.0, help="position-servo stiffness (N·m/rad)")
    ap.add_argument("--squeeze-mm", type=float, default=0.5, help="commanded fingertip penetration along the contact normal (mm)")
    ap.add_argument("--libraries", nargs="*", default=["two_finger", "three_contact"], help="export: which libraries to (re)compute; the others are kept from the existing archive")
    ap.add_argument("--drop-fingers", nargs="*", default=[], help="simulate: fingers removed from the scene (their libraries are skipped)")
    ap.add_argument("--tag", default="", help="simulate: suffix of the output JSON (simulation<tag>.json)")
    ap.add_argument("--controller", choices=["position", "force", "force_sym"], default="position", help="simulate: Phase 8.6 position servo, the Phase 9.3 force-floor controller, or the symmetric regulator")
    ap.add_argument("--force-scale", type=float, default=1.0, help="simulate/force: set-point = force_scale x exact allocation normal force")
    ap.add_argument("--k-i", type=float, default=0.2, help="simulate/force: integral admittance gain (m per N·s)")
    ap.add_argument("--k-p", type=float, default=0.0, help="simulate/force: proportional admittance gain (m per N)")
    ap.add_argument("--leak", type=float, default=0.1, help="simulate/force: leak rate (fraction of k_i) back toward the baseline squeeze when the force exceeds the set-point")
    ap.add_argument("--f-min", type=float, default=0.05, help="simulate/force: minimum normal-force set-point of every assigned lane (N)")
    ap.add_argument("--continue-after-failure", action="store_true", help="simulate: do not stop a trajectory at its first pose/contact failure (always on for the force controller)")
    ap.add_argument("--design-dir", default=None, help="export: directory with <name>.design.json files for names not in the built-in list")
    ap.add_argument("--seed-config", default="default", help="export: seed-search configuration (admissible_baseline.SEED_CONFIGS)")
    ap.add_argument("--archive", default=None, help="simulate: directory of the exported trajectories and designs (default --out)")
    ap.add_argument("--pad", default=None, help="simulate: compliant-fingertip parameter set (label in --pad-file, e.g. 20A_2mm; default rigid 8.6/9.3)")
    ap.add_argument("--pad-file", default="dkarma/configs/compliant_pads.json", help="simulate: the frozen compliant-pad parameter sets (Phase 10.1)")
    ap.add_argument("--pad-variant", choices=["patch", "rigid_friction", "patch_friction_only"], default="patch", help="simulate: pad friction (diagnostic D1: rigid_friction)")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    task = load_task_spec()
    global DESIGN_DIR
    DESIGN_DIR = args.design_dir
    if args.stage == "export":
        export_all(out, args.designs, args.budget, libraries=tuple(args.libraries), seed_config=args.seed_config)
    elif args.stage == "augment":
        for name in args.designs:
            augment_archive(Path(args.src), out, name)
    elif args.stage == "compare":
        compare_controllers(out)
    else:
        try:
            import mujoco  # noqa: F401
        except ImportError:
            raise SystemExit("mujoco is not installed in this environment; the owner must confirm the install first (SPEC §11.1 item 8.6)")
        from dkarma.core.actuation import load_actuation_spec
        from dkarma.experiments.compliant_pad import load_pads
        pad = load_pads(args.pad_file)[args.pad] if args.pad else None
        contact_model = dict(model="rigid 8.6/9.3 (solref 0.002 1, solimp 0.95 0.99 0.001, condim 4, torsional mu*3 mm)") if pad is None else \
            dict(model=f"compliant pad {args.pad} ({args.pad_variant} friction)", pad=pad.as_dict(), pad_variant=args.pad_variant, pad_file=args.pad_file)
        res = {}
        for name in args.designs:
            rows = simulate_design(out, name, task, load_actuation_spec().torque_limit_nm, kp=args.kp, squeeze_mm=args.squeeze_mm,
                                   drop_fingers=tuple(args.drop_fingers), controller=args.controller, force_scale=args.force_scale,
                                   k_i_m_per_Ns=args.k_i, k_p_m_per_N=args.k_p, leak=args.leak, f_min_n=args.f_min,
                                   stop_at_first_failure=(False if (args.continue_after_failure or args.controller != "position") else None),
                                   pad=pad, pad_variant=args.pad_variant, archive=(Path(args.archive) if args.archive else None))
            res[name] = result_record(rows, settings=dict(controller=args.controller, kp=args.kp, squeeze_mm=args.squeeze_mm, settle_s=0.3, sub_s=0.05, pos_tol_m=0.004, rot_tol_rad=0.2,
                                           drop_tol_m=0.02, force_scale=args.force_scale, k_i_m_per_Ns=args.k_i, k_p_m_per_N=args.k_p, leak=args.leak, f_min_n=args.f_min,
                                           drop_fingers=list(args.drop_fingers), torque_ceiling_nm=load_actuation_spec().torque_limit_nm,
                                           contact_model=contact_model, archive=(args.archive or str(out))))
            print(name, res[name]["confusion"], flush=True)
            print(name, "with orientation:", res[name]["confusion_with_orientation"], flush=True)
            print(name, "separated:", json.dumps(res[name]["separated"], default=float), flush=True)
            (out / f"simulation{args.tag}.json").write_text(json.dumps(res, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
