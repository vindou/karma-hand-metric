"""Admissibility of a design (SPEC.md §8.1 D18, owner's decision §16 item 10, 2026-10-13; D19, §16 item 14, 2026-10-23).

A design is admissible only if no two modelled fingers can occupy the same space. The
configuration-independent part is the base separation: any two finger bases at least
``r_a + r_b + eps_col`` apart, with ``r`` the radius of each finger's base capsule (24.1 mm for
11.8 mm capsules and the 0.5 mm clearance; 33.30 mm for the 16.40 mm servo links of D19). The reach
bound (250 mm per finger, the sum of a finger's link lengths including the tip) is the global
hand envelope. Configuration-dependent clearance between all fingers is the self-collision gate
of every evaluator (``dkarma.core.contacts.self_clearance``, with each capsule's own radius;
``karma_jax`` with ``build_flat_model(..., self_collision_all=True)``).

D19 (Phase 11) adds the servo envelope (:class:`ServoEnvelope`): every non-terminal link holds an
XL330 case along its axis, so its capsule radius is at least ``r_servo`` = 16.40 mm and it is at least
34 mm long (link 0 included: the owner withdrew its relaxation on 2026-10-24); the fingertip holds no
servo and keeps the task's contact radius (11.8 mm).

Smooth penalties for the optimizer live in :mod:`dkarma.optimization.objectives`; this module is
the host-side hard check every optimizer applies to a candidate before it is evaluated or
returned (CLAUDE.md rule 11 b).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .morphology import HandParams, DEFAULT_CAPSULE_RADIUS_M
from .topology import Topology

EPS_COL_M = 0.0005
MIN_BASE_SEPARATION_M = 2.0 * DEFAULT_CAPSULE_RADIUS_M + EPS_COL_M      # 0.0241 m (D18)
REACH_MAX_M = 0.25

R_SERVO_M = 0.0164                       # D19: half the 20 x 26 mm case diagonal, 16.40 mm as written (16.401 mm exact)
SERVO_LEN_MIN_M = 0.034                  # D19: the 34 mm case along links 0-2
D19_MIN_BASE_SEPARATION_M = 2.0 * R_SERVO_M + EPS_COL_M                # 0.0333 m


@dataclass(frozen=True)
class ServoEnvelope:
    """D19: every non-terminal (servo) link at least ``servo_radius_m`` thick and ``servo_len_min_m`` long, the fingertip
    at the contact radius."""
    servo_radius_m: float = R_SERVO_M
    servo_len_min_m: float = SERVO_LEN_MIN_M
    tip_radius_m: float = DEFAULT_CAPSULE_RADIUS_M


D19_ENVELOPE = ServoEnvelope()


def base_separations(params: HandParams, topo: Topology) -> dict:
    """Pairwise distances (m) between finger bases, keyed by finger-name pairs."""
    b = np.asarray(params.base_xyz, float)
    names = topo.fingers
    return {(names[i], names[j]): float(np.linalg.norm(b[i] - b[j])) for i in range(len(names)) for j in range(i + 1, len(names))}


def required_base_separations(params: HandParams, topo: Topology) -> dict:
    """D18's rule with each base capsule's own radius: ``r_a + r_b + eps_col`` per finger pair (m)."""
    r = np.asarray(params.cap_radius, float)
    r0 = [float(r[topo.first_joint[f]]) for f in range(topo.n_fingers)]
    names = topo.fingers
    return {(names[i], names[j]): r0[i] + r0[j] + EPS_COL_M for i in range(len(names)) for j in range(i + 1, len(names))}


def finger_reaches(params: HandParams, topo: Topology) -> dict:
    """Sum of link lengths (tip included) per finger (m)."""
    ll = np.asarray(params.link_len, float)
    return {name: float(ll[f, : topo.n_joints[f]].sum()) for f, name in enumerate(topo.fingers)}


def envelope_check(params: HandParams, topo: Topology, env: ServoEnvelope = D19_ENVELOPE, tol: float = 1e-9) -> dict:
    """The servo envelope: radii and length floors per capsule, with the smallest margins (m)."""
    r = np.asarray(params.cap_radius, float)
    ll = np.asarray(params.link_len, float)
    viol, servo_r, floor_len, tip_r = [], [], [], []
    for f, name in enumerate(topo.fingers):
        n = topo.n_joints[f]
        for k in range(n):
            c = int(topo.first_joint[f]) + k
            if k == n - 1:
                tip_r.append(r[c])
                if abs(r[c] - env.tip_radius_m) > tol:
                    viol.append(f"{name} fingertip radius {r[c] * 1e3:.2f} mm != contact radius {env.tip_radius_m * 1e3:.2f} mm")
                continue
            servo_r.append(r[c])
            if r[c] < env.servo_radius_m - tol:
                viol.append(f"{name} link {k} radius {r[c] * 1e3:.2f} mm < {env.servo_radius_m * 1e3:.2f} mm")
            floor_len.append(ll[f, k])
            if ll[f, k] < env.servo_len_min_m - tol:
                viol.append(f"{name} link {k} length {ll[f, k] * 1e3:.1f} mm < {env.servo_len_min_m * 1e3:.0f} mm")
    return dict(ok=not viol, violations=viol, min_servo_radius_m=(min(servo_r) if servo_r else None),
                min_servo_link_len_m=(min(floor_len) if floor_len else None),
                servo_len_margin_m=((min(floor_len) - env.servo_len_min_m) if floor_len else None),
                tip_radius_m=(sorted(set(float(x) for x in tip_r))), envelope=env.__dict__)


def admissibility(params: HandParams, topo: Topology, min_sep_m: float | None = None, reach_max_m: float = REACH_MAX_M,
                  envelope: ServoEnvelope | None = None) -> dict:
    """Hard D18 check of a design: ``dict(admissible, min_base_separation_m, max_reach_m,
    base_separations, reaches, violations)``. ``min_sep_m`` (one threshold for every pair) defaults to D18's
    rule with the design's own base-capsule radii, which is ``MIN_BASE_SEPARATION_M`` for uniform 11.8 mm
    capsules and 33.30 mm under D19. ``envelope`` (D19) adds the servo-envelope check and its record."""
    seps = base_separations(params, topo)
    reaches = finger_reaches(params, topo)
    req = {k: min_sep_m for k in seps} if min_sep_m is not None else required_base_separations(params, topo)
    min_sep = min(seps.values()) if seps else float("inf")
    max_reach = max(reaches.values())
    viol = []
    short = [k for k, v in seps.items() if v < req[k]]
    if short:
        k = min(short, key=lambda k: seps[k] - req[k])
        viol.append(f"base separation {seps[k] * 1e3:.1f} mm < {req[k] * 1e3:.1f} mm")
    if max_reach > reach_max_m:
        viol.append(f"reach {max_reach * 1e3:.1f} mm > {reach_max_m * 1e3:.0f} mm")
    out = dict(admissible=not viol, min_base_separation_m=min_sep, max_reach_m=max_reach,
               base_separations={f"{a}-{b}": v for (a, b), v in seps.items()}, reaches=reaches, violations=viol,
               min_sep_required_m=(max(req.values()) if req else (min_sep_m or MIN_BASE_SEPARATION_M)), reach_max_m=reach_max_m)
    if envelope is not None:
        env = envelope_check(params, topo, envelope)
        out["violations"] = viol + env["violations"]
        out["admissible"] = not out["violations"]
        out["envelope"] = env
        ll = np.asarray(params.link_len, float)
        out["margins_m"] = dict(base_separation=(min(seps[k] - req[k] for k in seps) if seps else None), reach=reach_max_m - max_reach,
                                servo_link_len=env["servo_len_margin_m"])
        out["lengths_m"] = dict(shortest_servo_link=env["min_servo_link_len_m"], shortest_tip=min(float(ll[f, n - 1]) for f, n in enumerate(topo.n_joints)),
                                longest=float(ll[np.asarray(topo.joint_mask)].max()))
    return out


__all__ = ["EPS_COL_M", "MIN_BASE_SEPARATION_M", "REACH_MAX_M", "R_SERVO_M", "SERVO_LEN_MIN_M", "D19_MIN_BASE_SEPARATION_M",
           "ServoEnvelope", "D19_ENVELOPE", "base_separations", "required_base_separations", "finger_reaches", "envelope_check",
           "admissibility"]
