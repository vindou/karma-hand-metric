"""Phase 10.1 (SPEC.md §11.3): a compliant silicone-fingertip approximation for the MuJoCo scene of 8.6/9.3.

Simulation model only. The pad is the outer ``t`` metres of the terminal-link capsule radius (11.8 mm), so the
zero-load geometry, the morphologies, the exported trajectories and every exact verdict are unchanged: in MuJoCo
the capsule keeps its radius and the pad appears only in the contact law of the (terminal capsule, sphere) pairs.

Stated assumptions (documented in ``docs/phase10_report.md`` §10.1 until measured on fabricated fingertips):

* material: platinum-cure silicone of Shore A hardness ``H``, incompressible (nu = 0.5), Young's modulus from
  Gent's relation (A. N. Gent, Rubber Chem. Technol. 31, 1958):
  ``E = 0.0981 (56 + 7.62336 H) / (0.137505 (254 - 2.54 H))`` MPa;
* contact law: the rigid 15 mm glass sphere indenting the pad on a capsule cap (sphere-sphere geometry, measured on
  the exported states: 76 % of three-contact states contact a hemispherical cap), effective radius
  ``R* = (1/R_s + 1/r_l)^-1 = 6.60 mm``; Hertz with the bonded-thin-layer correction of Dimitriadis et al. (Biophys.
  J. 82, 2002, incompressible layer bonded to a rigid substrate):
  ``F = 16/9 E R*^(1/2) d^(3/2) (1 + 1.133 c + 1.283 c^2 + 0.769 c^3 + 0.0975 c^4)``, ``c = sqrt(R* d) / t``;
* patch: contact radius ``a = sqrt(R* d)`` at the reference load ``F_ref`` (the median exact-allocation normal force of
  the exported three-contact states, 0.26 N); torsional friction of a Hertzian patch ``(3 pi / 16) mu a`` (m);
  rolling resistance from hysteresis ``(3 / 16) alpha a`` (m, Johnson, Contact Mechanics §9) with loss fraction
  ``alpha = 0.2``; sliding friction ``mu = 0.6`` (the frozen task value);
* MuJoCo realization: an explicit ``<pair>`` per terminal capsule and the sphere (``condim 6``, elliptic cone) in the
  direct ``solref = (-s, -b)`` format. MuJoCo's static penetration under a normal force ``F`` is
  ``d = (1 - imp) dmax^2 A F / (imp^2 s)`` with ``A`` the pair's configuration-independent ``diagApprox`` (sum of the
  two bodies' translational ``body_invweight0``); ``s = s_u A`` per pair makes the static law independent of ``A``,
  and the impedance ``imp(d)`` (``solimp``) is fitted so that it reproduces the target law over 0.02-2 N.
  The fit is restricted to ``imp >= 0.9``: MuJoCo regularizes every row of a contact with the same impedance, and a
  soft normal row (``imp ~ 0.1``) lets the sphere creep under static friction at ``(1 - imp) a0 / (imp B_f)`` (measured
  5.3 mm/s at 0.3 N normal / 0.1 N tangential), while the noslip post-solver does not act on ``condim 4/6`` contacts in
  MuJoCo 3.15 (measured). With ``imp >= 0.9`` and the friction damping set deadbeat (``dt imp B_f = imp / dmax``),
  the creep stays at the rigid 9.3 contact's level. The normal damping ratio is ``zeta = b / (2 sqrt(s))``.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

R_SPHERE_M = 0.015
R_LINK_M = 0.0118
R_STAR_M = 1.0 / (1.0 / R_SPHERE_M + 1.0 / R_LINK_M)
F_REF_N = 0.26                      # median exact-allocation normal force over the exported three-contact states
FIT_RANGE_N = (0.02, 2.0)           # operating range of the normal force (allocations up to 2.8 N are < 0.1 %)
HYSTERESIS_LOSS = 0.2               # alpha, unfilled platinum silicone (tan delta ~ 0.05-0.1)
IMP_MIN = 0.9                       # lower bound of the fitted impedance (friction creep, see module docstring)
TIMESTEP_S = 5e-4                   # the 8.6/9.3 scene timestep


def gent_modulus_pa(shore_a: float) -> float:
    """Young's modulus (Pa) from Shore A hardness, Gent (1958)."""
    return 0.0981 * (56.0 + 7.62336 * shore_a) / (0.137505 * (254.0 - 2.54 * shore_a)) * 1e6


def target_force(d, E, t, R=R_STAR_M):
    """Normal force (N) of the pad law at indentation ``d`` (m): Hertz with the bonded-layer correction."""
    d = np.asarray(d, float)
    c = np.sqrt(R * d) / t
    return 16.0 / 9.0 * E * np.sqrt(R) * d ** 1.5 * (1.0 + 1.133 * c + 1.283 * c ** 2 + 0.769 * c ** 3 + 0.0975 * c ** 4)


def _invert(f, F, lo=1e-12, hi=0.05):
    from scipy.optimize import brentq
    if f(lo) >= F:
        return lo
    if f(hi) <= F:
        return hi
    return brentq(lambda d: f(d) - F, lo, hi, xtol=1e-16, rtol=1e-12)


def target_indentation(F, E, t, R=R_STAR_M):
    return _invert(lambda d: float(target_force(d, E, t, R)), F)


def mj_impedance(x, d0, dmax, mid, power):
    """MuJoCo's ``solimp`` impedance at ``x = |pos| / width`` (``getimpedance``, clamped as ``mjMINIMP/mjMAXIMP``)."""
    x = np.clip(np.asarray(x, float), 0.0, 1.0)
    if power == 1:
        y = x
    else:
        y = np.where(x <= mid, x ** power / mid ** (power - 1), 1.0 - (1.0 - x) ** power / (1.0 - mid) ** (power - 1))
    return np.clip(d0 + y * (dmax - d0), 1e-4, 0.9999)


def mj_static_force(d, s_u, d0, dmax, width, mid, power):
    """Static normal force per unit ``A`` scaling: ``F = s_u imp^2 d / ((1 - imp) dmax^2)`` (``s = s_u A``)."""
    imp = mj_impedance(np.asarray(d, float) / width, d0, dmax, mid, power)
    return s_u * imp ** 2 * np.asarray(d, float) / ((1.0 - imp) * dmax ** 2)


@dataclass(frozen=True)
class PadParams:
    """One compliant-pad parameter set: material, thickness and its MuJoCo realization."""
    shore_a: float
    thickness_m: float
    E_pa: float
    s_u: float            # MuJoCo stiffness per unit diagApprox (s = s_u A, direct solref format)
    d0: float             # solimp
    dmax: float
    width_m: float
    mid: float
    power: float
    zeta: float           # normal damping ratio (b = 2 zeta sqrt(s))
    mu: float
    torsion_m: float      # torsional friction coefficient (m)
    rolling_m: float      # rolling friction coefficient (m)
    patch_radius_m: float  # at F_REF_N
    indentation_ref_m: float
    fit_max_rel_err: float  # max relative indentation error of the realization over FIT_RANGE_N
    friction_damping_b: float  # direct-format solreffriction damping (deadbeat at the scene timestep)
    label: str = ""

    def pair_attrs(self, A: float) -> str:
        """``<pair>`` attributes for a (terminal capsule, sphere) pair whose ``diagApprox`` is ``A``."""
        s = self.s_u * A
        b = 2.0 * self.zeta * np.sqrt(s)
        return (f'condim="6" friction="{self.mu} {self.mu} {self.torsion_m:.6g} {self.rolling_m:.6g} {self.rolling_m:.6g}" '
                f'solref="{-s:.8g} {-b:.8g}" solimp="{self.d0:.8g} {self.dmax:.8g} {self.width_m:.8g} {self.mid:.8g} {self.power:.8g}" '
                f'solreffriction="-1 {-self.friction_damping_b:.8g}"')

    def as_dict(self):
        return asdict(self)


def fit_realization(E, t, imp_min=IMP_MIN, F_range=FIT_RANGE_N, n=40):
    """Least-squares fit (log indentation) of MuJoCo's static law to the pad law over ``F_range`` with
    ``imp >= imp_min``. Returns ``(s_u, d0, dmax, width, mid, power, max_rel_err)``."""
    from scipy.optimize import least_squares
    Fs = np.geomspace(F_range[0], F_range[1], n)
    d_t = np.array([target_indentation(F, E, t) for F in Fs])

    def unpack(z):
        s_u, d0, fr, width, mid, power = z
        return s_u, d0, d0 + fr * (0.999 - d0), width, mid, power

    def res(z):
        s_u, d0, dmax, width, mid, power = unpack(z)
        d_m = np.array([_invert(lambda d: float(mj_static_force(d, s_u, d0, dmax, width, mid, power)), F) for F in Fs])
        return np.log(d_m) - np.log(d_t)

    lb = [1.0, imp_min, 0.0, 1e-5, 0.05, 1.0]
    ub = [1e8, 0.99, 1.0, 5e-3, 0.95, 6.0]
    best = None
    for z0 in ([300.0, 0.92, 0.5, 4e-4, 0.5, 1.5], [100.0, 0.9, 0.8, 3e-4, 0.3, 2.0], [1000.0, 0.95, 0.5, 6e-4, 0.7, 1.0]):
        r = least_squares(res, z0, bounds=(lb, ub), max_nfev=3000, x_scale=[100.0, 0.01, 0.1, 1e-4, 0.1, 0.5])
        if best is None or r.cost < best.cost:
            best = r
    s_u, d0, dmax, width, mid, power = unpack(best.x)
    return s_u, d0, dmax, width, mid, power, float(np.exp(np.abs(best.fun)).max() - 1.0)


def make_pad(shore_a: float, thickness_m: float, zeta: float = 1.0, mu: float = 0.6, f_ref: float = F_REF_N,
             alpha: float = HYSTERESIS_LOSS, timestep: float = TIMESTEP_S, label: str = "") -> PadParams:
    E = gent_modulus_pa(shore_a)
    s_u, d0, dmax, width, mid, power, err = fit_realization(E, thickness_m)
    d_ref = target_indentation(f_ref, E, thickness_m)
    a = float(np.sqrt(R_STAR_M * d_ref))
    return PadParams(shore_a=shore_a, thickness_m=thickness_m, E_pa=E, s_u=s_u, d0=d0, dmax=dmax, width_m=width, mid=mid, power=power,
                     zeta=zeta, mu=mu, torsion_m=3.0 * np.pi / 16.0 * mu * a, rolling_m=3.0 / 16.0 * alpha * a, patch_radius_m=a,
                     indentation_ref_m=d_ref, fit_max_rel_err=err, friction_damping_b=1.0 / timestep, label=label or f"{shore_a:g}A_{thickness_m * 1e3:g}mm")


CENTRAL = "20A_2mm"
GRID = [(h, t) for h in (10.0, 20.0, 40.0) for t in (1e-3, 2e-3, 3e-3)]      # 3 hardness x 3 thickness, central included


def load_pads(path="dkarma/configs/compliant_pads.json") -> dict:
    """The frozen parameter sets (label -> ``PadParams``)."""
    import json
    from pathlib import Path
    rec = json.loads(Path(path).read_text())
    return {k: PadParams(**v) for k, v in rec["pads"].items()}


def main():
    """Fit and freeze the Phase 10.1 parameter sets: the 3 x 3 grid (central 20A / 2 mm) and the damping
    diagnostic D2 (central law, zeta 0.3)."""
    import argparse
    import json
    import time
    from pathlib import Path
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="dkarma/configs/compliant_pads.json")
    args = ap.parse_args()
    pads = {}
    for h, t in GRID:
        p = make_pad(h, t)
        pads[p.label] = p.as_dict()
        print(f"{p.label}: E {p.E_pa / 1e6:.3f} MPa, d(0.26 N) {p.indentation_ref_m * 1e3:.3f} mm, a {p.patch_radius_m * 1e3:.2f} mm, "
              f"torsion {p.torsion_m * 1e3:.3f} mm, rolling {p.rolling_m * 1e6:.1f} um, fit error {p.fit_max_rel_err * 100:.1f} %", flush=True)
    c = PadParams(**pads[CENTRAL])
    d2 = PadParams(**{**c.as_dict(), "zeta": 0.3, "label": f"{CENTRAL}_zeta0.3"})
    pads[d2.label] = d2.as_dict()
    import mujoco
    import scipy
    rec = dict(created=time.strftime("%Y-%m-%d %H:%M %Z"), central=CENTRAL, grid=[f"{h:g}A_{t * 1e3:g}mm" for h, t in GRID], diagnostics=[d2.label],
               assumptions=dict(R_star_m=R_STAR_M, F_ref_n=F_REF_N, fit_range_n=list(FIT_RANGE_N), hysteresis_loss=HYSTERESIS_LOSS, imp_min=IMP_MIN,
                                timestep_s=TIMESTEP_S, modulus="Gent (1958) Shore A", law="Hertz + Dimitriadis et al. (2002) bonded incompressible layer"),
               versions=dict(mujoco=mujoco.__version__, scipy=scipy.__version__, numpy=np.__version__), pads=pads)
    Path(args.out).write_text(json.dumps(rec, indent=1))
    print(f"wrote {args.out}")


__all__ = ["PadParams", "make_pad", "fit_realization", "gent_modulus_pa", "target_force", "target_indentation", "mj_impedance",
           "mj_static_force", "load_pads", "R_STAR_M", "F_REF_N", "FIT_RANGE_N", "CENTRAL", "GRID"]


if __name__ == "__main__":
    main()
