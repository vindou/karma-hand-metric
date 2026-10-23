"""Kinematic features computable from any KaRMA robot YAML + URDF, for the
16 bundled hands and for optimized designs alike (arm A analysis).

    python optimize/features.py --out optimize/runs/features.csv [--designs optimize/runs/T/best.json ...]

Features (pinch fingers only):
  knuckle_dist_mm        thumb-index knuckle distance (first active joints, neutral pose)
  thumb_wire_mm / index_wire_mm   knuckle-to-tip chain length
  L_ref_mm               shipped L_ref
  knuckle_over_wire      knuckle_dist / mean wire        (palm compactness)
  distal_frac_thumb/_index  tip segment / wire           (how much of the finger is the last segment)
  rom_mean_deg / rom_max_deg   joint range over active joints
  obliq_thumb_deg / obliq_index_deg   mean |angle between joint axis and the plane perpendicular to the link|
                         (0 = pure flexion/abduction axes, 90 = pure twist axes)
  n_dof                  active thumb + index joints
  opposition_deg         angle between thumb and index knuckle->tip directions at neutral
  tip_gap_neutral_over_L tip-to-tip distance at neutral / L_ref
"""
from __future__ import annotations
import argparse, csv, json, math, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import yaml
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from karma.config import load_config                                           # noqa: E402
from karma.robot import load_robot, update_kinematics, capsule_endpoints_world   # noqa: E402
from karma.lref import compute_lref_breakdown                                   # noqa: E402
from optimize.lref_variants import lref_pinch_breakdown                         # noqa: E402
from optimize.evaluate_full import load_design                                  # noqa: E402


def _urdf_joints(urdf_path):
    root = ET.parse(str(urdf_path)).getroot()
    J = {}
    for j in root.findall("joint"):
        o = j.find("origin"); ax = j.find("axis"); lim = j.find("limit")
        J[j.attrib["name"]] = dict(
            type=j.attrib.get("type", "fixed"), parent=j.find("parent").attrib["link"], child=j.find("child").attrib["link"],
            xyz=np.fromstring(o.attrib.get("xyz", "0 0 0"), sep=" ") if o is not None else np.zeros(3),
            axis=np.fromstring(ax.attrib.get("xyz", "0 0 1"), sep=" ") if ax is not None else np.array([0, 0, 1.0]),
            lo=float(lim.attrib.get("lower", 0)) if lim is not None else 0.0,
            hi=float(lim.attrib.get("upper", 0)) if lim is not None else 0.0,
            mimic=j.find("mimic") is not None)
    return J


def finger_features(robot: dict, J: dict, finger: str):
    """Per-finger: ROMs, obliquities, distal fraction (from the URDF chain of active joints)."""
    active = robot["fingers"][finger]["active_joints"]
    seed_link = robot["seed"][f"{finger}_link"]
    links_cfg = robot.get("links", {}) or {}
    by_parent = {}
    for n, j in J.items():
        by_parent.setdefault(j["parent"], []).append(n)
    roms, obliq = [], []
    for k, jn in enumerate(active):
        j = J[jn]
        roms.append(math.degrees(j["hi"] - j["lo"]))
        # link direction in this joint's child frame: towards the next joint in the chain (through fixed joints),
        # or the tip for the last one
        nxt = None
        if k + 1 < len(active):
            # walk down from child link to the next active joint accumulating fixed offsets
            link = j["child"]; acc = np.zeros(3); found = None; depth = 0
            while found is None and depth < 6:
                depth += 1
                cands = by_parent.get(link, [])
                if not cands: break
                if active[k + 1] in cands:
                    found = acc + J[active[k + 1]]["xyz"]; break
                # follow the child that leads to the next active joint (take the one with the longest offset as heuristic)
                mv = [c for c in cands if J[c]["type"] != "fixed"]
                c = (mv or cands)[0]
                acc = acc + J[c]["xyz"]; link = J[c]["child"]
            nxt = found
        if nxt is None or np.linalg.norm(nxt) < 1e-9:
            tl = links_cfg.get(seed_link, {}).get("tip_length_m") if isinstance(links_cfg.get(seed_link), dict) else None
            nxt = np.array([tl or 0.02, 0, 0])  # direction unknown for terminal links of bundled hands -> skip obliquity
            if k == len(active) - 1 and finger_features.skip_terminal_obliq:
                continue
        d = nxt / np.linalg.norm(nxt); a = j["axis"] / np.linalg.norm(j["axis"])
        obliq.append(math.degrees(math.asin(min(1.0, abs(float(np.dot(a, d)))))))
    return roms, obliq


finger_features.skip_terminal_obliq = True


def hand_features(yaml_path: Path) -> dict:
    robot = yaml.safe_load(Path(yaml_path).read_text())
    J = _urdf_joints(Path(yaml_path).parent / robot["urdf_path"])
    sh = compute_lref_breakdown(yaml_path); pb = lref_pinch_breakdown(yaml_path)
    wt, wi = pb["finger_wires_m"]
    f = {"L_ref_mm": sh["L_ref_m"] * 1e3, "knuckle_dist_mm": pb["mxPr_m"] * 1e3, "thumb_wire_mm": wt * 1e3, "index_wire_mm": wi * 1e3,
         "knuckle_over_wire": pb["mxPr_m"] / max(pb["mF_m"], 1e-9), "wire_over_L": pb["mF_m"] / sh["L_ref_m"]}
    links_cfg = robot.get("links", {}) or {}
    for fin, w in (("thumb", wt), ("index", wi)):
        tl = links_cfg.get(robot["seed"][f"{fin}_link"], {})
        tl = tl.get("tip_length_m") if isinstance(tl, dict) else None
        f[f"distal_frac_{fin}"] = (tl / w) if (tl and w > 0) else float("nan")
    rt, ot = finger_features(robot, J, "thumb"); ri, oi = finger_features(robot, J, "index")
    f["rom_mean_deg"] = float(np.mean(rt + ri)); f["rom_max_deg"] = float(np.max(rt + ri))
    f["obliq_thumb_deg"] = float(np.mean(ot)) if ot else float("nan"); f["obliq_index_deg"] = float(np.mean(oi)) if oi else float("nan")
    f["n_dof"] = len(rt) + len(ri)
    # neutral-pose geometry via the metric's own robot model
    cfg = load_config(REPO / "karma_config.yaml", yaml_path); ctx = load_robot(cfg.urdf_path, cfg)
    q0 = np.zeros(ctx.model.nq); update_kinematics(ctx, q0)
    names = list(ctx.model.names)
    kt = np.array(ctx.data.oMi[names.index(cfg.thumb_joint_names[0])].translation)
    ki = np.array(ctx.data.oMi[names.index(cfg.index_joint_names[0])].translation)
    _, tt = capsule_endpoints_world(ctx, cfg.seed_thumb_link); _, ti = capsule_endpoints_world(ctx, cfg.seed_index_link)
    dt, di = tt - kt, ti - ki
    f["opposition_deg"] = math.degrees(math.acos(np.clip(np.dot(dt, di) / (np.linalg.norm(dt) * np.linalg.norm(di) + 1e-12), -1, 1)))
    f["tip_gap_neutral_over_L"] = float(np.linalg.norm(tt - ti) / sh["L_ref_m"])
    return f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="*", default=[])
    ap.add_argument("--out", default=str(REPO / "optimize" / "runs" / "features.csv"))
    a = ap.parse_args()
    pub = {r["robot"]: r for r in yaml.safe_load((REPO / "results/16_hand_batch/summary.yaml").read_text())["rows"]}
    rows = []
    for p in sorted((REPO / "robots").glob("robot_*.yaml")):
        if "template" in p.name: continue
        name = p.stem[6:]
        try:
            f = hand_features(p)
        except Exception as e:
            print(f"{name}: {e}"); continue
        rows.append({"hand": name, "kind": "published", "KaRMA_T": pub[name]["KaRMA_T"], "KaRMA_R": pub[name]["KaRMA_R"], **f})
    for d in a.designs:
        d = Path(d); hand = load_design(str(d)); run = d.resolve().parent if d.name == "best.json" else d.parent
        tmp = run / "features_tmp"; tmp.mkdir(exist_ok=True); _, yp = hand.write(tmp)
        f = hand_features(yp)
        sj = run / "final" / "summary.json"
        s = json.loads(sj.read_text()) if sj.exists() else {}
        rows.append({"hand": run.name, "kind": "optimized", "KaRMA_T": s.get("KaRMA_T", float("nan")), "KaRMA_R": s.get("KaRMA_R", float("nan")), **f})
    keys = list(rows[0].keys())
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(f"wrote {a.out} ({len(rows)} rows)")
    # Spearman of each feature with published T across the bundled hands
    from scipy.stats import spearmanr
    pubrows = [r for r in rows if r["kind"] == "published"]
    T = np.array([r["KaRMA_T"] for r in pubrows])
    print(f"\n{'feature':<26s} {'rho(T)':>7s}   {'published median [min,max]':<32s} optimized designs")
    for k in keys[4:]:
        v = np.array([r[k] for r in pubrows], dtype=float); ok = np.isfinite(v)
        rho = spearmanr(v[ok], T[ok]).correlation if ok.sum() > 3 else float("nan")
        opt = ", ".join(f"{r[k]:.2f}" for r in rows if r["kind"] == "optimized")
        print(f"{k:<26s} {rho:7.2f}   {np.nanmedian(v):8.2f} [{np.nanmin(v):7.2f},{np.nanmax(v):7.2f}]   {opt}")


if __name__ == "__main__":
    main()
