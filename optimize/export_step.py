"""Export a design (URDF + robot YAML) to STEP for Onshape, using the metric's own geometry.

    python optimize/export_step.py --design-dir optimize/runs/A1/best_design
    python optimize/export_step.py --design-dir optimize/runs/A1/final --pose seed   # posed at the seed pinch, with the sphere

Writes, next to the URDF (or --out-dir):
  <name>.step        one STEP assembly with NAMED solids at the chosen pose:
                       <link>              capsule exactly as KaRMA models it (radius = scaled link radius, tip = tip_length_m)
                       <joint>__axis       a reference pin along the joint's rotation axis (cylinder; mate connectors snap to it)
                       <joint>__arrow      a small cone on the +axis end (right-hand positive rotation)
                       palm                a bar between the knuckles (root joints) on the base link
                       test_sphere         the metric's sphere at the seed centre (only with --pose seed)
  <name>_links/<link>.step   each link capsule alone, in its OWN link frame (origin = inbound joint, +x along the chain)
                       for building an Onshape assembly with revolute mates
  <name>_joints.csv  joint table: parent, child, origin and axis in the parent-link frame (URDF), limits in degrees,
                       and world pose at the exported configuration
Units: millimetres. Onshape: Import -> "Combine into single part studio" off (keep as assembly) to preserve names.
"""
from __future__ import annotations
import argparse, csv, sys
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import cadquery as cq
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from karma.config import load_config                                  # noqa: E402
from karma.robot import load_robot, update_kinematics, capsule_endpoints_world  # noqa: E402
from optimize.viz import hand_pose                                    # noqa: E402

M = 1e3  # metres -> mm


def capsule_solid(a, b, r):
    """Capsule between world points a, b (mm) with radius r (mm)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    seg = b - a; L = float(np.linalg.norm(seg))
    sph_a = cq.Workplane("XY").sphere(r).translate(tuple(a))
    if L < 1e-6:
        return sph_a
    d = seg / L
    cyl = cq.Solid.makeCylinder(r, L, cq.Vector(*a), cq.Vector(*d))
    sph_b = cq.Workplane("XY").sphere(r).translate(tuple(b))
    return cq.Workplane("XY").add(cyl).union(sph_a).union(sph_b)


def pin_solid(origin, axis, r, length):
    o = np.asarray(origin, float); d = np.asarray(axis, float); d = d / np.linalg.norm(d)
    start = o - 0.5 * length * d
    return cq.Workplane("XY").add(cq.Solid.makeCylinder(r, length, cq.Vector(*start), cq.Vector(*d)))


def cone_solid(tip_base, axis, r, h):
    o = np.asarray(tip_base, float); d = np.asarray(axis, float); d = d / np.linalg.norm(d)
    return cq.Workplane("XY").add(cq.Solid.makeCone(r, 0.0, h, cq.Vector(*o), cq.Vector(*d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design-dir", required=True, help="directory holding <name>.urdf and robot_<name>.yaml")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--pose", choices=["neutral", "seed"], default="neutral", help="seed needs a current.pkl in the directory")
    ap.add_argument("--pin-length-r", type=float, default=4.6, help="axis pin length in link radii")
    a = ap.parse_args()
    dd = Path(a.design_dir); out = Path(a.out_dir or dd); out.mkdir(parents=True, exist_ok=True)
    yml = next(dd.glob("robot_*.yaml")); cfg = load_config(REPO / "karma_config.yaml", yml); ctx = load_robot(cfg.urdf_path, cfg)
    name = cfg.robot_name
    model = ctx.model
    q = np.zeros(model.nq); centre = None; pair = None
    if a.pose == "seed":
        from karma.storage import load_result
        res, _ = load_result(dd / "current.pkl"); q = np.asarray(res.seed_q); centre = np.asarray(res.seed_centre); pair = tuple(res.seed_contact_pair)
    g = hand_pose(ctx, cfg, q, centre, pair)
    r = cfg.link_radius_m * M
    thumb = set(cfg.thumb_contact_links)

    asm = cq.Assembly(name=name)
    for lname, pa, pb, is_thumb, _ in g["capsules"]:
        asm.add(capsule_solid(pa * M, pb * M, r), name=lname, color=cq.Color(0.88, 0.48, 0.16) if is_thumb else cq.Color(0.18, 0.44, 0.82))
    pin_r, pin_len = 0.18 * r, a.pin_length_r * r
    for j in g["joints"]:
        o, ax = j["origin"] * M, j["axis"]
        asm.add(pin_solid(o, ax, pin_r, pin_len), name=f"{j['name']}__axis", color=cq.Color(0.15, 0.15, 0.15))
        asm.add(cone_solid(o + 0.5 * pin_len * ax, ax, 2.2 * pin_r, 3.0 * pin_r), name=f"{j['name']}__arrow", color=cq.Color(0.9, 0.1, 0.3))
    K = g["knuckles"] * M
    if len(K) >= 2:
        asm.add(capsule_solid(K[0], K[1], 0.45 * r), name="palm", color=cq.Color(0.45, 0.45, 0.45))
    if centre is not None:
        asm.add(cq.Workplane("XY").sphere(cfg.sphere_radius_m * M).translate(tuple(centre * M)), name="test_sphere", color=cq.Color(0.25, 0.25, 0.25))
    step_path = out / f"{name}.step"
    asm.save(str(step_path), exportType="STEP")
    print(f"wrote {step_path}  ({len(g['capsules'])} links, {len(g['joints'])} joint pins, pose={a.pose})")

    # per-link STEP in the link's own frame: origin at the inbound joint, +x along the capsule axis
    link_dir = out / f"{name}_links"; link_dir.mkdir(exist_ok=True)
    for lname, pa, pb, is_thumb, _ in g["capsules"]:
        L = float(np.linalg.norm(pb - pa)) * M
        solid = capsule_solid(np.zeros(3), np.array([L, 0.0, 0.0]), r)
        cq.exporters.export(solid, str(link_dir / f"{lname}.step"))
    print(f"wrote {len(g['capsules'])} per-link STEP files to {link_dir}/ (local frames, +x along the link)")

    # joint table: URDF (parent-frame) data plus world pose at the exported configuration
    root = ET.parse(str(cfg.urdf_path)).getroot()
    urdf = {}
    for jt in root.findall("joint"):
        o = jt.find("origin"); axl = jt.find("axis"); lim = jt.find("limit")
        urdf[jt.attrib["name"]] = dict(
            type=jt.attrib.get("type"), parent=jt.find("parent").attrib["link"], child=jt.find("child").attrib["link"],
            xyz=o.attrib.get("xyz", "0 0 0") if o is not None else "0 0 0", rpy=o.attrib.get("rpy", "0 0 0") if o is not None else "0 0 0",
            axis=axl.attrib.get("xyz", "0 0 1") if axl is not None else "", lower=lim.attrib.get("lower", "") if lim is not None else "",
            upper=lim.attrib.get("upper", "") if lim is not None else "")
    csv_path = out / f"{name}_joints.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["joint", "type", "parent_link", "child_link", "origin_xyz_in_parent_mm", "origin_rpy_in_parent_rad", "axis_in_joint_frame",
                    "lower_deg", "upper_deg", "q_at_export_deg", "world_origin_mm", "world_axis", "link_radius_mm"])
        for j in g["joints"]:
            u = urdf.get(j["name"], {})
            xyz_mm = " ".join(f"{float(v) * M:.3f}" for v in u.get("xyz", "0 0 0").split())
            w.writerow([j["name"], u.get("type", ""), u.get("parent", ""), u.get("child", ""), xyz_mm, u.get("rpy", ""), u.get("axis", ""),
                        f"{np.degrees(j['lo']):.2f}", f"{np.degrees(j['hi']):.2f}", f"{np.degrees(j['q']):.2f}",
                        " ".join(f"{v:.3f}" for v in j["origin"] * M), " ".join(f"{v:.4f}" for v in j["axis"]), f"{r:.3f}"])
    print(f"wrote {csv_path}")
    # tip lengths for reference
    for lname, pa, pb, _, _ in g["capsules"]:
        print(f"   {lname:<10s} capsule axis {float(np.linalg.norm(pb - pa)) * M:6.2f} mm  (+ radius {r:.2f} mm at each end)")


if __name__ == "__main__":
    main()
