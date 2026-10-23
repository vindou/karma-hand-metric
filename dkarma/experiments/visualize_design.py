"""Figures and animations of one Phase 10 design (default: the design selected by criterion v2, Full-D18 start31).

Observation only. Every simulated frame comes from ``simulation_replay.simulate_design`` with the stored 10.2 settings
(compliant pads 20A/2 mm, force-floor controller); the replayed rows are checked against the stored 10.2 rows before
anything is written. Every number comes from the Phase 10 JSON files. Rendering uses a separate copy of the MuJoCo
scene with visual-only additions (materials, lights, a floor, the palm-frame axes); the simulated model is unchanged
and the copy only receives ``qpos``.

Stages (``--stages``): ``views`` (the hand at one grasp, perspective and orthographic views with a scale bar),
``compare`` (anthropomorphic start, the run's initial design and the optimized design at the same joint angles),
``gallery`` (the 12 three-contact grasps), ``animate`` (the 10.2 replay as MP4/GIF: one seed's commands in sequence,
and all twelve commands of that seed side by side), ``charts`` (per-command success, wrench capacity, standing among
the 48 10.2 designs, load sweep).

    MUJOCO_GL=osmesa JAX_PLATFORMS=cpu python -m dkarma.experiments.visualize_design
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")      # EGL is not available to this user on linbox01; OSMesa renders on the CPU

import numpy as np  # noqa: E402

P10 = Path("workspace/phase10")
START_ARCHIVE = Path("workspace/phase9/9.3")      # the anthropomorphic start's scene design (8.6/9.3 archive)
TAU = 0.108

# palette (dataviz reference instance): fingers take categorical slots 1-3 in fixed order; status colors only for pass/fail
FINGER = {"index": "#2a78d6", "thumb": "#eb6834", "middle": "#1baf7a"}
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
GOOD, BAD, PLAN = "#0ca30c", "#d03b3b", "#6250d6"
RAMP = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab",
        "#184f95", "#104281", "#0d366b"]
CMD_LABELS = ["+ê1", "−ê1", "+ê2", "−ê2", "+ê3", "−ê3"]   # simulation_replay.export_three_contact: commands u = ±frame axes


def rgb(h, a=1.0):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)] + [a]


def paths(key):
    arm, st = key.split("/")
    k = int(st.split("_")[0].replace("start", ""))
    name = f"{arm}_start{k}"
    sim = P10 / "10.2/sim/default"
    return dict(arm=arm, st=st, k=k, name=name, archive=sim, sim=sim / "20A_2mm" / f"simulation_force__{name}.json",
                suite=P10 / "10.2/suite" / f"{arm}__{st}" / "suite.json", sweep=P10 / "10.2/sweeps" / f"{arm}__{st}.json",
                run=P10 / "10.2" / arm / st, comparison=P10 / "10.2/comparison.json", summary=P10 / "10.2/summary.json")


def scene_design(design_json: Path) -> dict:
    """The MJCF-ready design dict (``fingers``, ``cap_radius``, ``topology``) of a dkarma ``*.design.json``."""
    from dkarma.export import design_from_dict
    from dkarma.core import params_to_fingers
    topo, params, _ = design_from_dict(json.loads(Path(design_json).read_text()))
    return dict(fingers=params_to_fingers(topo, params), cap_radius=np.asarray(params.cap_radius).tolist(),
                topology=dict(fingers=list(topo.fingers), n_joints=list(topo.n_joints), pair=list(topo.pair)))


# ── visual scene ─────────────────────────────────────────────────────────────

def visual_xml(design: dict, floor_z: float = -0.115) -> str:
    """The 8.6/9.3 scene of ``design`` with visual-only additions: offscreen buffer, lights, materials, a floor (group 3,
    no collisions) and the palm-frame axes at the origin (group 2). Same joints and ``qpos`` layout as the simulated scene."""
    from dkarma.core import load_task_spec
    from dkarma.experiments.simulation_replay import mjcf_from_design
    xml = mjcf_from_design(design, load_task_spec(), TAU)
    f = lambda h: " ".join(f"{c:.3f}" for c in rgb(h)[:3])  # noqa: E731
    vis = ('  <visual><headlight ambient=".36 .36 .36" diffuse=".38 .38 .38" specular=".04 .04 .04"/>'
           '<quality shadowsize="4096" offsamples="8"/><global offwidth="1920" offheight="1440"/><map znear="0.002" zfar="40"/></visual>\n'
           '  <asset>\n'
           '    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.988 0.988 0.984" rgb2="0.988 0.988 0.984" width="256" height="256"/>\n'
           '    <texture name="grid" type="2d" builtin="checker" rgb1=".955 .955 .948" rgb2=".925 .925 .915" width="512" height="512" mark="edge" markrgb=".88 .88 .86"/>\n'
           '    <material name="floor" texture="grid" texrepeat="10 10" reflectance="0.0" emission="0.45"/>\n'
           '    <texture name="ball" type="cube" builtin="checker" rgb1="0.97 0.97 0.96" rgb2="0.22 0.22 0.24" width="128" height="128"/>\n'
           '    <material name="ball" texture="ball" specular="0.3" shininess="0.4"/>\n'
           + "".join(f'    <material name="m_{fn}" rgba="{f(c)} 1" specular="0.2" shininess="0.3"/>\n' for fn, c in FINGER.items())
           + '  </asset>\n')
    xml = xml.replace('<compiler angle="radian"/>', '<compiler angle="radian"/>\n' + vis)
    for fn in FINGER:
        xml = re.sub(rf'(name="{fn}_c\d+")', rf'\1 material="m_{fn}"', xml)
    xml = xml.replace('name="sphere_geom"', 'name="sphere_geom" material="ball"')
    world = (f'    <light name="key" pos="0.20 -0.12 0.32" dir="-0.16 0.12 -0.38" cutoff="45" exponent="2" castshadow="true" diffuse=".42 .42 .42" specular=".1 .1 .1"/>\n'
             f'    <geom name="floor" type="plane" pos="0.03 0 {floor_z}" size="0.2 0.2 0.01" material="floor" contype="0" conaffinity="0" group="3"/>\n'
             '    <geom name="palm_x" type="capsule" fromto="0 0 0 0.025 0 0" size="0.0011" rgba=".85 .2 .2 1" contype="0" conaffinity="0" group="2"/>\n'
             '    <geom name="palm_y" type="capsule" fromto="0 0 0 0 0.025 0" size="0.0011" rgba=".15 .65 .15 1" contype="0" conaffinity="0" group="2"/>\n'
             '    <geom name="palm_z" type="capsule" fromto="0 0 0 0 0 0.025" size="0.0011" rgba=".15 .3 .85 1" contype="0" conaffinity="0" group="2"/>\n')
    return xml.replace("  <worldbody>\n", "  <worldbody>\n" + world, 1)


_ALIVE = []        # OSMesa: a renderer freed while another is in use corrupts the latter's output, so none is freed


class Stage:
    """A rendering copy of a design's scene."""

    def __init__(self, design: dict, width: int, height: int, floor: bool = True, ortho_extent_m: float | None = None):
        import mujoco
        self.mj = mujoco
        self.m = mujoco.MjModel.from_xml_string(visual_xml(design))
        if ortho_extent_m is not None:
            self.m.vis.global_.orthographic = 1
            self.m.vis.global_.fovy = ortho_extent_m           # orthographic: the vertical extent of the view in metres
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, height, width, max_geom=20000)
        self.opt = mujoco.MjvOption(); self.opt.geomgroup[3] = int(floor)
        topo = design["topology"]
        self.order = [f"{fn}_j{k}" for fn, nj in zip(topo["fingers"], topo["n_joints"]) for k in range(nj)]
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in self.order])
        self.sq = self.m.joint("sphere_free").qposadr[0]
        self.w, self.h = width, height
        _ALIVE.append(self)

    def pose(self, q=None, p=None, quat=(1.0, 0.0, 0.0, 0.0), qpos=None):
        if qpos is not None:
            self.d.qpos[:] = qpos
        else:
            self.d.qpos[:] = 0.0
            if q is not None:
                self.d.qpos[self.qadr] = q
            self.d.qpos[self.sq:self.sq + 3] = p if p is not None else [0.0, 0.0, -10.0]   # no object: far below the floor
            self.d.qpos[self.sq + 3:self.sq + 7] = quat
        self.mj.mj_forward(self.m, self.d)

    def camera(self, lookat, azimuth, elevation, distance=0.25):
        c = self.mj.MjvCamera(); c.type = self.mj.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = lookat; c.azimuth = azimuth; c.elevation = elevation; c.distance = distance
        return c

    def render(self, cam, decor=None):
        self.r.update_scene(self.d, cam, self.opt)
        if decor is not None:
            decor(self.r.scene)
        img = self.r.render().copy()
        gc = self.r.scene.camera[0]
        self.fwd, self.up = np.array(gc.forward), np.array(gc.up)
        self.right = np.cross(self.fwd, self.up)
        return img


def _geom(scene, gtype, size, pos, mat, rgba):
    import mujoco
    if scene.ngeom >= scene.maxgeom:
        return None
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(), np.asarray(rgba, np.float32))
    scene.ngeom += 1
    return g


def _line(scene, a, b, radius, rgba):
    import mujoco
    g = _geom(scene, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.eye(3), rgba)
    if g is not None:
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, np.asarray(a, float), np.asarray(b, float))


def _ball(scene, pos, radius, rgba):
    import mujoco
    _geom(scene, mujoco.mjtGeom.mjGEOM_SPHERE, [radius, radius, radius], pos, np.eye(3), rgba)


# ── 2-D overlays ─────────────────────────────────────────────────────────────

def font(size, bold=False):
    from PIL import ImageFont
    from matplotlib import font_manager
    return ImageFont.truetype(font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight="bold" if bold else "normal")), size)


def _hex(h, a=255):
    return tuple(int(c * 255) for c in rgb(h)[:3]) + (a,)


def _panel(img, box, alpha=228):
    from PIL import Image, ImageDraw
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    ImageDraw.Draw(ov).rectangle(box, fill=_hex(SURF, alpha))
    return Image.alpha_composite(img.convert("RGBA"), ov)


def _axis_name(v):
    i = int(np.argmax(np.abs(v)))
    return ("+" if v[i] > 0 else "−") + "xyz"[i]


def _view_glyph(draw, stage, x0, y0, fnt):
    """Image-plane directions of the palm-frame axes and of gravity for the current (orthographic) view."""
    right, up = _axis_name(stage.right), _axis_name(stage.up)
    draw.line([(x0, y0), (x0 + 34, y0)], fill=_hex(INK2), width=2); draw.polygon([(x0 + 40, y0), (x0 + 32, y0 - 4), (x0 + 32, y0 + 4)], fill=_hex(INK2))
    draw.line([(x0, y0), (x0, y0 - 34)], fill=_hex(INK2), width=2); draw.polygon([(x0, y0 - 40), (x0 - 4, y0 - 32), (x0 + 4, y0 - 32)], fill=_hex(INK2))
    draw.text((x0 + 44, y0 - 8), right, font=fnt, fill=_hex(INK2)); draw.text((x0 - 6, y0 - 60), up, font=fnt, fill=_hex(INK2))
    g = np.array([np.dot([0, 0, -1], stage.right), np.dot([0, 0, -1], stage.up)])
    gx = x0 + 110
    if np.linalg.norm(g) < 0.3:                 # gravity along the view axis
        into = np.dot([0, 0, -1], stage.fwd) > 0
        draw.ellipse([gx - 9, y0 - 29, gx + 9, y0 - 11], outline=_hex(INK2), width=2)
        if into:
            draw.line([(gx - 6, y0 - 26), (gx + 6, y0 - 14)], fill=_hex(INK2), width=2); draw.line([(gx - 6, y0 - 14), (gx + 6, y0 - 26)], fill=_hex(INK2), width=2)
        else:
            draw.ellipse([gx - 3, y0 - 23, gx + 3, y0 - 17], fill=_hex(INK2))
        draw.text((gx + 18, y0 - 28), "gravity " + ("into" if into else "out of") + " page", font=fnt, fill=_hex(INK2))
    else:
        g = g / np.linalg.norm(g); a = (gx, y0 - 20); b = (gx + 30 * g[0], y0 - 20 - 30 * g[1])
        draw.line([a, b], fill=_hex(INK2), width=2)
        n = np.array([-g[1], -g[0]]) * 4
        tip = (gx + 36 * g[0], y0 - 20 - 36 * g[1])
        draw.polygon([tip, (b[0] + n[0], b[1] + n[1]), (b[0] - n[0], b[1] - n[1])], fill=_hex(INK2))
        draw.text((gx + 16, y0 - 8), "gravity", font=fnt, fill=_hex(INK2))


def _scale_bar(draw, x0, y0, px_per_m, fnt, length_mm=20):
    L = px_per_m * length_mm * 1e-3
    draw.line([(x0, y0), (x0 + L, y0)], fill=_hex(INK), width=3)
    for x in (x0, x0 + L):
        draw.line([(x, y0 - 6), (x, y0 + 6)], fill=_hex(INK), width=2)
    draw.text((x0 + L + 10, y0 - 9), f"{length_mm} mm", font=fnt, fill=_hex(INK))


def _finger_legend(draw, x, y, fnt, order=("index", "middle", "thumb")):
    for fn in order:
        draw.rounded_rectangle([x, y + 3, x + 22, y + 15], radius=4, fill=_hex(FINGER[fn]))
        draw.text((x + 30, y), fn, font=fnt, fill=_hex(INK)); x += 30 + draw.textlength(fn, font=fnt) + 26
    return x


# ── data ─────────────────────────────────────────────────────────────────────

class Data:
    def __init__(self, key):
        from dkarma.experiments.simulation_replay import load_trajectories
        self.key, self.P = key, paths(key)
        self.design = json.loads((self.P["archive"] / f"{self.P['name']}_design.json").read_text())
        self.sim = json.loads(self.P["sim"].read_text())[self.P["name"]]
        self.rows = {(r["seed"], r["kind"], r["command"]): r for r in self.sim["rows"]}
        self.trajs = {(t["seed"], t["kind"], t["command"]): t for t in load_trajectories(self.P["archive"], self.P["name"])}
        suite = json.loads(self.P["suite"].read_text())
        self.suite = suite
        dflt = suite["per_seed_config"]["default"]
        self.alpha = np.asarray(dflt["wrench_alpha_3"]); self.seed_q = np.asarray(dflt["seed_q_3"]); self.seed_p = np.asarray(dflt["seed_p_3"])
        self.seeds = sorted({s for s, _, _ in self.rows})
        for s in self.seeds:                    # the export's seeds are the suite's default-configuration seeds, in order
            assert np.array_equal(self.trajs[(s, "T", 0)]["q"][0], self.seed_q[s]) and np.array_equal(self.trajs[(s, "T", 0)]["p"][0], self.seed_p[s])
        self.topo = self.design["topology"]
        self.full_caps = [f"{fn}_c{k}" for fn, nj in zip(self.topo["fingers"], self.topo["n_joints"]) for k in range(nj)]

    def strict(self, s, kind, u):
        return int(self.rows[(s, kind, u)]["n_sim_rot"])

    def seed_total(self, s):
        return sum(self.strict(s, kd, u) for kd in "TR" for u in range(6))

    def settle_mm(self, s):
        return 1e3 * self.rows[(s, "T", 0)]["settle_err_m"]


def display(key):
    arm, st = key.split("/")
    return {"full_d18": "Full-D18", "armAprime": "arm A′"}.get(arm, arm) + " " + st.split("_")[0]


def cmd_label(kind, u, n_prim, long=True):
    ax = CMD_LABELS[u]
    if not long:
        return f"{'T' if kind == 'T' else 'R'} {ax}"
    return (f"translate along {ax}, {n_prim} × 10 mm" if kind == "T" else f"rotate about {ax}, {n_prim} × 0.15 rad")


# ── stage: views ─────────────────────────────────────────────────────────────

LOOK = (0.035, 0.004, -0.032)


def stage_views(D: Data, out: Path, seed: int):
    from PIL import Image, ImageDraw
    tw, th, ext = 640, 480, 0.165
    q, p = D.seed_q[seed], D.seed_p[seed]
    persp = Stage(D.design, tw, th, floor=True); persp.pose(q, p)
    ortho = Stage(D.design, tw, th, floor=False, ortho_extent_m=ext); ortho.pose(q, p)
    tiles = [("3/4 view (perspective)", persp, persp.camera((0.04, 0.004, -0.045), 150, -24, 0.25)),
             ("front: from +x, looking back at the palm", ortho, ortho.camera(LOOK, 180, 0, 0.5)),
             ("side: from +y", ortho, ortho.camera(LOOK, -90, 0, 0.5)),
             ("top: from +z", ortho, ortho.camera(LOOK, 180, -89.9, 0.5)),
             ("bottom: from −z (the grasp side)", ortho, ortho.camera(LOOK, 180, 89.9, 0.5))]
    head = 92
    W, H = 3 * tw, 2 * th + head
    canvas = Image.new("RGBA", (W, H), _hex(SURF))
    dr = ImageDraw.Draw(canvas)
    f_t, f_s, f_l = font(26, True), font(17), font(16)
    dr.text((24, 16), f"{display(D.key)}: the hand at grasp seed {seed} (exact seed state; sphere ⌀ 30 mm, 35 g)", font=f_t, fill=_hex(INK))
    dr.text((24, 54), "Capsule links r = 11.8 mm as modelled; the palm itself is not modelled. Palm frame at the origin: "
                      "x red, y green, z blue. Gravity along −z.", font=f_s, fill=_hex(INK2))
    for i, (label, st, cam) in enumerate(tiles):
        img = Image.fromarray(st.render(cam)).convert("RGBA")
        if st is ortho:
            img = _panel(img, [0, th - 78, tw, th], alpha=215)
        d = ImageDraw.Draw(img)
        d.text((14, 10), label, font=f_l, fill=_hex(INK))
        if st is ortho:
            _view_glyph(d, st, 26, th - 22, font(14)); _scale_bar(d, tw - 190, th - 26, th / ext, font(14))
        canvas.paste(img, ((i % 3) * tw, head + (i // 3) * th))
    # sixth tile: legend and dimensions
    x0, y0 = 2 * tw + 28, head + th + 22
    _finger_legend(dr, x0, y0, f_l)
    fp = json.loads((Path(D.P["run"]) / "optimized_morphology/optimized_morphology.design.json").read_text())["finger_params"]
    adm = D.suite["admissibility"]
    lines = [("link lengths, base → tip (mm)", True)]
    for fn in ("index", "middle", "thumb"):
        lines.append((f"  {fn:<7}" + " / ".join(f"{1e3 * x:.1f}" for x in fp[fn]["link_len"]), False))
    lines.append(("joint ranges (deg), joint 0 → 3", True))
    for fn in ("index", "middle", "thumb"):
        lines.append((f"  {fn:<7}" + "   ".join(f"{-np.degrees(a):.0f} … {np.degrees(b):.0f}".replace("-", "−") for a, b in zip(fp[fn]["lim_lo"], fp[fn]["lim_hi"])), False))
    lines.append((f"min. base separation {1e3 * adm['min_base_separation_m']:.1f} mm (≥ 24.1); reach {1e3 * adm['max_reach_m']:.1f} mm (≤ 250)", False))
    lines.append(("one XL330-M077-T per joint, ceiling 0.108 N·m", False))
    y = y0 + 40
    fm = font(15); fmb = font(15, True)
    for text, bold in lines:
        dr.text((x0, y), text, font=fmb if bold else fm, fill=_hex(INK if bold else INK2)); y += 26 if bold else 23
    canvas.convert("RGB").save(out / "hand_views.png", optimize=True)


# ── stage: compare ───────────────────────────────────────────────────────────

def stage_compare(D: Data, out: Path):
    from PIL import Image, ImageDraw
    designs = [("anthropomorphic start", "the root of every start (8.4)", json.loads((START_ARCHIVE / "start_design.json").read_text())),
               ("start31, initial", "Sobol point 41, scale 0.25", scene_design(D.P["run"] / "initial_morphology/initial_morphology.design.json")),
               ("start31, optimized (selected)", "Full KaRMA + Adam, 40 iterations", D.design)]
    q30 = None
    tw, th, ext, head, foot = 560, 440, 0.16, 96, 150
    W, H = 3 * tw, 2 * th + head + foot
    canvas = Image.new("RGBA", (W, H), _hex(SURF)); dr = ImageDraw.Draw(canvas)
    dr.text((24, 16), "What the optimizer changed: same joint angles (0°, 30°, 30°, 30° per finger), no object, same scale", font=font(26, True), fill=_hex(INK))
    dr.text((24, 54), "Orthographic views; top row from +x (looking back at the palm), bottom row from +y. Palm frame at the origin: x red, y green, z blue.",
            font=font(17), fill=_hex(INK2))
    from dkarma.experiments.simulation_replay import mjcf_from_design  # noqa: F401  (same scene builder for all three)
    for j, (title, sub, dsg) in enumerate(designs):
        st = Stage(dsg, tw, th, floor=False, ortho_extent_m=ext)
        q30 = np.concatenate([[0.0, np.radians(30), np.radians(30), np.radians(30)] for _ in dsg["topology"]["fingers"]])
        st.pose(q30, None)
        for i, (az, el) in enumerate([(180, 0), (-90, 0)]):
            img = Image.fromarray(st.render(st.camera(LOOK, az, el, 0.5))).convert("RGBA"); d = ImageDraw.Draw(img)
            if i == 0:
                d.text((14, 10), title, font=font(18, True), fill=_hex(INK)); d.text((14, 36), sub, font=font(15), fill=_hex(INK2))
            _view_glyph(d, st, 26, th - 26, font(14))
            if j == 2:
                _scale_bar(d, tw - 190, th - 26, th / ext, font(14))
            canvas.paste(img, (j * tw, head + i * th))
        dr.line([(j * tw, head), (j * tw, head + 2 * th)], fill=_hex(GRID), width=1)
    vh = [l.split(",") for l in (D.P["run"] / "validation_history.csv").read_text().strip().splitlines()]
    cols = vh[0]; v = {int(r[cols.index("iter")]): float(r[cols.index("exact_score")]) for r in vh[1:]}
    y = head + 2 * th + 18
    _finger_legend(dr, 24, y, font(16))
    st3 = json.loads(Path("workspace/phase10/10.3/stask_report.json").read_text())["designs"]
    dr.text((24, y + 36), f"Exact Full score, 12-seed suite (mean over 3 seed configurations): anthropomorphic start {st3['start/start0_nominal']['mean_over_configs']['full_all']:.4f}, "
                          f"selected {st3[D.key]['mean_over_configs']['full_all']:.4f}.", font=font(16), fill=_hex(INK2))
    dr.text((24, y + 62), "The run's own validation (6 seeds): " + ", ".join(f"iteration {k} {s:.4f}" for k, s in sorted(v.items()))
            + "; the initial design was not scored by the run.", font=font(16), fill=_hex(INK2))
    dr.text((24, y + 88), "Joint angles are equal, but each design has its own joint axes, so the same angles curl the fingers differently.",
            font=font(16), fill=_hex(INK2))
    canvas.convert("RGB").save(out / "design_before_after.png", optimize=True)


# ── stage: gallery ───────────────────────────────────────────────────────────

def stage_gallery(D: Data, out: Path, seed: int):
    from PIL import Image, ImageDraw
    tw, th, cap, head, ext = 480, 360, 58, 96, 0.125
    st = Stage(D.design, tw, th, floor=False, ortho_extent_m=ext)
    cam = st.camera((0.048, 0.004, -0.030), 180, -12, 0.5)
    W, H = 4 * tw, 3 * (th + cap) + head
    canvas = Image.new("RGBA", (W, H), _hex(SURF)); dr = ImageDraw.Draw(canvas)
    amin = D.alpha.min(1)
    dr.text((24, 16), f"{display(D.key)}: the 12 three-contact grasps found by the seed search (default configuration)", font=font(26, True), fill=_hex(INK))
    dr.text((24, 54), f"Same orthographic camera, from +x and slightly above. α_min: the grasp's capacity in its weakest of 14 directions (× the standard load); "
                      f"robust (α_min ≥ 1): {int((amin >= 1).sum())} of 12. Primitives: strict successes in the 10.2 replay, of 84.",
            font=font(17), fill=_hex(INK2))
    for s in D.seeds:
        st.pose(D.seed_q[s], D.seed_p[s])
        img = Image.fromarray(st.render(cam)).convert("RGBA"); d = ImageDraw.Draw(img)
        d.text((12, 8), f"seed {s}", font=font(20, True), fill=_hex(INK))
        if s == seed:
            d.text((tw - 12, 10), "▶ animated", font=font(17, True), fill=_hex(INK), anchor="ra")
        if s == 0:
            _scale_bar(d, tw - 150, th - 22, th / ext, font(14))
        x, y = (s % 4) * tw, head + (s // 4) * (th + cap)
        canvas.paste(img, (x, y))
        dr.line([(x, y), (x, y + th + cap)], fill=_hex(GRID), width=1)
        dr.text((x + 12, y + th + 6), f"α_min {amin[s]:.2f} · settle drift in simulation {D.settle_mm(s):.2f} mm", font=font(15), fill=_hex(INK2))
        dr.text((x + 12, y + th + 30), f"{D.seed_total(s)} of 84 primitives", font=font(15, True), fill=_hex(INK))
    canvas.convert("RGB").save(out / "grasp_gallery.png", optimize=True)


# ── stage: animate ───────────────────────────────────────────────────────────

class Recorder:
    """``on_step`` hook of ``simulate_design``: a snapshot every ``stride`` physics steps and at every primitive end."""

    def __init__(self, D: Data, stride: int):
        self.D, self.stride, self.clips = D, stride, {}

    def __call__(self, model, data, t, k, h, a):
        from scipy.spatial.transform import Rotation, Slerp
        from dkarma.experiments.simulation_replay import _normal_forces
        key = (t["seed"], t["kind"], t["command"])
        c = self.clips.get(key)
        if c is None:
            caps = [self.D.full_caps[i] for i in t["contact_caps"]]
            c = self.clips[key] = dict(n=0, frames=[], fingers=[cp.split("_c")[0] for cp in caps], gids=[model.geom(cp).id for cp in caps],
                                       sphere=model.geom("sphere_geom").id, sid=model.body("sphere").id, dt=model.opt.timestep,
                                       fstar=np.maximum(t["fn_exact"][:, :len(caps)], 0.05))   # the controller's set-points (force_scale 1, f_min 0.05 N)
        c["n"] += 1
        end = k >= 0 and a == 1.0 and h == (k + 1) * t["n_sub"]
        if c["n"] % self.stride and not end:
            return
        f, touch = _normal_forces(model, data, c["gids"], c["sphere"], with_touch=True)
        if k < 0:
            p_cmd, R_cmd, fs = t["p"][0], t["R"][0], c["fstar"][0]
        else:
            p_cmd = (1 - a) * t["p"][h - 1] + a * t["p"][h]
            R_cmd = Slerp([0, 1], Rotation.from_matrix([t["R"][h - 1], t["R"][h]]))([a]).as_matrix()[0]
            fs = (1 - a) * c["fstar"][h - 1] + a * c["fstar"][h]
        c["frames"].append(dict(qpos=data.qpos.copy(), x=data.xpos[c["sid"]].copy(), R=data.xmat[c["sid"]].reshape(3, 3).copy(), f=f, touch=touch,
                                fstar=np.asarray(fs), p_cmd=np.asarray(p_cmd), R_cmd=np.asarray(R_cmd), t=c["n"] * c["dt"], k=k, h=h, end=end))


def replay(D: Data, keys, stride):
    """Replay ``keys`` with the stored 10.2 settings and record snapshots; checks the rows against the stored ones."""
    from dkarma.core import load_task_spec
    from dkarma.experiments.simulation_replay import simulate_design
    from dkarma.experiments.compliant_pad import load_pads
    S = D.sim["settings"]
    assert S["controller"] == "force" and S["contact_model"]["pad"]["label"] == "20A_2mm"
    rec = Recorder(D, stride)
    with tempfile.TemporaryDirectory() as tmp:
        rows = simulate_design(Path(tmp), D.P["name"], load_task_spec(), S["torque_ceiling_nm"], settle_s=S["settle_s"], sub_s=S["sub_s"],
                               pos_tol_m=S["pos_tol_m"], kp=S["kp"], squeeze_mm=S["squeeze_mm"], rot_tol_rad=S["rot_tol_rad"], controller=S["controller"],
                               force_scale=S["force_scale"], k_i_m_per_Ns=S["k_i_m_per_Ns"], k_p_m_per_N=S["k_p_m_per_N"], leak=S["leak"],
                               f_min_n=S["f_min_n"], drop_tol_m=S["drop_tol_m"], stop_at_first_failure=False,
                               pad=load_pads(S["contact_model"]["pad_file"])[S["contact_model"]["pad"]["label"]], pad_variant=S["contact_model"]["pad_variant"],
                               archive=Path(S["archive"]), only=lambda t: (t["seed"], t["kind"], t["command"]) in set(keys), on_step=rec)
    for r in rows:
        ref = D.rows[(r["seed"], r["kind"], r["command"])]
        assert r["n_sim_rot"] == ref["n_sim_rot"] and [i["err_m"] for i in r["info"]] == [i["err_m"] for i in ref["info"]], \
            f"replay differs from the stored 10.2 row {r['seed'], r['kind'], r['command']}"
    return rec.clips


def _verdicts(row, tol_rot):
    """Per primitive: (passes strict, counted to the first failure, reason)."""
    out = []
    for k, i in enumerate(row["info"]):
        ok = bool(i["pose_ok"] and i["retain_ok"])
        why = "" if ok else ("drop" if i["drop"] else "position" if not i["pos_ok"] else "orientation" if i["rot_err_rad"] > tol_rot else "contact lost")
        out.append((ok, k < row["n_sim_rot"], why))
    return out


def _lost(row):
    """The 9.3 drop criterion (centre > 20 mm from the plan, or no assigned fingertip in contact): where and how it was met."""
    k = row["n_drop"]
    if k >= row["n_primitives"]:
        return None
    return k, ("released" if not any(row["info"][k]["contacts"]) else "> 20 mm off the plan")


def _decor_fn(t, fr, trail, cam_fwd, cam_up, r_s):
    """Planned path / axis, planned centre, orientation needles (sphere and plan), centre trail."""
    plan, ink = rgb(PLAN, 0.9), rgb(INK, 0.95)
    if t["kind"] == "R":
        from scipy.spatial.transform import Rotation
        ax = Rotation.from_matrix(t["R"][-1]).as_rotvec(); ax = ax / max(np.linalg.norm(ax), 1e-12)
        b0 = np.cross(ax, cam_fwd); b0 = b0 / np.linalg.norm(b0) if np.linalg.norm(b0) > 0.2 else cam_up
    else:
        ax, b0 = None, cam_up

    def decor(scene):
        p0 = t["p"][0]
        if ax is not None:
            _line(scene, p0 - 0.030 * ax, p0 + 0.030 * ax, 0.0006, plan)
        else:
            _line(scene, t["p"][0], t["p"][-1], 0.0006, plan)
            for k in range(t["n_primitives"] + 1):
                _ball(scene, t["p"][k * t["n_sub"]], 0.0012, plan)
            for a, b in zip(trail[:-1], trail[1:]):
                _line(scene, a, b, 0.0005, ink)
        _ball(scene, fr["p_cmd"], 0.0016, plan)
        _ball(scene, fr["p_cmd"], r_s + 0.0008, rgb(PLAN, 0.12))          # the planned sphere (ghost)
        _line(scene, fr["x"], fr["x"] + 1.9 * r_s * (fr["R"] @ b0), 0.0008, ink)
        _line(scene, fr["p_cmd"], fr["p_cmd"] + 1.9 * r_s * (fr["R_cmd"] @ b0), 0.0008, rgb(PLAN, 0.6))
    return decor


def _hud(img, D, s, t, row, fr, verd, title_left, scale, compact=False):
    """Top band (design, grasp, command, conditions, time) and bottom band (primitive verdicts, contact forces, errors)."""
    from PIL import ImageDraw
    W, H = img.size
    z = lambda v: int(round(v * scale))  # noqa: E731
    top, bot = z(78), z(176)
    img = _panel(img, [0, 0, W, top]); img = _panel(img, [0, H - bot, W, H])
    d = ImageDraw.Draw(img)
    K = t["n_primitives"]
    d.text((z(20), z(12)), title_left, font=font(z(22), True), fill=_hex(INK))
    d.text((z(20), z(46)), "MuJoCo replay of the planned trajectory · silicone pads Shore 20A, 2 mm · force-floor controller · 5× slow motion",
           font=font(z(15)), fill=_hex(INK2))
    phase = "settling" if fr["k"] < 0 else f"primitive {fr['k'] + 1} of {K}"
    tx = f"t = {fr['t']:.3f} s"
    d.text((W - z(20) - d.textlength(tx, font=font(z(20), True)), z(12)), tx, font=font(z(20), True), fill=_hex(INK))
    d.text((W - z(20) - d.textlength(phase, font=font(z(15))), z(46)), phase, font=font(z(15)), fill=_hex(INK2))
    # legend for the 3-D marks
    lx, ly = W - z(300), top + z(12)
    d.ellipse([lx + z(6), ly + z(1), lx + z(22), ly + z(17)], fill=_hex(PLAN, 70), outline=_hex(PLAN, 160))
    d.text((lx + z(34), ly), "planned sphere (ghost)", font=font(z(14)), fill=_hex(INK2)); ly += z(22)
    for txt, col in (("planned path / rotation axis", PLAN), ("sphere orientation (needle)", INK), ("planned orientation (needle)", PLAN)):
        d.line([(lx, ly + z(9)), (lx + z(26), ly + z(9))], fill=_hex(col, 230 if col == INK else 200), width=max(2, z(3)))
        d.text((lx + z(34), ly), txt, font=font(z(14)), fill=_hex(INK2)); ly += z(22)
    d.text((lx, ly + z(2)), "ê1, ê2, ê3: axes of this grasp's frame", font=font(z(14)), fill=_hex(INK2))
    # primitive verdicts
    y0 = H - bot + z(10)
    d.text((z(20), y0), "Primitives: pose within 4 mm / 0.2 rad and all three fingertips in contact at the primitive end", font=font(z(14)), fill=_hex(INK2))
    bw, gap = z(30), z(6)
    done = fr["k"] if fr["k"] >= 0 else 0
    done = done + (1 if fr["end"] else 0) if fr["k"] >= 0 else 0
    for k in range(K):
        x = z(20) + k * (bw + gap); y = y0 + z(26)
        if k < done:
            ok, counted, why = verd[k]
            col = (GOOD if counted else MUTED) if ok else BAD
            d.rounded_rectangle([x, y, x + bw, y + bw], radius=z(5), fill=_hex(col))
            d.text((x + bw / 2, y + bw / 2), "✓" if ok else "✗", font=font(z(20), True), fill=(255, 255, 255, 255), anchor="mm")
        else:
            d.rounded_rectangle([x, y, x + bw, y + bw], radius=z(5), outline=_hex(AXIS), width=max(1, z(2)))
    counted_so_far = sum(1 for k in range(done) if verd[k][1])
    first_fail = next((k for k in range(done) if not verd[k][0]), None)
    d.text((z(20), y0 + z(26) + bw + z(8)), f"counted (to the first failure): {counted_so_far} of {K}", font=font(z(16), True), fill=_hex(INK))
    fail = []
    if first_fail is not None:
        fail.append(f"primitive {first_fail + 1} failed: {verd[first_fail][2]}")
    lost = _lost(row)
    if lost is not None and done > lost[0]:
        fail.append(f"sphere {lost[1]} in primitive {lost[0] + 1} (drop criterion)")
    if fail:
        d.text((z(20), y0 + z(26) + bw + z(31)), " · ".join(fail), font=font(z(14), True), fill=_hex(INK))
    err = np.linalg.norm(fr["x"] - fr["p_cmd"]) * 1e3
    rot = np.degrees(np.arccos(np.clip((np.trace(fr["R"].T @ fr["R_cmd"]) - 1) / 2, -1, 1)))
    d.text((z(20), y0 + z(26) + bw + z(52)), f"centre error {err:.1f} mm (tolerance 4) · orientation error {rot:.1f}° (tolerance 11.5°)",
           font=font(z(14)), fill=_hex(INK2))
    rr = [r for r in D.sim["rows"] if r["kind"] == t["kind"]]          # context: this command kind over all seeds
    ls = [_lost(r) for r in rr]
    d.text((z(20), y0 + z(26) + bw + z(72)), f"{'rotations' if t['kind'] == 'R' else 'translations'}, all {len(D.seeds)} seeds: "
           f"{sum(r['n_sim_rot'] == r['n_primitives'] for r in rr)}/{len(rr)} complete all {K} primitives · "
           f"{sum(1 for l in ls if l and l[1] == 'released')}/{len(rr)} release the sphere · "
           f"{sum(1 for l in ls if l and l[1] != 'released')}/{len(rr)} end > 20 mm off the plan", font=font(z(13)), fill=_hex(INK2))
    # contact forces
    fx = W - z(470)
    d.text((fx, y0), "contact normal force on the sphere (tick: the controller's floor)", font=font(z(14)), fill=_hex(INK2))
    fmax, bl = 4.0, z(250)
    order = sorted(range(len(fr["f"])), key=lambda i: ("index", "middle", "thumb").index(row_fingers(t, D)[i]))
    for j, i in enumerate(order):
        fn = row_fingers(t, D)[i]; y = y0 + z(28) + j * z(34)
        d.rounded_rectangle([fx, y + z(4), fx + z(18), y + z(18)], radius=z(3), fill=_hex(FINGER[fn]))
        d.text((fx + z(26), y), fn, font=font(z(15)), fill=_hex(INK))
        bx = fx + z(96)
        d.rectangle([bx, y + z(6), bx + bl, y + z(16)], fill=_hex(GRID))
        w = bl * min(fr["f"][i], fmax) / fmax
        if w > 0:
            d.rectangle([bx, y + z(6), bx + w, y + z(16)], fill=_hex(FINGER[fn]))
        tk = bx + bl * min(fr["fstar"][i], fmax) / fmax
        d.line([(tk, y + z(1)), (tk, y + z(21))], fill=_hex(INK), width=max(2, z(2)))
        lab = f"{fr['f'][i]:.2f} N" if fr["touch"][i] else "no contact"
        d.text((bx + bl + z(10), y), lab, font=font(z(15), True), fill=_hex(INK if fr["touch"][i] else BAD))
    return img


def row_fingers(t, D):
    return [D.full_caps[i].split("_c")[0] for i in t["contact_caps"]]


def _encode(frames_dir: Path, mp4: Path, fps: int):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames_dir / "%05d.png"), "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", "-preset", "slow", "-movflags", "+faststart", str(mp4)], check=True)


def _gif(mp4: Path, gif: Path, width: int, fps: float, t0: float = 0.0, dur: float | None = None):
    span = ["-ss", f"{t0}"] + (["-t", f"{dur}"] if dur else [])
    flt = f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + span + ["-i", str(mp4), "-filter_complex", flt, "-loop", "0", str(gif)], check=True)


def stage_animate(D: Data, out: Path, seed: int, fps: int = 25, slow: float = 5.0):
    from PIL import Image
    stride = int(round(1.0 / (fps * slow) / 5e-4))      # physics steps per frame: 5x slow motion at 25 fps = 16 steps of 0.5 ms
    seq = [("R", 0), ("T", 1), ("R", 1), ("T", 0)]
    keys = [(seed, kd, u) for kd in "TR" for u in range(6)]
    clips = replay(D, keys, stride)
    r_s = 0.015
    W, H = 1280, 720
    st = Stage(D.design, W, H, floor=False)          # the floor has no collision; a dropped sphere would fall through it
    cam = st.camera((0.05, 0.002, -0.036), 180, -14, 0.215)
    st.pose(D.seed_q[seed], D.seed_p[seed]); st.render(cam)
    cam_fwd, cam_up = st.fwd.copy(), st.up.copy()
    tol_rot = D.sim["settings"]["rot_tol_rad"]
    # 1. the sequence video
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp); n = 0
        bounds = []
        for kd, u in seq:
            t = D.trajs[(seed, kd, u)]; row = D.rows[(seed, kd, u)]; verd = _verdicts(row, tol_rot)
            title = f"{display(D.key)} · grasp seed {seed} · {cmd_label(kd, u, t['n_primitives'])}"
            frames = clips[(seed, kd, u)]["frames"]
            trail = []
            bounds.append((kd, u, n / fps))
            for j, fr in enumerate(frames):
                if fr["k"] >= 0 and j % 2 == 0:
                    trail.append(fr["x"].copy())
                st.pose(qpos=fr["qpos"])
                img = Image.fromarray(st.render(cam, _decor_fn(t, fr, trail + [fr["x"]], cam_fwd, cam_up, r_s)))
                img = _hud(img, D, seed, t, row, fr, verd, title, 1.0)
                reps = 15 if j == 0 else (38 if j == len(frames) - 1 else 1)        # hold the first and the last frame
                for _ in range(reps):
                    img.convert("RGB").save(tmp / f"{n:05d}.png"); n += 1
        _encode(tmp, out / f"manipulation_seed{seed}.mp4", fps)
    t_r0 = bounds[0][2]; t_t1 = bounds[1][2]
    _gif(out / f"manipulation_seed{seed}.mp4", out / f"manipulation_seed{seed}_rotation.gif", 800, 12.5, t_r0, t_t1 - t_r0)
    # 2. all twelve commands side by side
    tw, th, head = 400, 300, 64
    sm = Stage(D.design, tw, th, floor=False)
    camg = sm.camera((0.05, 0.002, -0.036), 180, -14, 0.25)
    T = max(len(clips[k]["frames"]) for k in keys)
    order = [(seed, "R", u) for u in range(6)] + [(seed, "T", u) for u in range(6)]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        from PIL import ImageDraw
        for j in range(T + 40):
            canvas = Image.new("RGBA", (3 * tw * 4 // 3, 3 * th + head), _hex(SURF))
            dc = ImageDraw.Draw(canvas)
            jj = min(j, T - 1)
            tsim = clips[order[-1]]["frames"][min(jj, len(clips[order[-1]]["frames"]) - 1)]["t"]
            dc.text((20, 12), f"{display(D.key)} · grasp seed {seed}: all twelve commands from the same grasp (same replay as the 10.2 endpoint)", font=font(20, True), fill=_hex(INK))
            dc.text((20, 40), f"top rows: rotations, 10 × 0.15 rad · bottom row: translations, 4 × 10 mm · boxes: strict pass / fail per primitive · "
                              f"violet ghost: planned sphere · 5× slow motion · t = {tsim:.3f} s",
                    font=font(15), fill=_hex(INK2))
            for i, key in enumerate(order):
                t = D.trajs[key]; row = D.rows[key]; verd = _verdicts(row, tol_rot)
                frs = clips[key]["frames"]; fr = frs[min(jj, len(frs) - 1)]; finished = jj >= len(frs) - 1
                sm.pose(qpos=fr["qpos"])
                img = Image.fromarray(sm.render(camg, _decor_fn(t, fr, [], cam_fwd, cam_up, r_s))).convert("RGBA")
                img = _panel(img, [0, th - 38, tw, th], alpha=215); img = _panel(img, [0, 0, tw, 52], alpha=200)
                d = ImageDraw.Draw(img)
                d.text((10, 8), cmd_label(key[1], key[2], t["n_primitives"], long=False), font=font(16, True), fill=_hex(INK))
                K = t["n_primitives"]; done = (fr["k"] + (1 if fr["end"] else 0)) if fr["k"] >= 0 else 0
                bw, gap = 20, 4
                for k in range(K):
                    x, y = 10 + k * (bw + gap), th - 30
                    if k < done:
                        ok, counted, _ = verd[k]
                        d.rounded_rectangle([x, y, x + bw, y + bw], radius=3, fill=_hex((GOOD if counted else MUTED) if ok else BAD))
                    else:
                        d.rounded_rectangle([x, y, x + bw, y + bw], radius=3, outline=_hex(AXIS), width=1)
                if finished:
                    d.text((tw - 12, 8), f"{row['n_sim_rot']} of {K} counted", font=font(15, True), fill=_hex(INK), anchor="ra")
                lost = _lost(row)
                if lost is not None and done > lost[0]:
                    d.text((tw - 12, 30), f"✗ {lost[1]} in primitive {lost[0] + 1}", font=font(14, True), fill=_hex(BAD), anchor="ra")
                canvas.paste(img, ((i % 4) * tw, head + (i // 4) * th))
            canvas.convert("RGB").save(tmp / f"{j:05d}.png")
        _encode(tmp, out / f"all_commands_seed{seed}.mp4", fps)


# ── stage: charts ────────────────────────────────────────────────────────────

def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "axes.titleweight": "bold",
                         "axes.titlesize": 13, "axes.titlelocation": "left"})
    return plt


def stage_charts(D: Data, out: Path, seed: int):
    plt = _mpl()
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    cmap = LinearSegmentedColormap.from_list("ramp", RAMP)

    def ink_on(v, vmax):           # white text on the dark half of the ramp
        return "white" if v / vmax > 0.55 else INK

    # 1. per-command strict success
    cols = [("R", u) for u in range(6)] + [("T", u) for u in range(6)]
    M = np.array([[D.strict(s, kd, u) / (10 if kd == "R" else 4) for kd, u in cols] for s in D.seeds])
    N = np.array([[D.strict(s, kd, u) for kd, u in cols] for s in D.seeds])
    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, str(N[i, j]), ha="center", va="center", fontsize=10.5, color=ink_on(M[i, j], 1))
        ax.text(12.0, i, f"{D.seed_total(i)} / 84", ha="left", va="center", fontsize=10.5, color=INK, fontweight="bold" if i == seed else "normal")
    ax.set_xticks(range(12), [f"R {CMD_LABELS[u]}" for _, u in cols[:6]] + [f"T {CMD_LABELS[u]}" for _, u in cols[6:]])
    ax.set_yticks(range(12), [f"seed {s}" + ("  ▶" if s == seed else "") for s in D.seeds])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(5.5, color=SURF, lw=4)
    ax.text(2.5, -1.0, "rotations (of 10 primitives)", ha="center", color=INK2); ax.text(8.5, -1.0, "translations (of 4 primitives)", ha="center", color=INK2)
    ax.text(12.0, -1.0, "seed total", ha="left", color=INK2)
    ax.set_xlim(-0.5, 13.3); ax.set_ylim(11.5, -1.6)
    E = np.mean([D.seed_total(s) for s in D.seeds])
    fig.suptitle(f"Simulated three-contact manipulation, per grasp and command: E = {E:.1f} primitives per seed", x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.02, 0.905, "Strict successes counted to the first failure (pose within 4 mm / 0.2 rad, all three fingertips in contact). "
                          "MuJoCo, silicone pads 20A/2 mm, force-floor controller; default seed configuration. ▶ = the animated seed.",
             fontsize=10, color=INK2, wrap=True)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.07)
    fig.savefig(out / "command_success.png", dpi=150); plt.close(fig)

    # 2. wrench capacity
    A = D.alpha
    names = ["+Fx", "−Fx", "+Fy", "−Fy", "+Fz", "−Fz", "+Mx", "−Mx", "+My", "−My", "+Mz", "−Mz", "mix 1", "mix 2"]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    vmax = 50.0
    im = ax.imshow(np.clip(A, 1, vmax), cmap=cmap, norm=LogNorm(1, vmax), aspect="auto")
    amin = A.min(1)
    for i in range(A.shape[0]):
        j = int(np.argmin(A[i]))
        ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.6))
        ax.text(14.0, i, f"{amin[i]:.2f}", ha="left", va="center", fontsize=10.5, color=INK)
    ax.set_xticks(range(14), names); ax.set_yticks(range(12), [f"seed {s}" for s in D.seeds]); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.axvline(5.5, color=SURF, lw=4); ax.axvline(11.5, color=SURF, lw=4)
    ax.text(14.0, -0.9, "weakest", ha="left", color=INK2)
    ax.set_xlim(-0.5, 15.0); ax.set_ylim(11.5, -1.4)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01, ticks=[1, 2, 5, 10, 20, 50]); cb.ax.set_yticklabels(["1", "2", "5", "10", "20", "≥ 50"])
    cb.outline.set_visible(False); cb.ax.tick_params(colors=INK2, length=0)
    fig.suptitle(f"Wrench capacity: every grasp carries at least {amin.min():.2f} × the standard load in each of 14 directions",
                 x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.02, 0.905, "α(d): the largest multiple of the standard load along d that the grasp carries on top of the object's weight (exact SLSQP, 0.108 N·m ceiling). "
                          "Standard load: the 35 g weight (forces, palm-frame axes) or weight × 15 mm (moments); mix 1, 2: fixed mixed directions. "
                          "Outlined: each seed's weakest direction. Robust grasp (S_task): α ≥ 1 in all 14.",
             fontsize=10, color=INK2, wrap=True)
    fig.subplots_adjust(left=0.08, right=0.94, top=0.82, bottom=0.07)
    fig.savefig(out / "wrench_capacity.png", dpi=150); plt.close(fig)

    # 3. standing among the 48 10.2 designs
    comp = json.loads(D.P["comparison"].read_text())
    E_ = comp["measures"]["primary"]["per_start"]
    summ = json.loads(D.P["summary"].read_text())["per_design"]
    starts = sorted(E_, key=lambda k: E_[k]["full_d18"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 8.4), sharey=True, gridspec_kw=dict(wspace=0.08))
    for ax, get, xl in ((a1, lambda st, arm: E_[st][arm], "E: strict simulated primitives per budgeted seed"),
                        (a2, lambda st, arm: summ[f"{arm}/{st}"]["full_exact_score"][0], "exact Full score (mean of 3 seed configurations)")):
        for i, st in enumerate(starts):
            a, f = get(st, "armAprime"), get(st, "full_d18")
            sel = st == D.P["st"]
            if sel:
                ax.axhspan(i - 0.5, i + 0.5, color="#f0efec", zorder=0)
            ax.plot([a, f], [i, i], color=AXIS, lw=1.2, zorder=1)
            ax.scatter([a], [i], s=46, color=MUTED, edgecolor=SURF, linewidth=1.5, zorder=2)
            ax.scatter([f], [i], s=46 if not sel else 80, color=FINGER["index"], edgecolor=SURF if not sel else INK, linewidth=1.5, zorder=3)
        ax.set_xlabel(xl); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
        ax.spines["left"].set_visible(False)
    a1.set_yticks(range(len(starts)), [s.split("_")[0] + ("  (selected)" if s == D.P["st"] else "") for s in starts])
    for lbl in a1.get_yticklabels():
        if "(selected)" in lbl.get_text():
            lbl.set_fontweight("bold"); lbl.set_color(INK)
    a1.set_ylim(-0.8, len(starts) - 0.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=FINGER["index"], markersize=8, label="Full-D18 (Full KaRMA + Adam)"),
                        Line2D([], [], marker="o", ls="", color=MUTED, markersize=8, label="arm A′ (original KaRMA + CMA-ES)")],
               loc="upper left", bbox_to_anchor=(0.02, 0.925), ncol=2, frameon=False)
    fig.suptitle("The selected design among the 48 designs of the 10.2 comparison (24 new starts, two arms)", x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.87, bottom=0.07)
    fig.savefig(out / "standing_among_designs.png", dpi=150); plt.close(fig)

    # 4. load sweep
    sw = json.loads(D.P["sweep"].read_text())["rows"]
    lab = {("torque", 0.215): "stall 0.215 N·m", ("torque", 0.108): "primary ceiling 0.108 N·m", ("torque", 0.043): "continuous 0.043 N·m",
           ("mass", 0.108): "70 g object, 0.108 N·m"}
    rows = [(lab[(r["kind"], r["tau_bar"])], r["full_exact_score"], r["seeds_hold"]) for r in sw]
    fig, ax = plt.subplots(figsize=(9.5, 4.0))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, [r[1] for r in rows], height=0.42, color=FINGER["index"])
    for yy, (l, v, n) in zip(y, rows):
        ax.text(v + 0.0004, yy, f"{v:.4f} · {n} of 12 seeds hold", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(y, [r[0] for r in rows]); ax.tick_params(length=0)
    ax.set_xlim(0, max(r[1] for r in rows) * 1.45); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.spines["left"].set_visible(False)
    ax.set_xlabel("exact Full score (default seed configuration, 12 seeds)")
    fig.suptitle("Load sweep: torque ceiling and object mass", x=0.02, ha="left", fontweight="bold", fontsize=14)
    fig.text(0.02, 0.83, "Exact evaluation (no simulation). A seed holds when the exact equilibrium at its seed state is feasible under that ceiling and mass.",
             fontsize=10, color=INK2)
    fig.subplots_adjust(left=0.25, right=0.98, top=0.76, bottom=0.18)
    fig.savefig(out / "load_sweep.png", dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", default="full_d18/start31_sobol41_s0.25")
    ap.add_argument("--out", default="results/fabrication_candidates/selected_full_d18_start31_sobol41_s0.25/figures")
    ap.add_argument("--seed", type=int, default=8, help="the grasp seed shown in the views and the animations")
    ap.add_argument("--stages", nargs="*", default=["views", "compare", "gallery", "charts", "animate"])
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if shutil.which("ffmpeg") is None and "animate" in args.stages:
        raise SystemExit("ffmpeg is needed for the animate stage")
    D = Data(args.key)
    for s in args.stages:
        dict(views=lambda: stage_views(D, out, args.seed), compare=lambda: stage_compare(D, out), gallery=lambda: stage_gallery(D, out, args.seed),
             charts=lambda: stage_charts(D, out, args.seed), animate=lambda: stage_animate(D, out, args.seed))[s]()
        print(f"[{s}] done", flush=True)
    for st in reversed(_ALIVE):        # close the GL contexts while the interpreter is still intact
        st.r.close()
    print("done")


if __name__ == "__main__":
    main()
