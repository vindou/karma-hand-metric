import marimo

__generated_with = "0.25.1"
app = marimo.App(width="full", app_title="KaRMA optimization explorer")


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import json, math, sys, functools
    from pathlib import Path
    import numpy as np
    import plotly.graph_objects as go
    import yaml

    # repo root: the directory holding karma_config.yaml (works whether marimo gives __file__ or not)
    _start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    REPO = next((p for p in [_start, *_start.parents] if (p / "karma_config.yaml").exists()), _start)
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    RUNS = REPO / "optimize" / "runs"
    return REPO, RUNS, functools, go, json, mo, np, yaml


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # KaRMA optimization explorer

    Interactive view of the hands produced by optimizing a two-finger kinematic tree against the
    **KaRMA** metric, next to the 16 published hands. Pick a hand, pose it at any reached voxel,
    and inspect the orientation bins the pinch can reach there. Lower sections compare runs,
    show optimizer progress, and the arm F re-ranking. Capsules are the metric's own procedural
    geometry (thumb orange, index blue); the grey bar joins the two knuckles; the dark sphere is the
    test object.
    """)
    return


@app.cell(hide_code=True)
def _(REPO, RUNS, json, yaml):
    # ── catalogue every scored hand: optimization runs first, then the published 16 ──
    catalogue = {}
    pub_rows = {r["robot"]: r for r in yaml.safe_load((REPO / "results/16_hand_batch/summary.yaml").read_text())["rows"]}
    if RUNS.exists():
        for _d in sorted(RUNS.iterdir()):
            _pkl = _d / "final" / "current.pkl"
            if not _pkl.exists() and (_d / "current.pkl").exists():
                _pkl = _d / "current.pkl"
            if not _pkl.exists():
                continue
            _ymls = list(_pkl.parent.glob("robot_*.yaml"))
            if not _ymls:
                continue
            _sj = _pkl.parent / "summary.json"
            _summ = json.loads(_sj.read_text()) if _sj.exists() else {}
            catalogue[f"run: {_d.name}"] = dict(pkl=_pkl, yaml=_ymls[0], kind="optimized", run=_d.name, summary=_summ)
    for _name, _row in sorted(pub_rows.items(), key=lambda kv: -kv[1]["KaRMA_T"]):
        _pkl = REPO / "results/16_hand_batch" / f"robot_{_name}" / "current.pkl"
        _yml = REPO / "robots" / f"robot_{_name}.yaml"
        if _pkl.exists() and _yml.exists():
            catalogue[f"published: {_name}"] = dict(pkl=_pkl, yaml=_yml, kind="published", run=_name,
                                                  summary={"KaRMA_T": _row["KaRMA_T"], "KaRMA_R": _row["KaRMA_R"], "KaRMA_S": _row["KaRMA_S"],
                                                           "n_voxels": _row["n_voxels"], "L_ref_mm": _row["L_ref_mm"], "n_seeds": _row["n_seeds"]})
    return catalogue, pub_rows


@app.cell(hide_code=True)
def _(REPO, functools, np):
    from karma.storage import load_result
    from karma.config import load_config
    from karma.robot import load_robot, update_kinematics, capsule_endpoints_world
    from karma.viser_geometry import capsule_mesh, sphere_mesh
    from karma.orientation import healpix_bin_to_axis, healpix_npix_lines

    class HandData:
        """A scored hand: metric result + the robot model needed to draw it."""

        def __init__(self, pkl, yml):
            self.res, self.meta = load_result(pkl)
            self.cfg = load_config(REPO / "karma_config.yaml", yml)
            self.ctx = load_robot(self.cfg.urdf_path, self.cfg)
            cov = self.res.rotational_coverage_per_voxel
            nodes = {}
            for (v, _b), node in self.res.state_nodes.items():
                nodes.setdefault(v, node)
            seed_c = np.asarray(self.res.seed_centre)
            self.voxels = []
            for v, node in nodes.items():
                self.voxels.append(dict(voxel=v, centre=np.asarray(node.centre_world), q=np.asarray(node.q),
                                        pair=tuple(node.contact_pair), coverage=float(cov.get(v, 0.0)),
                                        bins=set(self.res.voxel_ori_bins.get(v, ())),
                                        dist=float(np.linalg.norm(np.asarray(node.centre_world) - seed_c))))
            self.voxels.sort(key=lambda d: (-d["coverage"], d["dist"]))
            # smooth order: first-visit order of an Euler tour of the Phase-1 search tree
            from optimize.voxel_tour import tour_order
            try:
                _order, _tnodes = tour_order(self.res)
                _byv = {d["voxel"]: d for d in self.voxels}
                self.voxels_tour = []
                for v in _order:
                    d = dict(_byv[v]); d["q"] = _tnodes[v]["q"]; self.voxels_tour.append(d)
            except Exception:
                self.voxels_tour = list(self.voxels)
            self.seed_q = np.asarray(self.res.seed_q)
            self.seed_centre = seed_c
            self.seed_pair = tuple(self.res.seed_contact_pair)
            from optimize.viz import seed_frame
            self.R_grid = seed_frame(self.res)
            self.total_bins = healpix_npix_lines(self.cfg.healpix_nside)

        def capsules(self, q):
            update_kinematics(self.ctx, np.array(q, dtype=float))
            thumb = set(self.cfg.thumb_contact_links)
            out = []
            for name in self.ctx.link_capsules:
                a, b = capsule_endpoints_world(self.ctx, name)
                out.append((name, np.asarray(a).copy(), np.asarray(b).copy(), name in thumb))
            roots = [j for j in range(1, self.ctx.model.njoints) if self.ctx.model.parents[j] == 0]
            knuckles = np.array([self.ctx.data.oMi[j].translation for j in roots])
            return out, knuckles

    @functools.lru_cache(maxsize=8)
    def load_hand(pkl_str, yml_str):
        return HandData(pkl_str, yml_str)

    return capsule_mesh, healpix_bin_to_axis, load_hand, sphere_mesh


@app.cell(hide_code=True)
def _(catalogue, mo):
    _default = next((k for k in catalogue if k == "run: A1"), next(iter(catalogue), None))
    hand_pick = mo.ui.dropdown(options=list(catalogue), value=_default, label="hand")
    show_cloud = mo.ui.checkbox(value=True, label="show reachable voxels")
    color_by = mo.ui.dropdown(options=["orientation coverage", "distance from seed"], value="orientation coverage", label="colour voxels by")
    order_by = mo.ui.dropdown(options=["search-tree tour (smooth)", "orientation coverage"], value="search-tree tour (smooth)", label="slider order")
    pose_mode = mo.ui.radio(options=["selected voxel", "seed pinch", "farthest voxel"], value="selected voxel",
                            label="pose the hand at (with 'selected voxel' the slider below moves the hand)")
    mo.hstack([hand_pick, pose_mode, order_by, color_by, show_cloud], justify="start", gap=2)
    return color_by, hand_pick, order_by, pose_mode, show_cloud


@app.cell(hide_code=True)
def _(catalogue, hand_pick, load_hand, mo, order_by):
    entry = catalogue[hand_pick.value]
    hd = load_hand(str(entry["pkl"]), str(entry["yaml"]))
    vox_list = hd.voxels_tour if order_by.value.startswith("search-tree") else hd.voxels
    n_vox = len(vox_list)
    _lab = ("position along the search-tree tour (drag slowly: consecutive voxels are one rolling step apart)"
            if order_by.value.startswith("search-tree") else "rank by orientation coverage, 0 = best")
    voxel_rank = mo.ui.slider(start=0, stop=max(n_vox - 1, 0), value=0, step=1, show_value=True,
                              label=f"selected voxel: {_lab} ({n_vox} voxels)")
    voxel_rank
    return entry, hd, vox_list, voxel_rank


@app.cell(hide_code=True)
def _(entry, hd, mo):
    _s = entry["summary"]
    _sens = hd.res.seed_sensitivity or {}
    mo.md(
        f"**{entry['run']}** ({entry['kind']}) · KaRMA-T **{hd.res.translational_score:.4f}** · KaRMA-R **{hd.res.global_rotational_score:.3f}** · "
        f"KaRMA-S **{_sens.get('karma_s', float('nan')):.3f}** · {hd.res.n_voxels_reached} voxels · L_ref {hd.res.l_ref_m * 1e3:.1f} mm · "
        f"voxel {hd.cfg.voxel_size_m * 1e3:.2f} mm · sphere radius {hd.cfg.sphere_radius_m * 1e3:.2f} mm · "
        f"{_sens.get('n_seeds_evaluated', '?')} pinches evaluated · top-3 seeds {_sens.get('ensemble_t', '?')}"
        + (f" · backend {_s.get('backend')}" if _s.get("backend") else "")
    )
    return


@app.cell(hide_code=True)
def _(
    capsule_mesh,
    color_by,
    go,
    hd,
    np,
    pose_mode,
    show_cloud,
    sphere_mesh,
    vox_list,
    voxel_rank,
):
    from optimize.viz import hand_pose, voxel_cubes, rot_z_to, THUMB_C, INDEX_C, THUMB_DIM, INDEX_DIM, PALM_C

    sel = vox_list[voxel_rank.value] if vox_list else None
    if pose_mode.value == "seed pinch" or sel is None:
        _q, _centre, _pair = hd.seed_q, hd.seed_centre, hd.seed_pair
    elif pose_mode.value == "farthest voxel":
        _far = max(hd.voxels, key=lambda d: d["dist"]); _q, _centre, _pair = _far["q"], _far["centre"], _far["pair"]
    else:
        _q, _centre, _pair = sel["q"], sel["centre"], sel["pair"]

    _g = hand_pose(hd.ctx, hd.cfg, _q, _centre, _pair)
    _r_l = hd.cfg.link_radius_m * 1e3
    _traces = []
    for _name, _a, _b, _is_thumb, _is_contact in _g["capsules"]:
        _a, _b = _a * 1e3, _b * 1e3
        _seg = _b - _a; _L = np.linalg.norm(_seg)
        _verts, _faces = capsule_mesh(_r_l, max(_L / 2, 1e-6), n_rings=10, n_segments=18)
        _verts = _verts @ (rot_z_to(_seg) if _L > 1e-9 else np.eye(3)).T + 0.5 * (_a + _b)
        _col = (THUMB_C if _is_contact else THUMB_DIM) if _is_thumb else (INDEX_C if _is_contact else INDEX_DIM)
        _traces.append(go.Mesh3d(x=_verts[:, 0], y=_verts[:, 1], z=_verts[:, 2], i=_faces[:, 0], j=_faces[:, 1], k=_faces[:, 2],
                                 color=_col, opacity=1.0, name=_name + (" (in contact)" if _is_contact else ""), hoverinfo="name",
                                 lighting=dict(ambient=0.5, diffuse=0.75, specular=0.2, roughness=0.6), flatshading=False, showlegend=False))
    _alen = 4.6 * _r_l
    for _j in _g["joints"]:
        _o = _j["origin"] * 1e3; _ax = _j["axis"]
        _p0, _p1 = _o - 0.5 * _alen * _ax, _o + 0.5 * _alen * _ax
        _c = "#7a3b00" if _j["is_thumb"] else "#0b2f73"
        _hover = (f"{_j['name']}<br>q = {np.degrees(_j['q']):.1f}° of [{np.degrees(_j['lo']):.0f}°, {np.degrees(_j['hi']):.0f}°]"
                  f"<br>axis (world) = ({_ax[0]:+.2f}, {_ax[1]:+.2f}, {_ax[2]:+.2f})" + ("" if _j["active"] else "<br>(not an active joint)"))
        _traces.append(go.Scatter3d(x=[_p0[0], _p1[0]], y=[_p0[1], _p1[1]], z=[_p0[2], _p1[2]], mode="lines",
                                    line=dict(color=_c, width=7), text=[_hover, _hover], hoverinfo="text", showlegend=False))
        _traces.append(go.Cone(x=[_p1[0]], y=[_p1[1]], z=[_p1[2]], u=[_ax[0]], v=[_ax[1]], w=[_ax[2]], sizemode="absolute", sizeref=0.9 * _r_l,
                               anchor="tail", colorscale=[[0, _c], [1, _c]], showscale=False, hoverinfo="skip", showlegend=False))
        _traces.append(go.Scatter3d(x=[_o[0]], y=[_o[1]], z=[_o[2]], mode="markers", marker=dict(size=4, color=_c), hoverinfo="skip", showlegend=False))
    if len(_g["knuckles"]) >= 2:
        _K = _g["knuckles"] * 1e3
        _traces.append(go.Scatter3d(x=_K[:, 0], y=_K[:, 1], z=_K[:, 2], mode="lines", line=dict(color=PALM_C, width=12),
                                    name="palm (knuckle to knuckle)", hoverinfo="name"))
    _sv, _sf = sphere_mesh(hd.cfg.sphere_radius_m * 1e3, 14, 20)
    _sv = np.asarray(_sv) + np.asarray(_centre) * 1e3
    _traces.append(go.Mesh3d(x=_sv[:, 0], y=_sv[:, 1], z=_sv[:, 2], i=_sf[:, 0], j=_sf[:, 1], k=_sf[:, 2], color="#3a3a3a", opacity=1.0,
                             name="sphere", hoverinfo="name", lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3), showlegend=False))
    for _ct in _g["contacts"]:
        _pt = _ct["point"] * 1e3; _n = _ct["normal"]; _nl = 1.5 * hd.cfg.sphere_radius_m * 1e3
        _traces.append(go.Scatter3d(x=[_pt[0], _pt[0] + _nl * _n[0]], y=[_pt[1], _pt[1] + _nl * _n[1]], z=[_pt[2], _pt[2] + _nl * _n[2]],
                                    mode="lines+markers", line=dict(color="#00c2a8", width=5), marker=dict(size=[6, 0], color="#00c2a8"),
                                    name=f"contact {_ct['link']} (gap {_ct['gap']*1e3:+.3f} mm)", hoverinfo="name"))
    if show_cloud.value and hd.voxels:
        _P = np.array([d["centre"] for d in hd.voxels]) * 1e3
        _edge = hd.cfg.voxel_size_m * 1e3
        if color_by.value == "orientation coverage":
            _vals = np.array([d["coverage"] for d in hd.voxels]); _cmin, _cmax, _ctitle = 0.0, max(0.25, float(_vals.max())), "orientation coverage"
        else:
            _vals = np.array([d["dist"] * 1e3 for d in hd.voxels]); _cmin, _cmax, _ctitle = 0.0, float(_vals.max()), "distance from seed [mm]"
        _verts, _faces, _owner = voxel_cubes(_P, _edge, hd.R_grid, shrink=0.9)
        _text = np.array([f"voxel {d['voxel']}<br>coverage {d['coverage']:.3f} ({len(d['bins'])}/{hd.total_bins} bins)<br>{d['dist']*1e3:.1f} mm from seed<br>contact {d['pair']}"
                          for d in hd.voxels])[_owner]
        _traces.append(go.Mesh3d(x=_verts[:, 0], y=_verts[:, 1], z=_verts[:, 2], i=_faces[:, 0], j=_faces[:, 1], k=_faces[:, 2],
                                 intensity=_vals[_owner], colorscale="Magma", cmin=_cmin, cmax=_cmax, opacity=0.35, flatshading=True,
                                 colorbar=dict(title=_ctitle, len=0.6), text=_text, hoverinfo="text", name="reachable voxels",
                                 lighting=dict(ambient=0.7, diffuse=0.5, specular=0.05), showlegend=False))
        if sel is not None:
            _v2, _f2, _ = voxel_cubes(sel["centre"][None, :] * 1e3, _edge, hd.R_grid, shrink=1.05)
            _traces.append(go.Mesh3d(x=_v2[:, 0], y=_v2[:, 1], z=_v2[:, 2], i=_f2[:, 0], j=_f2[:, 1], k=_f2[:, 2], color="#00e5ff", opacity=0.9,
                                     name=f"selected voxel {sel['voxel']}", hoverinfo="name", showlegend=False))
    fig_hand = go.Figure(_traces)
    fig_hand.update_layout(scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]",
                                      camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))),
                           height=760, margin=dict(l=0, r=0, t=30, b=0), legend=dict(x=0.01, y=0.99),
                           title=f"{pose_mode.value} · sphere centre {np.round(np.asarray(_centre)*1e3, 1).tolist()} mm · contact {_pair}")
    fig_hand
    return (sel,)


@app.cell(hide_code=True)
def _(go, hd, healpix_bin_to_axis, mo, np, sel):
    # ── orientation bins _reached at the selected voxel, drawn on the hemisphere of pinch-axis lines ──
    _nside = hd.cfg.healpix_nside
    _all_axes = np.array([healpix_bin_to_axis(_b, _nside) for _b in range(hd.total_bins)])
    _reached = np.zeros(hd.total_bins, bool)
    if sel is not None:
        for _b in sel["bins"]:
            if 0 <= _b < hd.total_bins:
                _reached[_b] = True
    _tr = []
    _tr.append(go.Scatter3d(x=_all_axes[~_reached, 0], y=_all_axes[~_reached, 1], z=_all_axes[~_reached, 2], mode="markers",
                           marker=dict(size=3, color="#bbbbbb"), name="not _reached", hoverinfo="skip"))
    _tr.append(go.Scatter3d(x=_all_axes[_reached, 0], y=_all_axes[_reached, 1], z=_all_axes[_reached, 2], mode="markers",
                           marker=dict(size=6, color="#fca636"), name=f"reached ({int(_reached.sum())}/{hd.total_bins})",
                           text=[f"bin {_b}" for _b in np.where(_reached)[0]], hoverinfo="text"))
    fig_ori = go.Figure(_tr)
    fig_ori.update_layout(scene=dict(aspectmode="cube", xaxis_title="x", yaxis_title="y", zaxis_title="z"), height=430,
                          margin=dict(l=0, r=0, t=40, b=0),
                          title="Pinch-axis directions (sphere body frame) _reached at the selected voxel · 228 HEALPix line bins (antipodes identified, upper hemisphere shown)")
    _cov_hist = go.Figure(go.Histogram(x=[d["coverage"] for d in hd.voxels], nbinsx=30, marker_color="#b73779"))
    _cov_hist.update_layout(height=430, title="Per-voxel orientation coverage over all _reached voxels (KaRMA-R averages the best five)",
                           xaxis_title="fraction of 228 bins", yaxis_title="voxels", margin=dict(l=0, r=0, t=40, b=0))
    mo.hstack([fig_ori, _cov_hist], widths=[1, 1])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Operating range (animated)

    Each active joint in turn is swept lower limit → upper limit → back while the others hold the
    base pose; the active joint's axis is highlighted in red and the two fingertips leave a red trail.
    The last segment moves every joint on its own sinusoid. Press **play** or drag the slider; frames
    are rendered in the browser.
    """)
    return


@app.cell
def _(mo):
    base_pose = mo.ui.radio(options=["seed pinch", "neutral (q = 0)"], value="seed pinch", label="base pose")
    fpj = mo.ui.slider(start=8, stop=40, value=12, step=4, show_value=True, label="frames per joint")
    frame_ms = mo.ui.slider(start=30, stop=200, value=60, step=10, show_value=True, label="ms per frame")
    mo.hstack([base_pose, fpj, frame_ms], justify="start", gap=2)
    return base_pose, fpj, frame_ms


@app.cell
def _(base_pose, fpj, frame_ms, hd, np):
    from optimize.plotly_hand import range_animation_figure
    _base = hd.seed_q if base_pose.value.startswith("seed") else np.zeros(hd.ctx.model.nq)
    fig_range = range_animation_figure(hd.ctx, hd.cfg, _base, frames_per_joint=fpj.value, wiggle_frames=max(24, 3 * fpj.value),
                                       frame_ms=frame_ms.value, title=f"operating range · base = {base_pose.value}")
    fig_range
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Manual pose: one slider per active joint (degrees)
    """)
    return


@app.cell
def _(hd, mo, np):
    _model = hd.ctx.model
    _names = list(_model.names)
    _lo = np.degrees(np.array(_model.lowerPositionLimit)); _hi = np.degrees(np.array(_model.upperPositionLimit))
    _seed = np.degrees(hd.seed_q)
    joint_rows = []
    for _jn in hd.cfg.finger_joint_names:
        _j = _names.index(_jn); _iq = _model.joints[_j].idx_q
        joint_rows.append((_jn, _iq))
    joint_sliders = mo.ui.array([mo.ui.slider(start=float(np.floor(_lo[_iq])), stop=float(np.ceil(_hi[_iq])), step=1.0,
                                              value=float(np.clip(_seed[_iq], _lo[_iq], _hi[_iq])), show_value=True, label=_jn)
                                 for _jn, _iq in joint_rows])
    _nt = len(hd.cfg.thumb_joint_names)
    mo.hstack([mo.vstack(list(joint_sliders)[:_nt]), mo.vstack(list(joint_sliders)[_nt:])], gap=4)
    return joint_rows, joint_sliders


@app.cell
def _(go, hd, joint_rows, joint_sliders, np):
    from optimize.plotly_hand import hand_traces, scene_layout
    _q = np.zeros(hd.ctx.model.nq)
    for (_jn, _iq), _v in zip(joint_rows, joint_sliders.value):
        _q[_iq] = np.radians(_v)
    _tr, _ = hand_traces(hd.ctx, hd.cfg, _q, None, None, dim_non_contact=False)
    fig_manual = go.Figure(_tr)
    fig_manual.update_layout(**scene_layout("manual pose (no sphere): drag the sliders above", height=620))
    fig_manual
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Voxel sweep (animated)

    The hand walks the metric's Phase-1 search tree in first-visit order, so consecutive frames are one
    rolling step apart. Visited voxel centres accumulate as points over the faint full cloud; the cyan cube
    is the current voxel. Subsampled to the frame budget below (the mp4 from `animate_sweep.py` has every step).
    """)
    return


@app.cell
def _(mo):
    sweep_frames = mo.ui.slider(start=40, stop=300, value=80, step=20, show_value=True, label="frame budget")
    sweep_frames
    return (sweep_frames,)


@app.cell
def _(hd, sweep_frames):
    from optimize.plotly_hand import voxel_sweep_figure
    fig_sweep = voxel_sweep_figure(hd.ctx, hd.cfg, hd.res, max_frames=sweep_frames.value, title="voxel sweep along the search tree")
    fig_sweep
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## All hands: published and optimized
    """)
    return


@app.cell(hide_code=True)
def _(catalogue, go, mo):
    _rows = []
    for _key, _e in catalogue.items():
        _s = _e["summary"]
        if "KaRMA_T" in _s:
            _rows.append(dict(hand=_e["run"], kind=_e["kind"], KaRMA_T=_s["KaRMA_T"], KaRMA_R=_s["KaRMA_R"], KaRMA_S=_s.get("KaRMA_S"),
                             voxels=_s.get("n_voxels"), L_ref_mm=_s.get("L_ref_mm"), pinches=_s.get("n_seeds"), backend=_s.get("backend", "reference")))
    _rows.sort(key=lambda r: -r["KaRMA_T"])
    _bar = go.Figure()
    for _kind, _col in (("published", "#888888"), ("optimized", "#d9822b")):
        _rr = [r for r in _rows if r["kind"] == _kind]
        _bar.add_bar(x=[r["hand"] for r in _rr], y=[r["KaRMA_T"] for r in _rr], name=_kind, marker_color=_col,
                    text=[f"{r['KaRMA_T']:.3f}" for r in _rr], textposition="outside")
    _bar.update_layout(yaxis_type="log", yaxis_title="KaRMA-T (log)", height=420, title="KaRMA-T, every scored hand",
                      xaxis={"categoryorder": "array", "categoryarray": [r["hand"] for r in _rows]}, margin=dict(l=0, r=0, t=40, b=0))
    _sc = go.Figure()
    for _kind, _col in (("published", "#888888"), ("optimized", "#d9822b")):
        _rr = [r for r in _rows if r["kind"] == _kind]
        _sc.add_scatter(x=[r["KaRMA_T"] for r in _rr], y=[r["KaRMA_R"] for r in _rr], mode="markers+text", text=[r["hand"] for r in _rr],
                       textposition="top center", name=_kind, marker=dict(size=10, color=_col))
    _sc.update_layout(xaxis_type="log", xaxis_title="KaRMA-T (log)", yaxis_title="KaRMA-R", height=420, title="Translation vs rotation",
                     margin=dict(l=0, r=0, t=40, b=0))
    mo.vstack([mo.hstack([_bar, _sc], widths=[1, 1]), mo.ui.table(_rows, selection=None, pagination=False)])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Optimizer progress (proxy objective per generation)
    """)
    return


@app.cell
def _(RUNS, go, json, np):
    _prog = go.Figure()
    for _d in sorted(RUNS.iterdir()) if RUNS.exists() else []:
        _ev = _d / "evals.jsonl"
        if not _ev.exists():
            continue
        _E = [json.loads(l) for l in _ev.read_text().splitlines() if l.strip()]
        _E = [e for e in _E if e["gen"] >= 0]
        if not _E:
            continue
        _G = max(e["gen"] for e in _E) + 1
        _best = np.maximum.accumulate([max(e["objective"] for e in _E if e["gen"] == g) for g in range(_G)])
        _mean = [np.mean([e["objective"] for e in _E if e["gen"] == g]) for g in range(_G)]
        _meta = json.loads((_d / "meta.json").read_text()) if (_d / "meta.json").exists() else {}
        _lab = f"{_d.name} ({_meta.get('args', {}).get('objective', '?')}, {_meta.get('backend', 'reference')}/{_meta.get('proxy', 'loop')})"
        _prog.add_scatter(x=list(range(_G)), y=_best, mode="lines", name=_lab + " _best-so-far", line=dict(width=3))
        _prog.add_scatter(x=list(range(_G)), y=_mean, mode="lines", name=_lab + " population _mean", line=dict(width=1, dash="dot"))
    _prog.add_hline(y=0.0969, line_dash="dash", annotation_text="LEAP official T", annotation_position="bottom right")
    _prog.update_layout(height=480, xaxis_title="generation", yaxis_title="proxy objective", margin=dict(l=0, r=0, t=30, b=0),
                       legend=dict(font=dict(size=10)))
    _prog
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Arm F: re-ranking the 16 published hands under alternative L_ref definitions
    """)
    return


@app.cell
def _(RUNS, go, mo, pub_rows, yaml):
    def _load(name):
        p = RUNS / name / "summary.yaml"
        return {_r["robot"]: _r for _r in yaml.safe_load(p.read_text())["rows"]} if p.exists() else {}
    _ship, _pinch, _noadv = _load("armF_shipped"), _load("armF_pinch"), _load("armF_noadvance")
    _frows = []
    for _n, _r in sorted(pub_rows.items(), key=lambda kv: -kv[1]["KaRMA_T"]):
        _frows.append(dict(hand=_n, published_T=_r["KaRMA_T"], shipped_here_T=_ship.get(_n, {}).get("KaRMA_T"),
                          pinch_only_T=_pinch.get(_n, {}).get("KaRMA_T"), no_advance_T=_noadv.get(_n, {}).get("KaRMA_T"),
                          L_ref_published=_r["L_ref_mm"], L_ref_pinch=_pinch.get(_n, {}).get("L_ref_mm"),
                          published_R=_r["KaRMA_R"], pinch_only_R=_pinch.get(_n, {}).get("KaRMA_R")))
    _f = go.Figure()
    _ok = [_r for _r in _frows if _r["pinch_only_T"]]
    _f.add_scatter(x=[_r["published_T"] for _r in _ok], y=[_r["pinch_only_T"] for _r in _ok], mode="markers+text", text=[_r["hand"] for _r in _ok],
                  textposition="top center", marker=dict(size=9, color="#3b7dd8"), name="pinch-only L_ref")
    _lo = min([_r["published_T"] for _r in _ok] + [1e-4]); _hi = max([_r["pinch_only_T"] for _r in _ok] + [0.2])
    _f.add_scatter(x=[_lo, _hi], y=[_lo, _hi], mode="lines", line=dict(color="#999999", dash="dash"), name="no change")
    _f.update_layout(xaxis_type="log", yaxis_type="log", xaxis_title="published KaRMA-T (shipped L_ref)", yaxis_title="KaRMA-T with _pinch-only L_ref",
                    height=480, margin=dict(l=0, r=0, t=30, b=0))
    mo.vstack([_f, mo.ui.table(_frows, selection=None, pagination=False)])
    return


@app.cell
def _(RUNS, mo):
    import csv as _csv
    _fp = RUNS / "features.csv"
    _feat_rows = list(_csv.DictReader(open(_fp))) if _fp.exists() else []
    mo.vstack([mo.md("## Kinematic features (optimize/features.py)"),
               mo.ui.table(_feat_rows, selection=None, pagination=True) if _feat_rows else mo.md("_features.csv not generated yet_")])
    return


if __name__ == "__main__":
    app.run()
