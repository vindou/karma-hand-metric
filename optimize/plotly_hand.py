"""Plotly traces for a KaRMA hand pose (shared by the explorer notebook's cells)."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from karma.viser_geometry import capsule_mesh, sphere_mesh
from optimize.viz import hand_pose, rot_z_to, THUMB_C, INDEX_C, THUMB_DIM, INDEX_DIM, PALM_C

AXIS_LEN_R = 4.6          # joint-axis arrow length in link radii
THUMB_AX, INDEX_AX = "#7a3b00", "#0b2f73"


def hand_traces(ctx, cfg, q, centre=None, pair=None, highlight_joint=None, dim_non_contact=True, show_axes=True):
    """Capsules (thumb orange / index blue, non-contact links dimmed), joint-axis arrows,
    palm bar, sphere and contact normals, all in millimetres."""
    g = hand_pose(ctx, cfg, q, centre, pair)
    r_l = cfg.link_radius_m * 1e3
    traces = []
    for name, a, b, is_thumb, is_contact in g["capsules"]:
        a, b = a * 1e3, b * 1e3
        seg = b - a; L = np.linalg.norm(seg)
        verts, faces = capsule_mesh(r_l, max(L / 2, 1e-6), n_rings=10, n_segments=18)
        verts = verts @ (rot_z_to(seg) if L > 1e-9 else np.eye(3)).T + 0.5 * (a + b)
        dim = dim_non_contact and pair is not None and not is_contact
        col = (THUMB_DIM if dim else THUMB_C) if is_thumb else (INDEX_DIM if dim else INDEX_C)
        traces.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                color=col, opacity=1.0, name=name + (" (in contact)" if is_contact else ""), hoverinfo="name",
                                lighting=dict(ambient=0.5, diffuse=0.75, specular=0.2, roughness=0.6), flatshading=False, showlegend=False))
    if show_axes:
        alen = AXIS_LEN_R * r_l
        for j in g["joints"]:
            o = j["origin"] * 1e3; ax = j["axis"]
            hl = highlight_joint is not None and j["name"] == highlight_joint
            p0, p1 = o - 0.5 * alen * ax, o + 0.5 * alen * ax
            c = ("#ff2d55" if hl else (THUMB_AX if j["is_thumb"] else INDEX_AX))
            hover = (f"{j['name']}<br>q = {np.degrees(j['q']):.1f}° of [{np.degrees(j['lo']):.0f}°, {np.degrees(j['hi']):.0f}°]"
                     f"<br>axis (world) = ({ax[0]:+.2f}, {ax[1]:+.2f}, {ax[2]:+.2f})" + ("" if j["active"] else "<br>(not an active joint)"))
            traces.append(go.Scatter3d(x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode="lines",
                                       line=dict(color=c, width=11 if hl else 7), text=[hover, hover], hoverinfo="text", showlegend=False))
            traces.append(go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[ax[0]], v=[ax[1]], w=[ax[2]], sizemode="absolute",
                                  sizeref=(1.2 if hl else 0.9) * r_l, anchor="tail", colorscale=[[0, c], [1, c]], showscale=False,
                                  hoverinfo="skip", showlegend=False))
            traces.append(go.Scatter3d(x=[o[0]], y=[o[1]], z=[o[2]], mode="markers", marker=dict(size=6 if hl else 4, color=c),
                                       hoverinfo="skip", showlegend=False))
    if len(g["knuckles"]) >= 2:
        K = g["knuckles"] * 1e3
        traces.append(go.Scatter3d(x=K[:, 0], y=K[:, 1], z=K[:, 2], mode="lines", line=dict(color=PALM_C, width=12),
                                   name="palm (knuckle to knuckle)", hoverinfo="name"))
    if centre is not None:
        sv, sf = sphere_mesh(cfg.sphere_radius_m * 1e3, 14, 20)
        sv = np.asarray(sv) + np.asarray(centre) * 1e3
        traces.append(go.Mesh3d(x=sv[:, 0], y=sv[:, 1], z=sv[:, 2], i=sf[:, 0], j=sf[:, 1], k=sf[:, 2], color="#3a3a3a", opacity=1.0,
                                name="sphere", hoverinfo="name", lighting=dict(ambient=0.5, diffuse=0.7, specular=0.3), showlegend=False))
        for ct in g["contacts"]:
            pt = ct["point"] * 1e3; n = ct["normal"]; nl = 1.5 * cfg.sphere_radius_m * 1e3
            traces.append(go.Scatter3d(x=[pt[0], pt[0] + nl * n[0]], y=[pt[1], pt[1] + nl * n[1]], z=[pt[2], pt[2] + nl * n[2]],
                                       mode="lines+markers", line=dict(color="#00c2a8", width=5), marker=dict(size=[6, 0], color="#00c2a8"),
                                       name=f"contact {ct['link']} (gap {ct['gap']*1e3:+.3f} mm)", hoverinfo="name"))
    return traces, g


def scene_layout(title: str, height: int = 760):
    return dict(scene=dict(aspectmode="data", xaxis_title="x [mm]", yaxis_title="y [mm]", zaxis_title="z [mm]",
                           camera=dict(eye=dict(x=1.4, y=-1.6, z=0.9))),
                height=height, margin=dict(l=0, r=0, t=30, b=0), legend=dict(x=0.01, y=0.99), title=title)


# ── animations (client-side Plotly frames) ──────────────────────────────────

def _lean_traces(ctx, cfg, q, highlight=None, trail=None, rings=4, segs=8):
    """Fixed-layout trace list for animation frames: capsules, axis shafts+cones+dots, palm, two trails."""
    g = hand_pose(ctx, cfg, q)
    r_l = cfg.link_radius_m * 1e3
    tr = []
    for name, a, b, is_thumb, _ in g["capsules"]:
        a, b = a * 1e3, b * 1e3
        seg = b - a; L = np.linalg.norm(seg)
        verts, faces = capsule_mesh(r_l, max(L / 2, 1e-6), n_rings=rings, n_segments=segs)
        verts = np.round(verts @ (rot_z_to(seg) if L > 1e-9 else np.eye(3)).T + 0.5 * (a + b), 2)
        tr.append(go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            color=THUMB_C if is_thumb else INDEX_C, name=name, hoverinfo="name", flatshading=False,
                            lighting=dict(ambient=0.5, diffuse=0.75, specular=0.2), showlegend=False))
    alen = AXIS_LEN_R * r_l
    for j in g["joints"]:
        o = j["origin"] * 1e3; ax = j["axis"]; hl = (j["name"] == highlight)
        L_ = alen * (1.35 if hl else 1.0)
        p0, p1 = o - 0.5 * L_ * ax, o + 0.5 * L_ * ax
        c = "#ff2d55" if hl else (THUMB_AX if j["is_thumb"] else INDEX_AX)
        txt = f"{j['name']} {np.degrees(j['q']):+.0f}° of [{np.degrees(j['lo']):+.0f}°, {np.degrees(j['hi']):+.0f}°]"
        tr.append(go.Scatter3d(x=[p0[0], p1[0]], y=[p0[1], p1[1]], z=[p0[2], p1[2]], mode="lines", line=dict(color=c, width=11 if hl else 7),
                               text=[txt, txt], hoverinfo="text", showlegend=False))
        tr.append(go.Cone(x=[p1[0]], y=[p1[1]], z=[p1[2]], u=[ax[0]], v=[ax[1]], w=[ax[2]], sizemode="absolute", sizeref=(1.2 if hl else 0.9) * r_l,
                          anchor="tail", colorscale=[[0, c], [1, c]], showscale=False, hoverinfo="skip", showlegend=False))
        tr.append(go.Scatter3d(x=[o[0]], y=[o[1]], z=[o[2]], mode="markers", marker=dict(size=6 if hl else 4, color=c), hoverinfo="skip", showlegend=False))
    K = g["knuckles"] * 1e3
    tr.append(go.Scatter3d(x=K[:, 0], y=K[:, 1], z=K[:, 2], mode="lines", line=dict(color=PALM_C, width=12), hoverinfo="skip", showlegend=False))
    tips = {name: b * 1e3 for name, a, b, _, _ in g["capsules"]}
    for link in (cfg.seed_thumb_link, cfg.seed_index_link):
        T = np.array(trail[link]) if trail and len(trail.get(link, [])) > 1 else np.zeros((0, 3))
        tr.append(go.Scatter3d(x=T[:, 0], y=T[:, 1], z=T[:, 2], mode="lines", line=dict(color="#ff2d55", width=4), hoverinfo="skip", showlegend=False))
    return tr, tips


def _animated_figure(frames_data, labels, title, height=760, frame_ms=60, scene_pts=None, static_traces=()):
    """Static traces are sent once (trace indices 0..k-1); each frame only updates the animated traces."""
    static_traces = list(static_traces); k = len(static_traces)
    anim_idx = list(range(k, k + len(frames_data[0])))
    fig = go.Figure(data=static_traces + list(frames_data[0]),
                    frames=[go.Frame(data=d, name=str(i), traces=anim_idx) for i, d in enumerate(frames_data)])
    steps = [dict(method="animate", args=[[str(i)], dict(mode="immediate", frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0))],
                  label=lab) for i, lab in enumerate(labels)]
    layout = scene_layout(title, height)
    if scene_pts is not None and len(scene_pts):
        P = np.asarray(scene_pts); c = 0.5 * (P.max(0) + P.min(0)); r = 0.5 * float((P.max(0) - P.min(0)).max()) * 1.05 + 10
        layout["scene"].update(xaxis=dict(range=[c[0] - r, c[0] + r], title="x [mm]"), yaxis=dict(range=[c[1] - r, c[1] + r], title="y [mm]"),
                               zaxis=dict(range=[c[2] - r, c[2] + r], title="z [mm]"), aspectmode="cube")
    layout.update(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02, xanchor="left", yanchor="bottom",
                          buttons=[dict(label="▶ play", method="animate",
                                        args=[None, dict(frame=dict(duration=frame_ms, redraw=True), transition=dict(duration=0), fromcurrent=True, mode="immediate")]),
                                   dict(label="❚❚ pause", method="animate",
                                        args=[[None], dict(frame=dict(duration=0, redraw=False), transition=dict(duration=0), mode="immediate")])])],
        sliders=[dict(active=0, steps=steps, x=0.15, y=0.02, len=0.8, currentvalue=dict(prefix="", font=dict(size=12)), pad=dict(t=10))],
    )
    fig.update_layout(**layout)
    return fig


def range_animation_figure(ctx, cfg, base_q, frames_per_joint=16, wiggle_frames=48, frame_ms=60, title="operating range"):
    from optimize.viz import joint_sweep_plan
    plan = joint_sweep_plan(ctx, cfg, base_q, frames_per_joint, wiggle_frames, fps=1000 / frame_ms)
    frames, labels, pts = [], [], []
    trail = {cfg.seed_thumb_link: [], cfg.seed_index_link: []}
    last = object()
    for q, hl, lab in plan:
        if hl != last:
            trail = {k: [] for k in trail}; last = hl
        tr, tips = _lean_traces(ctx, cfg, q, hl, trail)
        for k in trail:
            trail[k].append(tips[k])
        frames.append(tr); labels.append(lab)
        for t in tr[: len(ctx.link_capsules)]:
            pts.append([t.x.min(), t.y.min(), t.z.min()]); pts.append([t.x.max(), t.y.max(), t.z.max()])
    return _animated_figure(frames, labels, title, frame_ms=frame_ms, scene_pts=np.array(pts))


def voxel_sweep_figure(ctx, cfg, res, max_frames=200, frame_ms=70, title="voxel sweep"):
    """Hand walking the Phase-1 search tree (first-visit order), subsampled to max_frames;
    visited voxel centres accumulate as points, the full cloud stays as a faint backdrop."""
    from optimize.voxel_tour import tour_order
    from optimize.viz import seed_frame, voxel_cubes
    order, nodes = tour_order(res)
    stride = max(1, int(np.ceil(len(order) / max_frames)))
    idx = list(range(0, len(order), stride))
    if idx[-1] != len(order) - 1:
        idx.append(len(order) - 1)
    P = np.array([nodes[v]["centre"] for v in order]) * 1e3
    cov = np.array([nodes[v]["coverage"] for v in order]); vmax = max(0.25, float(cov.max()))
    edge = cfg.voxel_size_m * 1e3; R = seed_frame(res)
    # static backdrop: all voxels as faint cubes (one mesh, not per frame)
    verts, faces, owner = voxel_cubes(P, edge, R, shrink=0.85)
    backdrop = go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                         intensity=cov[owner], colorscale="Magma", cmin=0, cmax=vmax, opacity=0.08, flatshading=True, hoverinfo="skip",
                         showscale=False, showlegend=False, name="all voxels")
    frames, labels, pts = [], [], []
    for n_i, i in enumerate(idx):
        v = order[i]
        tr, _ = _lean_traces(ctx, cfg, nodes[v]["q"])
        c = nodes[v]["centre"] * 1e3
        sv, sf = sphere_mesh(cfg.sphere_radius_m * 1e3, 10, 14); sv = np.asarray(sv) + c
        tr.append(go.Mesh3d(x=sv[:, 0], y=sv[:, 1], z=sv[:, 2], i=sf[:, 0], j=sf[:, 1], k=sf[:, 2], color="#3a3a3a", hoverinfo="skip", showlegend=False))
        Pv = np.round(P[: i + 1], 1)
        tr.append(go.Scatter3d(x=Pv[:, 0], y=Pv[:, 1], z=Pv[:, 2], mode="markers",
                               marker=dict(size=3, color=np.round(cov[: i + 1], 3), colorscale="Magma", cmin=0, cmax=vmax, opacity=0.9), hoverinfo="skip", showlegend=False))
        v2, f2, _ = voxel_cubes(c[None, :], edge, R, shrink=1.05)
        tr.append(go.Mesh3d(x=v2[:, 0], y=v2[:, 1], z=v2[:, 2], i=f2[:, 0], j=f2[:, 1], k=f2[:, 2], color="#00e5ff", opacity=0.9, hoverinfo="skip", showlegend=False))
        frames.append(tr); labels.append(f"voxel {i + 1}/{len(order)} · coverage {cov[i]:.2f}")
    pts = np.vstack([P, np.zeros((1, 3))])
    return _animated_figure(frames, labels, title, frame_ms=frame_ms, scene_pts=pts, static_traces=[backdrop])
