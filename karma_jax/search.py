"""Level-synchronous, multi-seed translation BFS and batched rotation exploration.

Both searches drive the batched kernels in :mod:`karma_jax.steps` from host-side
bookkeeping that reproduces the reference's sequential order exactly:

* **Translation BFS** (:class:`TranslationSearch`). The reference runs one FIFO
  BFS per seed. FIFO order equals level-by-level order, and within a level the
  order is (queue position, primitive index), so expanding a whole level of
  every seed in one batch and then inserting the discoveries in attempt order
  gives the same discovered set, the same representative configuration per
  voxel ("first discoverer wins"), the same parents and the same effort
  statistics, including the mid-level stop at ``budget``.
* **Rotation exploration** (:class:`RotationSearch`). Each (voxel, direction)
  is an independent stream of up to 50 tilt steps; the set of bins a voxel
  reaches is order-independent. Streams are stepped together with host-side
  compaction of dead lanes; the stored configuration per bin is the first one
  reached in the reference's (direction, step) order.

Sphere positions never accumulate rounding: a voxel's centre is reconstructed
on the host in fp64 as ``origin + R_seed @ (h * voxel)`` (Section 5.1 of the
plan), and successor voxels are ``voxel + unit(primitive)`` with no rounding at
all. All per-level bookkeeping is vectorized numpy; nothing loops over
attempts in Python.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
import jax
import jax.numpy as jnp

from .model import FlatModel, DeviceModel
from .kinematics import forward_kinematics
from .geometry import capsule_pass, contact_kinematics
from .rolling import StepParams
from .qp import QPSettings, default_settings
from .steps import (GraspState, Constraints, make_batched_step, make_batched_rotation_step,
                    STEP_FAIL_NAMES, ROT_FAIL_NAMES)
from .healpix import axes_to_bins

VoxelIdx = tuple[int, int, int]

# Primitive k moves along seed axis k // 2 with sign (+, -); k ^ 1 is its reverse.
_PRIM_DIRS = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=np.int64)

ROTATION_STEP_RAD = 0.15
MAX_ROT_STEPS = 50

# Batch sizes are padded to one of a few buckets so that only a handful of
# shapes ever compile (each compile is seconds, each trace ~0.5 s).
_BUCKETS = (1024, 4096, 16384, 32768)


def _bucket(n: int, lo: int = 1024, hi: int = 1 << 15) -> int:
    for b in _BUCKETS:
        if b >= n and lo <= b <= hi:
            return b
    return min(max(n, lo), hi)


def step_params_from_cfg(cfg, dtype) -> StepParams:
    slip = cfg.rotation_slip_threshold_mm / 1000.0 if cfg.rotation_slip_threshold_mm is not None else np.inf
    f = lambda v: jnp.asarray(v, dtype)  # noqa: E731
    return StepParams(
        sphere_radius=f(cfg.sphere_radius_m), link_radius=f(cfg.link_radius_m), eps_g=f(cfg.eps_g_m),
        max_joint_step=f(cfg.max_joint_step_rad), qp_regularization=f(cfg.qp_regularization),
        slip_threshold=f(slip), friction_coeff=f(cfg.friction_coeff), eps_col=f(cfg.eps_col_m),
    )


def reorthogonalize_np(R: np.ndarray) -> np.ndarray:
    """Batched SVD projection onto SO(3) with det +1 (reference ``_reorthogonalize``). ``[...,3,3]``."""
    R = np.asarray(R, float)
    U, _, Vt = np.linalg.svd(R)
    Rc = U @ Vt
    neg = np.linalg.det(Rc) < 0
    if np.any(neg):
        U = U.copy()
        U[neg, :, -1] *= -1
        Rc = U @ Vt
    return Rc


def _pad_rows(a: np.ndarray, n: int) -> np.ndarray:
    pad = n - a.shape[0]
    return np.concatenate([a, np.repeat(a[-1:], pad, axis=0)], axis=0) if pad else a


# ── Translation BFS ──────────────────────────────────────────────────────────

@dataclass
class SeedSpec:
    q: np.ndarray            # [nq] full configuration (reference layout)
    p: np.ndarray            # [3] sphere centre
    pair: tuple[int, int]    # capsule indices (thumb, index)
    R_seed: np.ndarray       # [3,3] grid frame (columns = axes)


@dataclass
class BFSNode:
    voxel: VoxelIdx
    q: np.ndarray            # [nq] fp64
    p: np.ndarray            # [3] fp64 lattice point
    R: np.ndarray            # [3,3] fp64, orthonormalized
    pair: tuple[int, int]
    parent: int              # node index, -1 for the seed
    arrival_prim: int        # -1 for the seed


@dataclass
class BFSResult:
    """Discovered voxels of one seed, stored as arrays (``nodes`` materializes objects)."""
    vox: np.ndarray          # [N,3] int64 voxel indices, discovery order
    q: np.ndarray            # [N,nq]
    p: np.ndarray            # [N,3]
    R: np.ndarray            # [N,3,3]
    pair: np.ndarray         # [N,2] int32
    parent: np.ndarray       # [N] int64
    prim: np.ndarray         # [N] int64
    attempts: int = 0
    fail_moves: int = 0
    revisits: int = 0
    unique_boundary: int = 0
    frontier_size: int = 0
    fail_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def n_voxels(self) -> int:
        return int(self.vox.shape[0])

    @cached_property
    def nodes(self) -> list[BFSNode]:
        return [BFSNode(tuple(int(v) for v in self.vox[i]), self.q[i], self.p[i], self.R[i],
                        (int(self.pair[i, 0]), int(self.pair[i, 1])), int(self.parent[i]), int(self.prim[i]))
                for i in range(self.n_voxels)]

    @cached_property
    def voxel_to_node(self) -> dict[VoxelIdx, int]:
        return {tuple(int(v) for v in self.vox[i]): i for i in range(self.n_voxels)}


class _SeedRun:
    """Growable per-seed storage and BFS state."""

    def __init__(self, spec: SeedSpec, nq: int, h: float):
        self.spec = spec
        self.h = h
        self.prims = np.stack([float(d[0] + d[1] + d[2]) * h * spec.R_seed[:, int(np.argmax(np.abs(d)))]
                               for d in _PRIM_DIRS])                       # [6,3]
        self.vox = np.zeros((1, 3), np.int64)
        self.q = np.asarray(spec.q, float).reshape(1, nq).copy()
        self.p = np.asarray(spec.p, float).reshape(1, 3).copy()
        self.R = np.eye(3)[None]
        self.pair = np.asarray(spec.pair, np.int32).reshape(1, 2)
        self.parent = np.array([-1], np.int64)
        self.prim = np.array([-1], np.int64)
        self.keys: set[tuple[int, int, int]] = {(0, 0, 0)}
        self.frontier = np.array([0], np.int64)
        self.done = False
        self.attempts = self.fail_moves = self.revisits = self.unique_boundary = 0
        self.fail_counts = np.zeros(len(STEP_FAIL_NAMES), np.int64)
        self.frontier_size = 0

    @property
    def n(self) -> int:
        return int(self.vox.shape[0])

    def level_attempts(self):
        """(node indices, primitive indices) of this level's attempts in reference order."""
        F = self.frontier
        node = np.repeat(F, 6)
        prim = np.tile(np.arange(6), F.shape[0])
        arr = self.prim[node]
        keep = ~((arr >= 0) & (prim == (arr ^ 1)))
        return node[keep], prim[keep]

    def lattice(self, vox: np.ndarray) -> np.ndarray:
        return self.spec.p[None, :] + (self.h * vox.astype(float)) @ self.spec.R_seed.T

    def result(self) -> BFSResult:
        res = BFSResult(self.vox, self.q, self.p, self.R, self.pair, self.parent, self.prim,
                        attempts=self.attempts, fail_moves=self.fail_moves, revisits=self.revisits,
                        unique_boundary=self.unique_boundary, frontier_size=self.frontier_size)
        for k, c in enumerate(self.fail_counts):
            if c:
                res.fail_reasons[STEP_FAIL_NAMES[k]] = int(c)
        return res


class TranslationSearch:
    """Batched multi-seed translation BFS (Phase 1 and the seed-ranking trial BFS)."""

    def __init__(self, flat: FlatModel, cfg, dtype=jnp.float32, cons: Constraints = Constraints(),
                 qp_settings: QPSettings | None = None, max_batch: int = 1 << 15):
        self.flat = flat
        self.cfg = cfg
        self.dtype = dtype
        self.dm: DeviceModel = flat.to_device(dtype)
        self.params = step_params_from_cfg(cfg, dtype)
        self.qp_settings = qp_settings or default_settings(dtype)
        self.n_sub = max(1, int(np.ceil(cfg.voxel_size_m / cfg.step_size_m)))
        self.max_batch = max_batch
        self._step = make_batched_step(self.dm, self.n_sub, self.params, self.qp_settings, cons)

    def _run_step(self, q, p, R, pair, dp):
        """Run the batched step on host arrays, in bucketed chunks. Returns host arrays."""
        B = q.shape[0]
        outs = []
        for s in range(0, B, self.max_batch):
            e = min(B, s + self.max_batch)
            n = e - s
            nb = _bucket(n, hi=self.max_batch)
            st = GraspState(jnp.asarray(_pad_rows(q[s:e], nb), self.dtype),
                            jnp.asarray(_pad_rows(p[s:e], nb), self.dtype),
                            jnp.asarray(_pad_rows(R[s:e], nb), self.dtype),
                            jnp.asarray(_pad_rows(pair[s:e], nb), jnp.int32))
            out = jax.device_get(self._step(st, jnp.asarray(_pad_rows(dp[s:e], nb), self.dtype)))
            outs.append(dict(ok=np.asarray(out.ok)[:n], code=np.asarray(out.code)[:n],
                             q=np.asarray(out.state.q)[:n], R=np.asarray(out.state.R)[:n],
                             pair=np.asarray(out.state.pair)[:n]))
        return {k: np.concatenate([o[k] for o in outs], axis=0) for k in outs[0]}

    def run(self, seeds: list[SeedSpec], budget: int) -> list[BFSResult]:
        J = self.flat.J_pad
        nq = self.flat.n_joints
        h = self.cfg.voxel_size_m
        runs = [_SeedRun(sp, nq, h) for sp in seeds]

        while True:
            # ── gather this level's attempts (all alive seeds, reference order) ──
            chunks = []
            for si, run in enumerate(runs):
                if run.done:
                    continue
                if run.frontier.size == 0 or run.n >= budget:
                    run.done = True
                    run.frontier_size = int(run.frontier.size)
                    continue
                node, prim = run.level_attempts()
                chunks.append((si, node, prim))
            if not chunks:
                break
            B = sum(c[1].shape[0] for c in chunks)
            q = np.zeros((B, J))
            p = np.empty((B, 3))
            R = np.empty((B, 3, 3))
            pair = np.empty((B, 2), np.int32)
            dp = np.empty((B, 3))
            off = 0
            for si, node, prim in chunks:
                run = runs[si]
                n = node.shape[0]
                sl = slice(off, off + n)
                q[sl, :nq] = run.q[node]
                p[sl] = run.p[node]
                R[sl] = run.R[node]
                pair[sl] = run.pair[node]
                dp[sl] = run.prims[prim]
                off += n
            out = self._run_step(q, p, R, pair, dp)

            # ── insert discoveries per seed in attempt order, with the budget rule ──
            off = 0
            for si, node, prim in chunks:
                run = runs[si]
                n = node.shape[0]
                sl = slice(off, off + n)
                off += n
                ok = out["ok"][sl]
                codes = out["code"][sl]
                vox_new = run.vox[node] + _PRIM_DIRS[prim]                       # [n,3]
                keys = vox_new[:, 0] * (1 << 42) + vox_new[:, 1] * (1 << 21) + vox_new[:, 2]
                # candidates: ok attempts whose voxel is new (not discovered, first in level)
                ok_idx = np.nonzero(ok)[0]
                is_new = np.zeros(n, bool)
                if ok_idx.size:
                    _, first = np.unique(keys[ok_idx], return_index=True)
                    cand = np.sort(ok_idx[first])
                    known = np.array([tuple(v) in run.keys for v in vox_new[cand]], bool) if cand.size else np.zeros(0, bool)
                    cand = cand[~known]
                    is_new[cand] = True
                new_idx = np.nonzero(is_new)[0]                                   # ascending attempt order
                room = budget - run.n
                stop = new_idx.size >= room
                if stop:
                    cutoff = int(new_idx[room - 1])                                # last counted attempt
                    new_idx = new_idx[:room]
                else:
                    cutoff = n - 1
                counted = np.arange(n) <= cutoff
                run.attempts += int(counted.sum())
                fails = counted & ~ok
                run.fail_moves += int(fails.sum())
                run.fail_counts += np.bincount(codes[fails], minlength=len(STEP_FAIL_NAMES))
                run.revisits += int((counted & ok & ~is_new).sum())
                # boundary: expanded nodes (first attempt counted) with no new voxel
                expanded_nodes = np.unique(node[counted])
                nodes_with_new = np.unique(node[new_idx]) if new_idx.size else np.zeros(0, np.int64)
                run.unique_boundary += int(np.setdiff1d(expanded_nodes, nodes_with_new).size)

                if new_idx.size:
                    k = new_idx.size
                    vnew = vox_new[new_idx]
                    base = run.n
                    run.vox = np.concatenate([run.vox, vnew])
                    run.q = np.concatenate([run.q, out["q"][sl][new_idx][:, :nq].astype(float)])
                    run.p = np.concatenate([run.p, run.lattice(vnew)])
                    run.R = np.concatenate([run.R, reorthogonalize_np(out["R"][sl][new_idx])])
                    run.pair = np.concatenate([run.pair, out["pair"][sl][new_idx].astype(np.int32)])
                    run.parent = np.concatenate([run.parent, node[new_idx]])
                    run.prim = np.concatenate([run.prim, prim[new_idx]])
                    run.keys.update(tuple(int(x) for x in v) for v in vnew)
                    next_frontier = np.arange(base, base + k)
                else:
                    next_frontier = np.zeros(0, np.int64)
                if stop:
                    # nodes of this level not yet expanded stay queued ahead of the new ones
                    remaining = run.frontier[~np.isin(run.frontier, expanded_nodes)]
                    run.frontier = np.concatenate([remaining, next_frontier])
                    run.frontier_size = int(run.frontier.size)
                    run.done = True
                else:
                    run.frontier = next_frontier
        for run in runs:
            if not run.done:
                run.frontier_size = int(run.frontier.size)
        return [r.result() for r in runs]


# ── Rotation exploration (Phase 2) ───────────────────────────────────────────

@dataclass
class VoxelStart:
    q: np.ndarray
    p: np.ndarray
    R: np.ndarray
    pair: tuple[int, int]


@dataclass
class VoxelRotationResult:
    initial_bin: int
    configs: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]   # bin -> (q, R, pair)
    fail_reasons: dict[str, int] = field(default_factory=dict)

    @property
    def bins(self) -> set[int]:
        return set(self.configs.keys())


def rotation_primitives_np(pinch_axis_w: np.ndarray, step: float = ROTATION_STEP_RAD) -> np.ndarray:
    """Reference ``compute_rotation_primitives`` for ``[V,3]`` pinch axes -> ``[V,4,3]``."""
    g = pinch_axis_w / np.linalg.norm(pinch_axis_w, axis=-1, keepdims=True)
    ref = np.where((np.abs(g[:, 0]) < 0.9)[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    perp1 = np.cross(g, ref)
    perp1 /= np.linalg.norm(perp1, axis=-1, keepdims=True)
    perp2 = np.cross(g, perp1)
    return np.stack([step * perp1, -step * perp1, step * perp2, -step * perp2], axis=1)


class RotationSearch:
    def __init__(self, flat: FlatModel, cfg, dtype=jnp.float32, cons: Constraints = Constraints(),
                 qp_settings: QPSettings | None = None, max_batch: int = 1 << 15):
        self.flat = flat
        self.cfg = cfg
        self.dtype = dtype
        self.dm = flat.to_device(dtype)
        self.params = step_params_from_cfg(cfg, dtype)
        self.qp_settings = qp_settings or default_settings(dtype)
        self.max_batch = max_batch
        self._rot = make_batched_rotation_step(self.dm, self.params, self.qp_settings, cons)
        dm, r_s, r_l = self.dm, self.params.sphere_radius, self.params.link_radius

        def normals(q, p, pair):
            fk = forward_kinematics(dm, q)
            cp = capsule_pass(dm, fk, p, r_s, r_l)
            return contact_kinematics(dm, fk, cp, pair[0], p, r_s).normal, \
                contact_kinematics(dm, fk, cp, pair[1], p, r_s).normal
        self._normals = jax.jit(jax.vmap(normals))

    def _pad_q(self, qs: np.ndarray) -> np.ndarray:
        out = np.zeros((qs.shape[0], self.flat.J_pad))
        out[:, : self.flat.n_joints] = qs
        return out

    def contact_normals(self, q, p, pair):
        """Thumb and index contact normals for host arrays ``q [N,nq], p [N,3], pair [N,2]``."""
        N = q.shape[0]
        nb = _bucket(N, hi=self.max_batch)
        outs = []
        for s in range(0, N, self.max_batch):
            e = min(N, s + self.max_batch)
            nb = _bucket(e - s, hi=self.max_batch)
            n_t, n_i = self._normals(jnp.asarray(_pad_rows(self._pad_q(q[s:e]), nb), self.dtype),
                                     jnp.asarray(_pad_rows(p[s:e], nb), self.dtype),
                                     jnp.asarray(_pad_rows(pair[s:e], nb), jnp.int32))
            outs.append((np.asarray(n_t, float)[: e - s], np.asarray(n_i, float)[: e - s]))
        return np.concatenate([o[0] for o in outs]), np.concatenate([o[1] for o in outs])

    def _run_rot(self, q, p, R, pair, dth):
        B = q.shape[0]
        outs = []
        for s in range(0, B, self.max_batch):
            e = min(B, s + self.max_batch)
            n = e - s
            nb = _bucket(n, hi=self.max_batch)
            st = GraspState(jnp.asarray(_pad_rows(q[s:e], nb), self.dtype),
                            jnp.asarray(_pad_rows(p[s:e], nb), self.dtype),
                            jnp.asarray(_pad_rows(R[s:e], nb), self.dtype),
                            jnp.asarray(_pad_rows(pair[s:e], nb), jnp.int32))
            out = jax.device_get(self._rot(st, jnp.asarray(_pad_rows(dth[s:e], nb), self.dtype)))
            outs.append(dict(ok=np.asarray(out.ok)[:n], code=np.asarray(out.code)[:n],
                             q=np.asarray(out.state.q)[:n], R=np.asarray(out.state.R)[:n],
                             pair=np.asarray(out.state.pair)[:n], axis=np.asarray(out.axis_body)[:n]))
        return {k: np.concatenate([o[k] for o in outs], axis=0) for k in outs[0]}

    def run(self, voxels: list[VoxelStart]) -> list[VoxelRotationResult]:
        nside = self.cfg.healpix_nside
        nq = self.flat.n_joints
        V = len(voxels)
        if V == 0:
            return []
        q0 = np.stack([v.q for v in voxels])
        p0 = np.stack([v.p for v in voxels])
        R0 = np.stack([v.R for v in voxels])
        pair0 = np.array([v.pair for v in voxels], dtype=np.int32)
        n_t, _ = self.contact_normals(q0, p0, pair0)
        pinch_w = -n_t
        init_axis_b = np.einsum("vji,vj->vi", R0, pinch_w / np.linalg.norm(pinch_w, axis=1, keepdims=True))
        init_bins = axes_to_bins(init_axis_b, nside)

        # streams: 4 per voxel
        S = 4 * V
        s_vox = np.repeat(np.arange(V), 4)
        s_dir = np.tile(np.arange(4), V)
        dth = rotation_primitives_np(pinch_w).reshape(S, 3)
        q = self._pad_q(q0[s_vox])
        p = p0[s_vox]
        R = R0[s_vox]
        pair = pair0[s_vox]
        alive = np.arange(S)
        fail_counts = np.zeros((V, len(ROT_FAIL_NAMES)), np.int64)
        ev_stream, ev_step, ev_bin, ev_q, ev_R, ev_pair = [], [], [], [], [], []

        for step in range(MAX_ROT_STEPS):
            if alive.size == 0:
                break
            out = self._run_rot(q[alive], p[alive], R[alive], pair[alive], dth[alive])
            ok = out["ok"]
            dead = alive[~ok]
            if dead.size:
                np.add.at(fail_counts, (s_vox[dead], out["code"][~ok]), 1)
            live = alive[ok]
            if live.size:
                Rn = reorthogonalize_np(out["R"][ok])
                q[live] = out["q"][ok]
                R[live] = Rn
                pair[live] = out["pair"][ok]
                ev_stream.append(live)
                ev_step.append(np.full(live.size, step, np.int64))
                ev_bin.append(axes_to_bins(out["axis"][ok], nside))
                ev_q.append(q[live, :nq].copy())
                ev_R.append(Rn)
                ev_pair.append(pair[live].copy())
            alive = live

        results = [VoxelRotationResult(int(init_bins[v]), {int(init_bins[v]): (q0[v].copy(), R0[v].copy(),
                                                                                 (int(pair0[v, 0]), int(pair0[v, 1])))})
                   for v in range(V)]
        for v in range(V):
            for name_idx, c in enumerate(fail_counts[v]):
                if c:
                    results[v].fail_reasons[ROT_FAIL_NAMES[name_idx]] = int(c)
        if ev_stream:
            st = np.concatenate(ev_stream)
            stp = np.concatenate(ev_step)
            bn = np.concatenate(ev_bin)
            qq = np.concatenate(ev_q)
            RR = np.concatenate(ev_R)
            pp = np.concatenate(ev_pair)
            vv = s_vox[st]
            dd = s_dir[st]
            # first-reached config per (voxel, bin) in the reference's (direction, step) order
            order = np.lexsort((stp, dd, vv))
            key = vv[order] * 100000 + bn[order]
            _, first = np.unique(key, return_index=True)
            for i in order[first]:
                b = int(bn[i])
                res = results[int(vv[i])]
                if b not in res.configs:
                    res.configs[b] = (qq[i], RR[i], (int(pp[i, 0]), int(pp[i, 1])))
        return results
