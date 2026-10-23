"""Orientation targets (SPEC.md §4.5, D16): the reference's 228 antipodal pinch-axis bins and
deterministic Hopf-fibration grids on SO(3).

Pinch-axis bins
    HEALPix ``nside = 6`` (432 pixels, RING order). The reference canonicalizes a line
    direction to the ``z >= 0`` hemisphere and bins it with ``healpy.ang2pix``; the
    reachable pixels are exactly the first ``6 nside^2 + 2 nside = 228`` RING indices
    (north cap, northern belt and the equator ring). :func:`pinch_axis_bin_centres`
    returns their unit centre vectors ``[228, 3]``; the smooth kernel uses
    ``(a . b)^2`` so the antipodal identification is automatic.

SO(3) grids
    :func:`hopf_grid` builds the Yershova–Jain–LaValle–Mitchell grid: HEALPix
    pixel centres on S^2 (``nside``) times ``m`` equispaced points on the fibre circle,
    mapped to quaternions by the Hopf coordinates
    ``(cos(th/2) cos(psi/2), cos(th/2) sin(psi/2), sin(th/2) cos(ph + psi/2), sin(th/2) sin(ph + psi/2))``.
    Levels: ``(nside, m) = (1, 6) -> 72``, ``(2, 12) -> 576`` (the smooth target set),
    ``(4, 24) -> 4608`` (the exact bin set). The fibre offset is half a cell so no
    sample sits on a HEALPix pixel boundary's antipode.

Everything is host-side numpy computed once; the returned arrays are constants.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

PINCH_NSIDE = 6
PINCH_N_BINS = 6 * PINCH_NSIDE ** 2 + 2 * PINCH_NSIDE     # 228
HOPF_LEVELS = {72: (1, 6), 576: (2, 12), 4608: (4, 24)}


@lru_cache(maxsize=None)
def pinch_axis_bin_centres(nside: int = PINCH_NSIDE) -> np.ndarray:
    """Unit centre vectors of the ``6 nside^2 + 2 nside`` canonical line bins, ``[N, 3]``."""
    import healpy as hp
    n = 6 * nside * nside + 2 * nside
    x, y, z = hp.pix2vec(nside, np.arange(n))
    v = np.stack([x, y, z], axis=1)
    assert np.all(v[:, 2] >= -1e-12), "canonical bins lie in the z >= 0 hemisphere"
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def pinch_axis_bins(axes: np.ndarray, nside: int = PINCH_NSIDE) -> np.ndarray:
    """Exact bin index of line directions ``[N, 3]`` (the reference's binning)."""
    from karma_jax.healpix import axes_to_bins
    return axes_to_bins(np.asarray(axes, float).reshape(-1, 3), nside)


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """``[..., 4]`` unit quaternions (w, x, y, z) -> ``[..., 3, 3]``."""
    q = np.asarray(q, float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


@lru_cache(maxsize=None)
def hopf_grid(n: int = 576) -> np.ndarray:
    """``[n, 4]`` unit quaternions of the Hopf grid with ``n`` in ``HOPF_LEVELS``."""
    import healpy as hp
    nside, m = HOPF_LEVELS[n]
    npix = hp.nside2npix(nside)
    theta, phi = hp.pix2ang(nside, np.arange(npix))
    psi = (np.arange(m) + 0.5) * (2.0 * np.pi / m)
    th = np.repeat(theta, m)
    ph = np.repeat(phi, m)
    ps = np.tile(psi, npix)
    q = np.stack([np.cos(th / 2) * np.cos(ps / 2), np.cos(th / 2) * np.sin(ps / 2),
                  np.sin(th / 2) * np.cos(ph + ps / 2), np.sin(th / 2) * np.sin(ph + ps / 2)], axis=1)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def hopf_rotations(n: int = 576) -> np.ndarray:
    """``[n, 3, 3]`` rotation matrices of the Hopf grid."""
    return quat_to_matrix(hopf_grid(n))


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """Geodesic angle between rotations (broadcasts over leading axes)."""
    tr = np.einsum("...ij,...ij->...", Ra, Rb)
    return np.arccos(np.clip((tr - 1.0) / 2.0, -1.0, 1.0))


def nearest_neighbour_angles(R: np.ndarray) -> np.ndarray:
    """Per-sample geodesic distance to its nearest other sample (for uniformity checks)."""
    q = R
    n = q.shape[0]
    out = np.empty(n)
    for i in range(n):
        d = rotation_angle(q[i][None], q)
        d[i] = np.inf
        out[i] = d.min()
    return out


def so3_bin_index(R: np.ndarray, n: int = 4608) -> np.ndarray:
    """Exact SO(3) binning: index of the nearest grid rotation (``R`` is ``[N, 3, 3]``)."""
    grid = hopf_rotations(n)
    q = quat_to_matrix  # noqa: F841 (documentation: grid is in matrix form)
    R = np.asarray(R, float).reshape(-1, 3, 3)
    out = np.empty(R.shape[0], dtype=np.int64)
    for i in range(R.shape[0]):
        out[i] = int(np.argmin(rotation_angle(R[i][None], grid)))
    return out


__all__ = ["PINCH_NSIDE", "PINCH_N_BINS", "HOPF_LEVELS", "pinch_axis_bin_centres", "pinch_axis_bins",
           "quat_to_matrix", "hopf_grid", "hopf_rotations", "rotation_angle", "nearest_neighbour_angles",
           "so3_bin_index"]
