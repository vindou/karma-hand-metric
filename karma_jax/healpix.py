"""Vectorized twist-invariant orientation binning (host side, numpy + healpy).

Same math as ``karma.orientation.axis_to_healpix_bin`` applied to ``[N,3]``
arrays at once: normalize, canonicalize each line direction to the z >= 0
hemisphere (ties broken on y, then x), convert to HEALPix angles, and call
``healpy.ang2pix`` (RING scheme).
"""
from __future__ import annotations

import numpy as np
import healpy as hp


def canonicalize_axes(axes: np.ndarray) -> np.ndarray:
    a = np.asarray(axes, dtype=float).reshape(-1, 3)
    n = np.linalg.norm(a, axis=1)
    a = np.where(n[:, None] < 1e-12, 0.0, a / np.where(n < 1e-12, 1.0, n)[:, None])
    flip = (a[:, 2] < 0) | ((a[:, 2] == 0) & (a[:, 1] < 0)) | ((a[:, 2] == 0) & (a[:, 1] == 0) & (a[:, 0] < 0))
    return np.where(flip[:, None], -a, a)


def axes_to_bins(axes: np.ndarray, nside: int) -> np.ndarray:
    """``[N,3]`` directions -> ``[N]`` HEALPix line-bin indices."""
    a = canonicalize_axes(axes)
    theta = np.arccos(np.clip(a[:, 2], -1.0, 1.0))
    phi = np.arctan2(a[:, 1], a[:, 0]) % (2.0 * np.pi)
    return np.asarray(hp.ang2pix(nside, theta, phi), dtype=np.int64)
