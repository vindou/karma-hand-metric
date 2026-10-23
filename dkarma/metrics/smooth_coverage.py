"""Smooth coverage of a trajectory library (SPEC.md §4.4, §4.5).

Inputs are the flattened states of all evaluated trajectories: positions
``p [N, 3]``, body-frame pinch axes ``a [N, 3]`` or body rotations ``R [N, 3, 3]``,
and per-state log-weights ``logw [N]`` (``log W_k`` of :func:`dkarma.solvers.rollout.energies`,
``-inf`` or very negative for padded/invalid states). All aggregates are
``o = 1 - prod (1 - w k)`` in log space, so they stay in ``[0, 1]``, are
monotone in every weight, and never underflow.

    translation   a_v = exp(-|p - v|^2 / (2 sigma_p^2)),      K_T~ = mean_v o_v     (fraction of the task cube)
    pinch axis    k_b = exp(-(1 - (a . b)^2) / (2 sigma_R^2)), K_R~ = mean_b o_b     (fraction of the 228 line bins)
    SO(3)         k_t = exp(-(3 - tr(R^T T)) / (2 sigma^2)),   K_SO3~ = mean_t o_t  (fraction of the targets)

``(3 - tr(R^T T)) / 2 = 1 - cos(angle)`` ≈ ``angle^2 / 2``, so the SO(3) kernel is
``exp(-angle^2 / (2 sigma^2))`` near the target without an ``arccos`` cusp.
Region-restricted orientation coverage (SPEC §4.5: fixed spatial regions rather
than an exact top-five voxel selection) multiplies the state weights by a
position kernel around each region centre.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp


def _log_not_occupied(log_terms):
    """``sum log(1 - exp(log_terms))`` with the terms clamped below 1 (stable)."""
    t = jnp.minimum(log_terms, jnp.log(1.0 - 1e-7))
    return jnp.sum(jnp.log1p(-jnp.exp(t)), axis=0)


def occupancy(log_kernel, logw):
    """``o = 1 - prod_n (1 - w_n k_n)`` for ``log_kernel [N, M]`` and ``logw [N]`` -> ``[M]``."""
    return -jnp.expm1(_log_not_occupied(log_kernel + logw[:, None]))


def position_log_kernel(p, centres, sigma_p):
    """``[N, V]`` log of ``exp(-|p - v|^2 / (2 sigma_p^2))``."""
    d2 = jnp.sum((p[:, None, :] - centres[None, :, :]) ** 2, axis=-1)
    return -d2 / (2.0 * sigma_p ** 2)


def pinch_axis_log_kernel(axes, bin_centres, sigma_r):
    """``[N, B]`` log of ``exp(-(1 - (a . b)^2) / (2 sigma_r^2))`` (antipodally symmetric)."""
    c2 = jnp.einsum("na,ba->nb", axes, bin_centres) ** 2
    return -(1.0 - c2) / (2.0 * sigma_r ** 2)


def so3_log_kernel(R, targets, sigma):
    """``[N, T]`` log of ``exp(-(3 - tr(R^T T)) / (2 sigma^2))``."""
    tr = jnp.einsum("nij,tij->nt", R, targets)
    return -(3.0 - tr) / (2.0 * sigma ** 2)


class Coverage(NamedTuple):
    occupancy: object   # per cell / bin / target
    fraction: object    # mean occupancy


def translation_coverage(p, logw, centres, sigma_p) -> Coverage:
    o = occupancy(position_log_kernel(p, centres, sigma_p), logw)
    return Coverage(o, jnp.mean(o))


def pinch_axis_coverage(axes, logw, bin_centres, sigma_r) -> Coverage:
    o = occupancy(pinch_axis_log_kernel(axes, bin_centres, sigma_r), logw)
    return Coverage(o, jnp.mean(o))


def so3_coverage(R, logw, targets, sigma) -> Coverage:
    o = occupancy(so3_log_kernel(R, targets, sigma), logw)
    return Coverage(o, jnp.mean(o))


def region_orientation_coverage(p, logw, region_centres, sigma_p, log_kernel_orient):
    """Orientation coverage restricted to spatial regions: for region ``r``, state weights are
    multiplied by the position kernel around its centre. Returns ``[R]`` fractions."""
    logk_pos = position_log_kernel(p, region_centres, sigma_p)        # [N, R]

    def one(logk_r):
        return jnp.mean(occupancy(log_kernel_orient, logw + logk_r))
    return jax.vmap(one, in_axes=1)(logk_pos)


def soft_seed_aggregate(scores, beta: float, tau: float):
    """``(1 - beta) mean_s K_s + beta softmin_s K_s`` with a log-sum-exp soft minimum (SPEC §4.6)."""
    smin = -tau * jax.scipy.special.logsumexp(-scores / tau) + tau * jnp.log(scores.shape[0])
    return (1.0 - beta) * jnp.mean(scores) + beta * smin


__all__ = ["occupancy", "position_log_kernel", "pinch_axis_log_kernel", "so3_log_kernel", "Coverage",
           "translation_coverage", "pinch_axis_coverage", "so3_coverage", "region_orientation_coverage",
           "soft_seed_aggregate"]
