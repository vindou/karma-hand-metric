"""Shared fixtures: reference robot contexts and flat models for the shipped hands."""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
METRIC_CFG = ROOT / "karma_config.yaml"

# Hands covering every joint model (RX/RY/RZ/unaligned), mimics, tip-offset and
# distal-frame capsules, and the largest frame count. KARMA_JAX_TEST_ALL_HANDS=1
# runs all 16.
DEFAULT_HANDS = ["leap", "allegro", "shadowhand", "inspire", "svh", "orcahand", "dclaw"]


def hand_names() -> list[str]:
    if os.environ.get("KARMA_JAX_TEST_ALL_HANDS"):
        return sorted(
            Path(p).stem.replace("robot_", "")
            for p in glob.glob(str(ROOT / "robots" / "robot_*.yaml"))
            if "template" not in p
        )
    return DEFAULT_HANDS


_cache: dict[str, tuple] = {}


def load_hand(name: str):
    """(flat_model, cfg, ctx) for a hand, cached per session."""
    if name not in _cache:
        from karma_jax.model import load_flat_model
        _cache[name] = load_flat_model(METRIC_CFG, ROOT / "robots" / f"robot_{name}.yaml")
    return _cache[name]


@pytest.fixture(params=hand_names())
def hand(request):
    return load_hand(request.param)


def random_configs(ctx, n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    lo = np.array(ctx.model.lowerPositionLimit)
    hi = np.array(ctx.model.upperPositionLimit)
    return lo + rng.rand(n, ctx.model.nq) * (hi - lo)
