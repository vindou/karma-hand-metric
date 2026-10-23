"""Phase 0 fixture tests: every fixture loads on both code paths and the frozen
scores file is complete. Runs on CPU (set ``JAX_PLATFORMS=cpu`` when the GPU is
busy). The expensive re-score is opt-in:

    KARMA_FIXTURE_RESCORE=1 pytest dkarma/tests -q -k rescore      # floor hands, both backends, ~1-2 min on CPU
"""
from __future__ import annotations

import filecmp
import glob
import os
from pathlib import Path

import numpy as np
import pytest

from dkarma import fixtures as FX

ROOT = FX.ROOT
ALL = FX.fixtures()
FLOOR_HANDS = ["inspire", "svh", "ability_hand_right"]   # <= 4 voxels; both backends reproduce them exactly


def test_scores_file_is_complete():
    shipped = sorted(Path(p).stem.replace("robot_", "") for p in glob.glob(str(ROOT / "robots" / "robot_*.yaml"))
                     if "template" not in p)
    assert sorted(n for n, f in ALL.items() if f.kind == "shipped") == shipped
    assert sorted(n for n, f in ALL.items() if f.kind == "design") == ["A1", "T", "baseline"]
    for name, fx in ALL.items():
        assert fx.robot_yaml.exists(), name
        assert fx.recorded, f"{name}: no recorded scores"
        for key, rec in fx.recorded.items():
            assert key in ("reference", "jax_fp32", "jax_fp64"), (name, key)
            for f in ("n_voxels", "KaRMA_T", "KaRMA_R", "source", "machine", "date"):
                assert f in rec, (name, key, f)
    # the shipped hands carry both the paper's and the JAX sweep's numbers
    for name, fx in ALL.items():
        if fx.kind == "shipped":
            assert set(fx.recorded) >= {"reference", "jax_fp32"}, name


def test_design_fixtures_are_byte_copies_of_the_scored_files():
    data = FX.load_scores()["fixtures"]
    for name in ("baseline", "T", "A1"):
        src = ROOT / data[name]["copied_from"]
        if not src.exists():
            pytest.skip(f"{src} not present")
        for f in (ROOT / data[name]["robot_yaml"]).parent.iterdir():
            assert filecmp.cmp(f, src / f.name, shallow=False), f


@pytest.mark.parametrize("name", sorted(ALL))
def test_fixture_loads_and_flattens(name):
    import jax
    import jax.numpy as jnp
    from karma.robot import update_kinematics
    from karma_jax import kinematics as K
    jax.config.update("jax_enable_x64", True)
    flat, cfg, ctx = FX.load_flat(name)                      # reference loader + exact-padding flat model
    assert flat.J_pad == flat.n_joints and flat.M_pad == flat.n_active
    dm = flat.to_device(jnp.float64)
    f = jax.jit(lambda q: K.forward_kinematics(dm, q))        # closed-over model, as the kernels do
    rng = np.random.RandomState(0)
    lo, hi = np.array(ctx.model.lowerPositionLimit), np.array(ctx.model.upperPositionLimit)
    for _ in range(3):
        q = lo + rng.rand(ctx.model.nq) * (hi - lo)
        fk = f(jnp.asarray(q))
        update_kinematics(ctx, q.copy())
        P_ref = np.stack([np.array(ctx.data.oMf[k].translation) for k in range(ctx.model.nframes)])
        np.testing.assert_allclose(np.asarray(fk.p_frame), P_ref, rtol=0, atol=1e-10)


@pytest.mark.skipif(not os.environ.get("KARMA_FIXTURE_RESCORE"), reason="set KARMA_FIXTURE_RESCORE=1")
@pytest.mark.parametrize("backend", ["reference", "jax"])
@pytest.mark.parametrize("name", FLOOR_HANDS)
def test_rescore_floor_hands(name, backend, tmp_path):
    from dkarma.fixtures.rescore import score_one
    if backend == "jax":
        pytest.importorskip("karma_jax")
    row = score_one(name, backend, "fp32", tmp_path)
    rec = ALL[name].recorded_for(backend)
    cmp = FX.compare(row, rec, "exact")
    assert cmp["ok"], cmp
