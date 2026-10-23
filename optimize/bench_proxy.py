"""Calibrate the fast proxy: timing breakdown and comparison with the official metric."""
from __future__ import annotations
import argparse, logging, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import numpy as np
from optimize.hand_family import HandFamily, FamilyConfig
from optimize.fast_eval import evaluate, FastEvalConfig, generate_seeds, METRIC_CFG

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir", default="/tmp/karma_bench")
    ap.add_argument("--official", action="store_true", help="also run karma.metric.compute_metric")
    ap.add_argument("--rotation-voxels", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    fam = HandFamily(FamilyConfig())
    x0 = fam.anthropomorphic_x0()
    hand = fam.decode(x0)
    print(f"genotype dim = {fam.dim}")
    print(hand.describe())

    fe = FastEvalConfig(rotation_voxels=a.rotation_voxels)
    r = evaluate(hand, a.workdir, fe)
    print("\nproxy result:")
    for k, v in r.to_dict().items():
        print(f"  {k}: {v}")

    if a.official:
        from karma.config import load_config
        from karma.robot import load_robot
        from karma.metric import compute_metric
        _, yaml_path = hand.write(a.workdir)
        cfg = load_config(METRIC_CFG, yaml_path); ctx = load_robot(cfg.urdf_path, cfg)
        t0 = time.perf_counter()
        res = compute_metric(cfg, ctx)
        print(f"\nofficial: voxels={res.n_voxels_reached} T={res.translational_score:.6f} "
              f"R={res.global_rotational_score:.4f} S={(res.seed_sensitivity or {}).get('karma_s')} "
              f"n_seeds={(res.seed_sensitivity or {}).get('n_seeds_evaluated')} "
              f"ensemble_T={(res.seed_sensitivity or {}).get('ensemble_t')} ({time.perf_counter()-t0:.1f}s)")

if __name__ == "__main__":
    main()
