"""Arm E: re-score hands (bundled robots and/or optimized designs) under an
alternative metric config, e.g. a different sphere radius.

    python optimize/run_arm_e.py --metric-config optimize/configs/karma_sphere_006.yaml \
        --robots leap allegro --designs optimize/runs/T/best.json --out optimize/runs/armE_r006
"""
from __future__ import annotations
import argparse, dataclasses, logging, sys, time
from datetime import datetime
from pathlib import Path
import yaml
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from optimize import backend as BK            # noqa: E402
from optimize.evaluate_full import load_design  # noqa: E402


def one(name, yaml_path, metric_cfg, out, backend):
    t0 = time.perf_counter()
    m = BK.Model(metric_cfg, yaml_path, backend, out_dir=out / "workspace" / name)
    cfg = m.cfg
    res = m.official()
    sens = res.seed_sensitivity or {}
    return {"robot": name, "sphere_radius_mm": round(cfg.sphere_radius_m * 1e3, 2), "L_ref_mm": round(res.l_ref_m * 1e3, 1),
            "n_voxels": res.n_voxels_reached, "KaRMA_T": round(res.translational_score, 6),
            "KaRMA_R": round(res.global_rotational_score, 6), "KaRMA_S": sens.get("karma_s"),
            "n_seeds": sens.get("n_seeds_evaluated"), "runtime_s": round(time.perf_counter() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric-config", required=True)
    ap.add_argument("--robots", nargs="*", default=[])
    ap.add_argument("--designs", nargs="*", default=[], help="design json files (named by their parent run dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--backend", choices=["jax", "reference"], default=None)
    a = ap.parse_args()
    backend = a.backend or BK.default_backend()
    logging.basicConfig(level=logging.WARNING)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done = {}
    if (out / "summary.yaml").exists():
        done = {r["robot"]: r for r in yaml.safe_load((out / "summary.yaml").read_text())["rows"]}
    jobs = [(r, REPO / "robots" / f"robot_{r}.yaml") for r in a.robots]
    for d in a.designs:
        hand = load_design(d); name = "opt_" + Path(d).resolve().parent.name
        hand.name = name
        dd = out / "designs" / name; dd.mkdir(parents=True, exist_ok=True)
        _, yp = hand.write(dd); jobs.append((name, yp))
    rows = []
    for name, yp in jobs:
        if name in done:
            rows.append(done[name]); continue
        try:
            row = one(name, yp, a.metric_config, out, backend)
        except Exception as e:
            row = {"robot": name, "error": str(e)}
        rows.append(row)
        print(f"[{Path(a.metric_config).stem}] {name:<22s} {row}", flush=True)
        (out / "summary.yaml").write_text(yaml.safe_dump(
            {"metric_config": str(a.metric_config), "timestamp": datetime.now().isoformat(), "rows": rows}, sort_keys=False))


if __name__ == "__main__":
    main()
