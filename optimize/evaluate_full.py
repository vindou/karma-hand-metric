"""Score a design with the OFFICIAL KaRMA pipeline (karma.metric.compute_metric).

    python optimize/evaluate_full.py --design optimize/runs/T/best_design/opt_hand.json --out optimize/runs/T/final

Writes <out>/current.{yaml,pkl} (viewable with `python run_viser_app.py --result <out>/current.pkl`),
<out>/summary.json, and prints the hand in the paper's Table I ordering.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from karma.results import write_outputs       # noqa: E402
from karma.lref import compute_lref_breakdown  # noqa: E402
from optimize.hand_family import HandSpec, HandFamily, FamilyConfig  # noqa: E402

PUBLISHED = REPO / "results" / "16_hand_batch" / "summary.yaml"


def load_design(path: str) -> HandSpec:
    p = Path(path)
    d = json.loads(p.read_text())
    if "thumb" in d:
        return HandSpec.from_json(p.read_text())
    if "design" in d:                      # best.json from run_optimization
        return HandSpec.from_json(json.dumps(d["design"]))
    if "x" in d:
        return HandFamily(FamilyConfig()).decode(d["x"])
    raise ValueError(f"unrecognized design file {path}")


def score(hand: HandSpec, out: Path, name: str | None = None, workers: int | None = None,
          metric_cfg: Path | None = None, lref_variant: str | None = None,
          backend: str | None = None, precision: str = "fp32") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    if name:
        hand.name = name
    _, yaml_path = hand.write(out)
    if lref_variant:
        from optimize import lref_variants
        lref_variants.patch(lref_variant)
    from optimize import backend as BK
    backend = backend or BK.default_backend()
    m = BK.Model(metric_cfg or (REPO / "karma_config.yaml"), yaml_path, backend, out_dir=out)
    cfg = m.cfg
    t0 = time.perf_counter()
    res = m.official(precision=precision, rotation_workers=workers)
    elapsed = time.perf_counter() - t0
    write_outputs(cfg, res)
    sens = res.seed_sensitivity or {}
    lref = compute_lref_breakdown(yaml_path)
    summary = {
        "robot": hand.name,
        "L_ref_mm": round(res.l_ref_m * 1e3, 1),
        "mxPr_mm": round(lref["mxPr_m"] * 1e3, 1),
        "mF_mm": round(lref["mF_m"] * 1e3, 1),
        "n_voxels": res.n_voxels_reached,
        "KaRMA_T": round(res.translational_score, 6),
        "KaRMA_R": round(res.global_rotational_score, 6),
        "KaRMA_S": sens.get("karma_s"),
        "best_voxels": sens.get("best", res.n_voxels_reached),
        "median_voxels": sens.get("median"),
        "n_seeds": sens.get("n_seeds_evaluated"),
        "ensemble_t": sens.get("ensemble_t"),
        "ensemble_r": sens.get("ensemble_r"),
        "runtime_s": round(elapsed, 1),
        "sphere_radius_mm": round(cfg.sphere_radius_m * 1e3, 2),
        "backend": backend + ("" if backend == "reference" else f"/{precision}"),
        "phase1_fail_reasons": res.phase1_fail_reasons,
        "phase2_fail_reasons": res.phase2_fail_reasons,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_table(extra_rows: list[dict]) -> None:
    rows = yaml.safe_load(PUBLISHED.read_text())["rows"]
    rows = [{**r, "_pub": True} for r in rows] + [{**r, "_pub": False} for r in extra_rows]
    rows.sort(key=lambda r: -r["KaRMA_T"])
    print(f"\n  {'Hand':<22s} {'L_ref':>6s} {'Voxels':>7s} {'KaRMA-T':>9s} {'KaRMA-R':>9s} {'KaRMA-S':>8s}")
    print(f"  {'-'*22} {'-'*6} {'-'*7} {'-'*9} {'-'*9} {'-'*8}")
    for r in rows:
        mark = "  " if r["_pub"] else "* "
        s = f"{r['KaRMA_S']:.3f}" if r.get("KaRMA_S") is not None else "-"
        print(f"{mark}{r['robot']:<22s} {r['L_ref_mm']:>6.1f} {r['n_voxels']:>7d} {r['KaRMA_T']:>9.5f} "
              f"{r['KaRMA_R']:>9.4f} {s:>8s}")
    print("  (* = this exercise; published rows from results/16_hand_batch/summary.yaml)\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, help="HandSpec .json, run best.json, or genotype json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--metric-config", default=None, help="alternative karma_config.yaml (arm E)")
    ap.add_argument("--lref-variant", default=None, choices=["shipped", "pinch", "noadvance"])
    ap.add_argument("--backend", choices=["jax", "reference"], default=None)
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp32")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO if a.v else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    hand = load_design(a.design)
    print(hand.describe())
    s = score(hand, Path(a.out), a.name, a.workers, Path(a.metric_config) if a.metric_config else None, a.lref_variant,
              a.backend, a.precision)
    print(json.dumps({k: v for k, v in s.items() if not k.startswith("phase")}, indent=2))
    print_table([s])


if __name__ == "__main__":
    main()
