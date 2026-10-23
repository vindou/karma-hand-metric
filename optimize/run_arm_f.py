"""Arm F: re-score the 16 bundled hands under an alternative L_ref definition.

    python optimize/run_arm_f.py --variant pinch --out optimize/runs/armF_pinch

Writes <out>/summary.yaml (same row format as results/16_hand_batch/summary.yaml)
and prints the ranking against the published one (Spearman rho, pairwise inversions).
"""
from __future__ import annotations
import argparse, logging, sys, time
from datetime import datetime
from pathlib import Path
import yaml
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from optimize import lref_variants                      # noqa: E402
from optimize import backend as BK                       # noqa: E402

PUBLISHED = REPO / "results" / "16_hand_batch" / "summary.yaml"


def rank_compare(pub_rows, new_rows, key="KaRMA_T"):
    from scipy.stats import spearmanr
    names = [r["robot"] for r in new_rows]
    pub = {r["robot"]: r[key] for r in pub_rows}
    a = np.array([pub[n] for n in names]); b = np.array([r[key] for r in new_rows])
    rho = spearmanr(a, b).correlation
    inv = sum(1 for i in range(len(names)) for j in range(i + 1, len(names))
              if (a[i] - a[j]) * (b[i] - b[j]) < 0)
    return rho, inv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=list(lref_variants.VARIANTS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", nargs="*", default=None, help="subset of hand names")
    ap.add_argument("--backend", choices=["jax", "reference"], default=None)
    ap.add_argument("--precision", default="fp32")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.WARNING)
    lref_variants.patch(a.variant)
    backend = a.backend or BK.default_backend()

    robots = sorted(p for p in (REPO / "robots").glob("robot_*.yaml") if "template" not in p.name)
    if a.only:
        robots = [p for p in robots if p.stem[6:] in a.only]
    done = {}
    if (out / "summary.yaml").exists():
        for r in yaml.safe_load((out / "summary.yaml").read_text())["rows"]:
            done[r["robot"]] = r
    rows = []
    t_all = time.perf_counter()
    for p in robots:
        name = p.stem[6:]
        if name in done:
            rows.append(done[name]); continue
        t0 = time.perf_counter()
        m = BK.Model(REPO / "karma_config.yaml", p, backend, out_dir=out / "workspace" / name)
        cfg = m.cfg
        try:
            res = m.official(precision=a.precision)
            sens = res.seed_sensitivity or {}
            row = {"robot": name, "L_ref_mm": round(res.l_ref_m * 1e3, 1), "n_voxels": res.n_voxels_reached,
                   "KaRMA_T": round(res.translational_score, 6), "KaRMA_R": round(res.global_rotational_score, 6),
                   "KaRMA_S": sens.get("karma_s"), "best_voxels": sens.get("best", res.n_voxels_reached),
                   "median_voxels": sens.get("median"), "n_seeds": sens.get("n_seeds_evaluated"),
                   "runtime_s": round(time.perf_counter() - t0, 1)}
        except Exception as e:
            row = {"robot": name, "error": str(e), "L_ref_mm": round(cfg.l_ref_m * 1e3, 1)}
        rows.append(row)
        print(f"[{a.variant}] {name:<20s} L_ref={row.get('L_ref_mm')} vox={row.get('n_voxels')} "
              f"T={row.get('KaRMA_T')} R={row.get('KaRMA_R')} ({row.get('runtime_s', 0)}s)", flush=True)
        (out / "summary.yaml").write_text(yaml.safe_dump(
            {"variant": a.variant, "backend": backend, "timestamp": datetime.now().isoformat(), "rows": rows}, sort_keys=False))

    ok = [r for r in rows if "error" not in r]
    pub = yaml.safe_load(PUBLISHED.read_text())["rows"]
    if len(ok) >= 3:
        rho_t, inv_t = rank_compare(pub, ok, "KaRMA_T"); rho_r, inv_r = rank_compare(pub, ok, "KaRMA_R")
        print(f"\nranking vs published ({len(ok)} hands): KaRMA-T Spearman rho={rho_t:.3f}, inversions={inv_t}; "
              f"KaRMA-R rho={rho_r:.3f}, inversions={inv_r}   (total {(time.perf_counter()-t_all)/60:.0f} min)")


if __name__ == "__main__":
    main()
