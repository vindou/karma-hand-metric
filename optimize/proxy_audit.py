"""Proxy audit: official-score a stratified sample of designs from a run and
compare with the proxy objective the optimizer saw.

    python optimize/proxy_audit.py --run optimize/runs/T --n 20 --out optimize/runs/audit_T
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from optimize.hand_family import HandFamily, FamilyConfig   # noqa: E402
from optimize.evaluate_full import score                     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", required=True); ap.add_argument("--min-obj", type=float, default=0.005)
    ap.add_argument("--backend", choices=["jax", "reference"], default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(a.run) / "meta.json").read_text())
    fam = HandFamily(FamilyConfig(**meta["family"]))
    evals = [json.loads(l) for l in (Path(a.run) / "evals.jsonl").read_text().splitlines() if l.strip()]
    evals = [e for e in evals if e["objective"] >= a.min_obj and not e["result"]["error"]]
    evals.sort(key=lambda e: e["objective"])
    # stratified by objective quantile, always include the top one
    idx = sorted(set(np.linspace(0, len(evals) - 1, a.n).round().astype(int).tolist()) | {len(evals) - 1})
    rows_path = out / "audit.jsonl"
    done = set()
    if rows_path.exists():
        done = {json.loads(l)["key"] for l in rows_path.read_text().splitlines() if l.strip()}
    with open(rows_path, "a") as f:
        for i in idx:
            e = evals[i]; key = f"g{e['gen']}_{e['idx']}"
            if key in done:
                continue
            hand = fam.decode(np.array(e["x"]))
            t0 = time.perf_counter()
            s = score(hand, out / key, name=key, backend=a.backend)
            row = {"key": key, "proxy_T": e["result"]["karma_t"], "proxy_top": e["result"]["top3_voxels"],
                   "official_T": s["KaRMA_T"], "official_R": s["KaRMA_R"], "official_S": s["KaRMA_S"],
                   "official_voxels": s["n_voxels"], "n_seeds": s["n_seeds"], "runtime_s": round(time.perf_counter() - t0, 1)}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(row, flush=True)
    rows = [json.loads(l) for l in rows_path.read_text().splitlines() if l.strip()]
    from scipy.stats import spearmanr, pearsonr
    p = np.array([r["proxy_T"] for r in rows]); o = np.array([r["official_T"] for r in rows])
    print(f"\n{len(rows)} designs: Spearman rho = {spearmanr(p, o).correlation:.3f}, Pearson r = {pearsonr(p, o)[0]:.3f}, "
          f"median official/proxy = {np.median(o / np.maximum(p, 1e-9)):.2f}")
    top5p = set(np.argsort(-p)[:5]); top5o = set(np.argsort(-o)[:5])
    print(f"top-5 agreement: {len(top5p & top5o)}/5")


if __name__ == "__main__":
    main()
