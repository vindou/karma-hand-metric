"""Plot an optimization run: objective vs generation, and how key design
quantities (knuckle distance, thumb wire, L_ref, joint ranges) drifted.

    python optimize/plot_progress.py --run optimize/runs/T
"""
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from optimize.hand_family import HandFamily, FamilyConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    meta = json.loads((run / "meta.json").read_text())
    fam = HandFamily(FamilyConfig(**meta["family"]))
    evals = [json.loads(l) for l in (run / "evals.jsonl").read_text().splitlines() if l.strip()]
    gens = np.array([e["gen"] for e in evals]); obj = np.array([e["objective"] for e in evals])
    T = np.array([e["result"]["karma_t"] for e in evals])
    G = gens.max() + 1
    best_per_gen = np.array([obj[gens == g].max() for g in range(G)])
    mean_per_gen = np.array([obj[gens == g].mean() for g in range(G)])
    best_so_far = np.maximum.accumulate(best_per_gen)

    # design quantities of the per-generation best
    def quant(x):
        h = fam.decode(np.array(x))
        d = np.linalg.norm(h.thumb.knuckle_xyz) * 1e3
        rom_t = np.degrees(np.array(h.thumb.lim_lo) + np.array(h.thumb.lim_hi)).mean()
        rom_i = np.degrees(np.array(h.index.lim_lo) + np.array(h.index.lim_hi)).mean()
        return d, h.thumb.wire_m * 1e3, rom_t, rom_i
    Q = []
    for g in range(G):
        idx = np.where(gens == g)[0]; ib = idx[np.argmax(obj[idx])]
        Q.append(quant(evals[ib]["x"]))
    Q = np.array(Q)
    L = np.array([evals[np.where(gens == g)[0][np.argmax(obj[gens == g])]]["result"]["l_ref_m"] * 1e3 for g in range(G)])

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5))
    ax = axs[0]
    ax.scatter(gens, obj, s=6, alpha=0.25, color="#999999", label="every evaluation")
    ax.plot(range(G), best_per_gen, color="#3b7dd8", lw=1, label="generation best")
    ax.plot(range(G), best_so_far, color="#d9822b", lw=2, label="best so far")
    ax.axhline(0.0969, color="k", ls="--", lw=0.8); ax.text(0.5, 0.0969, " LEAP official T", va="bottom", fontsize=8)
    ax.set_xlabel("generation"); ax.set_ylabel(f"objective ({meta['args']['objective']}, proxy)"); ax.set_ylim(bottom=-0.01)
    ax.legend(fontsize=8); ax.set_title("CMA-ES progress")
    ax = axs[1]
    ax.plot(range(G), Q[:, 0], label="thumb-index knuckle distance [mm]")
    ax.plot(range(G), Q[:, 1], label="thumb wire [mm] (index pinned at 100)")
    ax.plot(range(G), L, label="L_ref [mm]")
    ax.set_xlabel("generation"); ax.legend(fontsize=8); ax.set_title("geometry of the generation-best design")
    ax = axs[2]
    ax.plot(range(G), Q[:, 2], label="mean thumb joint range [deg]")
    ax.plot(range(G), Q[:, 3], label="mean index joint range [deg]")
    ax.axhline(150, color="k", ls=":", lw=0.8); ax.text(0.5, 150, " cap", va="bottom", fontsize=8)
    ax.set_xlabel("generation"); ax.legend(fontsize=8); ax.set_title("joint ranges of the generation-best design")
    fig.tight_layout()
    out = a.out or (run / "progress.png")
    fig.savefig(out, dpi=130); print(f"wrote {out}")


if __name__ == "__main__":
    main()
