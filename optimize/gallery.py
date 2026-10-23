"""Tile every scored winner's official render (final/hand.png) and quick render
(best_design/best.png) into one contact sheet: optimize/runs/GALLERY.png.

    python optimize/gallery.py
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
RUNS = Path(__file__).resolve().parent / "runs"
ORDER = ["baseline", "T", "A1", "A2", "A3", "D_R", "D_TR", "B_2x2", "B_3x3", "B_5x5", "C_cap45", "C_cap60", "C_cap90"]


def main():
    rows = []
    for name in ORDER:
        official = RUNS / name / ("hand.png" if name == "baseline" else "final/hand.png")
        quick = RUNS / name / "best_design" / "best.png"
        if official.exists() or quick.exists():
            rows.append((name, official if official.exists() else None, quick if quick.exists() else None))
    if not rows:
        print("nothing to tile yet"); return
    fig, axs = plt.subplots(len(rows), 2, figsize=(22, 6.2 * len(rows)))
    if len(rows) == 1:
        axs = [axs]
    for (name, off, qk), (ax0, ax1) in zip(rows, axs):
        for ax, img, lab in ((ax0, off, "official metric: seed pinch + reachable set"), (ax1, qk, "quick render: neutral pose + best-seed cloud")):
            ax.axis("off")
            if img is not None:
                ax.imshow(mpimg.imread(img))
                ax.set_title(f"{name}: {lab}", fontsize=13, loc="left")
            else:
                ax.set_title(f"{name}: {lab} (pending)", fontsize=13, loc="left", color="grey")
    fig.tight_layout()
    out = RUNS / "GALLERY.png"
    fig.savefig(out, dpi=70)
    print(f"wrote {out} ({len(rows)} designs)")


if __name__ == "__main__":
    main()
