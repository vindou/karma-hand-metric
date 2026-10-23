"""Re-score regression fixtures on a backend and compare with the recorded numbers.

    # reference backend (paper's pinned CPU stack), the three low-DOF floor hands
    ~/mamba/envs/karma-hand-metric/bin/python -m dkarma.fixtures.rescore --backend reference \\
        --fixtures inspire svh ability_hand_right
    # JAX fp32 on the GPU, everything (needs the GPU free; ~10 min warm), record the result
    ~/mamba/envs/karma-jax/bin/python -m dkarma.fixtures.rescore --backend jax --all --record \\
        --note "Phase 0 padding fixes"
    # CPU-only JAX (slow; while the optimize queue owns the GPU)
    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.fixtures.rescore --backend jax --fixtures inspire

Writes a YAML report (default ``workspace/phase0/rescore_<backend>_<timestamp>.yaml``)
with every row, its comparison and the tolerance level used, and prints one
line per fixture. ``--record`` additionally stores each row under
``recorded.<backend key>`` in ``scores.yaml`` with provenance (machine, date,
``--note``). Comparison levels: ``exact``, ``practical``, ``jax`` (see
``dkarma.fixtures.TOLERANCES``); the default is ``practical`` for the
reference backend and ``jax`` for JAX.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import platform
import sys
import time
from pathlib import Path

import yaml

from . import (ROOT, SCORES_PATH, BACKEND_KEYS, fixtures, load, load_flat, scores_from_result,
               compare, format_row, load_scores)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def _machine(backend: str) -> str:
    cpu = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    s = f"{cpu or platform.processor() or platform.machine()} ({__import__('os').cpu_count()} cores)"
    if backend == "jax":
        import jax
        dev = jax.devices()[0]
        s = f"{getattr(dev, 'device_kind', dev.platform)} [{dev.platform}] + " + s
    return s


def score_one(name: str, backend: str, precision: str, out_dir: Path, rotation_workers: int | None = None) -> dict:
    t0 = time.perf_counter()
    if backend == "jax":
        from karma_jax.metric import compute_metric
        flat, cfg, ctx = load_flat(name, out_dir=out_dir)        # exact padding (the corrected default)
        res = compute_metric(cfg, flat, precision=precision)
    else:
        from karma.metric import compute_metric
        cfg, ctx = load(name, out_dir=out_dir)
        res = compute_metric(cfg, ctx, n_rotation_workers=rotation_workers if rotation_workers is not None
                             else cfg.rotation_workers)
    return scores_from_result(name, res, time.perf_counter() - t0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["reference", "jax"], default="reference")
    ap.add_argument("--precision", choices=["fp32", "fp64"], default="fp32", help="JAX search precision")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fixtures", nargs="+", help="fixture names (see scores.yaml)")
    g.add_argument("--all", action="store_true")
    g.add_argument("--shipped", action="store_true")
    g.add_argument("--designs", action="store_true")
    ap.add_argument("--level", choices=["exact", "practical", "jax"], default=None)
    ap.add_argument("--out", default=None, help="report YAML path")
    ap.add_argument("--record", action="store_true", help="store the rows in scores.yaml")
    ap.add_argument("--note", default="", help="code-state note stored with --record")
    ap.add_argument("--rotation-workers", type=int, default=None, help="reference backend only")
    a = ap.parse_args(argv)

    fx = fixtures()
    if a.fixtures:
        names = a.fixtures
    elif a.shipped:
        names = [n for n, f in fx.items() if f.kind == "shipped"]
    elif a.designs:
        names = [n for n, f in fx.items() if f.kind == "design"]
    else:
        names = list(fx)
    unknown = [n for n in names if n not in fx]
    if unknown:
        ap.error(f"unknown fixtures {unknown}; known: {sorted(fx)}")

    key = BACKEND_KEYS[a.backend] if a.backend == "reference" else f"jax_{a.precision}"
    level = a.level or ("practical" if a.backend == "reference" else "jax")
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = (Path(a.out).resolve() if a.out else ROOT / "workspace" / "phase0" / f"rescore_{key}_{stamp}.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    work = out.parent / f"runs_{key}_{stamp}"
    machine = _machine(a.backend)
    print(f"backend {a.backend} ({key}), level {level}, machine {machine}")
    print(f"{'fixture':<20s} {'backend':<9s} {'voxels':>6s} {'KaRMA-T':>9s} {'KaRMA-R':>8s} {'KaRMA-S':>7s} {'time':>9s}")

    rows, n_ok, n_exact, n_cmp = [], 0, 0, 0
    for name in names:
        row = score_one(name, a.backend, a.precision, work / name, a.rotation_workers)
        rec = fx[name].recorded.get(key)
        cmp = compare(row, rec, level) if rec else None
        if cmp:
            n_cmp += 1
            n_ok += cmp["ok"]
            n_exact += cmp["exact"]
        print(format_row(name, row, cmp), flush=True)
        rows.append({"fixture": name, "row": row, "recorded_key": key, "comparison": cmp})

    report = {"backend": a.backend, "recorded_key": key, "level": level, "machine": machine,
              "date": stamp, "note": a.note, "summary": {"n": len(rows), "compared": n_cmp, "within_tolerance": n_ok,
                                                        "exact": n_exact}, "results": rows}
    out.write_text(yaml.safe_dump(report, sort_keys=False, default_flow_style=None, width=100))
    print(f"\n{n_ok}/{n_cmp} within '{level}' tolerance ({n_exact} exact); {len(rows) - n_cmp} without a record. "
          f"Report: {_rel(out)}")

    if a.record:
        data = load_scores()
        for r in rows:
            entry = {"source": f"dkarma.fixtures.rescore ({_rel(out)})", "machine": machine,
                     "date": stamp[:4] + "-" + stamp[4:6] + "-" + stamp[6:8], "code": a.note}
            entry.update({k: v for k, v in r["row"].items() if k not in ("robot", "backend", "precision")})
            data["fixtures"][r["fixture"]].setdefault("recorded", {})[key] = entry
        text = SCORES_PATH.read_text()
        header = "".join(line for line in text.splitlines(True) if line.startswith("#"))
        SCORES_PATH.write_text(header + yaml.safe_dump(data, sort_keys=False, default_flow_style=None, width=100))
        print(f"recorded {len(rows)} rows under '{key}' in {SCORES_PATH.relative_to(ROOT)}")
    return 0 if n_ok == n_cmp else 1


if __name__ == "__main__":
    sys.exit(main())
