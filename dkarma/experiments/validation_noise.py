"""Phase 8.2 companion: how much does the exact validation score move for design changes of one
Adam step? The exact library counts cells and bins on a fresh seed set; both are discrete in the
design, so the score of the iterate at iteration ``k`` and at ``k + 1`` can differ by far more
than the optimizer's progress. This script takes a design (the best design of a study run),
perturbs its unconstrained coordinates by Gaussian vectors of the recorded Adam step norm (and
of 3x that), validates each perturbation exactly (fresh seeds, the study's configuration) and
reports the spread.

    JAX_PLATFORMS=cpu ~/mamba/envs/karma-jax/bin/python -m dkarma.experiments.validation_noise \\
        --run workspace/phase8/8.2/diff2/phase7/start0_nominal --n 8 --out workspace/phase8/8.2/validation_noise.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax
import jax.flatten_util
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from dkarma.core import load_task_spec, encode  # noqa: E402
from dkarma.solvers.seed_projection import SeedConfig  # noqa: E402
from dkarma.optimization import make_objective, load_run  # noqa: E402
from dkarma.optimization.objectives import exact_validation  # noqa: E402
from dkarma.experiments.continuation_study import objective_spec, BOUNDS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--objective", default="diff2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="workspace/phase8/8.2/validation_noise.json")
    ap.add_argument("--budget", type=int, default=None, help="validation seed budget (default: the objective's)")
    args = ap.parse_args()
    run = Path(args.run)
    rec = json.loads((run / "run.json").read_text())
    steps = [h["step_norm"] for h in rec["training_history"] if h["step_norm"] > 0]
    step = float(np.median(steps))
    topo, params, _ = load_run(run)
    spec = objective_spec(args.objective)
    task = load_task_spec()
    obj = make_objective(topo, params, BOUNDS, task, spec["cfg"], SeedConfig(budget=args.budget or spec["budget"]))
    validate = exact_validation(obj, run / "noise_val", official=False)
    x0, unravel = jax.flatten_util.ravel_pytree(encode(params, topo, BOUNDS))
    rng = np.random.RandomState(args.seed)
    out = dict(run=str(run), median_step_norm=step, budget=args.budget or spec["budget"], base=validate(params), perturbations=[])
    print(f"base exact {out['base']['exact_score']:.4f}; median Adam step norm {step:.4f}", flush=True)
    for scale_mult in (1.0, 3.0):
        for i in range(args.n):
            d = rng.randn(x0.shape[0]); d = d / np.linalg.norm(d) * step * scale_mult
            v = validate(obj.params_of(unravel(x0 + jnp.asarray(d))))
            out["perturbations"].append(dict(scale=step * scale_mult, i=i, **v))
            print(f"  step x{scale_mult:g} #{i}: exact {v['exact_score']:.4f} (K_T {v['exact_K_T']:.5f} K_R {v['exact_K_R']:.4f} seeds {v['n_seeds']})", flush=True)
            Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    for scale_mult in (1.0, 3.0):
        vals = np.array([p["exact_score"] for p in out["perturbations"] if abs(p["scale"] - step * scale_mult) < 1e-12])
        b = out["base"]["exact_score"]
        print(f"step x{scale_mult:g}: mean {vals.mean():.4f} sd {vals.std(ddof=1):.4f} rel sd {vals.std(ddof=1) / b:.3f} range [{vals.min():.4f}, {vals.max():.4f}] vs base {b:.4f}")
    out["summary"] = {f"x{m:g}": dict(sd=float(np.std([p["exact_score"] for p in out["perturbations"] if abs(p["scale"] - step * m) < 1e-12], ddof=1)))
                      for m in (1.0, 3.0)}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float))
    print("done")


if __name__ == "__main__":
    main()
