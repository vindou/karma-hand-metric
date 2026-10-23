"""Experiment artifacts (SPEC.md §13): everything needed to regenerate an optimized morphology."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from ..core.morphology import Topology, HandParams
from ..export import write_design, design_to_dict, design_from_dict


def _jsonable(o):
    if isinstance(o, dict):
        return {k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def save_run(out_dir, topo: Topology, params0: HandParams, result, config: dict, actuator_spec: dict | None = None,
             seeds=None) -> Path:
    """Write the §13 layout: config.yaml, initial/optimized morphology (URDF + robot YAML + design
    JSON), actuator_spec.yaml, parameter/objective/validation histories, random seeds, summary."""
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(_jsonable(config), sort_keys=False))
    write_design(topo, params0, out / "initial_morphology", "initial_morphology")
    write_design(topo, result.best_params, out / "optimized_morphology", "optimized_morphology")
    if actuator_spec is not None:
        (out / "actuator_spec.yaml").write_text(yaml.safe_dump(_jsonable(actuator_spec), sort_keys=False))
    import jax.flatten_util
    flat_best, _ = jax.flatten_util.ravel_pytree(result.best_phi)
    np.savez(out / "parameter_history.npz", best_phi=np.asarray(flat_best),
             smooth=np.asarray([h.get("smooth", h.get("best", np.nan)) for h in result.training_history]))
    with open(out / "objective_history.csv", "w") as fh:
        keys = sorted({k for h in result.training_history for k in h})
        fh.write(",".join(keys) + "\n")
        for h in result.training_history:
            fh.write(",".join(str(h.get(k, "")) for k in keys) + "\n")
    with open(out / "validation_history.csv", "w") as fh:
        rows = [dict(iter=h.get("iter", h.get("gen")), smooth=h.get("smooth"), **(h.get("validation") or {})) for h in result.validation_history]
        keys = sorted({k for r in rows for k in r})
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    (out / "random_seeds.json").write_text(json.dumps({"optimizer_seed": result.meta.get("seed"), "seed_config": _jsonable(config.get("seed_config", {}))}, indent=1))
    if seeds is not None:
        np.savez(out / "seeds.npz", q=np.asarray(seeds.q), p=np.asarray(seeds.p), frame=np.asarray(seeds.frame), valid=np.asarray(seeds.valid))
    (out / "constraint_diagnostics.json").write_text(json.dumps(_jsonable(result.constraint_history), indent=1))
    (out / "results_summary.json").write_text(json.dumps(_jsonable(dict(best_validation=result.best_validation, meta=result.meta,
                                                                       n_training=len(result.training_history))), indent=1))
    return out


def load_run(out_dir):
    out = Path(out_dir)
    topo, params, _ = design_from_dict(json.loads((out / "optimized_morphology" / "optimized_morphology.design.json").read_text()))
    summary = json.loads((out / "results_summary.json").read_text())
    return topo, params, summary


__all__ = ["save_run", "load_run"]
