"""Reference grasp states for the Phase 0 designs, read from the optimizer's official
result pickles (seed and BFS states with q, sphere centre, body rotation, contact pair),
together with the reference's L_ref-scaled step parameters for that design."""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from dkarma.core import build_hand_model
from dkarma.core.contacts import ContactAssignment
from dkarma.export import load_reference
from karma_jax.search import step_params_from_cfg

ROOT = Path(__file__).resolve().parents[2]
PICKLES = {"T": ROOT / "optimize/runs/T/final/current.pkl",
           "A1": ROOT / "optimize/runs/A1/final/current.pkl",
           "baseline": ROOT / "optimize/runs/baseline/current.pkl"}

_cache: dict = {}


def available(name: str) -> bool:
    return name in PICKLES and PICKLES[name].exists()


def load(name: str, tmpdir, n_nodes: int = 6, seed: int = 0):
    """``dict(topo, params, hm, cfg, sp, asg, seed_state, nodes)``: the design's model with the
    reference's link radius, its StepParams (fp64), the seed grasp and ``n_nodes`` BFS states
    (``GraspState``-like tuples ``(q, p, R)``) drawn deterministically from the result."""
    from .conftest import load_design
    from dkarma.solvers.rollout import GraspState
    key = (name, n_nodes, seed)
    if key in _cache:
        return _cache[key]
    import sys
    sys.path.insert(0, str(ROOT))
    from karma.storage import load_result
    topo, params, _ = load_design(name)
    cfg, ctx, flat = load_reference(topo, params, tmpdir, name=name)
    params = params._replace(cap_radius=jnp.full((topo.C,), cfg.link_radius_m))
    hm = build_hand_model(topo, params)
    sp = step_params_from_cfg(cfg, jnp.float64)
    r = load_result(str(PICKLES[name]))
    res = next(x for x in (r if isinstance(r, tuple) else [r]) if hasattr(x, "seed_q"))
    names = topo.capsule_names
    pair = (names.index(res.seed_contact_pair[0]), names.index(res.seed_contact_pair[1]))
    asg = ContactAssignment(caps=jnp.asarray(pair, jnp.int32), valid=jnp.array([True, True]))
    seed_state = GraspState(jnp.asarray(res.seed_q, jnp.float64), jnp.asarray(res.seed_centre, jnp.float64),
                            jnp.eye(3, dtype=jnp.float64))
    rng = np.random.RandomState(seed)
    keys = list(res.state_nodes.keys())
    pick = rng.choice(len(keys), size=min(n_nodes, len(keys)), replace=False)
    nodes = []
    for i in pick:
        nd = res.state_nodes[keys[i]]
        cp = (names.index(nd.contact_pair[0]), names.index(nd.contact_pair[1]))
        nodes.append((GraspState(jnp.asarray(nd.q, jnp.float64), jnp.asarray(nd.centre_world, jnp.float64),
                                 jnp.asarray(nd.R_sphere, jnp.float64)), cp))
    out = dict(topo=topo, params=params, hm=hm, cfg=cfg, sp=sp, asg=asg, seed_state=seed_state, nodes=nodes,
               seed_frame=np.asarray(res.seed_frame))
    _cache[key] = out
    return out
