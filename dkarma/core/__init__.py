"""dkarma.core: topology, morphology -> model, kinematics, contact geometry (SPEC.md §3, §9)."""
from .topology import Topology  # noqa: F401
from .morphology import (  # noqa: F401
    DEFAULT_CAPSULE_RADIUS_M, HandParams, Phi, Bounds, FixedParams, decode, encode, fixed_from_params,
    rotvec_to_matrix, axis_from_angles, HandModel, build_hand_model, build_hand_model_from_phi,
    design_in_axes, params_from_fingers, params_to_fingers, from_hand_spec, to_hand_spec_dict,
    random_params, capsule_radii, anthropomorphic,
)
from .object_model import ObjectModel, TaskVolume, OrientationTargets, TaskSpec, load_task_spec  # noqa: F401
from . import kinematics, geometry, contacts  # noqa: F401
