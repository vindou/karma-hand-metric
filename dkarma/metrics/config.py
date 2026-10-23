"""Metric configuration: the four variants as configurations of one evaluator (SPEC.md §7)."""
from __future__ import annotations

from dataclasses import dataclass, field

from ..solvers.rollout import EnergyWeights, RollConfig
from .commands import CommandConfig


@dataclass(frozen=True)
class SmoothConfig:
    """Kernel widths and aggregation of the smooth objective (continuation knobs, SPEC §8.4)."""
    sigma_p_m: float = 0.006            # translation kernel width (grid 10 mm)
    sigma_r_rad: float = 0.09           # pinch-axis kernel width (bins ~10 deg)
    sigma_so3_rad: float = 0.15         # SO(3) kernel width (576 targets, D16)
    sigma_region_m: float = 0.008       # region kernel for region-restricted orientation coverage
    beta_seed: float = 0.5              # (1 - beta) mean + beta softmin over seeds
    tau_seed: float = 0.02
    lambda_T: float = 1.0               # objective weights on the (normalized) coverages
    lambda_R: float = 1.0
    ref_T: float = 0.10                 # normalization scales of a "good" hand (fractions)
    ref_R: float = 0.30


@dataclass(frozen=True)
class MetricConfig:
    """Which physical constraints are on; everything else is shared (SPEC §7)."""
    name: str = "diff"
    contacts: int = 2
    rolling: bool = True
    torque_limits: bool = False
    force_equilibrium: bool = False
    orientation_space: str = "pinch_axis"     # "pinch_axis" | "SO3"
    exact: bool = False
    commands: CommandConfig = field(default_factory=CommandConfig)
    roll: RollConfig = field(default_factory=lambda: RollConfig(mode="hard"))
    energy: EnergyWeights = field(default_factory=EnergyWeights)
    smooth: SmoothConfig = field(default_factory=SmoothConfig)


DIFF_KARMA = MetricConfig(name="diff", contacts=2, torque_limits=False, force_equilibrium=False, orientation_space="pinch_axis")
ACT_KARMA = MetricConfig(name="act", contacts=2, torque_limits=True, force_equilibrium=True, orientation_space="pinch_axis",
                         roll=RollConfig(mode="hard", actuation=True))
MC_KARMA = MetricConfig(name="mc", contacts=3, torque_limits=False, force_equilibrium=False, orientation_space="SO3")
FULL_KARMA = MetricConfig(name="full", contacts=3, torque_limits=True, force_equilibrium=True, orientation_space="SO3",
                          roll=RollConfig(mode="hard", actuation=True))

PAIRWISE_KARMA = MetricConfig(name="pairwise", contacts=2, orientation_space="pinch_axis")

__all__ = ["SmoothConfig", "MetricConfig", "DIFF_KARMA", "ACT_KARMA", "MC_KARMA", "FULL_KARMA", "PAIRWISE_KARMA"]
