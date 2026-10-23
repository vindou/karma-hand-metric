"""dkarma: differentiable, actuation-aware, multi-contact KaRMA (see ``SPEC.md``).

Built phase by phase. Phase 0 (audit and baseline freeze) ships only
:mod:`dkarma.fixtures`: the regression fixtures (16 shipped hands + the three
optimizer designs baseline, T, A1) with their recorded exact scores under the
reference and JAX backends, a loader, a comparison with documented tolerances,
and a re-scoring CLI. Nothing here is a metric.
"""
from __future__ import annotations

__version__ = "0.0.1"
