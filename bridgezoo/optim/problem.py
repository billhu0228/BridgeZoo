"""Problem definition for cable strand and pretension optimization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping


@dataclass(frozen=True)
class CableBounds:
    strand_min: int = 1
    strand_max: int = 60
    stress_lower_mpa: float = 800.0
    stress_upper_mpa: float = 1200.0
    tension_bound_stress_mpa: float = 1600.0


@dataclass(frozen=True)
class ObjectiveWeights:
    shape: float = 1.0
    total_strands: float = 0.02
    stress_uniform: float = 0.2
    stress_violation: float = 100.0
    shape_scale_m: float = 1.0e-3
    stress_scale_mpa: float = 100.0
    strand_scale: float = 20.0


@dataclass(frozen=True)
class TargetLine:
    """Target vertical displacement line for completed construction."""

    uy_by_node: Mapping[int, float] | None = None
    default_uy: float = 0.0
    function: Callable[[int, float], float] | None = None

    def uy(self, node_id: int, x: float) -> float:
        if self.function is not None:
            return float(self.function(node_id, x))
        if self.uy_by_node is not None and node_id in self.uy_by_node:
            return float(self.uy_by_node[node_id])
        return float(self.default_uy)


@dataclass(frozen=True)
class CableOptimizationProblem:
    n_seg: int
    model_kwargs: dict = field(default_factory=dict)
    bounds: CableBounds = field(default_factory=CableBounds)
    weights: ObjectiveWeights = field(default_factory=ObjectiveWeights)
    target_line: TargetLine = field(default_factory=TargetLine)
    strand_area: float = 1.4e-4
    backend: str = "direct"

    def builder_kwargs(self) -> dict:
        data = dict(self.model_kwargs)
        data["n_seg"] = self.n_seg
        data.setdefault("strand_area", self.strand_area)
        return data
