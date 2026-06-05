"""Objective and metric calculations for cable optimization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bridgezoo.optim.problem import CableBounds, ObjectiveWeights


@dataclass(frozen=True)
class ObjectiveBreakdown:
    shape: float
    total_strands: float
    stress_uniform: float
    stress_violation: float

    @property
    def total(self) -> float:
        return self.shape + self.total_strands + self.stress_uniform + self.stress_violation


def stress_violation_mpa(stress_mpa: np.ndarray, bounds: CableBounds) -> np.ndarray:
    low = np.maximum(0.0, bounds.stress_lower_mpa - stress_mpa)
    high = np.maximum(0.0, stress_mpa - bounds.stress_upper_mpa)
    return low + high


def objective_breakdown(
    *,
    shape_rmse_m: float,
    total_strands: int,
    stress_std_mpa: float,
    stress_violation_rms_mpa: float,
    weights: ObjectiveWeights,
) -> ObjectiveBreakdown:
    shape = weights.shape * (shape_rmse_m / weights.shape_scale_m) ** 2
    strands = weights.total_strands * (total_strands / weights.strand_scale)
    uniform = weights.stress_uniform * (stress_std_mpa / weights.stress_scale_mpa) ** 2
    violation = weights.stress_violation * (stress_violation_rms_mpa / weights.stress_scale_mpa) ** 2
    return ObjectiveBreakdown(
        shape=float(shape),
        total_strands=float(strands),
        stress_uniform=float(uniform),
        stress_violation=float(violation),
    )
