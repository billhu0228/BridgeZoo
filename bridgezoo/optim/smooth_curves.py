"""Low-dimensional smooth curves for staged 3D cable design variables."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


CURVE_FAMILIES = ("bernstein", "piecewise-linear")


def build_smooth_curve_basis(
    n_seg: int,
    control_points: int,
    family: str,
) -> np.ndarray:
    """Return a bounded interpolation basis from inner to outer cable stage.

    Every row is nonnegative and sums to one.  Consequently, control values
    inside a physical bound produce a full interpolated curve inside the same
    bound.  ``bernstein`` with four controls is a cubic Bezier-like curve;
    ``piecewise-linear`` uses equally spaced control stations.
    """

    if isinstance(n_seg, bool) or n_seg < 1:
        raise ValueError("n_seg must be positive")
    if isinstance(control_points, bool) or control_points < 1:
        raise ValueError("curve control_points must be positive")
    if family not in CURVE_FAMILIES:
        raise ValueError(
            f"curve family must be one of {', '.join(CURVE_FAMILIES)}"
        )
    count = min(int(control_points), int(n_seg))
    if count == 1:
        return np.ones((n_seg, 1), dtype=float)

    stage_x = np.linspace(0.0, 1.0, n_seg)
    if family == "bernstein":
        degree = count - 1
        return np.column_stack(
            [
                math.comb(degree, index)
                * stage_x**index
                * (1.0 - stage_x) ** (degree - index)
                for index in range(count)
            ]
        )

    control_x = np.linspace(0.0, 1.0, count)
    basis = np.zeros((n_seg, count), dtype=float)
    for row, value in enumerate(stage_x):
        right = int(np.searchsorted(control_x, value, side="right"))
        if right == 0:
            basis[row, 0] = 1.0
        elif right >= count:
            basis[row, -1] = 1.0
        else:
            left = right - 1
            fraction = (value - control_x[left]) / (
                control_x[right] - control_x[left]
            )
            basis[row, left] = 1.0 - fraction
            basis[row, right] = fraction
    return basis


def build_stage_major_curve_basis(
    n_seg: int,
    control_points: int,
    family: str,
) -> np.ndarray:
    """Build independent backstay/main-stay curve blocks in stage-major order."""

    one_group = build_smooth_curve_basis(n_seg, control_points, family)
    count = one_group.shape[1]
    basis = np.zeros((2 * n_seg, 2 * count), dtype=float)
    basis[0::2, :count] = one_group
    basis[1::2, count:] = one_group
    return basis


def _isotonic_non_decreasing(values: np.ndarray) -> np.ndarray:
    """Project a short vector onto the non-decreasing cone using PAVA."""

    blocks: list[list[float | int]] = []
    for index, raw in enumerate(np.asarray(values, dtype=float)):
        blocks.append([float(raw), 1.0, index, index + 1])
        while len(blocks) >= 2 and blocks[-2][0] > blocks[-1][0]:
            right = blocks.pop()
            left = blocks.pop()
            weight = float(left[1]) + float(right[1])
            level = (
                float(left[0]) * float(left[1])
                + float(right[0]) * float(right[1])
            ) / weight
            blocks.append([level, weight, int(left[2]), int(right[3])])
    projected = np.empty(len(values), dtype=float)
    for level, _, start, stop in blocks:
        projected[int(start) : int(stop)] = float(level)
    return projected


@dataclass(frozen=True)
class SmoothStrandCurve3D:
    family: str
    control_coordinates: np.ndarray
    backstay_control_strands: np.ndarray
    main_stay_control_strands: np.ndarray
    interpolated_strands: np.ndarray


def project_strands_to_smooth_curve(
    strands,
    *,
    n_seg: int,
    control_points: int,
    family: str,
    lower: int,
    upper: int,
) -> SmoothStrandCurve3D:
    """Fit, monotonize, interpolate and round two cable-count curves.

    Stage zero is the innermost cable and the final stage is the outermost.
    Monotone control values therefore enforce that outer cable groups never
    contain fewer strands than inner groups of the same type.
    """

    values = np.asarray(strands, dtype=float)
    if values.size != 2 * n_seg or not np.all(np.isfinite(values)):
        raise ValueError("strand curve input must contain 2*n_seg finite values")
    if lower < 1 or upper < lower:
        raise ValueError("invalid strand curve bounds")
    basis = build_smooth_curve_basis(n_seg, control_points, family)
    controls = []
    interpolated = []
    for group_values in (values[0::2], values[1::2]):
        fitted, *_ = np.linalg.lstsq(basis, group_values, rcond=None)
        fitted = np.clip(_isotonic_non_decreasing(fitted), lower, upper)
        rounded = np.rint(basis @ fitted).astype(int)
        rounded = np.maximum.accumulate(np.clip(rounded, lower, upper))
        controls.append(fitted)
        interpolated.append(rounded)

    stage_major = np.empty(2 * n_seg, dtype=int)
    stage_major[0::2] = interpolated[0]
    stage_major[1::2] = interpolated[1]
    count = basis.shape[1]
    return SmoothStrandCurve3D(
        family=family,
        control_coordinates=np.linspace(0.0, 1.0, count),
        backstay_control_strands=controls[0],
        main_stay_control_strands=controls[1],
        interpolated_strands=stage_major,
    )


__all__ = [
    "CURVE_FAMILIES",
    "SmoothStrandCurve3D",
    "build_smooth_curve_basis",
    "build_stage_major_curve_basis",
    "project_strands_to_smooth_curve",
]
