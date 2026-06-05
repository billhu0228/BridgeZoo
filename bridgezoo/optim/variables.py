"""Cable optimization variable helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CableLayout:
    """Stage-major cable ordering used by the optimizers.

    The order is ``right1, left1, right2, left2, ...`` so vectors map directly
    to the staged builder's independent ``strands`` / ``pretension`` format.
    """

    n_seg: int

    @property
    def cable_ids(self) -> tuple[int, ...]:
        ids = []
        for i in range(1, self.n_seg + 1):
            ids.extend((1000 + i, 2000 + i))
        return tuple(ids)

    @property
    def size(self) -> int:
        return 2 * self.n_seg

    def as_mapping(self, values) -> dict[int, float]:
        arr = np.asarray(values)
        if arr.size != self.size:
            raise ValueError(f"expected {self.size} values, got {arr.size}")
        return {cid: float(value) for cid, value in zip(self.cable_ids, arr)}

    def as_int_mapping(self, values) -> dict[int, int]:
        arr = np.asarray(values)
        if arr.size != self.size:
            raise ValueError(f"expected {self.size} values, got {arr.size}")
        return {cid: int(value) for cid, value in zip(self.cable_ids, arr)}

    def stage_pairs(self, values) -> list[tuple[float, float]]:
        arr = np.asarray(values, dtype=float)
        if arr.size != self.size:
            raise ValueError(f"expected {self.size} values, got {arr.size}")
        return [(float(arr[2 * k]), float(arr[2 * k + 1])) for k in range(self.n_seg)]


def validate_strand_vector(values, layout: CableLayout, lower: int, upper: int) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size != layout.size:
        raise ValueError(f"expected {layout.size} strand counts, got {arr.size}")
    if np.iscomplexobj(arr):
        raise ValueError("strand counts must be real integers")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("strand counts must be finite")
    rounded = np.rint(arr).astype(int)
    if not np.allclose(arr, rounded):
        raise ValueError("strand counts must be integers")
    if np.any(rounded < lower) or np.any(rounded > upper):
        raise ValueError(f"strand counts must be between {lower} and {upper}")
    return rounded


def validate_tension_vector(values, layout: CableLayout) -> np.ndarray:
    arr = np.asarray(values)
    if arr.size != layout.size:
        raise ValueError(f"expected {layout.size} pretensions, got {arr.size}")
    if np.iscomplexobj(arr):
        raise ValueError("pretensions must be real")
    arr = arr.astype(float)
    if not np.all(np.isfinite(arr)):
        raise ValueError("pretensions must be finite")
    if np.any(arr < 0.0):
        raise ValueError("pretensions must be non-negative")
    return arr
