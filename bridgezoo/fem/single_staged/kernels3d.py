"""Numerical kernels shared by the self-written 3D staged solver."""

from __future__ import annotations

import numpy as np


def frame_axes_3d(
    xyz_i: tuple[float, float, float],
    xyz_j: tuple[float, float, float],
    orientation: tuple[float, float, float],
) -> tuple[float, np.ndarray]:
    """Return member length and global-to-local direction-cosine matrix.

    ``orientation`` follows OpenSees' 3D ``vecxz`` convention: it lies in the
    local x-z plane.  Local y is ``orientation × local_x`` and local z follows
    from the right-hand rule.
    """

    delta = np.asarray(xyz_j, dtype=float) - np.asarray(xyz_i, dtype=float)
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError("3D frame/truss element has zero length")
    local_x = delta / length
    reference = np.asarray(orientation, dtype=float)
    local_y = np.cross(reference, local_x)
    norm_y = float(np.linalg.norm(local_y))
    if norm_y <= 1.0e-12:
        raise ValueError("frame orientation vector is parallel to its axis")
    local_y /= norm_y
    local_z = np.cross(local_x, local_y)
    rotation = np.vstack((local_x, local_y, local_z))
    return length, rotation


def frame_transform_3d(rotation: np.ndarray) -> np.ndarray:
    """12x12 transform satisfying ``d_local = T @ d_global``."""

    transform = np.zeros((12, 12), dtype=float)
    for start in (0, 3, 6, 9):
        transform[start : start + 3, start : start + 3] = rotation
    return transform


def frame_local_stiffness_3d(
    E: float,
    G: float,
    A: float,
    Iy: float,
    Iz: float,
    J: float,
    length: float,
) -> np.ndarray:
    """Euler-Bernoulli 3D frame stiffness in local coordinates."""

    stiffness = np.zeros((12, 12), dtype=float)

    axial = E * A / length
    stiffness[np.ix_([0, 6], [0, 6])] += axial * np.array([[1.0, -1.0], [-1.0, 1.0]])

    torsion = G * J / length
    stiffness[np.ix_([3, 9], [3, 9])] += torsion * np.array([[1.0, -1.0], [-1.0, 1.0]])

    # Local-y translation bends about z; rz has the same sign as dv/dx.
    bending_z = E * Iz / length**3
    matrix_z = np.array(
        [
            [12.0, 6.0 * length, -12.0, 6.0 * length],
            [6.0 * length, 4.0 * length**2, -6.0 * length, 2.0 * length**2],
            [-12.0, -6.0 * length, 12.0, -6.0 * length],
            [6.0 * length, 2.0 * length**2, -6.0 * length, 4.0 * length**2],
        ]
    )
    stiffness[np.ix_([1, 5, 7, 11], [1, 5, 7, 11])] += bending_z * matrix_z

    # Local-z translation bends about y; ry has the opposite sign to dw/dx.
    bending_y = E * Iy / length**3
    matrix_y = np.array(
        [
            [12.0, -6.0 * length, -12.0, -6.0 * length],
            [-6.0 * length, 4.0 * length**2, 6.0 * length, 2.0 * length**2],
            [-12.0, 6.0 * length, 12.0, 6.0 * length],
            [-6.0 * length, 2.0 * length**2, 6.0 * length, 4.0 * length**2],
        ]
    )
    stiffness[np.ix_([2, 4, 8, 10], [2, 4, 8, 10])] += bending_y * matrix_y
    return stiffness


def uniform_load_local_3d(q_local: np.ndarray, length: float) -> np.ndarray:
    """Consistent local nodal loads for uniform ``(qx, qy, qz)``."""

    qx, qy, qz = (float(value) for value in q_local)
    equivalent = np.zeros(12, dtype=float)
    equivalent[[0, 6]] = qx * length / 2.0
    equivalent[[1, 7]] = qy * length / 2.0
    equivalent[5] = qy * length**2 / 12.0
    equivalent[11] = -equivalent[5]
    equivalent[[2, 8]] = qz * length / 2.0
    equivalent[4] = -qz * length**2 / 12.0
    equivalent[10] = -equivalent[4]
    return equivalent


def truss_stiffness_3d(
    xyz_i: tuple[float, float, float],
    xyz_j: tuple[float, float, float],
    E: float,
    A: float,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return length, unit axis and 6x6 global translational stiffness."""

    delta = np.asarray(xyz_j, dtype=float) - np.asarray(xyz_i, dtype=float)
    length = float(np.linalg.norm(delta))
    if length <= 0.0:
        raise ValueError("3D truss element has zero length")
    axis = delta / length
    direction = np.hstack((-axis, axis))
    stiffness = E * A / length * np.outer(direction, direction)
    return length, axis, stiffness


__all__ = [
    "frame_axes_3d",
    "frame_transform_3d",
    "frame_local_stiffness_3d",
    "uniform_load_local_3d",
    "truss_stiffness_3d",
]
