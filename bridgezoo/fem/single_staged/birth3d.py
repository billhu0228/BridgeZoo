"""MIDAS-style real-displacement history for staged 3D erection.

``Current Step Displacement`` is produced by the active FEM model.  ``Real
Displacement`` additionally contains the virtual tangent displacement that a
future beam node accumulated before it was erected.  MIDAS retains that
history stage by stage; it does not recreate the complete reference only when
the node becomes active.

This module owns that history for both production backends.  Only main-girder
beam nodes receive pre-activation tangent transport.  Slab nodes inherit their
active beam master's rigid-body position when the slab is installed, while
cables and future slab stiffness never enter the tangent reference.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping

import numpy as np

from bridgezoo.fem.single_staged.model3d import BridgeModel3D, Node3D


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _as_columns(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim == 1:
        return array[:, None]
    if array.ndim != 2:
        raise ValueError("3D staged displacement must be a vector or column matrix")
    return array


def _transport(
    model: BridgeModel3D,
    source_id: int,
    target_id: int,
    source_value: np.ndarray,
) -> np.ndarray:
    """Rigidly transport one stage's beam-end motion to a future node."""

    source = model.nodes[source_id]
    target = model.nodes[target_id]
    values = _as_columns(source_value)
    offset = np.asarray(target.xyz) - np.asarray(source.xyz)
    transported = np.empty_like(values)
    transported[:3] = values[:3] - _skew(offset) @ values[3:]
    transported[3:] = values[3:]
    return transported


def _is_tangent_beam_node(node: Node3D) -> bool:
    return node.role.startswith("main_girder")


class TangentDisplacementHistory3D:
    """Accumulate virtual tangent displacement before beam-node activation.

    Values are stored as ``(6, ncase)`` columns internally.  ``batched`` only
    controls the shape written into the solver's mutable displacement mapping.
    The legacy empirical correction is deliberately restricted to its identity
    value: scaling a six-DOF history is not a valid MIDAS activation rule.
    """

    def __init__(
        self,
        model: BridgeModel3D,
        *,
        ncase: int = 1,
        batched: bool = False,
        correction_factor: float = 1.0,
    ) -> None:
        if isinstance(ncase, bool) or ncase <= 0:
            raise ValueError("3D tangent history requires a positive case count")
        if not math.isfinite(correction_factor) or not math.isclose(
            correction_factor, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                "flexible_birth_correction_factor is retired and must equal 1.0"
            )
        self.model = model
        self.ncase = ncase
        self.batched = batched
        self.virtual: dict[int, np.ndarray] = {
            node.id: np.zeros((6, ncase), dtype=float)
            for node in model.nodes.values()
            if _is_tangent_beam_node(node)
        }

    def _output_shape(self, value: np.ndarray) -> np.ndarray:
        return value.copy() if self.batched else value[:, 0].copy()

    def activate(
        self,
        stage_index: int,
        displacement: MutableMapping[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Activate nodes through ``stage_index`` at their stored real reference."""

        pending = sorted(
            (
                node
                for node in self.model.nodes.values()
                if node.activation_stage <= stage_index
                and node.id not in displacement
            ),
            key=lambda node: (node.activation_stage, node.id),
        )
        born: dict[int, np.ndarray] = {}
        for node in pending:
            if _is_tangent_beam_node(node):
                value = self.virtual[node.id]
            elif node.birth_master is not None:
                if node.birth_master not in displacement:
                    raise ValueError(
                        f"3D node {node.id} activates before birth master "
                        f"{node.birth_master} has a real displacement"
                    )
                value = _transport(
                    self.model,
                    node.birth_master,
                    node.id,
                    displacement[node.birth_master],
                )
            else:
                value = np.zeros((6, self.ncase), dtype=float)
            if value.shape != (6, self.ncase) or not np.all(np.isfinite(value)):
                raise ValueError(
                    f"3D node {node.id} has an invalid tangent activation reference"
                )
            stored = self._output_shape(value)
            displacement[node.id] = stored
            born[node.id] = stored.copy()
        return born

    def _active_ancestor(self, node: Node3D, stage_index: int) -> int | None:
        ancestor_id = node.birth_master
        visited = {node.id}
        while ancestor_id is not None:
            if ancestor_id in visited:
                raise ValueError(f"cyclic 3D birth-master chain at node {node.id}")
            visited.add(ancestor_id)
            ancestor = self.model.nodes[ancestor_id]
            if ancestor.activation_stage <= stage_index:
                return ancestor_id
            ancestor_id = ancestor.birth_master
        return None

    def accumulate(
        self,
        stage_index: int,
        stage_increment: Mapping[int, np.ndarray],
    ) -> None:
        """Add this stage's active beam motion to every future beam node."""

        for node in self.model.nodes.values():
            if not _is_tangent_beam_node(node) or node.activation_stage <= stage_index:
                continue
            ancestor_id = self._active_ancestor(node, stage_index)
            if ancestor_id is None:
                continue
            try:
                ancestor_increment = stage_increment[ancestor_id]
            except KeyError as exc:
                raise ValueError(
                    f"future 3D beam node {node.id} has no active tangent ancestor "
                    f"increment at stage {stage_index}"
                ) from exc
            transported = _transport(
                self.model,
                ancestor_id,
                node.id,
                ancestor_increment,
            )
            if transported.shape != (6, self.ncase) or not np.all(
                np.isfinite(transported)
            ):
                raise ValueError(
                    f"future 3D beam node {node.id} received an invalid tangent "
                    f"increment at stage {stage_index}"
                )
            self.virtual[node.id] += transported


__all__ = ["TangentDisplacementHistory3D"]
