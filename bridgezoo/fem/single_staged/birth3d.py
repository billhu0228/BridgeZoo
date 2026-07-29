"""Shared flexible birth geometry for incremental 3D construction.

MIDAS' current initial-tangent convention uses the actual stiffness of the
structure erected in the next stage.  This module mirrors that convention by
solving an unloaded auxiliary system for each newly added structural group:
previously committed interface degrees of freedom are prescribed, while the
new nodes are elastically extended through the actual frame/truss stiffness.
The auxiliary forces are discarded; the resulting displacement field is the
stress-free birth reference used by both production backends.
"""

from __future__ import annotations

from collections.abc import MutableMapping

import numpy as np

from bridgezoo.fem.single_staged.kernels3d import (
    frame_axes_3d,
    frame_local_stiffness_3d,
    frame_transform_3d,
    truss_stiffness_3d,
)
from bridgezoo.fem.single_staged.model3d import BridgeModel3D


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


def _rigid_birth(
    model: BridgeModel3D,
    node_id: int,
    displacement: MutableMapping[int, np.ndarray],
    ncase: int,
) -> np.ndarray:
    """Return the legacy rigid extrapolation used as a safe fallback."""

    node = model.nodes[node_id]
    if node.birth_master is None:
        return np.zeros((6, ncase), dtype=float)
    master = model.nodes[node.birth_master]
    master_u = _as_columns(displacement[node.birth_master])
    offset = np.asarray(node.xyz) - np.asarray(master.xyz)
    node_u = np.empty((6, ncase), dtype=float)
    node_u[:3] = master_u[:3] - _skew(offset) @ master_u[3:]
    node_u[3:] = master_u[3:]
    return node_u


def initialize_flexible_birth_3d(
    model: BridgeModel3D,
    stage_index: int,
    displacement: MutableMapping[int, np.ndarray],
    *,
    ncase: int | None = None,
) -> None:
    """Add nodes born by ``stage_index`` using an actual-stiffness extension.

    The mutable ``displacement`` values may be scalar ``(6,)`` vectors or
    batched ``(6, ncase)`` matrices.  Nodes tied by a rigid link retain the
    exact rigid-offset birth rule.  Other new nodes first receive a rigid
    fallback, then nodes belonging to the newly erected frame/truss group are
    replaced by the solution of its unloaded elastic extension problem.
    """

    existing_ids = set(displacement)
    pending = sorted(
        (
            node
            for node in model.nodes.values()
            if node.activation_stage <= stage_index and node.id not in displacement
        ),
        key=lambda node: (node.activation_stage, node.id),
    )
    if not pending:
        return

    if displacement:
        sample = _as_columns(next(iter(displacement.values())))
        ncase = sample.shape[1]
        scalar = np.asarray(next(iter(displacement.values()))).ndim == 1
    else:
        scalar = ncase is None
        ncase = 1 if ncase is None else ncase
    if ncase <= 0:
        raise ValueError("3D staged birth initialization requires a positive case count")

    pending_ids = {node.id for node in pending}
    stage_frames = [
        frame
        for frame in model.frames.values()
        if frame.activation_stage == stage_index
        and (frame.i in pending_ids or frame.j in pending_ids)
    ]
    stage_cables = [
        cable
        for cable in model.cables.values()
        if cable.activation_stage == stage_index
        and (cable.i in pending_ids or cable.j in pending_ids)
    ]
    stage_links = [
        link
        for link in model.rigid_links.values()
        if link.activation_stage == stage_index and link.slave in pending_ids
    ]
    slave_ids = {link.slave for link in stage_links}

    # Establish a complete, dependency-ordered fallback first.  Rigid-link
    # slaves are refreshed after the flexible solve so slab offsets remain
    # exact when their masters also receive a flexible birth displacement.
    unresolved = list(pending)
    while unresolved:
        progressed = False
        for node in list(unresolved):
            if node.birth_master is not None and node.birth_master not in displacement:
                continue
            value = _rigid_birth(model, node.id, displacement, ncase)
            displacement[node.id] = value[:, 0] if scalar else value
            unresolved.remove(node)
            progressed = True
        if not progressed:
            node_ids = [node.id for node in unresolved]
            raise ValueError(f"unresolved 3D node birth masters: {node_ids}")

    flexible_frames = [
        frame
        for frame in stage_frames
        if frame.i not in slave_ids and frame.j not in slave_ids
    ]
    flexible_cables = [
        cable
        for cable in stage_cables
        if cable.i not in slave_ids and cable.j not in slave_ids
    ]
    domain_ids = sorted(
        {
            node_id
            for element in (*flexible_frames, *flexible_cables)
            for node_id in (element.i, element.j)
        }
    )
    flexible_new_ids = [node_id for node_id in domain_ids if node_id in pending_ids]
    interface_ids = [node_id for node_id in domain_ids if node_id in existing_ids]

    if flexible_new_ids and interface_ids:
        position = {node_id: index for index, node_id in enumerate(domain_ids)}
        ndof = 6 * len(domain_ids)
        stiffness = np.zeros((ndof, ndof), dtype=float)

        def dofs(node_id: int) -> list[int]:
            start = 6 * position[node_id]
            return list(range(start, start + 6))

        for element in flexible_frames:
            node_i = model.nodes[element.i]
            node_j = model.nodes[element.j]
            element_dofs = dofs(element.i) + dofs(element.j)
            length, rotation = frame_axes_3d(
                node_i.xyz, node_j.xyz, element.orientation
            )
            transform = frame_transform_3d(rotation)
            section = element.section
            local = frame_local_stiffness_3d(
                element.material.E,
                element.material.G,
                section.A,
                section.Iy,
                section.Iz,
                section.J,
                length,
            )
            element_stiffness = transform.T @ local @ transform
            stiffness[np.ix_(element_dofs, element_dofs)] += element_stiffness
        for element in flexible_cables:
            node_i = model.nodes[element.i]
            node_j = model.nodes[element.j]
            _, _, translation = truss_stiffness_3d(
                node_i.xyz,
                node_j.xyz,
                element.material.E,
                element.area,
            )
            element_dofs = dofs(element.i)[:3] + dofs(element.j)[:3]
            stiffness[np.ix_(element_dofs, element_dofs)] += translation

        prescribed_dofs: list[int] = []
        prescribed_values: list[np.ndarray] = []
        for node_id in interface_ids:
            values = _as_columns(displacement[node_id])
            for component, full_dof in enumerate(dofs(node_id)):
                prescribed_dofs.append(full_dof)
                prescribed_values.append(values[component])

        # A newly created support without an erection master represents an
        # original-position anchor (the cable ground anchors in this model).
        # Supports on tangent-born nodes instead lock their birth position and
        # must not force the auxiliary field back to zero.
        for support in model.supports.values():
            if (
                support.activation_stage <= stage_index
                and support.node in flexible_new_ids
                and model.nodes[support.node].birth_master is None
            ):
                for component, restrained in enumerate(support.restraints):
                    if restrained:
                        prescribed_dofs.append(dofs(support.node)[component])
                        prescribed_values.append(np.zeros(ncase, dtype=float))

        if prescribed_dofs:
            prescribed = np.asarray(prescribed_dofs, dtype=int)
            known = np.vstack(prescribed_values)
            prescribed_set = set(prescribed_dofs)
            candidate = [
                full_dof
                for node_id in flexible_new_ids
                for full_dof in dofs(node_id)
                if full_dof not in prescribed_set
                and np.any(np.abs(stiffness[full_dof, :]) > 0.0)
            ]
            free = np.asarray(candidate, dtype=int)
            if free.size:
                free_stiffness = stiffness[np.ix_(free, free)]
                right_hand_side = -stiffness[np.ix_(free, prescribed)] @ known
                diagonal = np.diag(free_stiffness)
                if np.all(diagonal > 0.0):
                    scale = 1.0 / np.sqrt(diagonal)
                    equilibrated = (
                        scale[:, None] * free_stiffness * scale[None, :]
                    )
                    equilibrated_rhs = scale[:, None] * right_hand_side
                    solution, _, rank, _ = np.linalg.lstsq(
                        equilibrated,
                        equilibrated_rhs,
                        rcond=1.0e-12,
                    )
                    if rank == free.size:
                        full_solution = np.zeros((ndof, ncase), dtype=float)
                        full_solution[prescribed] = known
                        full_solution[free] = scale[:, None] * solution
                        for node_id in flexible_new_ids:
                            value = full_solution[dofs(node_id)]
                            displacement[node_id] = value[:, 0] if scalar else value

    for link in stage_links:
        master_u = _as_columns(displacement[link.master])
        offset = np.asarray(model.nodes[link.slave].xyz) - np.asarray(
            model.nodes[link.master].xyz
        )
        slave_u = np.empty((6, ncase), dtype=float)
        slave_u[:3] = master_u[:3] - _skew(offset) @ master_u[3:]
        slave_u[3:] = master_u[3:]
        displacement[link.slave] = slave_u[:, 0] if scalar else slave_u


__all__ = ["initialize_flexible_birth_3d"]
