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


def _kinematic_seed(
    model: BridgeModel3D,
    node_id: int,
    displacement: MutableMapping[int, np.ndarray],
    ncase: int,
) -> np.ndarray:
    """Return a dependency-ordered seed for the auxiliary elastic solve."""

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
    batched ``(6, ncase)`` matrices.  The auxiliary system includes the newly
    erected beam and cable stiffness, but deliberately excludes deck-slab
    strips.  Every new main-girder node must be obtained from this elastic
    extension problem.  A compatible rank-deficient system uses its
    minimum-norm elastic solution; a non-finite or non-equilibrating solution
    is an error and never triggers a rigid-tangent fallback.  Deck nodes tied
    by a rigid link retain the exact rigid-offset birth rule.
    """

    existing_ids = set(displacement)
    birth_values = dict(displacement)
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
    bootstrap_ids = {
        node.id for node in pending if node.activation_stage < stage_index
    }
    stage_pending_ids = pending_ids - bootstrap_ids
    # Nodes erected before the first requested analysis stage form its already
    # existing interface at their original, stress-free position.
    existing_ids.update(bootstrap_ids)
    stage_frames = [
        frame
        for frame in model.frames.values()
        if frame.activation_stage == stage_index
        and (frame.i in stage_pending_ids or frame.j in stage_pending_ids)
    ]
    stage_cables = [
        cable
        for cable in model.cables.values()
        if cable.activation_stage == stage_index
        and (cable.i in stage_pending_ids or cable.j in stage_pending_ids)
    ]
    stage_links = [
        link
        for link in model.rigid_links.values()
        if link.activation_stage == stage_index and link.slave in stage_pending_ids
    ]
    slave_ids = {link.slave for link in stage_links}

    # Establish dependency-ordered seed values without committing them to the
    # caller.  Structural-node seeds must be overwritten by the elastic solve
    # below; rigid-link slaves are refreshed afterward from their solved master.
    unresolved = list(pending)
    while unresolved:
        progressed = False
        for node in list(unresolved):
            if node.birth_master is not None and node.birth_master not in birth_values:
                continue
            value = _kinematic_seed(model, node.id, birth_values, ncase)
            birth_values[node.id] = value[:, 0] if scalar else value
            unresolved.remove(node)
            progressed = True
        if not progressed:
            node_ids = [node.id for node in unresolved]
            raise ValueError(f"unresolved 3D node birth masters: {node_ids}")

    flexible_frames = [
        frame for frame in stage_frames if not frame.group.startswith("deck_")
    ]
    flexible_cables = stage_cables
    domain_ids = sorted(
        {
            node_id
            for element in (*flexible_frames, *flexible_cables)
            for node_id in (element.i, element.j)
        }
    )
    flexible_new_ids = [
        node_id for node_id in domain_ids if node_id in stage_pending_ids
    ]
    interface_ids = [node_id for node_id in domain_ids if node_id in existing_ids]

    solved_flexible_ids: set[int] = set()
    if flexible_new_ids:
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
            values = _as_columns(birth_values[node_id])
            for component, full_dof in enumerate(dofs(node_id)):
                prescribed_dofs.append(full_dof)
                prescribed_values.append(values[component])

        # A rigid-link slave is a prescribed boundary if a non-slab beam happens
        # to use it.  Deck-slab strips themselves are excluded above.
        for node_id in sorted(slave_ids & set(domain_ids)):
            values = _as_columns(birth_values[node_id])
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

        if not prescribed_dofs:
            raise ValueError(
                f"3D flexible birth stage {stage_index} has no prescribed "
                "interface or support degrees of freedom"
            )

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
        full_solution = np.zeros((ndof, ncase), dtype=float)
        full_solution[prescribed] = known
        if free.size:
            free_stiffness = stiffness[np.ix_(free, free)]
            right_hand_side = -stiffness[np.ix_(free, prescribed)] @ known
            diagonal = np.diag(free_stiffness)
            if not np.all(np.isfinite(diagonal)) or np.any(diagonal <= 0.0):
                raise ValueError(
                    f"3D flexible birth stage {stage_index} has invalid "
                    "free-DOF stiffness"
                )
            scale = 1.0 / np.sqrt(diagonal)
            equilibrated = scale[:, None] * free_stiffness * scale[None, :]
            equilibrated_rhs = scale[:, None] * right_hand_side
            solution, _, _, _ = np.linalg.lstsq(
                equilibrated,
                equilibrated_rhs,
                rcond=1.0e-12,
            )
            residual = equilibrated @ solution - equilibrated_rhs
            residual_limit = 1.0e-10 * max(
                1.0,
                float(np.linalg.norm(equilibrated_rhs, ord=np.inf)),
            )
            if (
                not np.all(np.isfinite(solution))
                or not np.all(np.isfinite(residual))
                or float(np.linalg.norm(residual, ord=np.inf)) > residual_limit
            ):
                raise ValueError(
                    f"3D flexible birth stage {stage_index} does not equilibrate; "
                    "rigid-tangent fallback is disabled"
                )
            full_solution[free] = scale[:, None] * solution

        for node_id in flexible_new_ids:
            value = full_solution[dofs(node_id)]
            birth_values[node_id] = value[:, 0] if scalar else value
            solved_flexible_ids.add(node_id)

    required_flexible_ids = {
        node_id
        for node_id in stage_pending_ids
        if model.nodes[node_id].role.startswith("main_girder")
    }
    unsolved_ids = sorted(required_flexible_ids - solved_flexible_ids)
    if unsolved_ids:
        raise ValueError(
            f"3D flexible birth stage {stage_index} did not solve new structural "
            f"nodes {unsolved_ids}; rigid-tangent fallback is disabled"
        )

    for link in stage_links:
        master_u = _as_columns(birth_values[link.master])
        offset = np.asarray(model.nodes[link.slave].xyz) - np.asarray(
            model.nodes[link.master].xyz
        )
        slave_u = np.empty((6, ncase), dtype=float)
        slave_u[:3] = master_u[:3] - _skew(offset) @ master_u[3:]
        slave_u[3:] = master_u[3:]
        birth_values[link.slave] = slave_u[:, 0] if scalar else slave_u

    displacement.update(
        {node_id: birth_values[node_id] for node_id in pending_ids}
    )


__all__ = ["initialize_flexible_birth_3d"]
