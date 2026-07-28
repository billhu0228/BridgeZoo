"""Self-written linear 3D solver for :mod:`single_staged.model3d`.

The scalar and batched paths share one implementation.  For fixed geometry and
cable areas, cable pretension changes only the load vector.  The batch solver
therefore assembles and condenses the 3D stiffness once per stage, then solves
all pretension cases as one multiple-right-hand-side linear system.  This is
the accelerated kernel used to construct the exact affine optimization model.

Path-dependent stress-free birth and displacement lock-in are intentionally a
later milestone; see ``TODO.md``.
"""

from __future__ import annotations

import numpy as np

from bridgezoo.fem.single_staged.kernels3d import (
    frame_axes_3d,
    frame_local_stiffness_3d,
    frame_transform_3d,
    truss_stiffness_3d,
    uniform_load_local_3d,
)
from bridgezoo.fem.single_staged.model3d import (
    SingleStagedPlan3D,
    SolveResult3D,
    StagedResult3D,
)


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _cable_structure(cable) -> tuple:
    """Cable fields that affect topology or stiffness, excluding pretension."""

    return (
        cable.id,
        cable.i,
        cable.j,
        cable.material,
        cable.area,
        cable.group,
        cable.activation_stage,
    )


def _assert_same_structure_3d(reference: SingleStagedPlan3D, case: SingleStagedPlan3D) -> None:
    """Require plans to differ only in cable pretension values."""

    if reference.stages != case.stages:
        raise ValueError("batched 3D plans differ in construction stages")
    reference_model = reference.model
    case_model = case.model
    for name in ("nodes", "frames", "rigid_links", "supports"):
        if getattr(reference_model, name) != getattr(case_model, name):
            raise ValueError(f"batched 3D plans differ in {name}")
    if reference_model.nodal_loads != case_model.nodal_loads:
        raise ValueError("batched 3D plans differ in nodal loads")
    if reference_model.frame_loads != case_model.frame_loads:
        raise ValueError("batched 3D plans differ in frame loads")
    if set(reference_model.cables) != set(case_model.cables):
        raise ValueError("batched 3D plans differ in cable ids")
    for cable_id, reference_cable in reference_model.cables.items():
        if _cable_structure(reference_cable) != _cable_structure(case_model.cables[cable_id]):
            raise ValueError(f"batched 3D plans differ in cable structure at {cable_id}")


class SingleStagedDirectBatchSolver3D:
    """Linear 3D direct solver sharing one factorization across load cases.

    ``solve_stage_batch`` accepts plans with identical geometry, sections,
    cable areas, loads and restraints.  Only ``CableElement3D.pretension`` may
    differ.  The reduced free-DOF system is solved once with a ``(nf, ncase)``
    right-hand-side matrix, after which every case receives complete frame,
    cable and reaction recovery.
    """

    name = "direct3d"

    def run_batch(self, plans: list[SingleStagedPlan3D]) -> list[StagedResult3D]:
        if not plans:
            raise ValueError("run_batch requires at least one 3D plan")
        reference = plans[0]
        for plan in plans[1:]:
            _assert_same_structure_3d(reference, plan)
        results = [StagedResult3D(backend=self.name) for _ in plans]
        for stage in reference.stages:
            records = self.solve_stage_batch(
                plans,
                stage.index,
                stage.label,
                _structure_checked=True,
            )
            for result, record in zip(results, records):
                result.records.append(record)
        return results

    def solve_stage_batch(
        self,
        plans: list[SingleStagedPlan3D],
        stage_index: int,
        stage_label: str | None = None,
        *,
        _structure_checked: bool = False,
    ) -> list[SolveResult3D]:
        if not plans:
            raise ValueError("solve_stage_batch requires at least one 3D plan")
        reference = plans[0]
        if not _structure_checked:
            for plan in plans[1:]:
                _assert_same_structure_3d(reference, plan)
        model = reference.model
        ncase = len(plans)
        nodes = sorted(
            (node for node in model.nodes.values() if node.activation_stage <= stage_index),
            key=lambda node: node.id,
        )
        node_ids = {node.id for node in nodes}
        index = {node.id: position for position, node in enumerate(nodes)}
        ndof = 6 * len(nodes)
        stiffness = np.zeros((ndof, ndof), dtype=float)
        constant_load = np.zeros(ndof, dtype=float)

        def dofs(node_id: int) -> list[int]:
            base = 6 * index[node_id]
            return list(range(base, base + 6))

        active_frames = {
            element.id: element
            for element in model.frames.values()
            if element.activation_stage <= stage_index
        }
        active_cables = {
            element.id: element
            for element in model.cables.values()
            if element.activation_stage <= stage_index
        }
        active_links = [
            link for link in model.rigid_links.values() if link.activation_stage <= stage_index
        ]
        frame_loads = [
            item
            for item in model.frame_loads
            if item.activation_stage <= stage_index and item.member in active_frames
        ]
        loads_by_frame: dict[int, list] = {}
        for item in frame_loads:
            loads_by_frame.setdefault(item.member, []).append(item)

        frame_data: dict[int, tuple[np.ndarray, np.ndarray, list[int], np.ndarray]] = {}
        total_applied = np.zeros(3, dtype=float)
        for element in active_frames.values():
            node_i, node_j = model.nodes[element.i], model.nodes[element.j]
            length, rotation = frame_axes_3d(node_i.xyz, node_j.xyz, element.orientation)
            transform = frame_transform_3d(rotation)
            section = element.section
            local_stiffness = frame_local_stiffness_3d(
                element.material.E,
                element.material.G,
                section.A,
                section.Iy,
                section.Iz,
                section.J,
                length,
            )
            global_stiffness = transform.T @ local_stiffness @ transform
            element_dofs = dofs(element.i) + dofs(element.j)
            stiffness[np.ix_(element_dofs, element_dofs)] += global_stiffness

            fixed_end_local = np.zeros(12, dtype=float)
            for item in loads_by_frame.get(element.id, ()):
                global_q = np.asarray(item.global_vector, dtype=float)
                local_q = rotation @ global_q
                equivalent_local = uniform_load_local_3d(local_q, length)
                constant_load[element_dofs] += transform.T @ equivalent_local
                fixed_end_local += equivalent_local
                total_applied += global_q * length
            frame_data[element.id] = (
                local_stiffness,
                transform,
                element_dofs,
                fixed_end_local,
            )

        cable_data: dict[int, tuple[float, np.ndarray, list[int]]] = {}
        for element in active_cables.values():
            node_i, node_j = model.nodes[element.i], model.nodes[element.j]
            length, axis, translational_stiffness = truss_stiffness_3d(
                node_i.xyz,
                node_j.xyz,
                element.material.E,
                element.area,
            )
            element_dofs = dofs(element.i)[:3] + dofs(element.j)[:3]
            stiffness[np.ix_(element_dofs, element_dofs)] += translational_stiffness
            cable_data[element.id] = (length, axis, element_dofs)

        for item in model.nodal_loads:
            if item.activation_stage <= stage_index and item.node in node_ids:
                constant_load[dofs(item.node)] += np.asarray(item.values, dtype=float)
                total_applied += np.asarray(item.values[:3], dtype=float)

        # Each case shares the permanent-load column and differs only in the
        # initial cable-force equivalent nodal loads.
        load = np.repeat(constant_load[:, None], ncase, axis=1)
        for element_id, (_, axis, _) in cable_data.items():
            reference_element = active_cables[element_id]
            node_i_dofs = dofs(reference_element.i)[:3]
            node_j_dofs = dofs(reference_element.j)[:3]
            tensions = np.asarray(
                [plan.model.cables[element_id].pretension for plan in plans],
                dtype=float,
            )
            load[node_i_dofs, :] += axis[:, None] * tensions[None, :]
            load[node_j_dofs, :] -= axis[:, None] * tensions[None, :]

        # Exact rigid-link transformation: u_slave = u_master + theta x r,
        # theta_slave = theta_master.  It is constructed once for every case.
        slave_ids = {link.slave for link in active_links}
        independent_full_dofs = [
            full_dof
            for node in nodes
            if node.id not in slave_ids
            for full_dof in dofs(node.id)
        ]
        reduced_column = {
            full_dof: column for column, full_dof in enumerate(independent_full_dofs)
        }
        constraint_transform = np.zeros((ndof, len(independent_full_dofs)), dtype=float)
        for full_dof, column in reduced_column.items():
            constraint_transform[full_dof, column] = 1.0
        for link in active_links:
            if link.master in slave_ids:
                raise ValueError("chained 3D rigid links are not supported")
            master_dofs = dofs(link.master)
            slave_dofs = dofs(link.slave)
            offset = np.asarray(model.nodes[link.slave].xyz) - np.asarray(
                model.nodes[link.master].xyz
            )
            rotation_coupling = -_skew(offset)
            for component in range(3):
                constraint_transform[
                    slave_dofs[component], reduced_column[master_dofs[component]]
                ] = 1.0
                for rotation_component in range(3):
                    constraint_transform[
                        slave_dofs[component],
                        reduced_column[master_dofs[3 + rotation_component]],
                    ] += rotation_coupling[component, rotation_component]
                constraint_transform[
                    slave_dofs[3 + component],
                    reduced_column[master_dofs[3 + component]],
                ] = 1.0

        reduced_stiffness = constraint_transform.T @ stiffness @ constraint_transform
        reduced_load = constraint_transform.T @ load
        fixed = np.zeros(len(independent_full_dofs), dtype=bool)
        active_supports = [
            support
            for support in model.supports.values()
            if support.activation_stage <= stage_index and support.node in node_ids
        ]
        for support in active_supports:
            if support.node in slave_ids:
                raise ValueError("supports on rigid-link slave nodes are not supported")
            for component, restrained in enumerate(support.restraints):
                if restrained:
                    fixed[reduced_column[dofs(support.node)[component]]] = True

        # Truss-only nodes have unused rotational DOFs.
        for reduced_dof in range(len(independent_full_dofs)):
            if not fixed[reduced_dof] and not np.any(reduced_stiffness[reduced_dof, :]):
                fixed[reduced_dof] = True

        free = np.flatnonzero(~fixed)
        reduced_displacement = np.zeros((len(independent_full_dofs), ncase), dtype=float)
        converged = True
        if free.size:
            free_stiffness = reduced_stiffness[np.ix_(free, free)]
            free_load = reduced_load[free, :]
            try:
                # LAPACK gesv factors free_stiffness once and applies that
                # factorization to every right-hand side in free_load.
                reduced_displacement[free, :] = np.linalg.solve(
                    free_stiffness,
                    free_load,
                )
            except np.linalg.LinAlgError:
                converged = False
        displacement = constraint_transform @ reduced_displacement
        reduced_reaction = reduced_stiffness @ reduced_displacement - reduced_load

        label = stage_label or next(
            (stage.label for stage in reference.stages if stage.index == stage_index),
            f"stage{stage_index}",
        )
        results: list[SolveResult3D] = []
        for case_index, plan in enumerate(plans):
            case_displacement = displacement[:, case_index]
            result = SolveResult3D(
                backend=self.name,
                stage_index=stage_index,
                stage_label=label,
                converged=converged,
                applied_load=tuple(float(value) for value in total_applied),
            )
            for node in nodes:
                result.displacement[node.id] = tuple(
                    float(value) for value in case_displacement[dofs(node.id)]
                )

            for element_id, data in frame_data.items():
                local_stiffness, transform, element_dofs, fixed_end = data
                local_displacement = transform @ case_displacement[element_dofs]
                local_force = local_stiffness @ local_displacement - fixed_end
                result.frame_force[element_id] = tuple(float(value) for value in local_force)

            for element_id, (length, axis, element_dofs) in cable_data.items():
                element = plan.model.cables[element_id]
                translational_displacement = case_displacement[element_dofs]
                elongation = float(
                    axis
                    @ (
                        translational_displacement[3:]
                        - translational_displacement[:3]
                    )
                )
                force = (
                    element.pretension
                    + element.material.E * element.area / length * elongation
                )
                result.cable_force[element_id] = float(force)
                result.cable_stress[element_id] = float(force / element.area)

            for support in active_supports:
                values = []
                for component in range(6):
                    column = reduced_column[dofs(support.node)[component]]
                    values.append(
                        float(reduced_reaction[column, case_index])
                        if fixed[column]
                        else 0.0
                    )
                result.support_reaction[support.node] = tuple(values)
            results.append(result)
        return results


class SingleStagedDirectSolver3D:
    """Scalar facade over the multiple-right-hand-side 3D direct kernel."""

    name = "direct3d"

    def run(self, plan: SingleStagedPlan3D) -> StagedResult3D:
        return SingleStagedDirectBatchSolver3D().run_batch([plan])[0]

    def solve_stage(
        self,
        plan: SingleStagedPlan3D,
        stage_index: int,
        stage_label: str | None = None,
    ) -> SolveResult3D:
        return SingleStagedDirectBatchSolver3D().solve_stage_batch(
            [plan],
            stage_index,
            stage_label,
        )[0]


def solve_single_staged_3d(plan: SingleStagedPlan3D) -> StagedResult3D:
    return SingleStagedDirectSolver3D().run(plan)


__all__ = [
    "SingleStagedDirectBatchSolver3D",
    "SingleStagedDirectSolver3D",
    "solve_single_staged_3d",
]
