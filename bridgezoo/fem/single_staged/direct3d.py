"""Self-written linear 3D solver for :mod:`single_staged.model3d`.

This first-round backend performs a cumulative linear re-analysis for every
activation stage.  It includes 3D Euler-Bernoulli frames, linear truss cables,
physical distributed self-weight and exact rigid-link kinematic condensation.
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


class SingleStagedDirectSolver3D:
    """Dense direct-stiffness backend for the first 3D model milestone."""

    name = "direct3d"

    def run(self, plan: SingleStagedPlan3D) -> StagedResult3D:
        result = StagedResult3D(backend=self.name)
        for stage in plan.stages:
            result.records.append(self.solve_stage(plan, stage.index, stage.label))
        return result

    def solve_stage(
        self,
        plan: SingleStagedPlan3D,
        stage_index: int,
        stage_label: str | None = None,
    ) -> SolveResult3D:
        model = plan.model
        nodes = sorted(
            (node for node in model.nodes.values() if node.activation_stage <= stage_index),
            key=lambda node: node.id,
        )
        node_ids = {node.id for node in nodes}
        index = {node.id: position for position, node in enumerate(nodes)}
        ndof = 6 * len(nodes)
        stiffness = np.zeros((ndof, ndof), dtype=float)
        load = np.zeros(ndof, dtype=float)

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
                load[element_dofs] += transform.T @ equivalent_local
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
            if element.pretension:
                load[dofs(element.i)[:3]] += element.pretension * axis
                load[dofs(element.j)[:3]] -= element.pretension * axis
            cable_data[element.id] = (length, axis, element_dofs)

        for item in model.nodal_loads:
            if item.activation_stage <= stage_index and item.node in node_ids:
                load[dofs(item.node)] += np.asarray(item.values, dtype=float)
                total_applied += np.asarray(item.values[:3], dtype=float)

        # Exact rigid-link transformation: u_slave = u_master + theta x r,
        # theta_slave = theta_master.  Slaves are removed from the independent
        # coordinate set, avoiding penalty stiffness and its conditioning cost.
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
                raise ValueError("chained 3D rigid links are not supported in the first-round solver")
            master_dofs = dofs(link.master)
            slave_dofs = dofs(link.slave)
            offset = np.asarray(model.nodes[link.slave].xyz) - np.asarray(model.nodes[link.master].xyz)
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

        # Truss-only nodes have unused rotational DOFs.  Constrain any exactly
        # zero reduced row automatically, matching OpenSees' explicit fixities.
        for reduced_dof in range(len(independent_full_dofs)):
            if not fixed[reduced_dof] and not np.any(reduced_stiffness[reduced_dof, :]):
                fixed[reduced_dof] = True

        free = np.flatnonzero(~fixed)
        reduced_displacement = np.zeros(len(independent_full_dofs), dtype=float)
        converged = True
        if free.size:
            free_stiffness = reduced_stiffness[np.ix_(free, free)]
            free_load = reduced_load[free]
            try:
                reduced_displacement[free] = np.linalg.solve(free_stiffness, free_load)
            except np.linalg.LinAlgError:
                converged = False
        displacement = constraint_transform @ reduced_displacement

        label = stage_label or next(
            (stage.label for stage in plan.stages if stage.index == stage_index),
            f"stage{stage_index}",
        )
        result = SolveResult3D(
            backend=self.name,
            stage_index=stage_index,
            stage_label=label,
            converged=converged,
            applied_load=tuple(float(value) for value in total_applied),
        )
        for node in nodes:
            result.displacement[node.id] = tuple(
                float(value) for value in displacement[dofs(node.id)]
            )

        for element_id, (local_stiffness, transform, element_dofs, fixed_end) in frame_data.items():
            local_displacement = transform @ displacement[element_dofs]
            local_force = local_stiffness @ local_displacement - fixed_end
            result.frame_force[element_id] = tuple(float(value) for value in local_force)

        for element_id, (length, axis, element_dofs) in cable_data.items():
            element = active_cables[element_id]
            translational_displacement = displacement[element_dofs]
            elongation = float(axis @ (translational_displacement[3:] - translational_displacement[:3]))
            force = element.pretension + element.material.E * element.area / length * elongation
            result.cable_force[element_id] = float(force)
            result.cable_stress[element_id] = float(force / element.area)

        reduced_reaction = reduced_stiffness @ reduced_displacement - reduced_load
        for support in active_supports:
            values = []
            for component in range(6):
                column = reduced_column[dofs(support.node)[component]]
                values.append(float(reduced_reaction[column]) if fixed[column] else 0.0)
            result.support_reaction[support.node] = tuple(values)
        return result


def solve_single_staged_3d(plan: SingleStagedPlan3D) -> StagedResult3D:
    return SingleStagedDirectSolver3D().run(plan)


__all__ = ["SingleStagedDirectSolver3D", "solve_single_staged_3d"]
