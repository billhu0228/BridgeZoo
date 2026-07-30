"""Path-dependent linear 3D solver for the detailed construction plan.

Every construction substep is an incremental equilibrium solve.  Newly born
beam nodes inherit their stage-by-stage virtual tangent history as their
stress-free reference, while previously accumulated member force and real
displacement remain in the state.  This is what lets wet-deck weight be applied
to steelwork in substep 2 and retained after the slab joins the composite
system in substep 3.
"""

from __future__ import annotations

import numpy as np

from bridgezoo.fem.single_staged.birth3d import TangentDisplacementHistory3D
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
    """Cable fields affecting topology/stiffness, excluding A/B force values."""

    return (
        cable.id,
        cable.i,
        cable.j,
        cable.material,
        cable.area,
        cable.group,
        cable.activation_stage,
        cable.second_pretension_stage,
        cable.construction_stage,
    )


def _assert_same_structure_3d(reference: SingleStagedPlan3D, case: SingleStagedPlan3D) -> None:
    """Require batched plans to differ only in cable A/B pretension values."""

    if reference.stages != case.stages:
        raise ValueError("batched 3D plans differ in construction stages")
    if (
        reference.metadata["config"].flexible_birth_correction_factor
        != case.metadata["config"].flexible_birth_correction_factor
    ):
        raise ValueError("batched 3D plans differ in flexible birth correction")
    reference_model = reference.model
    case_model = case.model
    for name in ("nodes", "frames", "rigid_links", "supports"):
        if getattr(reference_model, name) != getattr(case_model, name):
            raise ValueError(f"batched 3D plans differ in {name}")
    if set(reference_model.cables) != set(case_model.cables):
        raise ValueError("batched 3D plans differ in cable ids")
    for cable_id, reference_cable in reference_model.cables.items():
        if _cable_structure(reference_cable) != _cable_structure(case_model.cables[cable_id]):
            raise ValueError(f"batched 3D plans differ in cable structure at {cable_id}")
    if reference_model.nodal_loads != case_model.nodal_loads:
        raise ValueError("batched 3D plans differ in nodal loads")
    if reference_model.frame_loads != case_model.frame_loads:
        raise ValueError("batched 3D plans differ in frame loads")


class SingleStagedDirectBatchSolver3D:
    """Incremental 3D solver sharing each stage factorization across cases."""

    name = "direct3d"

    def run_batch(self, plans: list[SingleStagedPlan3D]) -> list[StagedResult3D]:
        return self._run_until(plans, stop_stage=None, record_all=True)

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
        stage = next(
            (item for item in plans[0].stages if item.index == stage_index),
            None,
        )
        if stage is None:
            raise ValueError(f"unknown 3D construction stage {stage_index}")
        results = self._run_until(
            plans,
            stop_stage=stage_index,
            record_all=False,
            structure_checked=_structure_checked,
        )
        records = [result.final for result in results]
        if stage_label is not None:
            for record in records:
                record.stage_label = stage_label
        return records

    def _run_until(
        self,
        plans: list[SingleStagedPlan3D],
        *,
        stop_stage: int | None,
        record_all: bool,
        structure_checked: bool = False,
    ) -> list[StagedResult3D]:
        if not plans:
            raise ValueError("run_batch requires at least one 3D plan")
        reference = plans[0]
        if not structure_checked:
            for plan in plans[1:]:
                _assert_same_structure_3d(reference, plan)

        model = reference.model
        ncase = len(plans)
        displacement: dict[int, np.ndarray] = {}
        frame_birth: dict[int, np.ndarray] = {}
        frame_fixed_end: dict[int, np.ndarray] = {}
        cable_birth_i: dict[int, np.ndarray] = {}
        cable_birth_j: dict[int, np.ndarray] = {}
        cable_n0: dict[int, np.ndarray] = {}
        cumulative_reaction: dict[int, np.ndarray] = {}
        processed_frame_loads: set[int] = set()
        processed_nodal_loads: set[int] = set()
        total_applied = np.zeros(3, dtype=float)
        results = [StagedResult3D(backend=self.name) for _ in plans]
        tangent_history = TangentDisplacementHistory3D(
            model,
            ncase=ncase,
            batched=True,
            correction_factor=reference.metadata[
                "config"
            ].flexible_birth_correction_factor,
        )

        stages = [
            stage
            for stage in reference.stages
            if stop_stage is None or stage.index <= stop_stage
        ]
        if not stages or (stop_stage is not None and stages[-1].index != stop_stage):
            raise ValueError(f"unknown 3D construction stage {stop_stage}")

        for stage in stages:
            stage_index = stage.index
            born_this_stage = tangent_history.activate(
                stage_index,
                displacement,
            )

            active_nodes = sorted(displacement)
            node_ids = set(active_nodes)
            node_position = {node_id: index for index, node_id in enumerate(active_nodes)}

            def dofs(node_id: int) -> list[int]:
                base = 6 * node_position[node_id]
                return list(range(base, base + 6))

            for frame in model.frames.values():
                if frame.activation_stage <= stage_index and frame.id not in frame_birth:
                    frame_birth[frame.id] = np.vstack(
                        (displacement[frame.i].copy(), displacement[frame.j].copy())
                    ).reshape(12, ncase)
                    frame_fixed_end[frame.id] = np.zeros(12, dtype=float)
            for cable in model.cables.values():
                if cable.activation_stage <= stage_index and cable.id not in cable_n0:
                    cable_birth_i[cable.id] = displacement[cable.i][:3].copy()
                    cable_birth_j[cable.id] = displacement[cable.j][:3].copy()
                    cable_n0[cable.id] = np.asarray(
                        [
                            plan.model.cables[cable.id].resolved_pretension_a
                            for plan in plans
                        ],
                        dtype=float,
                    )

            active_frames = {
                frame_id: model.frames[frame_id] for frame_id in frame_birth
            }
            active_cables = {
                cable_id: model.cables[cable_id] for cable_id in cable_n0
            }
            active_links = [
                link
                for link in model.rigid_links.values()
                if link.activation_stage <= stage_index
            ]
            active_supports = [
                support
                for support in model.supports.values()
                if support.activation_stage <= stage_index and support.node in node_ids
            ]

            ndof = 6 * len(active_nodes)
            stiffness = np.zeros((ndof, ndof), dtype=float)
            incremental_load = np.zeros((ndof, ncase), dtype=float)
            frame_data: dict[int, tuple[np.ndarray, np.ndarray, list[int]]] = {}
            cable_data: dict[int, tuple[float, np.ndarray, list[int]]] = {}

            for element in active_frames.values():
                node_i, node_j = model.nodes[element.i], model.nodes[element.j]
                length, rotation = frame_axes_3d(
                    node_i.xyz, node_j.xyz, element.orientation
                )
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
                element_dofs = dofs(element.i) + dofs(element.j)
                stiffness[np.ix_(element_dofs, element_dofs)] += (
                    transform.T @ local_stiffness @ transform
                )
                frame_data[element.id] = (local_stiffness, transform, element_dofs)

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

            # A load object is consumed once as an increment.  A temporary
            # deck load expires from the plan at the composite substep, but
            # its committed displacement/internal-force effect intentionally
            # remains in the state rather than being unloaded.
            for load_index, item in enumerate(model.frame_loads):
                if load_index in processed_frame_loads or item.activation_stage > stage_index:
                    continue
                if item.member not in active_frames:
                    raise ValueError(f"3D frame load {load_index} precedes member activation")
                frame = active_frames[item.member]
                node_i, node_j = model.nodes[frame.i], model.nodes[frame.j]
                length, rotation = frame_axes_3d(
                    node_i.xyz, node_j.xyz, frame.orientation
                )
                transform = frame_transform_3d(rotation)
                global_q = np.asarray(item.global_vector, dtype=float)
                equivalent_local = uniform_load_local_3d(rotation @ global_q, length)
                element_dofs = dofs(frame.i) + dofs(frame.j)
                incremental_load[element_dofs, :] += (
                    transform.T @ equivalent_local
                )[:, None]
                frame_fixed_end[frame.id] += equivalent_local
                total_applied += global_q * length
                processed_frame_loads.add(load_index)

            for load_index, item in enumerate(model.nodal_loads):
                if load_index in processed_nodal_loads or item.activation_stage > stage_index:
                    continue
                if item.node not in node_ids:
                    raise ValueError(f"3D nodal load {load_index} precedes node activation")
                values = np.asarray(item.values, dtype=float)
                incremental_load[dofs(item.node), :] += values[:, None]
                total_applied += values[:3]
                processed_nodal_loads.add(load_index)

            for cable_id, (_, axis, _) in cable_data.items():
                reference_cable = active_cables[cable_id]
                increments = np.zeros(ncase, dtype=float)
                if reference_cable.activation_stage == stage_index:
                    increments += cable_n0[cable_id]
                if reference_cable.second_pretension_stage == stage_index:
                    b_values = np.asarray(
                        [plan.model.cables[cable_id].pretension_b for plan in plans],
                        dtype=float,
                    )
                    cable_n0[cable_id] += b_values
                    increments += b_values
                incremental_load[dofs(reference_cable.i)[:3], :] += (
                    axis[:, None] * increments[None, :]
                )
                incremental_load[dofs(reference_cable.j)[:3], :] -= (
                    axis[:, None] * increments[None, :]
                )

            slave_ids = {link.slave for link in active_links}
            independent_full_dofs = [
                full_dof
                for node_id in active_nodes
                if node_id not in slave_ids
                for full_dof in dofs(node_id)
            ]
            reduced_column = {
                full_dof: column
                for column, full_dof in enumerate(independent_full_dofs)
            }
            constraint_transform = np.zeros(
                (ndof, len(independent_full_dofs)), dtype=float
            )
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
            reduced_load = constraint_transform.T @ incremental_load
            fixed = np.zeros(len(independent_full_dofs), dtype=bool)
            for support in active_supports:
                if support.node in slave_ids:
                    raise ValueError("supports on rigid-link slave nodes are not supported")
                for component, restrained in enumerate(support.restraints):
                    if restrained:
                        fixed[reduced_column[dofs(support.node)[component]]] = True
                cumulative_reaction.setdefault(
                    support.node, np.zeros((6, ncase), dtype=float)
                )
            for reduced_dof in range(len(independent_full_dofs)):
                if not fixed[reduced_dof] and not np.any(reduced_stiffness[reduced_dof, :]):
                    fixed[reduced_dof] = True

            free = np.flatnonzero(~fixed)
            reduced_increment = np.zeros(
                (len(independent_full_dofs), ncase), dtype=float
            )
            converged = True
            if free.size:
                from scipy.linalg import cho_factor, cho_solve

                free_stiffness = reduced_stiffness[np.ix_(free, free)]
                free_load = reduced_load[free, :]
                try:
                    dof_scale = 1.0 / np.sqrt(np.diag(free_stiffness))
                    equilibrated_stiffness = (
                        dof_scale[:, None] * free_stiffness * dof_scale[None, :]
                    )
                    factor = cho_factor(
                        equilibrated_stiffness,
                        lower=True,
                        check_finite=False,
                    )
                    equilibrated_increment = cho_solve(
                        factor,
                        dof_scale[:, None] * free_load,
                        check_finite=False,
                    )
                    reduced_increment[free, :] = (
                        dof_scale[:, None] * equilibrated_increment
                    )
                except np.linalg.LinAlgError:
                    converged = False
            full_increment = constraint_transform @ reduced_increment
            reduced_reaction = (
                reduced_stiffness @ reduced_increment - reduced_load
            )
            stage_increment = {
                node_id: full_increment[dofs(node_id), :].copy()
                for node_id in active_nodes
            }
            for node_id in active_nodes:
                displacement[node_id] += stage_increment[node_id]
            tangent_history.accumulate(stage_index, stage_increment)
            for support in active_supports:
                for component, restrained in enumerate(support.restraints):
                    if restrained:
                        column = reduced_column[dofs(support.node)[component]]
                        cumulative_reaction[support.node][component] += reduced_reaction[column]

            stage_records: list[SolveResult3D] = []
            for case_index, plan in enumerate(plans):
                record = SolveResult3D(
                    backend=self.name,
                    stage_index=stage_index,
                    stage_label=stage.label,
                    converged=converged,
                    applied_load=tuple(float(value) for value in total_applied),
                )
                record.birth_displacement = {
                    node_id: tuple(
                        float(value)
                        for value in birth[:, case_index]
                    )
                    for node_id, birth in born_this_stage.items()
                }
                for node_id in active_nodes:
                    record.displacement[node_id] = tuple(
                        float(value) for value in displacement[node_id][:, case_index]
                    )
                for frame_id, (local_stiffness, transform, element_dofs) in frame_data.items():
                    current = np.concatenate(
                        (
                            displacement[active_frames[frame_id].i][:, case_index],
                            displacement[active_frames[frame_id].j][:, case_index],
                        )
                    )
                    relative = current - frame_birth[frame_id][:, case_index]
                    local_force = (
                        local_stiffness @ (transform @ relative)
                        - frame_fixed_end[frame_id]
                    )
                    record.frame_force[frame_id] = tuple(
                        float(value) for value in local_force
                    )
                for cable_id, (length, axis, _) in cable_data.items():
                    cable = plan.model.cables[cable_id]
                    elongation = float(
                        axis
                        @ (
                            displacement[cable.j][:3, case_index]
                            - cable_birth_j[cable_id][:, case_index]
                            - displacement[cable.i][:3, case_index]
                            + cable_birth_i[cable_id][:, case_index]
                        )
                    )
                    force = (
                        cable_n0[cable_id][case_index]
                        + cable.material.E * cable.area / length * elongation
                    )
                    record.cable_force[cable_id] = float(force)
                    record.cable_stress[cable_id] = float(force / cable.area)
                for support in active_supports:
                    record.support_reaction[support.node] = tuple(
                        float(value)
                        for value in cumulative_reaction[support.node][:, case_index]
                    )
                stage_records.append(record)

            if record_all or stage_index == stages[-1].index:
                for result, record in zip(results, stage_records):
                    result.records.append(record)

        return results


class SingleStagedDirectSolver3D:
    """Scalar facade over the multiple-right-hand-side incremental kernel."""

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
            [plan], stage_index, stage_label
        )[0]


def solve_single_staged_3d(plan: SingleStagedPlan3D) -> StagedResult3D:
    return SingleStagedDirectSolver3D().run(plan)


__all__ = [
    "SingleStagedDirectBatchSolver3D",
    "SingleStagedDirectSolver3D",
    "solve_single_staged_3d",
]
