"""OpenSees reference backend for the detailed incremental 3D plan.

The domain is rebuilt for each *increment*.  OpenSees therefore solves the
same active tangent system as the self-written backend with zero incremental
support displacements, while Python retains committed displacements, member
forces and reactions between substeps.  Rebuilding avoids confusing total
OpenSees nodal displacements with the stress-free birth state of newly erected
members.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np

from bridgezoo.fem.single_staged.birth3d import initialize_flexible_birth_3d
from bridgezoo.fem.single_staged.kernels3d import frame_axes_3d
from bridgezoo.fem.single_staged.model3d import (
    SingleStagedPlan3D,
    SolveResult3D,
    StagedResult3D,
)


@contextmanager
def _suppress(enabled: bool):
    if not enabled:
        yield
        return
    with open(os.devnull, "w") as devnull:
        stdout, stderr = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout, sys.stderr = stdout, stderr


class SingleStagedOpenSeesSolver3D:
    """Incremental linear OpenSees implementation of ``SingleStagedPlan3D``."""

    name = "opensees3d"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def run(self, plan: SingleStagedPlan3D) -> StagedResult3D:
        return self._run_until(plan, stop_stage=None, record_all=True)

    def solve_stage(
        self,
        plan: SingleStagedPlan3D,
        stage_index: int,
        stage_label: str | None = None,
    ) -> SolveResult3D:
        if not any(stage.index == stage_index for stage in plan.stages):
            raise ValueError(f"unknown 3D construction stage {stage_index}")
        record = self._run_until(
            plan, stop_stage=stage_index, record_all=False
        ).final
        if stage_label is not None:
            record.stage_label = stage_label
        return record

    def _run_until(
        self,
        plan: SingleStagedPlan3D,
        *,
        stop_stage: int | None,
        record_all: bool,
    ) -> StagedResult3D:
        import openseespy.opensees as ops

        model = plan.model
        stages = [
            stage for stage in plan.stages if stop_stage is None or stage.index <= stop_stage
        ]
        if not stages or (stop_stage is not None and stages[-1].index != stop_stage):
            raise ValueError(f"unknown 3D construction stage {stop_stage}")

        displacement: dict[int, np.ndarray] = {}
        cumulative_frame_force: dict[int, np.ndarray] = {}
        cumulative_cable_elastic_force: dict[int, float] = {}
        cable_n0: dict[int, float] = {}
        cumulative_reaction: dict[int, np.ndarray] = {}
        processed_frame_loads: set[int] = set()
        processed_nodal_loads: set[int] = set()
        total_applied = np.zeros(3, dtype=float)
        result = StagedResult3D(backend=self.name)

        for stage in stages:
            stage_index = stage.index
            initialize_flexible_birth_3d(model, stage_index, displacement)

            active_nodes = sorted(displacement)
            active_frames = {
                frame.id: frame
                for frame in model.frames.values()
                if frame.activation_stage <= stage_index
            }
            active_cables = {
                cable.id: cable
                for cable in model.cables.values()
                if cable.activation_stage <= stage_index
            }
            active_links = [
                link
                for link in model.rigid_links.values()
                if link.activation_stage <= stage_index
            ]
            active_supports = [
                support
                for support in model.supports.values()
                if support.activation_stage <= stage_index
            ]
            for frame_id in active_frames:
                cumulative_frame_force.setdefault(frame_id, np.zeros(12, dtype=float))
            for cable_id, cable in active_cables.items():
                if cable_id not in cable_n0:
                    cable_n0[cable_id] = cable.resolved_pretension_a
                    cumulative_cable_elastic_force[cable_id] = 0.0
            for support in active_supports:
                cumulative_reaction.setdefault(support.node, np.zeros(6, dtype=float))

            stage_frame_loads = []
            for load_index, item in enumerate(model.frame_loads):
                if load_index in processed_frame_loads or item.activation_stage > stage_index:
                    continue
                stage_frame_loads.append(item)
                processed_frame_loads.add(load_index)
            stage_nodal_loads = []
            for load_index, item in enumerate(model.nodal_loads):
                if load_index in processed_nodal_loads or item.activation_stage > stage_index:
                    continue
                stage_nodal_loads.append(item)
                processed_nodal_loads.add(load_index)
            cable_increments: dict[int, float] = {}
            for cable_id, cable in active_cables.items():
                increment = 0.0
                if cable.activation_stage == stage_index:
                    increment += cable.resolved_pretension_a
                if cable.second_pretension_stage == stage_index:
                    cable_n0[cable_id] += cable.pretension_b
                    increment += cable.pretension_b
                if increment:
                    cable_increments[cable_id] = increment

            with _suppress(not self.verbose):
                ops.wipe()
                ops.model("basic", "-ndm", 3, "-ndf", 6)
                for node_id in active_nodes:
                    node = model.nodes[node_id]
                    ops.node(node.id, node.x, node.y, node.z)
                for support in active_supports:
                    ops.fix(
                        support.node,
                        *(int(value) for value in support.restraints),
                    )
                for link in active_links:
                    ops.rigidLink("beam", link.master, link.slave)
                for frame in active_frames.values():
                    transform_tag = 100_000 + frame.id
                    ops.geomTransf("Linear", transform_tag, *frame.orientation)
                    section = frame.section
                    ops.element(
                        "elasticBeamColumn",
                        frame.id,
                        frame.i,
                        frame.j,
                        section.A,
                        frame.material.E,
                        frame.material.G,
                        section.J,
                        section.Iy,
                        section.Iz,
                        transform_tag,
                    )
                for material_index, cable in enumerate(
                    active_cables.values(), start=1
                ):
                    material_tag = 5_000_000 + material_index
                    ops.uniaxialMaterial("Elastic", material_tag, cable.material.E)
                    ops.element(
                        "Truss",
                        cable.id,
                        cable.i,
                        cable.j,
                        cable.area,
                        material_tag,
                    )

                has_load = bool(
                    stage_frame_loads or stage_nodal_loads or cable_increments
                )
                if has_load:
                    ops.timeSeries("Constant", 1)
                    ops.pattern("Plain", 1, 1)
                    for item in stage_nodal_loads:
                        ops.load(item.node, *item.values)
                        total_applied += np.asarray(item.values[:3], dtype=float)
                    for item in stage_frame_loads:
                        frame = active_frames[item.member]
                        node_i, node_j = model.nodes[frame.i], model.nodes[frame.j]
                        length, rotation = frame_axes_3d(
                            node_i.xyz, node_j.xyz, frame.orientation
                        )
                        global_q = np.asarray(item.global_vector, dtype=float)
                        local_q = rotation @ global_q
                        if np.any(local_q):
                            ops.eleLoad(
                                "-ele",
                                frame.id,
                                "-type",
                                "-beamUniform",
                                float(local_q[1]),
                                float(local_q[2]),
                                float(local_q[0]),
                            )
                        total_applied += global_q * length
                    for cable_id, increment in cable_increments.items():
                        cable = active_cables[cable_id]
                        vector = np.asarray(model.nodes[cable.j].xyz) - np.asarray(
                            model.nodes[cable.i].xyz
                        )
                        axis = vector / np.linalg.norm(vector)
                        ops.load(
                            cable.i,
                            float(axis[0] * increment),
                            float(axis[1] * increment),
                            float(axis[2] * increment),
                            0.0,
                            0.0,
                            0.0,
                        )
                        ops.load(
                            cable.j,
                            float(-axis[0] * increment),
                            float(-axis[1] * increment),
                            float(-axis[2] * increment),
                            0.0,
                            0.0,
                            0.0,
                        )

                ops.system("BandGeneral")
                ops.numberer("RCM")
                ops.constraints("Transformation")
                ops.integrator("LoadControl", 1.0)
                ops.algorithm("Linear")
                ops.analysis("Static")
                ok = ops.analyze(1)

                for node_id in active_nodes:
                    displacement[node_id] += np.asarray(
                        ops.nodeDisp(node_id), dtype=float
                    )
                for frame_id in active_frames:
                    response = ops.eleResponse(frame_id, "localForce")
                    cumulative_frame_force[frame_id] += np.asarray(
                        response, dtype=float
                    )
                for cable_id in active_cables:
                    cumulative_cable_elastic_force[cable_id] += self._truss_axial_force(
                        ops, cable_id
                    )
                ops.reactions()
                for support in active_supports:
                    cumulative_reaction[support.node] += np.asarray(
                        ops.nodeReaction(support.node), dtype=float
                    )

            record = SolveResult3D(
                backend=self.name,
                stage_index=stage_index,
                stage_label=stage.label,
                converged=(ok == 0),
                applied_load=tuple(float(value) for value in total_applied),
            )
            record.displacement = {
                node_id: tuple(float(value) for value in displacement[node_id])
                for node_id in active_nodes
            }
            record.frame_force = {
                frame_id: tuple(float(value) for value in values)
                for frame_id, values in cumulative_frame_force.items()
            }
            for cable_id, cable in active_cables.items():
                force = cable_n0[cable_id] + cumulative_cable_elastic_force[cable_id]
                record.cable_force[cable_id] = float(force)
                record.cable_stress[cable_id] = float(force / cable.area)
            record.support_reaction = {
                node_id: tuple(float(value) for value in values)
                for node_id, values in cumulative_reaction.items()
            }
            if record_all or stage_index == stages[-1].index:
                result.records.append(record)

        return result

    @staticmethod
    def _truss_axial_force(ops, tag: int) -> float:
        for response_name in ("axialForce", "basicForce", "force"):
            try:
                response = ops.eleResponse(tag, response_name)
            except Exception:
                response = None
            if response:
                return (
                    float(response[0])
                    if isinstance(response, (list, tuple))
                    else float(response)
                )
        return 0.0


__all__ = ["SingleStagedOpenSeesSolver3D"]
