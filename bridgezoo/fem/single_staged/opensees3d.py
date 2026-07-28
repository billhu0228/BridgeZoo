"""OpenSees 3D reference backend for the new single-staged IR."""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

import numpy as np

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
    """Linear OpenSees reference implementation consuming ``SingleStagedPlan3D``."""

    name = "opensees3d"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

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
        import openseespy.opensees as ops

        model = plan.model
        nodes = sorted(
            (node for node in model.nodes.values() if node.activation_stage <= stage_index),
            key=lambda node: node.id,
        )
        node_ids = {node.id for node in nodes}
        frames = {
            frame.id: frame
            for frame in model.frames.values()
            if frame.activation_stage <= stage_index
        }
        cables = {
            cable.id: cable
            for cable in model.cables.values()
            if cable.activation_stage <= stage_index
        }
        links = [link for link in model.rigid_links.values() if link.activation_stage <= stage_index]
        supports = {
            support.node: support
            for support in model.supports.values()
            if support.activation_stage <= stage_index and support.node in node_ids
        }
        frame_loads = [
            item
            for item in model.frame_loads
            if item.activation_stage <= stage_index and item.member in frames
        ]
        total_applied = np.zeros(3, dtype=float)

        with _suppress(not self.verbose):
            ops.wipe()
            ops.model("basic", "-ndm", 3, "-ndf", 6)
            for node in nodes:
                ops.node(node.id, node.x, node.y, node.z)

            for node in nodes:
                support = supports.get(node.id)
                restraints = support.restraints if support is not None else (False,) * 6
                if any(restraints):
                    ops.fix(node.id, *(int(value) for value in restraints))

            for link in links:
                ops.rigidLink("beam", link.master, link.slave)

            for frame in frames.values():
                transform_tag = 100000 + frame.id
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

            for material_index, cable in enumerate(cables.values(), start=1):
                # Material tags use their own compact namespace.  Deriving
                # them from large grouped element ids can make one cable's
                # InitStress tag collide with another cable's Elastic tag.
                elastic_tag = 5_000_000 + 2 * material_index
                initial_stress_tag = elastic_tag + 1
                ops.uniaxialMaterial("Elastic", elastic_tag, cable.material.E)
                ops.uniaxialMaterial(
                    "InitStressMaterial",
                    initial_stress_tag,
                    elastic_tag,
                    cable.pretension / cable.area,
                )
                ops.element("Truss", cable.id, cable.i, cable.j, cable.area, initial_stress_tag)

            ops.timeSeries("Constant", 1)
            ops.pattern("Plain", 1, 1)
            for item in model.nodal_loads:
                if item.activation_stage <= stage_index and item.node in node_ids:
                    ops.load(item.node, *item.values)
                    total_applied += np.asarray(item.values[:3], dtype=float)
            for item in frame_loads:
                frame = frames[item.member]
                node_i, node_j = model.nodes[frame.i], model.nodes[frame.j]
                length, rotation = frame_axes_3d(node_i.xyz, node_j.xyz, frame.orientation)
                global_q = np.asarray(item.global_vector, dtype=float)
                local_q = rotation @ global_q
                if np.any(local_q):
                    # OpenSees 3D order is Wy, Wz, optional Wx.
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

            ops.system("BandGeneral")
            ops.numberer("RCM")
            ops.constraints("Transformation")
            ops.integrator("LoadControl", 1.0)
            ops.algorithm("Linear")
            ops.analysis("Static")
            ok = ops.analyze(1)

            label = stage_label or next(
                (stage.label for stage in plan.stages if stage.index == stage_index),
                f"stage{stage_index}",
            )
            result = SolveResult3D(
                backend=self.name,
                stage_index=stage_index,
                stage_label=label,
                converged=(ok == 0),
                applied_load=tuple(float(value) for value in total_applied),
            )
            for node in nodes:
                result.displacement[node.id] = tuple(
                    float(value) for value in ops.nodeDisp(node.id)
                )
            for frame in frames.values():
                response = ops.eleResponse(frame.id, "localForce")
                result.frame_force[frame.id] = tuple(float(value) for value in response)
            for cable in cables.values():
                force = self._truss_axial_force(ops, cable.id)
                result.cable_force[cable.id] = force
                result.cable_stress[cable.id] = force / cable.area

            ops.reactions()
            for support in supports.values():
                reaction = ops.nodeReaction(support.node)
                result.support_reaction[support.node] = tuple(float(value) for value in reaction)
            ops.wipe()
            return result

    @staticmethod
    def _truss_axial_force(ops, tag: int) -> float:
        for response_name in ("axialForce", "basicForce", "force"):
            try:
                response = ops.eleResponse(tag, response_name)
            except Exception:
                response = None
            if response:
                return float(response[0]) if isinstance(response, (list, tuple)) else float(response)
        return 0.0


__all__ = ["SingleStagedOpenSeesSolver3D"]
