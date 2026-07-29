"""Solver-neutral 3D IR for the single-tower staged bridge.

The coordinate convention is ``x`` longitudinal, ``y`` transverse and ``z``
vertical.  Every structural node has six degrees of freedom ordered as
``(ux, uy, uz, rx, ry, rz)``.  Both 3D solvers consume this module directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from bridgezoo.fem.single_staged.sections3d import ElasticMaterial3D, FrameSection3D


Vector3 = tuple[float, float, float]
Vector6 = tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class Node3D:
    id: int
    x: float
    y: float
    z: float
    role: str
    activation_stage: int = 0
    birth_master: int | None = None

    @property
    def xyz(self) -> Vector3:
        return (self.x, self.y, self.z)


@dataclass(frozen=True)
class FrameElement3D:
    """Two-node Euler-Bernoulli space-frame element."""

    id: int
    i: int
    j: int
    material: ElasticMaterial3D
    section: FrameSection3D
    orientation: Vector3 = (0.0, 0.0, 1.0)
    group: str = "frame"
    activation_stage: int = 0


@dataclass(frozen=True)
class CableElement3D:
    """Linear 3D truss cable with an installation pretension."""

    id: int
    i: int
    j: int
    material: ElasticMaterial3D
    area: float
    pretension: float = 0.0
    group: str = "stay"
    activation_stage: int = 0
    pretension_a: float | None = None
    second_pretension_stage: int | None = None
    construction_stage: int = 0

    def __post_init__(self) -> None:
        if self.area <= 0.0:
            raise ValueError("cable area must be positive")
        if self.pretension < 0.0:
            raise ValueError("cable pretension must be nonnegative")
        if self.pretension_a is not None and not 0.0 <= self.pretension_a <= self.pretension:
            raise ValueError("cable pretension A must be between zero and total pretension")
        if (
            self.second_pretension_stage is not None
            and self.second_pretension_stage < self.activation_stage
        ):
            raise ValueError("cable pretension B cannot precede cable activation")

    @property
    def resolved_pretension_a(self) -> float:
        return self.pretension if self.pretension_a is None else self.pretension_a

    @property
    def pretension_b(self) -> float:
        return self.pretension - self.resolved_pretension_a


@dataclass(frozen=True)
class RigidLink3D:
    """Six-DOF rigid beam link from ``master`` to ``slave``."""

    id: int
    master: int
    slave: int
    activation_stage: int = 0


@dataclass(frozen=True)
class Support3D:
    node: int
    ux: bool = False
    uy: bool = False
    uz: bool = False
    rx: bool = False
    ry: bool = False
    rz: bool = False
    activation_stage: int = 0

    @property
    def restraints(self) -> tuple[bool, bool, bool, bool, bool, bool]:
        return (self.ux, self.uy, self.uz, self.rx, self.ry, self.rz)


@dataclass(frozen=True)
class NodalLoad3D:
    node: int
    values: Vector6
    load_case: str = "dead"
    activation_stage: int = 0


@dataclass(frozen=True)
class FrameLoad3D:
    """Uniform load per unit member length expressed in global coordinates."""

    member: int
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    load_case: str = "self_weight"
    activation_stage: int = 0
    deactivation_stage: int | None = None

    @property
    def global_vector(self) -> Vector3:
        return (self.qx, self.qy, self.qz)

    def is_defined_at(self, stage: int) -> bool:
        return self.activation_stage <= stage and (
            self.deactivation_stage is None or stage < self.deactivation_stage
        )


@dataclass(frozen=True)
class ConstructionStage3D:
    index: int
    label: str
    description: str = ""
    construction_stage: int = 0
    phase: str = ""


class BridgeModel3D:
    """Complete topology plus per-object activation stages."""

    def __init__(self, name: str):
        self.name = name
        self.nodes: dict[int, Node3D] = {}
        self.frames: dict[int, FrameElement3D] = {}
        self.cables: dict[int, CableElement3D] = {}
        self.rigid_links: dict[int, RigidLink3D] = {}
        self.supports: dict[int, Support3D] = {}
        self.nodal_loads: list[NodalLoad3D] = []
        self.frame_loads: list[FrameLoad3D] = []

    def add_node(self, node: Node3D) -> None:
        self._add_unique(self.nodes, node.id, node, "node")

    def add_frame(self, frame: FrameElement3D) -> None:
        self._require_nodes(frame.i, frame.j)
        self._add_unique(self.frames, frame.id, frame, "frame")

    def add_cable(self, cable: CableElement3D) -> None:
        self._require_nodes(cable.i, cable.j)
        self._add_unique(self.cables, cable.id, cable, "cable")

    def add_rigid_link(self, link: RigidLink3D) -> None:
        self._require_nodes(link.master, link.slave)
        if link.master == link.slave:
            raise ValueError("rigid-link master and slave must differ")
        if any(existing.slave == link.slave for existing in self.rigid_links.values()):
            raise ValueError(f"node {link.slave} is already a rigid-link slave")
        self._add_unique(self.rigid_links, link.id, link, "rigid link")

    def add_support(self, support: Support3D) -> None:
        self._require_nodes(support.node)
        self._add_unique(self.supports, support.node, support, "support")

    def add_nodal_load(self, load: NodalLoad3D) -> None:
        self._require_nodes(load.node)
        self.nodal_loads.append(load)

    def add_frame_load(self, load: FrameLoad3D) -> None:
        if load.member not in self.frames:
            raise KeyError(f"unknown loaded frame {load.member}")
        self.frame_loads.append(load)

    def active_node_ids(self, stage: int) -> set[int]:
        """Nodes participating in the cumulative model at ``stage``."""

        return {node.id for node in self.nodes.values() if node.activation_stage <= stage}

    def validate(self) -> None:
        """Validate activation ordering and rigid-link topology."""

        slave_ids = set()
        for node in self.nodes.values():
            if node.birth_master is not None:
                if node.birth_master not in self.nodes:
                    raise ValueError(
                        f"node {node.id} has unknown birth master {node.birth_master}"
                    )
                if self.nodes[node.birth_master].activation_stage > node.activation_stage:
                    raise ValueError(
                        f"node {node.id} activates before birth master {node.birth_master}"
                    )
        for collection in (self.frames.values(), self.cables.values()):
            for element in collection:
                for node_id in (element.i, element.j):
                    if self.nodes[node_id].activation_stage > element.activation_stage:
                        raise ValueError(
                            f"element {element.id} activates before node {node_id}"
                        )
        for link in self.rigid_links.values():
            if link.slave in slave_ids:
                raise ValueError(f"node {link.slave} has multiple rigid masters")
            slave_ids.add(link.slave)
            node_stage = max(
                self.nodes[link.master].activation_stage,
                self.nodes[link.slave].activation_stage,
            )
            if node_stage > link.activation_stage:
                raise ValueError(f"rigid link {link.id} activates before one of its nodes")
        for support in self.supports.values():
            if self.nodes[support.node].activation_stage > support.activation_stage:
                raise ValueError(f"support at node {support.node} activates before its node")
        for load in self.frame_loads:
            if self.frames[load.member].activation_stage > load.activation_stage:
                raise ValueError(f"load on frame {load.member} activates before the frame")
            if (
                load.deactivation_stage is not None
                and load.deactivation_stage <= load.activation_stage
            ):
                raise ValueError(
                    f"load on frame {load.member} must deactivate after it activates"
                )

    @staticmethod
    def _add_unique(mapping: dict, key: int, value, kind: str) -> None:
        if key in mapping:
            raise ValueError(f"duplicate {kind} id {key}")
        mapping[key] = value

    def _require_nodes(self, *node_ids: int) -> None:
        missing = [node_id for node_id in node_ids if node_id not in self.nodes]
        if missing:
            raise KeyError(f"unknown node ids: {missing}")

    def summary(self) -> str:
        return (
            f"<BridgeModel3D '{self.name}': {len(self.nodes)} nodes, "
            f"{len(self.frames)} frames, {len(self.cables)} cables, "
            f"{len(self.rigid_links)} rigid links, {len(self.supports)} supports>"
        )


@dataclass
class SingleStagedPlan3D:
    model: BridgeModel3D
    stages: list[ConstructionStage3D]
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        indices = [stage.index for stage in self.stages]
        if not indices or indices != sorted(set(indices)):
            raise ValueError("construction stage indices must be nonempty, unique and sorted")
        self.model.validate()

    @property
    def final_stage(self) -> ConstructionStage3D:
        return self.stages[-1]


@dataclass
class SolveResult3D:
    backend: str
    stage_index: int
    stage_label: str
    converged: bool = True
    displacement: dict[int, Vector6] = field(default_factory=dict)
    # Stress-free displacement assigned when nodes enter this stage.  This is
    # diagnostic state used to measure erection movement relative to the
    # inherited tangent, not an additional structural degree of freedom.
    birth_displacement: dict[int, Vector6] = field(default_factory=dict)
    frame_force: dict[int, tuple[float, ...]] = field(default_factory=dict)
    cable_force: dict[int, float] = field(default_factory=dict)
    cable_stress: dict[int, float] = field(default_factory=dict)
    support_reaction: dict[int, Vector6] = field(default_factory=dict)
    applied_load: Vector3 = (0.0, 0.0, 0.0)

    def uz(self, node_id: int) -> float:
        return self.displacement[node_id][2]


@dataclass
class StagedResult3D:
    backend: str
    records: list[SolveResult3D] = field(default_factory=list)

    @property
    def final(self) -> SolveResult3D:
        if not self.records:
            raise RuntimeError("3D staged result has no records")
        return self.records[-1]


__all__ = [
    "Vector3",
    "Vector6",
    "Node3D",
    "FrameElement3D",
    "CableElement3D",
    "RigidLink3D",
    "Support3D",
    "NodalLoad3D",
    "FrameLoad3D",
    "ConstructionStage3D",
    "BridgeModel3D",
    "SingleStagedPlan3D",
    "SolveResult3D",
    "StagedResult3D",
]
