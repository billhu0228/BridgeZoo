"""与求解器无关的二维结构模型定义（IR）。

设计目标：**一套结构定义，两种求解后端，结果一致**。

- 本模块只描述"结构是什么"（节点、梁单元、索单元、约束、荷载），不含任何求解器细节。
- :mod:`bridgezoo.fem.completed.direct`（自研直接刚度法）与
  :mod:`bridgezoo.fem.completed.opensees`（OpenSees）都消费同一个 :class:`StructuralModel`，
  返回同一种 :class:`SolveResult`，从而可交叉校核自研求解器的正确性。

约定
----
- 二维，每节点 3 自由度：``(ux, uy, rz)``。
- 单位自洽即可，推荐 SI（N, m, Pa）。
- 梁单元：Euler-Bernoulli 框架（``E, A, I``）。
- 索单元：仅受轴力的二节点杆（``E, A``），``pretension`` 为初始轴力（+ 受拉，单位 N），
  等效于 OpenSees 的 ``InitStressMaterial`` 初应力（σ0 = pretension / A）。
- 梁单元均布荷载 :class:`MemberUDL`：``wy`` 为**单元局部横向**线荷载（与 OpenSees
  ``eleLoad -beamUniform Wy`` 一致）；重力作用在水平梁上时 ``wy = -自重线荷载``。

参见 ``docs/DESIGN_MAPPO.md`` 第 4 节、``docs/ARCHITECTURE.md``。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Node:
    id: int
    x: float
    y: float


@dataclass
class FrameMember:
    """Euler-Bernoulli 梁单元。"""

    id: int
    i: int          # 起节点 id
    j: int          # 末节点 id
    E: float
    A: float
    I: float


@dataclass
class CableMember:
    """仅受轴力的索/杆单元。

    ``pretension`` 为安装初始轴力 (N, 正为拉)，对应 OpenSees 初应力 σ0 = pretension/A。
    """

    id: int
    i: int
    j: int
    E: float
    A: float
    pretension: float = 0.0


@dataclass
class Support:
    """节点约束（True 表示该方向固定）。"""

    node: int
    ux: bool = False
    uy: bool = False
    rz: bool = False


@dataclass
class NodalLoad:
    node: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0


@dataclass
class MemberUDL:
    """梁单元局部横向均布荷载 (per length)。与 OpenSees beamUniform Wy 一致。"""

    member: int
    wy: float


class StructuralModel:
    """与求解器无关的结构模型容器。"""

    def __init__(self, name: str = "model"):
        self.name = name
        self.nodes: dict[int, Node] = {}
        self.frames: dict[int, FrameMember] = {}
        self.cables: dict[int, CableMember] = {}
        self.supports: dict[int, Support] = {}
        self.nodal_loads: list[NodalLoad] = []
        self.member_udls: dict[int, MemberUDL] = {}

    # ----------------------------------------------------------- 建模 API
    def add_node(self, node_id: int, x: float, y: float) -> None:
        self.nodes[node_id] = Node(node_id, float(x), float(y))

    def add_frame(self, mid: int, i: int, j: int, E: float, A: float, I: float) -> None:
        self.frames[mid] = FrameMember(mid, i, j, E, A, I)

    def add_cable(self, mid: int, i: int, j: int, E: float, A: float, pretension: float = 0.0) -> None:
        self.cables[mid] = CableMember(mid, i, j, E, A, float(pretension))

    def add_support(self, node: int, ux=False, uy=False, rz=False) -> None:
        self.supports[node] = Support(node, bool(ux), bool(uy), bool(rz))

    def add_nodal_load(self, node: int, fx=0.0, fy=0.0, mz=0.0) -> None:
        self.nodal_loads.append(NodalLoad(node, float(fx), float(fy), float(mz)))

    def add_member_udl(self, member: int, wy: float) -> None:
        self.member_udls[member] = MemberUDL(member, float(wy))

    # ----------------------------------------------------------- 辅助
    def node_xy(self, node_id: int) -> tuple[float, float]:
        n = self.nodes[node_id]
        return n.x, n.y

    def summary(self) -> str:
        return (
            f"<StructuralModel '{self.name}': {len(self.nodes)} nodes, "
            f"{len(self.frames)} frames, {len(self.cables)} cables, "
            f"{len(self.supports)} supports, {len(self.nodal_loads)} nodal loads, "
            f"{len(self.member_udls)} UDLs>"
        )


@dataclass
class SolveResult:
    """两种后端统一返回的结果，便于逐项对比。

    Attributes
    ----------
    disp : dict[int, tuple[float, float, float]]
        节点位移 {node_id: (ux, uy, rz)}。
    frame_force : dict[int, tuple]
        梁单元**局部**端力 {id: (N_i, V_i, M_i, N_j, V_j, M_j)}。
    cable_force : dict[int, float]
        索轴力 {id: N}（+ 受拉）。
    cable_stress : dict[int, float]
        索应力 {id: Pa}。
    converged : bool
    backend : str
    """

    disp: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    frame_force: dict[int, tuple] = field(default_factory=dict)
    cable_force: dict[int, float] = field(default_factory=dict)
    cable_stress: dict[int, float] = field(default_factory=dict)
    converged: bool = True
    backend: str = ""

    def uy(self, node_id: int) -> float:
        return self.disp[node_id][1]
