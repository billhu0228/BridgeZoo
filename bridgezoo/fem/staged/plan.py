"""逐阶段施工的**后端无关**数据层:施工计划 IR + 结果容器 + 共享辅助。

延续"一套定义,两种后端,结果一致"的思路,但对象是**施工过程**:

- :class:`StagedPlan` —— 与求解器无关的施工计划:一串 :class:`BuildStep`,每个 step 是
  一次"激活构件 + 施加增量荷载 + 求解增量"的施工动作(装节段 / 张索)。
- :class:`StagedResult` / :class:`StagedStepRecord` —— 两种后端统一返回的结果容器,逐项可比。

两个后端(:mod:`bridgezoo.fem.staged.direct` 自研增量直接刚度法、
:mod:`bridgezoo.fem.staged.opensees` OpenSees 校核)都消费**同一个** StagedPlan,
返回**同一种** StagedResult。

切线激活(零应力诞生)
----------------------
新节点按附着节点的**变形后切线**安放(小转角线性)::

    ux_J = ux_I - dy * rz_I,   uy_J = uy_I + dx * rz_I,   rz_J = rz_I

其中 (dx, dy) 为新单元的设计矢量、I 为附着节点。这样新梁单元安装即零应力;之后各
增量步测得的相对变形才使其受力(位移锁定)。索按"安装时刻达到目标张力 T"处理,随
后续施工重分布而变化——即索力历程。
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ============================================================ 施工计划(IR)
@dataclass
class NewNode:
    id: int
    x: float
    y: float
    attach: int | None = None  # 切线附着节点;None 表示按零位移就位(如根部/锚点)


@dataclass
class NewFrame:
    id: int
    i: int
    j: int
    E: float
    A: float
    I: float
    udl_wy: float = 0.0  # 本步施加的**全局竖向**均布荷载(自重,向下为负;按单元方向投影到局部)


@dataclass
class NewCable:
    id: int
    i: int
    j: int
    E: float
    A: float
    tension: float = 0.0  # 安装时刻目标张力 (N)


@dataclass
class NodalLoad:
    node: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0


@dataclass
class BalanceDof:
    node: int
    dof: int  # 0=ux, 1=uy, 2=rz
    target: float = 0.0


@dataclass
class BuildStep:
    """一次施工动作(= 一次增量求解)。"""

    label: str
    new_nodes: list[NewNode] = field(default_factory=list)
    new_frames: list[NewFrame] = field(default_factory=list)
    new_cables: list[NewCable] = field(default_factory=list)
    nodal_loads: list[NodalLoad] = field(default_factory=list)
    balance_dofs: list[BalanceDof] = field(default_factory=list)
    record: bool = True


@dataclass
class CompletedState:
    """由分阶段施工计划派生的成桥（完成态）结构状态。"""

    label: str = "completed"
    nodes: list[NewNode] = field(default_factory=list)
    frames: list[NewFrame] = field(default_factory=list)
    cables: list[NewCable] = field(default_factory=list)
    supports: list[tuple[int, bool, bool, bool]] = field(default_factory=list)
    nodal_loads: list[NodalLoad] = field(default_factory=list)


@dataclass
class StagedPlan:
    name: str
    init_nodes: list[NewNode] = field(default_factory=list)        # 阶段0已存在的节点
    supports: list[tuple[int, bool, bool, bool]] = field(default_factory=list)
    steps: list[BuildStep] = field(default_factory=list)
    completed: CompletedState | None = None


# ============================================================ 结果
@dataclass
class StagedStepRecord:
    label: str
    disp: dict[int, tuple[float, float, float]]
    cable_force: dict[int, float]
    cable_stress: dict[int, float]
    applied_loads: dict[int, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class StagedResult:
    backend: str
    records: list[StagedStepRecord] = field(default_factory=list)
    # 静态几何信息(供可视化/后处理使用,免去外部按 id 反推坐标)
    coords: dict[int, tuple[float, float]] = field(default_factory=dict)   # {node_id: (x, y)}
    cable_nodes: dict[int, tuple[int, int]] = field(default_factory=dict)  # {cable_id: (i, j)}
    anchor_ids: list[int] = field(default_factory=list)                    # 塔上锚点 node id
    deck_ids: list[int] = field(default_factory=list)                      # 主梁 node id(含根部)

    def cable_stress_history(self) -> dict[int, list[tuple[str, float]]]:
        out: dict[int, list[tuple[str, float]]] = {}
        for rec in self.records:
            for cid, s in rec.cable_stress.items():
                out.setdefault(cid, []).append((rec.label, s))
        return out

    def final_disp(self) -> dict[int, tuple[float, float, float]]:
        return self.records[-1].disp if self.records else {}


# ============================================================ 共享辅助(两后端复用)
def _attach_geometry(result: StagedResult, plan: StagedPlan) -> None:
    """把施工计划的静态几何(节点坐标、索连接、锚点/梁节点分类)写入结果。"""
    nodes: list[NewNode] = list(plan.init_nodes)
    for step in plan.steps:
        nodes.extend(step.new_nodes)
    for nd in nodes:
        result.coords[nd.id] = (nd.x, nd.y)
        if nd.y > 1e-9:
            result.anchor_ids.append(nd.id)   # 塔上锚点(y>0)
        else:
            result.deck_ids.append(nd.id)     # 主梁节点(y≈0,含根部)
    for step in plan.steps:
        for cb in step.new_cables:
            result.cable_nodes[cb.id] = (cb.i, cb.j)
