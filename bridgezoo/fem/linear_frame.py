"""自写轻量线性 2D 框架求解器（RL 内核，里程碑 M1）。

设计动机
--------
MAPPO 训练需要数十万~数百万次环境步，每步都要做结构分析。若每步重建 OpenSees
模型，开销不可接受。本模块用直接刚度法（direct stiffness method）实现一个**线性**
2D 框架求解器，单次装配 + 求解控制在毫秒级。

核心物理点（务必遵守）
----------------------
正向逐阶段施工中，**结构刚度矩阵 K 随施工阶段与拉索股数不断变化**：

1. 每进入一个新阶段，新增的梁段/索按**上一阶段变形后的构型**接入（新单元的无应力
   长度在安装时刻定义，之后的增量才使其受力 —— 实现位移"锁定"/lock-in）。
2. 用当前激活集合（含当前股数 → 当前索面积）**重新装配** ``K_k``。
3. 组装本阶段**增量荷载** ``ΔF_k``（新增梁段自重 + 本阶段张拉等效力）。
4. 解 ``K_k · Δu_k = ΔF_k``（稀疏直接法，缓存符号结构）。
5. **累加** ``u ← u + Δu_k``，并更新各索轴力。

因此不能用脱离动作的单一全局影响矩阵；但每阶段拓扑固定、规模很小，逐阶段线性求解
即可。固定股数下张拉→响应是线性的，可选地为加速预计算"每阶段张拉影响矩阵"。

单元约定
--------
- 节点：3 自由度 (u, v, θ)。
- 梁：Euler-Bernoulli 框架单元，参数 (EA, EI, L)。
- 索：二节点桁架（只受拉），等效轴向刚度 ``E_s * A_s * n``，张拉以初应变/等效节点
  力施加；求解后轴力 ≤ 0 视为松弛（交由环境层做安全惩罚，本求解器只如实返回）。

参见 ``docs/DESIGN_MAPPO.md`` 第 3、4 节。校核见 ``scripts/validate_fem.py``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# TODO(M1): 大模型时切换为 scipy.sparse + splu；当前规模 (DOF<~150) 稠密即可。


@dataclass
class Node:
    """节点：编号、平面坐标，以及全局自由度索引 (u, v, θ)。"""

    id: int
    x: float
    y: float
    dofs: tuple[int, int, int] | None = None


@dataclass
class FrameElement:
    """梁单元（Euler-Bernoulli），属性 EA、EI。"""

    id: int
    n1: int
    n2: int
    EA: float
    EI: float

    def local_stiffness(self, dx: float, dy: float):
        """返回该单元在全局坐标系下的 6x6 刚度矩阵。

        TODO(M1): 标准框架单元刚度 + 坐标变换 T^T k T。
        """
        raise NotImplementedError("TODO(M1): FrameElement.local_stiffness")


@dataclass
class CableElement:
    """索单元（二节点桁架）。

    ``area = A_s * n`` 随股数 ``n`` 变化，故会改变全局刚度矩阵。``pretension`` 为
    安装/张拉时刻施加的初张力 (N)，以等效节点力 + 初应变方式进入增量荷载。
    """

    id: int
    n1: int
    n2: int
    Es: float
    area: float
    pretension: float = 0.0

    def axial_stiffness(self, dx: float, dy: float):
        """返回该索在全局坐标系下的 4x4 轴向刚度矩阵（仅平动自由度）。

        TODO(M1): k = (Es*area/L) * 外积(方向余弦)。
        """
        raise NotImplementedError("TODO(M1): CableElement.axial_stiffness")


@dataclass
class StagedFrameModel:
    """逐阶段（变刚度）线性框架求解器。

    典型用法（环境层每个施工阶段调用一次）::

        m = StagedFrameModel()
        m.add_node(...); m.add_frame(...); m.add_cable(...)
        m.set_support(node_id, ux, uy, rz)
        m.activate(stage_elements)            # 本阶段激活的单元集合
        m.apply_incremental_load(dF)          # 新增自重等
        m.apply_cable_pretension(elem_id, T)  # 本阶段张拉
        du = m.solve_increment()              # 解 K_k Δu = ΔF_k
        u_total, cable_forces = m.accumulate()

    Notes
    -----
    - ``self.u`` 维护累加的总位移；``solve_increment`` 只解增量并叠加。
    - 每次 ``activate`` 后重建自由度映射与刚度矩阵（刚度随阶段/股数变化）。
    """

    nodes: dict[int, Node] = field(default_factory=dict)
    frames: dict[int, FrameElement] = field(default_factory=dict)
    cables: dict[int, CableElement] = field(default_factory=dict)
    supports: dict[int, tuple[bool, bool, bool]] = field(default_factory=dict)
    active: set[int] = field(default_factory=set)
    u: np.ndarray | None = None  # 累加总位移 (n_dof,)

    # ------------------------------------------------------------------ 建模
    def add_node(self, node_id: int, x: float, y: float) -> None:
        self.nodes[node_id] = Node(node_id, x, y)

    def add_frame(self, elem_id: int, n1: int, n2: int, EA: float, EI: float) -> None:
        self.frames[elem_id] = FrameElement(elem_id, n1, n2, EA, EI)

    def add_cable(self, elem_id: int, n1: int, n2: int, Es: float, area: float) -> None:
        self.cables[elem_id] = CableElement(elem_id, n1, n2, Es, area)

    def set_support(self, node_id: int, ux: bool, uy: bool, rz: bool) -> None:
        """设置约束（True 表示该方向固定）。"""
        self.supports[node_id] = (ux, uy, rz)

    # -------------------------------------------------------------- 阶段求解
    def activate(self, element_ids: Iterable[int]) -> None:
        """设置当前阶段激活的单元集合并重建自由度编号/刚度结构。

        TODO(M1): 仅对激活单元涉及的节点编号自由度，组装当前 K_k。
        """
        raise NotImplementedError("TODO(M1): StagedFrameModel.activate")

    def apply_incremental_load(self, loads: dict[int, tuple[float, float, float]]) -> None:
        """累加本阶段增量节点荷载 (Fx, Fy, Mz)。

        TODO(M1): 含梁段自重 -> 等效节点力（也可用单元均布荷载固端力）。
        """
        raise NotImplementedError("TODO(M1): StagedFrameModel.apply_incremental_load")

    def apply_cable_pretension(self, elem_id: int, tension: float) -> None:
        """对指定索施加本阶段张拉力，折算为等效节点增量荷载。

        TODO(M1): 沿索方向施加 ±T 等效节点力，并记录用于轴力回算。
        """
        raise NotImplementedError("TODO(M1): StagedFrameModel.apply_cable_pretension")

    def solve_increment(self) -> np.ndarray:
        """解 ``K_k Δu_k = ΔF_k`` 并叠加到累加位移。返回本阶段增量 Δu。

        TODO(M1): 施加边界条件 -> 缩减自由度 -> 直接求解 -> 还原 -> self.u += du。
        """
        raise NotImplementedError("TODO(M1): StagedFrameModel.solve_increment")

    def accumulate(self) -> tuple[np.ndarray, dict[int, float]]:
        """返回 (各梁节点累加竖向位移, {索单元id: 轴力})。

        TODO(M1): 由 self.u 提取竖向位移；由索两端相对位移 + 初张力回算轴力。
        """
        raise NotImplementedError("TODO(M1): StagedFrameModel.accumulate")
