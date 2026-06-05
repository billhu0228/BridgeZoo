"""由桥梁几何生成正向施工阶段序列（里程碑 M1/M2）。

把 :class:`bridgezoo.envs.geometry.BridgeGeometry` 翻译成一串"施工阶段"，每个阶段
描述：本阶段新增/激活的节点与单元、边界条件的变化、本阶段的张拉动作槽位。环境层据此
驱动 :class:`bridgezoo.fem.staged.direct.StagedDirectSolver` 逐阶段求解。

施工序列（对称悬臂拼装 + 二次张拉，N = 单侧索对数）::

    阶段 0       : 索塔 + 0# 梁段就位，仅自重
    阶段 1..N    : 安装第 k 对梁段 + 第 k 对索 → 第一次张拉 T1_k
    阶段 N+1     : 合龙（中跨闭合，边界条件切换）
    阶段 N+1+j   : 对第 j 对索第二次张拉/调索 T2_j  (j = 1..N)
    成桥阶段     : 全部恒载就位，评价线形/索力/股数

参见 ``docs/DESIGN_MAPPO.md`` 第 3.1 节。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    """施工相位。"""

    ERECTION = "erection"        # 悬臂拼装 + 第一次张拉
    CLOSURE = "closure"          # 合龙
    ADJUSTMENT = "adjustment"    # 第二次张拉 / 调索
    FINAL = "final"              # 成桥评价


@dataclass
class ConstructionStage:
    """单个施工阶段的描述。

    Attributes
    ----------
    index : int
        阶段序号（0 起）。
    phase : Phase
        所属相位。
    new_nodes / new_frames / new_cables : list[int]
        本阶段**新增并激活**的节点 / 梁单元 / 索单元 id。
    active_cable : int | None
        本阶段进行张拉的索对编号（无张拉则 None）；对应当前可决策的智能体。
    support_changes : dict[int, tuple[bool, bool, bool]]
        本阶段边界条件变化（如合龙时跨中约束切换）。
    """

    index: int
    phase: Phase
    new_nodes: list[int]
    new_frames: list[int]
    new_cables: list[int]
    active_cable: int | None
    support_changes: dict[int, tuple[bool, bool, bool]]


def build_stages(geometry) -> list[ConstructionStage]:
    """由几何对象生成完整施工阶段序列。

    Parameters
    ----------
    geometry : bridgezoo.envs.geometry.BridgeGeometry

    Returns
    -------
    list[ConstructionStage]

    TODO(M1/M2): 依据 geometry 的节点/单元编号约定，按上面的施工序列生成阶段列表，
    与 :mod:`bridgezoo.fem.staged.builder` 的编号保持一致以便校核。
    """
    raise NotImplementedError("TODO(M1/M2): staged_builder.build_stages")
