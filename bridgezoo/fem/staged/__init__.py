"""逐阶段施工(变刚度 + 切线激活)子包 —— "一套计划,两种后端,结果一致"。

- :mod:`bridgezoo.fem.staged.plan` —— 与求解器无关的施工计划 IR + 结果容器 + 共享辅助。
- :mod:`bridgezoo.fem.staged.builder` —— 由参数构建 :class:`StagedPlan`(对称双悬臂 + 扇面索)。
- :mod:`bridgezoo.fem.staged.direct` —— 自研增量直接刚度法后端(RL 内核,线性小位移)。
- :mod:`bridgezoo.fem.staged.opensees` —— OpenSees 后端(切线激活),用于交叉校核。
- :mod:`bridgezoo.fem.staged.sequence` —— RL 环境的施工阶段序列(骨架,M1/M2)。

公共 API 在此重导出,故 ``from bridgezoo.fem.staged import StagedDirectSolver`` 等写法保持稳定。
"""

from bridgezoo.fem.staged.builder import build_staged_cantilever
from bridgezoo.fem.staged.direct import StagedDirectSolver
from bridgezoo.fem.staged.opensees import StagedOpenSeesSolver
from bridgezoo.fem.staged.plan import (
    BalanceDof,
    BuildStep,
    NewCable,
    NewFrame,
    NewNode,
    NodalLoad,
    StagedPlan,
    StagedResult,
    StagedStepRecord,
)

__all__ = [
    "build_staged_cantilever",
    "StagedDirectSolver",
    "StagedOpenSeesSolver",
    "StagedPlan",
    "StagedResult",
    "StagedStepRecord",
    "BuildStep",
    "NewNode",
    "NewFrame",
    "NewCable",
    "NodalLoad",
    "BalanceDof",
    "plan",
    "builder",
    "direct",
    "opensees",
    "sequence",
]
