"""结构有限元求解子包。

**一套结构定义,两种求解后端,结果一致**(用于校核自研求解器)。两种**分析模式**镜像对称:

共享核心(本层):

- :mod:`bridgezoo.fem.model` —— 与求解器无关的结构模型 IR(``StructuralModel`` /
  ``SolveResult``)。
- :mod:`bridgezoo.fem.kernels` —— 二维杆系单元的底层数值核(刚度/变换/等效荷载),
  供两种模式的自研直接刚度法共享。

一次成桥(完成态)—— :mod:`bridgezoo.fem.completed`:

- ``completed.direct`` —— 自研二维直接刚度法(``CompletedDirectSolver``,RL 内核)。
- ``completed.opensees`` —— OpenSees 线性后端(Truss + 初应力),交叉校核。

逐阶段施工(切线激活、变刚度)+ 成桥派生 —— :mod:`bridgezoo.fem.staged`:

- ``staged.plan`` —— 施工计划 IR(含成桥状态 ``CompletedState``)+ 结果容器 + 共享辅助。
- ``staged.builder`` —— 由参数构建 ``StagedPlan``(对称双悬臂 + 扇面索)。
- ``staged.direct`` —— 自研增量直接刚度法后端(RL 内核)。
- ``staged.opensees`` —— OpenSees 后端(切线激活),交叉校核。
- ``staged.completed`` —— 由施工计划派生成桥(完成态)``StructuralModel``(``build_completed_model``),
  成桥工况的唯一建模入口。
- ``staged.sequence`` —— RL 环境施工阶段序列(骨架,M1/M2)。

参见 ``docs/DESIGN_MAPPO.md`` 第 3、4 节、``docs/ARCHITECTURE.md``。
"""

from bridgezoo.fem.completed import CompletedDirectSolver, CompletedOpenSeesSolver
from bridgezoo.fem.model import SolveResult, StructuralModel

__all__ = [
    "StructuralModel",
    "SolveResult",
    "CompletedDirectSolver",
    "CompletedOpenSeesSolver",
    "model",
    "kernels",
    "completed",
    "staged",
]
