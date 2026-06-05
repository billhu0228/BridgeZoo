"""结构有限元求解子包。

**一套结构定义,两种求解后端,结果一致**(用于校核自研求解器),分两条主线:

共享核心(本层):

- :mod:`bridgezoo.fem.model` —— 与求解器无关的结构模型 IR(``StructuralModel`` /
  ``SolveResult``)。
- :mod:`bridgezoo.fem.linear_frame` —— 自研二维直接刚度法求解器(``DirectStiffnessSolver``,
  RL 内核);其低层单元刚度/变换辅助也被施工阶段子包复用。
- :mod:`bridgezoo.fem.opensees_backend` —— 消费 ``StructuralModel`` 的 OpenSees 线性后端
  (Truss + 初应力),与自研直接刚度法交叉校核(成桥/单阶段线性工况)。

逐阶段施工(切线激活、变刚度)+ 成桥派生 —— :mod:`bridgezoo.fem.staged`:

- ``staged.plan`` —— 施工计划 IR(含成桥状态 ``OneShotState``)+ 结果容器 + 共享辅助。
- ``staged.builder`` —— 由参数构建 ``StagedPlan``(对称双悬臂 + 扇面索)。
- ``staged.direct`` —— 自研增量直接刚度法后端(RL 内核)。
- ``staged.opensees`` —— OpenSees 后端(切线激活),交叉校核。
- ``staged.oneshot`` —— 由施工计划派生成桥(完成态)``StructuralModel``(``build_oneshot_model``),
  "一次成桥"工况的唯一建模入口。
- ``staged.sequence`` —— RL 环境施工阶段序列(骨架,M1/M2)。

参见 ``docs/DESIGN_MAPPO.md`` 第 3、4 节、``docs/ARCHITECTURE.md``。
"""

from bridgezoo.fem.model import StructuralModel, SolveResult

__all__ = [
    "StructuralModel",
    "SolveResult",
    "model",
    "linear_frame",
    "opensees_backend",
    "staged",
]
