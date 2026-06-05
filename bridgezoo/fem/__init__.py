"""结构有限元求解子包。

**一套结构定义,两种求解后端,结果一致**(用于校核自研求解器),分两条主线:

共享核心(本层):

- :mod:`bridgezoo.fem.model` —— 与求解器无关的结构模型 IR(``StructuralModel`` /
  ``SolveResult``)。
- :mod:`bridgezoo.fem.linear_frame` —— 自研二维直接刚度法求解器(``DirectStiffnessSolver``,
  RL 内核);其低层单元刚度/变换辅助也被施工阶段子包复用。

一次成桥(完成态)—— :mod:`bridgezoo.fem.oneshot`:

- ``oneshot.builder`` —— 由 ``BridgeGeometry`` 构建 ``StructuralModel``。
- ``oneshot.opensees_backend`` —— OpenSees 线性后端(Truss + 初应力),交叉校核。
- ``oneshot.opensees_ref`` —— 几何非线性"一次成桥"参考(corotTruss + Newton)。

逐阶段施工(切线激活、变刚度)—— :mod:`bridgezoo.fem.staged`:

- ``staged.plan`` —— 施工计划 IR + 结果容器 + 共享辅助。
- ``staged.builder`` —— 由参数构建 ``StagedPlan``(对称双悬臂 + 扇面索)。
- ``staged.direct`` —— 自研增量直接刚度法后端(RL 内核)。
- ``staged.opensees`` —— OpenSees 后端(切线激活),交叉校核。
- ``staged.sequence`` —— RL 环境施工阶段序列(骨架,M1/M2)。

参见 ``docs/DESIGN_MAPPO.md`` 第 3、4 节、``docs/ARCHITECTURE.md``。
"""

from bridgezoo.fem.model import StructuralModel, SolveResult

__all__ = [
    "StructuralModel",
    "SolveResult",
    "model",
    "linear_frame",
    "oneshot",
    "staged",
]
