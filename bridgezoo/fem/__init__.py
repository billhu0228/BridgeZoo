"""结构有限元求解子包。

**一套结构定义，两种求解后端，结果一致**（用于校核自研求解器）：

- :mod:`bridgezoo.fem.model` —— 与求解器无关的结构模型 IR（``StructuralModel`` /
  ``SolveResult``）。
- :mod:`bridgezoo.fem.builder` —— 由 ``BridgeGeometry`` 构建 ``StructuralModel``。
- :mod:`bridgezoo.fem.linear_frame` —— 自研二维直接刚度法后端（``DirectStiffnessSolver``）。
- :mod:`bridgezoo.fem.opensees_backend` —— OpenSees 后端（``OpenSeesSolver``），线性公式，
  用于交叉校核。
- :mod:`bridgezoo.fem.opensees_staged` —— 施工阶段（切线激活、几何非线性）OpenSees 模型。
- :mod:`bridgezoo.fem.opensees_ref` —— 历史"一次成桥"参考求解器（corotTruss）。
- :mod:`bridgezoo.fem.staged_builder` —— 施工阶段序列（骨架，M1/M2）。

参见 ``docs/DESIGN_MAPPO.md`` 第 3、4 节、``docs/ARCHITECTURE.md``。
"""

from bridgezoo.fem.model import StructuralModel, SolveResult

__all__ = [
    "StructuralModel",
    "SolveResult",
    "model",
    "builder",
    "linear_frame",
    "opensees_backend",
    "opensees_staged",
    "opensees_ref",
    "staged_builder",
]
