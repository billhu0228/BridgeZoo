"""一次成桥（完成态）分析子包 —— "一套结构定义，两种求解后端，结果一致"。

消费与求解器无关的 :class:`bridgezoo.fem.model.StructuralModel`（成桥完成态：所有梁段/
拉索一次激活），与 :mod:`bridgezoo.fem.staged`（分阶段施工）镜像对称：两者都提供
``direct`` / ``opensees`` 两个后端。

- :mod:`bridgezoo.fem.completed.direct` —— 自研二维直接刚度法（``CompletedDirectSolver``）。
- :mod:`bridgezoo.fem.completed.opensees` —— OpenSees 线性后端（Truss + 初应力），交叉校核。

公共 API 在此重导出，故 ``from bridgezoo.fem.completed import CompletedDirectSolver`` 等写法稳定。
"""

from bridgezoo.fem.completed.direct import CompletedDirectSolver
from bridgezoo.fem.completed.opensees import CompletedOpenSeesSolver

__all__ = [
    "CompletedDirectSolver",
    "CompletedOpenSeesSolver",
    "direct",
    "opensees",
]
