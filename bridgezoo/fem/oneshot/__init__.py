"""一次成桥(完成态)子包 —— "一套结构定义,两种求解后端,结果一致"。

- :mod:`bridgezoo.fem.oneshot.builder` —— 由 ``BridgeGeometry`` 构建 ``StructuralModel``。
- :mod:`bridgezoo.fem.oneshot.opensees_backend` —— OpenSees 线性后端(Truss + 初应力),
  与自研直接刚度法 :class:`bridgezoo.fem.linear_frame.DirectStiffnessSolver` 交叉校核。
- :mod:`bridgezoo.fem.oneshot.opensees_ref` —— 几何非线性"一次成桥"参考求解器
  (corotTruss + Newton),作为几何精确成桥参考。

结构 IR(``StructuralModel`` / ``SolveResult``)与自研求解器在上层 :mod:`bridgezoo.fem`。
"""

from bridgezoo.fem.oneshot.builder import build_cable_bridge
from bridgezoo.fem.oneshot.opensees_backend import OpenSeesSolver
from bridgezoo.fem.oneshot.opensees_ref import FEM, build_oneshot_fem

__all__ = [
    "build_cable_bridge",
    "OpenSeesSolver",
    "FEM",
    "build_oneshot_fem",
    "builder",
    "opensees_backend",
    "opensees_ref",
]
