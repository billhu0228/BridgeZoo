"""多智能体调索环境子包。

- :mod:`bridgezoo.envs.geometry` —— 桥梁几何与截面/材料参数（纯几何，已实现）。
- :mod:`bridgezoo.envs.cable_agent` —— 单根索智能体的状态/动作/局部观测定义。
- :mod:`bridgezoo.envs.cable_construction` —— 正向逐阶段施工 + 两次张拉的
  PettingZoo 并行环境（合作型 Dec-POMDP，CTDE）。

参见 ``docs/DESIGN_MAPPO.md`` 第 5 节。
"""

from bridgezoo.envs.geometry import BridgeGeometry

__all__ = ["BridgeGeometry", "cable_agent", "cable_construction"]
