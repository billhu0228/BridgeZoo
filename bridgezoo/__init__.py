"""BridgeZoo —— 基于 MAPPO 的二维斜拉桥正向施工调索研究框架。

本包提供：

- ``bridgezoo.fem``    : 结构有限元求解（自写线性变刚度前进分析求解器 + OpenSees 校核）。
- ``bridgezoo.envs``   : 正向逐阶段施工 + 两次张拉的多智能体调索环境（PettingZoo）。
- ``bridgezoo.mappo``  : 自写精简 MAPPO 算法（共享 actor + 中心化 critic，CTDE）。
- ``bridgezoo.render`` : 施工过程与成桥状态的可视化。

总体设计见 ``docs/DESIGN_MAPPO.md``，目录与模块职责见 ``docs/ARCHITECTURE.md``。

> 注意：本版本（v0.1.x）是对历史实验代码的重构起点，多数模块为带完整说明的
> 骨架（skeleton），具体实现按 ``TODO.md`` 的里程碑推进。历史代码已归档至 ``archive/``。
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
