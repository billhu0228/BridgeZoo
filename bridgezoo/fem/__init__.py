"""结构有限元求解子包。

包含两套求解器，职责分离：

- :mod:`bridgezoo.fem.linear_frame` —— 自写轻量线性 2D 框架求解器（**RL 内核**，
  追求速度），支持每施工阶段重装配刚度矩阵、线性增量求解并累加锁定位移。
- :mod:`bridgezoo.fem.staged_builder` —— 由桥梁几何参数生成施工阶段序列与每阶段
  激活的节点/单元集合。
- :mod:`bridgezoo.fem.opensees_ref` —— 基于 OpenSeesPy 的"一次成桥"参考求解器，
  **仅用于正确性校核**，不进入 RL 训练回路。

参见 ``docs/DESIGN_MAPPO.md`` 第 3、4 节。
"""

__all__ = ["linear_frame", "staged_builder", "opensees_ref"]
