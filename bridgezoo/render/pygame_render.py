"""施工过程与成桥状态的 pygame 可视化（里程碑 M2，可视化增强 M4）。

从历史 ``cablebridge2.py`` 的渲染逻辑重构而来，新增：

- 逐阶段拼装 + 两次张拉的动画（高亮当前活动索 / 当前相位）。
- 实时线形（带目标线形对照）、各索应力与股数条形图。
- 文本模式（``render_mode='text'``）打印阶段报表。

为保持骨架在无 pygame 环境下可被 import，pygame 仅在实例化/绘制时导入。
"""

from __future__ import annotations

import numpy as np

from bridgezoo.envs.geometry import BridgeGeometry


class BridgeRenderer:
    """斜拉桥施工/成桥可视化器（骨架）。

    TODO(M2):
      - __init__: 初始化 pygame 窗口、比例尺、坐标变换矩阵（参考历史实现）。
      - draw(state): 绘制变形主梁、索（按活动/普通着色）、塔、目标线形、文本面板。
      - close()。
    """

    def __init__(self, geometry: BridgeGeometry, screen_size=(1080, 600), fps: int = 10, def_scale: float = 10.0):
        self.geom = geometry
        self.screen_size = screen_size
        self.fps = fps
        self.def_scale = def_scale
        raise NotImplementedError("TODO(M2): BridgeRenderer.__init__")

    def draw(self, beam_deflection: np.ndarray, cable_stress, cable_strands, active_cable=None, phase: str = ""):
        raise NotImplementedError("TODO(M2): BridgeRenderer.draw")

    def close(self):
        pass
