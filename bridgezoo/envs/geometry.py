"""桥梁几何与截面/材料参数（已实现）。

本模块从历史代码 ``cablebridge2.py`` / ``test_ops.py`` 抽取并去重，是整个项目唯一的
几何真源（single source of truth）。FEM 求解器、施工阶段生成、渲染都从这里取参数。

约定：对称双塔三跨，二维，单侧 ``num_cables_per_side``（须为偶数）对索，左右对称、
以"索对"为单位。坐标系原点在跨中、x 向右、y 向上。

节点/单元编号约定（与 :mod:`bridgezoo.fem.oneshot.opensees_ref` 保持一致）：

- 梁节点：``1 .. 2*N+5``，按 x 升序。
- 索塔节点：左塔 ``1001+i``，右塔 ``3001+i`` （``i = 0 .. N/2-1``，自塔顶向下）。
- 梁单元：``1 .. 2*N+4``。
- 索单元：左塔→左/右梁 ``1001+i`` / ``2001+i``；右塔→ ``3001+i`` / ``4001+i``。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BridgeGeometry:
    """二维对称斜拉桥几何 + 截面/材料参数。

    Parameters
    ----------
    num_cables_per_side : int
        单侧索对数，必须为偶数且 > 1。
    anchor_height : float
        上塔柱高度（塔顶相对梁面，m）。
    beam_w, beam_h : float
        主梁矩形截面宽、高 (m)。
    beam_E : float
        主梁弹性模量 (Pa)。
    cable_Es : float
        拉索弹性模量 (Pa)。
    strand_area : float
        单股钢绞线截面积 (m²)。
    density, gravity : float
        主梁材料密度 (kg/m³) 与重力加速度 (m/s²)，用于自重线荷载。
    load_factor : float
        线荷载附加系数（历史模型按双幅取 2.0）。

    几何间距（沿用历史默认值，单位 m）::

        middle_spacing=10, outside_spacing=8, end_to_first_spacing=4,
        center_to_adjacent_spacing=2, vertical_spacing=2
    """

    num_cables_per_side: int = 6
    anchor_height: float = 20.0
    beam_w: float = 10.0
    beam_h: float = 1.0
    beam_E: float = 20e9
    cable_Es: float = 1.95e11
    strand_area: float = 1.4e-4
    density: float = 2400.0
    gravity: float = 9.806
    load_factor: float = 2.0

    middle_spacing: float = 10.0
    outside_spacing: float = 8.0
    end_to_first_spacing: float = 4.0
    center_to_adjacent_spacing: float = 2.0
    vertical_spacing: float = 2.0

    # ---- 派生量（__post_init__ 计算，勿手动赋值） ----
    span: float = field(init=False)
    side_span: float = field(init=False)
    beam_length: float = field(init=False)
    beam_area: float = field(init=False)
    beam_Iz: float = field(init=False)
    wg: float = field(init=False)
    num_beam_points: int = field(init=False)
    x_positions: np.ndarray = field(init=False)
    left_tower_top: np.ndarray = field(init=False)
    right_tower_top: np.ndarray = field(init=False)
    left_tower_base: np.ndarray = field(init=False)
    right_tower_base: np.ndarray = field(init=False)
    left_tower_pts: np.ndarray = field(init=False)
    right_tower_pts: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        N = self.num_cables_per_side
        if N <= 1 or N % 2 != 0:
            raise ValueError("num_cables_per_side 必须为偶数且 > 1")

        # 截面 / 自重
        self.beam_area = self.beam_h * self.beam_w
        self.beam_Iz = self.beam_w * self.beam_h ** 3 / 12.0
        self.wg = self.beam_area * self.density * self.gravity * self.load_factor

        # 跨径
        self.num_beam_points = N + 3  # 仅一侧的梁节点计数（历史约定）
        self.span = 2 * (N * 0.5 * self.middle_spacing + self.center_to_adjacent_spacing)
        self.side_span = self.end_to_first_spacing + N * 0.5 * self.outside_spacing
        self.beam_length = self.side_span * 2 + self.span

        # 主梁节点 x 坐标
        x1 = np.linspace(
            -0.5 * self.beam_length + self.end_to_first_spacing,
            -self.outside_spacing - self.span * 0.5,
            N // 2,
        )
        x2 = np.linspace(-self.span * 0.5, -self.center_to_adjacent_spacing, N // 2 + 1)
        x_positions = np.hstack(
            (-self.beam_length * 0.5, x1, x2, 0.0, -x2, -x1, 0.5 * self.beam_length)
        )
        # 与历史实现一致：共 2N+5 个节点，直接排序得到单调递增的节点坐标
        x_positions = np.round(x_positions, 9)
        x_positions.sort()
        self.x_positions = x_positions.astype(np.float64)

        # 索塔
        self.left_tower_top = np.array([-0.5 * self.span, self.anchor_height])
        self.right_tower_top = np.array([0.5 * self.span, self.anchor_height])
        self.left_tower_base = np.array([-0.5 * self.span, 0.0])
        self.right_tower_base = np.array([0.5 * self.span, 0.0])
        offs = np.array([[0.0, -self.vertical_spacing * i] for i in range(N // 2)])
        self.left_tower_pts = self.left_tower_top + offs
        self.right_tower_pts = self.right_tower_top + offs

    @property
    def num_beam_nodes(self) -> int:
        """全梁节点数（两侧），等于 ``len(x_positions)``。"""
        return len(self.x_positions)

    def summary(self) -> str:
        """单行参数摘要，便于日志/调试。"""
        return (
            f"N={self.num_cables_per_side} span={self.span:.1f}m "
            f"side={self.side_span:.1f}m L={self.beam_length:.1f}m "
            f"H={self.anchor_height:.1f}m nodes={self.num_beam_nodes} "
            f"EI={self.beam_E * self.beam_Iz:.3e} wg={self.wg:.3e}N/m"
        )
