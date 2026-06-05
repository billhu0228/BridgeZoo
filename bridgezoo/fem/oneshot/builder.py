"""由 :class:`bridgezoo.envs.geometry.BridgeGeometry` 构建与求解器无关的结构模型。

这是"**一套结构定义**"的落点：调用一次得到 :class:`StructuralModel`，即可分别交给
直接刚度法或 OpenSees 后端求解并对比。节点/单元编号沿用历史约定（见 geometry.py），
便于与 :func:`bridgezoo.fem.oneshot.opensees_ref.build_oneshot_fem` 对照。
"""

from __future__ import annotations

from bridgezoo.fem.model import StructuralModel


def build_cable_bridge(geometry, cable_sigma, cable_sizes) -> StructuralModel:
    """构建一次成桥（完成态）斜拉桥结构模型。

    Parameters
    ----------
    geometry : BridgeGeometry
    cable_sigma : Sequence[float]
        各索初应力 (MPa)，长度 N=num_cables_per_side。
    cable_sizes : Sequence[float|int]
        各索股数，长度 N。

    Returns
    -------
    StructuralModel
        含梁/索单元、约束、各梁单元自重均布荷载、各索预张力。
    """
    N = geometry.num_cables_per_side
    nbp = geometry.num_beam_points  # = N + 3
    Es = geometry.cable_Es
    As = geometry.strand_area

    m = StructuralModel(name=f"cable_bridge_N{N}")

    # 梁节点（1..2N+5）+ 梁单元（1..2N+4）
    xs = geometry.x_positions
    for i, x in enumerate(xs):
        m.add_node(i + 1, float(x), 0.0)
    for i in range(len(xs) - 1):
        m.add_frame(i + 1, i + 1, i + 2, geometry.beam_E, geometry.beam_area, geometry.beam_Iz)
        m.add_member_udl(i + 1, -geometry.wg)  # 自重（局部横向向下）

    # 索塔锚点 + 拉索
    def cable_pretension(sigma_mpa, ns):
        area = As * ns
        return sigma_mpa * 1e6 * area, area

    for i in range(N // 2):
        lx, ly = float(geometry.left_tower_pts[i, 0]), float(geometry.left_tower_pts[i, 1])
        rx, ry = float(geometry.right_tower_pts[i, 0]), float(geometry.right_tower_pts[i, 1])
        m.add_node(1001 + i, lx, ly)
        m.add_node(3001 + i, rx, ry)
        m.add_support(1001 + i, True, True, True)
        m.add_support(3001 + i, True, True, True)

        beam_index_left = i + 2
        beam_index_right = nbp + 1

        # 左塔两根、右塔两根；股数/应力按历史索引规则
        T1, A1 = cable_pretension(cable_sigma[i], cable_sizes[i])
        T2, A2 = cable_pretension(cable_sigma[N - 1 - i], cable_sizes[N - 1 - i])
        m.add_cable(1001 + i, 1001 + i, beam_index_left, Es, A1, T1)
        m.add_cable(2001 + i, 1001 + i, nbp - 1 - i, Es, A2, T2)
        m.add_cable(3001 + i, 3001 + i, beam_index_right + i, Es, A1, T1)
        m.add_cable(4001 + i, 3001 + i, nbp * 2 - 2 - i, Es, A2, T2)

    # 边界条件（与 opensees_ref.FEM.opensees 一致）
    m.add_support(1, ux=False, uy=True, rz=False)               # 桥台
    m.add_support(N // 2 + 2, ux=False, uy=True, rz=False)      # 索塔处梁
    m.add_support(N + 3, ux=True, uy=False, rz=True)            # 跨中
    m.add_support(N // 2 * 3 + 4, ux=False, uy=True, rz=False)  # 索塔处梁
    m.add_support(2 * N + 5, ux=False, uy=True, rz=False)       # 桥台

    return m
