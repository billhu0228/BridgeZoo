"""自研直接刚度法求解器测试。

两类：
1. **解析解对比**（不需 openseespy，任何解释器可跑）：简支梁/悬臂梁经典挠度。
2. **OpenSees 交叉校核**（``importorskip`` 门控）：同一 StructuralModel，两后端结果一致。
"""

import numpy as np
import pytest

from bridgezoo.fem.model import StructuralModel
from bridgezoo.fem.completed import CompletedDirectSolver


E = 2.0e11
I = 1.0e-4
A = 1.0e-2


def _solve(model):
    return CompletedDirectSolver().solve(model)


# ----------------------------------------------------------- 解析解对比
def test_cantilever_point_load():
    """悬臂梁端点集中力：δ = P L^3 / (3 EI)。"""
    L, P = 5.0, 1000.0
    m = StructuralModel("cantilever_P")
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_frame(1, 1, 2, E, A, I)
    m.add_support(1, True, True, True)
    m.add_nodal_load(2, fy=-P)
    r = _solve(m)
    expected = -P * L ** 3 / (3 * E * I)
    assert np.isclose(r.uy(2), expected, rtol=1e-6)


def test_cantilever_udl():
    """悬臂梁均布荷载：δ = w L^4 / (8 EI)。"""
    L, w = 5.0, 1000.0
    m = StructuralModel("cantilever_w")
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L, 0.0)
    m.add_frame(1, 1, 2, E, A, I)
    m.add_support(1, True, True, True)
    m.add_member_udl(1, -w)  # 向下
    r = _solve(m)
    expected = -w * L ** 4 / (8 * E * I)
    assert np.isclose(r.uy(2), expected, rtol=1e-6)


def test_simply_supported_udl():
    """简支梁均布荷载：跨中 δ = 5 w L^4 / (384 EI)（FE 节点值精确）。"""
    L, w = 10.0, 1000.0
    m = StructuralModel("ss_w")
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, L / 2, 0.0)
    m.add_node(3, L, 0.0)
    m.add_frame(1, 1, 2, E, A, I)
    m.add_frame(2, 2, 3, E, A, I)
    m.add_support(1, ux=True, uy=True)   # 铰
    m.add_support(3, uy=True)            # 滚动
    m.add_member_udl(1, -w)
    m.add_member_udl(2, -w)
    r = _solve(m)
    expected = -5 * w * L ** 4 / (384 * E * I)
    assert np.isclose(r.uy(2), expected, rtol=1e-6)


def test_cable_pretension_between_fixed_nodes():
    """两端固定的索：无伸长，轴力 = 预张力。"""
    m = StructuralModel("cable")
    m.add_node(1, 0.0, 0.0)
    m.add_node(2, 3.0, 4.0)  # 长度 5
    m.add_cable(1, 1, 2, E, A, pretension=1.0e5)
    m.add_support(1, True, True, True)
    m.add_support(2, True, True, True)
    r = _solve(m)
    assert np.isclose(r.cable_force[1], 1.0e5, rtol=1e-9)


# ----------------------------------------------------------- OpenSees 交叉校核
def test_cable_bridge_matches_opensees():
    """同一斜拉桥 StructuralModel：直接刚度法 vs OpenSees，逐项一致。

    成桥模型由施工计划派生（单塔双悬臂半桥），与施工阶段模型同源。
    """
    pytest.importorskip("openseespy", reason="需要 openseespy")
    from bridgezoo.fem.completed import CompletedOpenSeesSolver
    from bridgezoo.fem.staged import build_completed_model, build_staged_cantilever

    n, strand_area, strands = 6, 1.4e-4, 20
    pretension = 600.0 * 1e6 * strand_area * strands  # 初应力 600 MPa → 预张力
    plan = build_staged_cantilever(
        n_seg=n,
        strand_area=strand_area,
        strands=[strands] * n,
        pretension=[pretension] * n,
    )
    model, _ = build_completed_model(plan)

    rd = CompletedDirectSolver().solve(model)
    ro = CompletedOpenSeesSolver().solve(model)

    # 竖向位移
    for nid in model.nodes:
        assert np.isclose(rd.uy(nid), ro.uy(nid), atol=1e-9, rtol=1e-6)
    # 索力（按量级相对）
    scale = max(abs(v) for v in rd.cable_force.values())
    for cid in model.cables:
        assert abs(rd.cable_force[cid] - ro.cable_force[cid]) < 1e-6 * scale
