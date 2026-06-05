"""逐阶段施工求解器测试（对称双悬臂 + 扇面索）。

1. 自研直接刚度法的运行/记录与索力历程结构（不需 openseespy）。
2. OpenSees 交叉校核（importorskip）：同一施工计划，自研 vs OpenSees
   （linear 索单元逐项一致 < 0.5%；corot 几何精确 < 1%，小位移算例）。
"""

import math

import pytest

from bridgezoo.fem.staged import StagedDirectSolver, build_staged_cantilever

RIGHT_TIP, LEFT_TIP = 200, 201


def _plan(n=3, wg=5.0e4, **kw):
    pre = [2.0e6] * n
    return build_staged_cantilever(n_seg=n, wg=wg, strands=[20] * n, pretension=pre, **kw)


def test_direct_runs_and_records():
    n = 3
    r = StagedDirectSolver().run(_plan(n))
    # n 个 cable 记录 + 1 个 tip 记录
    assert len(r.records) == n + 1
    assert r.records[-1].label == "tip"
    hist = r.cable_stress_history()
    # 右侧首索(1001) 自 stage1 起，经 n 个 cable 步 + tip = n+1 条历程
    assert len(hist[1001]) == n + 1
    # 右侧第 n 索(1000+n) 仅在最后一个 cable 步与 tip 出现 = 2 条
    assert len(hist[1000 + n]) == 2
    # 几何信息已附带
    assert set(r.anchor_ids) == {301, 302, 303}
    assert RIGHT_TIP in r.deck_ids and LEFT_TIP in r.deck_ids
    assert r.cable_nodes[1001] == (301, 1)
    assert r.cable_nodes[2001] == (301, 101)


def test_direct_stress_history_evolves():
    """首索安装后，其应力随后续节段/张索而变化（历程非平凡，且全程受拉）。"""
    r = StagedDirectSolver().run(_plan(4))
    series = [s for _, s in r.cable_stress_history()[1001]]
    assert len(series) == 5  # n 个 cable 步 + tip
    assert max(series) - min(series) > 1e6  # 应力确有变化（>1 MPa）
    assert all(s > 0 for s in series)       # 全程受拉


def test_symmetric_plan_is_symmetric():
    """左右几何相同 → 两侧悬臂端挠度对称（自研，纯几何）。"""
    r = StagedDirectSolver().run(_plan(4))  # 默认左右对称
    last = r.records[-1]
    assert math.isclose(last.disp[RIGHT_TIP][1], last.disp[LEFT_TIP][1], rel_tol=1e-9, abs_tol=1e-9)


def _assert_match(rd, ro, rtol):
    hd = {c: v[-1][1] for c, v in rd.cable_stress_history().items()}
    ho = {c: v[-1][1] for c, v in ro.cable_stress_history().items()}
    scale = max(abs(v) for v in hd.values())
    for c in hd:
        assert abs(hd[c] - ho[c]) < rtol * scale
    for tip in (RIGHT_TIP, LEFT_TIP):
        da, db = rd.records[-1].disp[tip][1], ro.records[-1].disp[tip][1]
        assert abs(da - db) <= rtol * max(abs(da), abs(db), 1e-9)


def test_staged_matches_opensees_linear():
    """自研(线性) vs OpenSees(线性 Truss)，小位移下逐项吻合 < 0.5%。"""
    pytest.importorskip("openseespy", reason="需要 openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="linear").run(plan)
    _assert_match(rd, ro, rtol=5e-3)


def test_staged_matches_opensees_corot():
    """自研(线性) vs OpenSees(corotTruss 几何精确)，小位移下 < 1%。"""
    pytest.importorskip("openseespy", reason="需要 openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="corot").run(plan)
    _assert_match(rd, ro, rtol=1e-2)
