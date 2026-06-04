"""逐阶段施工求解器测试。

1. 自研直接刚度法的**自洽性**（不需 openseespy）：单段悬臂装索后，索力松弛量与端部
   位移、索刚度三者自洽。
2. **OpenSees 交叉校核**（importorskip）：同一施工计划，自研(线性) vs OpenSees(corot)，
   小位移下逐项吻合（< 1%）。
"""

import math

import numpy as np
import pytest

from bridgezoo.fem.staged import StagedDirectSolver, build_staged_cantilever


def _plan(n=3, **kw):
    wg = 4.7e5
    L, H = 8.0, 20.0
    pre = [wg * L * math.hypot(i * L, H) / H for i in range(1, n + 1)]
    return build_staged_cantilever(n_seg=n, seg_len=L, tower_height=H, wg=wg,
                                   strands=[20] * n, pretension=pre, **kw)


def test_direct_runs_and_records():
    r = StagedDirectSolver().run(_plan(3))
    assert len(r.records) == 3
    hist = r.cable_stress_history()
    # 最早安装的索应有最长历程
    assert len(hist[1001]) == 3
    assert len(hist[1003]) == 1


def test_direct_stress_history_monotone_first_cable():
    """首索随后续节段安装，应力单调上升（自重持续增加）。"""
    r = StagedDirectSolver().run(_plan(4))
    series = [s for _, s in r.cable_stress_history()[1001]]
    assert all(series[k + 1] >= series[k] - 1.0 for k in range(len(series) - 1))


def _plan_small():
    """小位移算例（线性≈几何精确），供两种后端交叉校核。"""
    n = 4
    L, H, wg = 8.0, 20.0, 5.0e4
    pre = [wg * L * math.hypot(i * L, H) / H for i in range(1, n + 1)]
    return n, build_staged_cantilever(n_seg=n, seg_len=L, tower_height=H, wg=wg,
                                      strands=[20] * n, pretension=pre)


def _assert_match(rd, ro, n, rtol):
    hd = {c: v[-1][1] for c, v in rd.cable_stress_history().items()}
    ho = {c: v[-1][1] for c, v in ro.cable_stress_history().items()}
    scale = max(abs(v) for v in hd.values())
    for c in hd:
        assert abs(hd[c] - ho[c]) < rtol * scale
    da = rd.records[-1].disp[n][1]
    db = ro.records[-1].disp[n][1]
    assert abs(da - db) <= rtol * max(abs(da), abs(db), 1e-9)


def test_staged_matches_opensees_linear():
    """同一计划：自研(线性) vs OpenSees(线性 Truss)，小位移下逐项吻合 < 0.5%。"""
    pytest.importorskip("openseespy", reason="需要 openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    n, plan = _plan_small()
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="linear").run(plan)
    _assert_match(rd, ro, n, rtol=5e-3)


def test_staged_matches_opensees_corot():
    """同一计划：自研(线性) vs OpenSees(corotTruss 几何精确)，小位移下 < 1%。"""
    pytest.importorskip("openseespy", reason="需要 openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    n, plan = _plan_small()
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="corot").run(plan)
    _assert_match(rd, ro, n, rtol=1e-2)
