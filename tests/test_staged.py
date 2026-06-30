import pytest

from bridgezoo.fem.staged import StagedDirectSolver, build_staged_cantilever

RIGHT_TIP, LEFT_TIP = 200, 201


def _plan(n=3, wg=5.0e4, **kw):
    pre = [2.0e6] * n
    return build_staged_cantilever(n_seg=n, wg=wg, strands=[20] * n, pretension=pre, **kw)


def test_first_increment_combines_segment_and_cable_only():
    plan = _plan(3)
    labels = [step.label for step in plan.steps]
    assert labels[:4] == ["cable1", "seg2", "cable2", "seg3"]

    first = plan.steps[0]
    assert first.record
    assert len(first.new_nodes) == 2
    assert len(first.new_frames) == 2
    assert len(first.new_cables) == 2

    second = plan.steps[1]
    assert second.label == "seg2"
    assert not second.record
    assert len(second.new_frames) == 2
    assert not second.new_cables


def test_direct_runs_and_records():
    n = 3
    r = StagedDirectSolver().run(_plan(n))
    assert len(r.records) == n + 1
    assert r.records[-1].label == "tip_free"
    hist = r.cable_stress_history()
    assert len(hist[1001]) == n + 1
    assert len(hist[1000 + n]) == 2
    assert set(r.anchor_ids) == {301, 302, 303}
    assert RIGHT_TIP in r.deck_ids and LEFT_TIP in r.deck_ids
    assert r.cable_nodes[1001] == (301, 1)
    assert r.cable_nodes[2001] == (301, 101)


def test_direct_stress_history_evolves():
    r = StagedDirectSolver().run(_plan(4))
    series = [s for _, s in r.cable_stress_history()[1001]]
    assert len(series) == 5
    assert max(series) - min(series) > 1e6
    assert all(s > 0 for s in series)


def _assert_tip_lock_in(result, n, left_end=4.0, right_end=4.0, tol=1e-10):
    """被锁自由度 = 切线诞生值(由附着节点位移外推,甲板 dy=0)。

    tol 取 1e-10:锁定是纯运动学复制,两后端都应在机器精度内保持诞生值,
    1e-10 m / rad 远小于任何物理位移量级,同时为浮点往返留余量。
    """
    prev, last = result.records[-2], result.records[-1]
    assert last.label == "tip_free"
    uyL, rzL = prev.disp[100 + n][1], prev.disp[100 + n][2]
    uxR, rzR = prev.disp[n][0], prev.disp[n][2]
    assert abs(last.disp[LEFT_TIP][1] - (uyL - left_end * rzL)) < tol
    assert abs(last.disp[RIGHT_TIP][0] - uxR) < tol
    assert abs(last.disp[RIGHT_TIP][2] - rzR) < tol


def test_plan_ends_with_tip_closure_supports():
    plan = _plan(3)
    last = plan.steps[-1]
    assert last.label == "tip_free"
    assert last.new_supports == [
        (LEFT_TIP, False, True, False),
        (RIGHT_TIP, True, False, True),
    ]
    assert all(not step.balance_dofs for step in plan.steps)


def test_half_bridge_boundary_conditions_are_applied():
    n = 4
    plan = _plan(n)
    supports = {nid: (ux, uy, rz) for nid, ux, uy, rz in plan.supports}
    assert supports[0] == (True, True, False)
    assert RIGHT_TIP not in supports
    assert LEFT_TIP not in supports

    r = StagedDirectSolver().run(plan)
    # 合龙支座把被锁自由度保持在切线诞生值(直接法为精确运动学,取 1e-12)
    _assert_tip_lock_in(r, n, tol=1e-12)

    # 锁定值本身非零(悬臂端诞生在变形切线上)
    last = r.records[-1]
    assert abs(last.disp[LEFT_TIP][1]) > 1e-12

    # 未锁自由度在端段自重下确实变化(右端 uy 偏离其切线诞生值)
    prev = r.records[-2]
    uy_birth_right = prev.disp[n][1] + 4.0 * prev.disp[n][2]
    assert abs(last.disp[RIGHT_TIP][1] - uy_birth_right) > 1e-9


def _assert_match(rd, ro, rtol, history_index=-1, record_index=-1):
    hd = {c: v[history_index][1] for c, v in rd.cable_stress_history().items()}
    ho = {c: v[history_index][1] for c, v in ro.cable_stress_history().items()}
    scale = max(abs(v) for v in hd.values())
    for c in hd:
        assert abs(hd[c] - ho[c]) < rtol * scale
    da_rec, db_rec = rd.records[record_index], ro.records[record_index]
    nodes = set(da_rec.disp) & set(db_rec.disp)
    y_scale = max(max(abs(da_rec.disp[nid][1]), abs(db_rec.disp[nid][1]), 1e-9) for nid in nodes)
    for nid in nodes:
        assert abs(da_rec.disp[nid][1] - db_rec.disp[nid][1]) <= rtol * y_scale


def test_staged_matches_opensees_linear():
    pytest.importorskip("openseespy", reason="requires openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="linear").run(plan)
    _assert_match(rd, ro, rtol=5e-3)
    _assert_tip_lock_in(ro, 4)


def test_staged_matches_opensees_corot():
    pytest.importorskip("openseespy", reason="requires openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="corot").run(plan)
    _assert_match(rd, ro, rtol=1e-2)
    _assert_tip_lock_in(ro, 4)


# ===================================================== 二期恒载（dw / phase2）

def test_phase2_step_only_when_dw_nonzero():
    """dw=0(默认)不追加阶段——既有行为逐字节不变;dw>0 末步为 phase2,施加于全主梁。"""
    plan0 = _plan(3)
    assert all(step.label != "phase2" for step in plan0.steps)
    assert all(not step.member_loads for step in plan0.steps)

    dw = 2.0e4
    plan = _plan(3, dw=dw)
    last = plan.steps[-1]
    assert last.label == "phase2"
    assert last.record
    # 二期荷载覆盖全部主梁单元(各节段 + 合龙端段),且 wy=-dw(向下,与自重同约定)
    girder_ids = {fr.id for step in plan.steps for fr in step.new_frames}
    assert {ml.member for ml in last.member_loads} == girder_ids
    assert all(ml.wy == -dw for ml in last.member_loads)
    # phase2 不引入新构件(纯加载步)
    assert not last.new_frames and not last.new_cables and not last.new_nodes


def test_phase2_adds_record_and_deflects_deck_down():
    n, dw = 4, 3.0e4
    r0 = StagedDirectSolver().run(_plan(n))
    rdw = StagedDirectSolver().run(_plan(n, dw=dw))
    # 多出一条 phase2 记录;dw=0 时记录数与既有断言一致
    assert r0.records[-1].label == "tip_free"
    assert rdw.records[-1].label == "phase2"
    assert len(rdw.records) == len(r0.records) + 1

    # phase2 相对其前一步(tip_free)的增量纯为二期荷载效应:主梁整体下挠。
    tip_free, phase2 = rdw.records[-2], rdw.records[-1]
    assert tip_free.label == "tip_free"
    assert phase2.disp[RIGHT_TIP][1] < tip_free.disp[RIGHT_TIP][1] - 1e-6


def test_phase2_delta_matches_completed_one_shot():
    """dw 进入「分阶段」与「成桥一次成型」两条路径的增量应逐位一致(机器精度)。

    分阶段的 phase2 增量 = 在已建成结构上施加 dw 的 K^-1·F_dw;成桥模型 dw>0 与 dw=0
    之差亦为同一 K^-1·F_dw(最终结构刚度与等效荷载完全相同)。比较「增量差」可绕开
    分阶段 vs 成桥本身固有的施工历史差异,得到机器精度等价的护栏。容差 1e-8(相对最大
    增量)远紧于物理量级,仅为两套求解器装配次序的浮点舍入留余量。
    """
    from bridgezoo.fem.staged import build_completed_model

    n, dw = 4, 3.0e4
    rdw = StagedDirectSolver().run(_plan(n, dw=dw))
    tip_free, phase2 = rdw.records[-2], rdw.records[-1]
    assert tip_free.label == "tip_free" and phase2.label == "phase2"
    staged_delta = {nid: phase2.disp[nid][1] - tip_free.disp[nid][1] for nid in phase2.disp}

    from bridgezoo.fem.completed import CompletedDirectSolver

    d0 = CompletedDirectSolver().solve(build_completed_model(_plan(n, dw=0.0))[0]).disp
    ddw = CompletedDirectSolver().solve(build_completed_model(_plan(n, dw=dw))[0]).disp
    completed_delta = {nid: ddw[nid][1] - d0[nid][1] for nid in ddw}

    common = set(staged_delta) & set(completed_delta)
    assert common
    scale = max(max(abs(staged_delta[nid]), 1e-9) for nid in common)
    for nid in common:
        assert abs(staged_delta[nid] - completed_delta[nid]) < 1e-8 * scale


def test_phase2_batch_matches_scalar():
    """含二期荷载的计划在批量多右端路径与逐个标量求解逐位一致(member_loads 恒定→结构级)。"""
    from bridgezoo.fem.staged import StagedDirectBatchSolver

    def mk(pre):
        return build_staged_cantilever(n_seg=3, wg=5.0e4, dw=2.0e4,
                                       strands=[20] * 3, pretension=[pre] * 3)

    plans = [mk(1.0e6), mk(2.5e6)]
    batch = StagedDirectBatchSolver().run_batch(plans)
    scalar = [StagedDirectSolver().run(p) for p in plans]
    for rs, rb in zip(scalar, batch):
        assert rs.records[-1].label == "phase2"
        for nid in rs.records[-1].disp:
            assert rb.records[-1].disp[nid] == pytest.approx(rs.records[-1].disp[nid], rel=1e-9, abs=1e-11)


def test_phase2_matches_opensees_linear():
    pytest.importorskip("openseespy", reason="requires openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4, dw=2.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="linear").run(plan)
    assert rd.records[-1].label == "phase2"
    _assert_match(rd, ro, rtol=5e-3)
