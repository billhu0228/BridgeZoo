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
