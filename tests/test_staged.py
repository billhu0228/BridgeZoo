import pytest

from bridgezoo.fem.staged import StagedDirectSolver, build_staged_cantilever

RIGHT_TIP, LEFT_TIP = 200, 201


def _plan(n=3, wg=5.0e4, **kw):
    pre = [2.0e6] * n
    return build_staged_cantilever(n_seg=n, wg=wg, strands=[20] * n, pretension=pre, **kw)


def test_direct_runs_and_records():
    n = 3
    r = StagedDirectSolver().run(_plan(n))
    assert len(r.records) == n + 2
    assert r.records[-2].label == "tip_free"
    assert r.records[-1].label == "closure_balance"
    hist = r.cable_stress_history()
    assert len(hist[1001]) == n + 2
    assert len(hist[1000 + n]) == 3
    assert set(r.anchor_ids) == {301, 302, 303}
    assert RIGHT_TIP in r.deck_ids and LEFT_TIP in r.deck_ids
    assert r.cable_nodes[1001] == (301, 1)
    assert r.cable_nodes[2001] == (301, 101)


def test_direct_stress_history_evolves():
    r = StagedDirectSolver().run(_plan(4))
    series = [s for _, s in r.cable_stress_history()[1001]]
    assert len(series) == 6
    assert max(series) - min(series) > 1e6
    assert all(s > 0 for s in series)


def test_half_bridge_boundary_conditions_are_applied():
    plan = _plan(4)
    supports = {nid: (ux, uy, rz) for nid, ux, uy, rz in plan.supports}
    assert supports[0] == (True, True, False)
    assert RIGHT_TIP not in supports
    assert LEFT_TIP not in supports

    r = StagedDirectSolver().run(plan)
    tip_free = r.records[-2]
    assert tip_free.label == "tip_free"
    assert abs(tip_free.disp[LEFT_TIP][1]) > 1e-12

    last = r.records[-1]
    assert last.label == "closure_balance"
    assert abs(last.disp[RIGHT_TIP][0]) < 1e-12
    assert abs(last.disp[RIGHT_TIP][2]) < 1e-12
    assert abs(last.disp[LEFT_TIP][1]) < 1e-12


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
    pytest.importorskip("openseespy", reason="requires openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="linear").run(plan)
    _assert_match(rd, ro, rtol=5e-3)


def test_staged_matches_opensees_corot():
    pytest.importorskip("openseespy", reason="requires openseespy")
    from bridgezoo.fem.staged import StagedOpenSeesSolver

    plan = _plan(4, wg=3.0e4)
    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element="corot").run(plan)
    _assert_match(rd, ro, rtol=1e-2)
