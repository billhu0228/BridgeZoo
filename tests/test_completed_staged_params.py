"""分阶段构建器派生的成桥(completed)状态 + 预张力映射的回归测试。

仅依赖在用代码:``bridgezoo.fem.staged.build_staged_cantilever`` 与
``scripts.staged_analysis.default_pretension``。原先依赖 ``archive/`` 的
``build_completed_from_staged_params`` 的若干用例已移除——``archive/`` 按项目约定
不得在测试中导入,其在用替代 ``build_completed_model`` 的覆盖见
``tests/test_completed_direct.py``。
"""

import pytest

from bridgezoo.fem.staged import build_staged_cantilever
from scripts.staged_analysis import default_pretension


def test_staged_builder_creates_completed_state():
    n = 4
    plan = build_staged_cantilever(n_seg=n, strands=[20] * n, pretension=[1.0e6] * n)

    assert plan.completed is not None
    assert len(plan.completed.nodes) == 2 * n + 3 + n
    assert len(plan.completed.frames) == 2 * n + 2
    assert len(plan.completed.cables) == 2 * n
    assert len(plan.completed.supports) == n + 3
    assert not plan.completed.nodal_loads


def test_staged_builder_keeps_legacy_pair_pretension():
    plan = build_staged_cantilever(n_seg=2, strands=[20, 20], pretension=[1.0e6, 2.0e6])
    cables = {cb.id: cb.tension for cb in plan.completed.cables}

    assert cables[1001] == 1.0e6
    assert cables[2001] == 1.0e6
    assert cables[1002] == 2.0e6
    assert cables[2002] == 2.0e6


def test_staged_builder_accepts_independent_left_right_pretension_pairs():
    plan = build_staged_cantilever(
        n_seg=2,
        strands=[20, 20],
        pretension=[(1.0e6, 1.5e6), (2.0e6, 2.5e6)],
    )
    cables = {cb.id: cb.tension for cb in plan.completed.cables}

    assert cables[1001] == 1.0e6
    assert cables[2001] == 1.5e6
    assert cables[1002] == 2.0e6
    assert cables[2002] == 2.5e6


def test_staged_builder_accepts_flat_independent_pretension():
    plan = build_staged_cantilever(n_seg=2, strands=[20, 20], pretension=[1.0e6, 1.5e6, 2.0e6, 2.5e6])
    cables = {cb.id: cb.tension for cb in plan.completed.cables}

    assert cables[1001] == 1.0e6
    assert cables[2001] == 1.5e6
    assert cables[1002] == 2.0e6
    assert cables[2002] == 2.5e6


def test_staged_builder_accepts_independent_left_right_strands():
    strand_area = 1.4e-4
    plan = build_staged_cantilever(
        n_seg=2,
        strand_area=strand_area,
        strands=[(20, 18), (22, 16)],
        pretension=[1.0e6, 1.5e6],
    )
    areas = {cb.id: cb.A for cb in plan.completed.cables}

    assert areas[1001] == strand_area * 20
    assert areas[2001] == strand_area * 18
    assert areas[1002] == strand_area * 22
    assert areas[2002] == strand_area * 16


def test_staged_builder_rejects_non_integer_strands():
    with pytest.raises(ValueError, match="strands"):
        build_staged_cantilever(n_seg=2, strands=[20.5, 20])
    with pytest.raises(ValueError, match="strands"):
        build_staged_cantilever(n_seg=2, strands=[20 + 1j, 20])


def test_default_pretension_uses_left_and_right_geometry():
    pretension = default_pretension(
        2,
        anchor_base=32.0,
        anchor_spacing=2.0,
        left_start=12.0,
        left_spacing=8.0,
        right_start=12.0,
        right_spacing=12.0,
        wg=1.0e5,
    )
    plan = build_staged_cantilever(n_seg=2, strands=[20, 20], pretension=pretension)
    cables = {cb.id: cb.tension for cb in plan.completed.cables}

    assert len(pretension) == 2
    assert pretension[0][0] != pretension[0][1]
    assert cables[1001] == pretension[0][0]
    assert cables[2001] == pretension[0][1]
    assert cables[1002] == pretension[1][0]
    assert cables[2002] == pretension[1][1]
