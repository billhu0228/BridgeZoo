import math

import pytest

from bridgezoo.fem.staged import StagedDirectSolver, build_staged_cantilever


def _tower_parts(plan):
    first = plan.steps[0]
    nodes = [node for node in first.new_nodes if node.role in {"tower_base", "tower", "anchor"}]
    frames = [frame for frame in first.new_frames if frame.group == "tower"]
    coords = {node.id: (node.x, node.y) for node in nodes}
    return nodes, frames, coords


def test_tower_mesh_keeps_anchors_and_interpolates_ei():
    plan = build_staged_cantilever(
        n_seg=2,
        anchor_base_height=4.0,
        anchor_spacing=2.0,
        anchor_top_free=4.0,
        tower_stiffness=[[0.0, 1.0e10], [10.0, 3.0e10]],
        tower_element_size=3.0,
        strands=[20, 20],
        pretension=[0.0, 0.0],
    )
    nodes, frames, coords = _tower_parts(plan)

    assert {node.id for node in nodes if node.role == "anchor"} == {301, 302}
    assert max(node.y for node in nodes) == 10.0
    assert len(frames) == len(nodes) - 1
    for frame in frames:
        z0, z1 = coords[frame.i][1], coords[frame.j][1]
        midpoint = 0.5 * (z0 + z1)
        assert z1 - z0 <= 3.0 + 1e-12
        assert frame.E * frame.I == pytest.approx(1.0e10 + 2.0e9 * midpoint)
        assert frame.udl_wy == 0.0


def test_tower_flexibility_changes_anchor_displacement():
    common = dict(
        n_seg=2,
        anchor_base_height=10.0,
        anchor_spacing=4.0,
        anchor_top_free=2.0,
        left_start=6.0,
        left_spacing=8.0,
        right_start=6.0,
        right_spacing=10.0,
        wg=5.0e4,
        strands=[20, 20],
        pretension=[(2.0e6, 0.5e6), (2.0e6, 0.5e6)],
        tower_element_size=2.0,
    )
    flexible = StagedDirectSolver().run(
        build_staged_cantilever(**common, tower_stiffness=[[0.0, 1.0e10]])
    )
    stiff = StagedDirectSolver().run(
        build_staged_cantilever(**common, tower_stiffness=[[0.0, 1.0e18]])
    )

    ux_flexible = flexible.records[-1].disp[302][0]
    ux_stiff = stiff.records[-1].disp[302][0]
    assert math.isfinite(ux_flexible)
    assert abs(ux_flexible) > 100.0 * abs(ux_stiff)
    assert set(flexible.anchor_ids) == {301, 302}
    assert 300 in flexible.tower_ids
    assert set(flexible.anchor_ids).issubset(flexible.tower_ids)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        ([], "at least one"),
        ([[0.0, 1.0e10], [0.0, 2.0e10]], "strictly increasing"),
        ([[0.0, -1.0]], "positive"),
    ],
)
def test_tower_stiffness_rejects_invalid_profiles(profile, message):
    with pytest.raises(ValueError, match=message):
        build_staged_cantilever(n_seg=2, tower_stiffness=profile)
