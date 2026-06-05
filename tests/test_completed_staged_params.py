from argparse import Namespace

from bridgezoo.fem.completed import CompletedDirectSolver
from bridgezoo.fem.staged import build_staged_cantilever
from scripts.plot_completed_balance import (
    MIDSPAN_NODE,
    TOWER_DECK_NODE,
    VERTICAL_SUPPORT_NODE,
    build_completed_from_staged_params,
)
from scripts.plot_staged_deck_growth import MODEL_DEFAULTS, default_pretension


def _args(**kw):
    data = dict(MODEL_DEFAULTS)
    data.update(strands=20)
    data.update(kw)
    return Namespace(**data)


def test_completed_uses_staged_model_geometry_defaults():
    args = _args()
    model, meta = build_completed_from_staged_params(args)
    n = args.n

    assert len(meta["anchor_ids"]) == n
    assert len(meta["deck_ids"]) == 2 * n + 3
    assert len(meta["cable_nodes"]) == 2 * n
    assert len(model.frames) == 2 * n + 2
    assert len(model.supports) == n + 3
    assert (model.supports[TOWER_DECK_NODE].ux, model.supports[TOWER_DECK_NODE].uy, model.supports[TOWER_DECK_NODE].rz) == (
        True,
        True,
        False,
    )
    assert (model.supports[MIDSPAN_NODE].ux, model.supports[MIDSPAN_NODE].uy, model.supports[MIDSPAN_NODE].rz) == (
        True,
        False,
        True,
    )
    assert (model.supports[VERTICAL_SUPPORT_NODE].ux, model.supports[VERTICAL_SUPPORT_NODE].uy, model.supports[VERTICAL_SUPPORT_NODE].rz) == (
        False,
        True,
        False,
    )
    assert not model.nodal_loads

    xs = [meta["coords"][nid][0] for nid in meta["deck_ids"]]
    assert min(xs) == -(args.left_start + (n - 1) * args.left_spacing + args.left_end)
    assert max(xs) == args.right_start + (n - 1) * args.right_spacing + args.right_end


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


def test_completed_projects_gravity_for_left_and_right_member_directions():
    args = _args(n=3)
    model, _ = build_completed_from_staged_params(args)

    assert model.member_udls[11].wy == -args.wg
    assert model.member_udls[111].wy == args.wg


def test_completed_solves_after_final_supports_are_active():
    args = _args(n=3)
    model, _ = build_completed_from_staged_params(args)
    result = CompletedDirectSolver().solve(model)

    assert result.converged
    assert abs(result.disp[TOWER_DECK_NODE][0]) < 1e-12
    assert abs(result.disp[TOWER_DECK_NODE][1]) < 1e-12
    assert abs(result.disp[VERTICAL_SUPPORT_NODE][1]) < 1e-12
    assert abs(result.disp[MIDSPAN_NODE][0]) < 1e-12
    assert abs(result.disp[MIDSPAN_NODE][2]) < 1e-12
