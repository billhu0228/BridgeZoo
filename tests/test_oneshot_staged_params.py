from argparse import Namespace

from bridgezoo.fem.linear_frame import DirectStiffnessSolver
from scripts.plot_oneshot_balance import (
    MIDSPAN_NODE,
    TOWER_DECK_NODE,
    VERTICAL_SUPPORT_NODE,
    build_oneshot_from_staged_params,
)
from scripts.plot_staged_deck_growth import MODEL_DEFAULTS


def _args(**kw):
    data = dict(MODEL_DEFAULTS)
    data.update(strands=20)
    data.update(kw)
    return Namespace(**data)


def test_oneshot_uses_staged_model_geometry_defaults():
    args = _args()
    model, meta = build_oneshot_from_staged_params(args)
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


def test_oneshot_projects_gravity_for_left_and_right_member_directions():
    args = _args(n=3)
    model, _ = build_oneshot_from_staged_params(args)

    assert model.member_udls[11].wy == -args.wg
    assert model.member_udls[111].wy == args.wg


def test_oneshot_solves_after_final_supports_are_active():
    args = _args(n=3)
    model, _ = build_oneshot_from_staged_params(args)
    result = DirectStiffnessSolver().solve(model)

    assert result.converged
    assert abs(result.disp[TOWER_DECK_NODE][0]) < 1e-12
    assert abs(result.disp[TOWER_DECK_NODE][1]) < 1e-12
    assert abs(result.disp[VERTICAL_SUPPORT_NODE][1]) < 1e-12
    assert abs(result.disp[MIDSPAN_NODE][0]) < 1e-12
    assert abs(result.disp[MIDSPAN_NODE][2]) < 1e-12
