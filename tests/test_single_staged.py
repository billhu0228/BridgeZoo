import numpy as np
import pytest

from bridgezoo.fem import single_staged, staged
from bridgezoo.optim import CableBounds, CableDesignEvaluator, CableOptimizationProblem
from scripts import optimize_cables, staged_analysis
from scripts.bridge_config import model_family_for_bridge_type, staged_api_for_bridge_type
from scripts.validate_staged import _stage_tip_y_errors


def _plan(module):
    return module.build_staged_cantilever(
        n_seg=2,
        wg=5.0e4,
        strands=[20, 20],
        pretension=[1.0e6, 1.5e6],
    )


def test_single_staged_is_an_independent_module_copy():
    assert single_staged.StagedPlan is not staged.StagedPlan
    assert single_staged.StagedDirectSolver is not staged.StagedDirectSolver
    assert single_staged.StagedPlan.__module__ == "bridgezoo.fem.single_staged.plan"
    assert single_staged.build_staged_cantilever.__module__ == "bridgezoo.fem.single_staged.builder"


def test_single_staged_has_fixed_backstays_and_one_right_girder_segment():
    n = 3
    plan = single_staged.build_staged_cantilever(
        n_seg=n,
        wg=5.0e4,
        dw=2.0e4,
        strands=[20] * n,
        pretension=[1.0e6] * n,
    )

    assert [step.label for step in plan.steps] == [
        "seg1",
        "cable1",
        "cable2",
        "cable3",
        "tip_free",
    ]
    supports = {node: (ux, uy, rz) for node, ux, uy, rz in plan.supports}
    assert supports[1] == (True, True, True)
    assert all(supports[400 + i] == (True, True, True) for i in range(1, n + 1))

    first = plan.steps[0]
    assert {node.id for node in first.new_nodes if node.role == "deck"} == {1, 101}
    assert {frame.id for frame in first.new_frames if frame.group == "deck"} == {11, 111}
    assert not first.new_cables

    cable_steps = plan.steps[1:-1]
    right_cables = [
        cable
        for step in cable_steps
        for cable in step.new_cables
        if cable.id < 2000
    ]
    left_cables = [
        cable
        for step in cable_steps
        for cable in step.new_cables
        if cable.id >= 2000
    ]
    assert [(cable.id, cable.i, cable.j) for cable in right_cables] == [
        (1000 + i, 300 + i, 400 + i) for i in range(1, n + 1)
    ]
    assert [(cable.id, cable.i, cable.j) for cable in left_cables] == [
        (2000 + i, 300 + i, 100 + i) for i in range(1, n + 1)
    ]
    assert [
        frame.id
        for step in plan.steps
        for frame in step.new_frames
        if 10 <= frame.id < 100
    ] == [11]

    last = plan.steps[-1]
    assert last.label == "tip_free"
    assert [node.id for node in last.new_nodes] == [201]
    assert [frame.id for frame in last.new_frames] == [190]
    assert not last.new_supports
    assert all(not step.member_loads for step in plan.steps)

    completed, meta = single_staged.build_completed_model(plan)
    assert set(meta["deck_ids"]) == {0, 1, 101, 102, 103, 201}
    assert all(400 + i not in meta["deck_ids"] for i in range(1, n + 1))
    assert completed.cables[1001].j == 401
    assert completed.supports[1].ux and completed.supports[1].uy and completed.supports[1].rz
    assert completed.supports[401].ux and completed.supports[401].uy and completed.supports[401].rz


def test_single_staged_solver_keeps_right_nodes_fixed_and_excludes_ground_anchors_from_deck():
    n = 3
    result = single_staged.StagedDirectSolver().run(
        single_staged.build_staged_cantilever(
            n_seg=n,
            wg=5.0e4,
            strands=[20] * n,
            pretension=[1.0e6] * n,
        )
    )

    assert [record.label for record in result.records] == [
        "seg1",
        "cable1",
        "cable2",
        "cable3",
        "tip_free",
    ]
    assert set(result.deck_ids) == {0, 1, 101, 102, 103, 201}
    assert set(result.anchor_ids) == {301, 302, 303}
    assert all(400 + i not in result.deck_ids for i in range(1, n + 1))

    for record in result.records:
        assert record.disp[1] == pytest.approx((0.0, 0.0, 0.0), abs=1e-14)
        for i in range(1, n + 1):
            support = 400 + i
            if support in record.disp:
                assert record.disp[support] == pytest.approx((0.0, 0.0, 0.0), abs=1e-14)

    rows = _stage_tip_y_errors(result, result, bridge_type="single")
    assert rows[0]["node"] == 101
    assert rows[-1]["node"] == 201


def test_optimizer_can_target_single_staged_family():
    problem = CableOptimizationProblem(
        n_seg=2,
        model_kwargs={"wg": 5.0e4},
        bounds=CableBounds(strand_min=1, strand_max=60),
        model_family="single_staged",
    )
    evaluator = CableDesignEvaluator(problem)
    plan = evaluator.build_plan(np.array([20, 20, 20, 20]), np.zeros(4))

    assert type(plan).__module__ == "bridgezoo.fem.single_staged.plan"
    assert evaluator.run_solver(plan).backend == "direct"


def test_canonical_scripts_follow_bridge_type_and_keep_current_defaults():
    normal_args = staged_analysis.parse_args(["--bridge", "model", "--render", "text"])
    single_args = staged_analysis.parse_args(["--bridge", "omo", "--render", "text"])
    optimize_args = optimize_cables.parse_args(["--bridge", "model", "--n", "2"])

    normal_builder, _, _ = staged_api_for_bridge_type(normal_args.bridge_defaults["bridge_type"])
    single_builder, _, _ = staged_api_for_bridge_type(single_args.bridge_defaults["bridge_type"])
    assert normal_builder.__module__ == "bridgezoo.fem.staged.builder"
    assert single_builder.__module__ == "bridgezoo.fem.single_staged.builder"
    assert normal_args.bridge_defaults["n"] == 6
    assert model_family_for_bridge_type(optimize_args.bridge_defaults["bridge_type"]) == "staged"
    assert optimize_args.n == 2
