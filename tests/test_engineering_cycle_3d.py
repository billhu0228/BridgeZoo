import json
import math
from dataclasses import asdict, replace
from io import StringIO

import numpy as np
import pytest

from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator3D,
    CableOptimizationProblem,
)
from bridgezoo.optim.engineering_cycle3d import (
    EngineeringCableCycleOptimizer3D,
    EngineeringCycleOptions3D,
    EngineeringProgress3D,
    EngineeringStageStatus3D,
)
from bridgezoo.optim.problem import ObjectiveWeights
from bridgezoo.optim.single_staged3d import StageAControlResponse3D
from scripts.bridge_config import load_single_staged_3d_config
from scripts.optimize_cables_3d_engineering import (
    _EngineeringProgressDisplay,
    main as engineering_cli,
)
from scripts.single_staged_3d import main as single_staged_cli


def _direct_optimizer(n_seg=2, options=None):
    config = replace(load_single_staged_3d_config("omo3d"), n_seg=n_seg)
    model_kwargs = asdict(config)
    for name in (
        "strands_per_cable",
        "pretension_per_cable",
        "pretension_a_ratio",
    ):
        model_kwargs.pop(name)
    problem = CableOptimizationProblem(
        n_seg=n_seg,
        model_kwargs=model_kwargs,
        bounds=CableBounds(
            strand_min=5,
            strand_max=500,
            tension_bound_stress_mpa=1600.0,
        ),
        weights=ObjectiveWeights(),
        strand_area=config.strand_area,
        backend="direct",
        model_family="single_staged_3d",
    )
    return EngineeringCableCycleOptimizer3D(
        CableDesignEvaluator3D(problem, config), options
    )


def test_engineering_cycle_writes_restartable_reproducible_design(tmp_path):
    pytest.importorskip("openseespy.opensees")
    out_dir = tmp_path / "engineering"
    common = [
        "--bridge",
        "omo3d",
        "--n",
        "1",
        "--cycles",
        "1",
        "--quiet",
        "--out",
        str(out_dir),
    ]

    assert engineering_cli(common) == 0
    design_path = out_dir / "best_design.json"
    checkpoint_path = out_dir / "engineering_checkpoint.json"
    design = json.loads(design_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))

    assert design["schema"] == "bridgezoo.cable_optimization_3d.v3"
    assert design["algorithm"] == "engineering_final_state_feedback_cycle"
    assert "camber_profile" not in design["engineering_cycle"]
    assert design["engineering_cycle"]["final_state_target"] == {
        "stage_label": "secondary_load",
        "deck_uz_mm": 0.0,
        "tower_anchor_dx_mm": 0.0,
        "cable_stress_mpa": 500.0,
    }
    assert design["final_state_controls"]["stage_label"] == "secondary_load"
    assert "deck_uz_mm" in design["final_state_controls"]
    assert "deck_errors_mm" not in design["final_state_controls"]
    assert design["engineering_cycle"]["strands_before_sizing"] == [100, 100]
    assert [item["phase"] for item in design["engineering_cycle"]["substage_controls"]] == [
        "A",
        "B",
    ]
    first_control = design["engineering_cycle"]["substage_controls"][0]
    assert first_control["normalized_local_residual_after"] < (
        first_control["normalized_local_residual_before"]
    )
    assert first_control[
        "FEM_cases_for_influence_matrix"
    ] == 0
    assert design["search"]["OpenSees_FEM_cases_this_cycle"] in (3, 4)
    assert design["engineering_cycle"]["tension_update_accepted"] is True
    assert design["engineering_cycle"]["strand_update_attempted"] is True
    assert design["engineering_cycle"]["strand_update_accepted"] is True
    assert design["engineering_cycle"]["update_accepted"] is True
    assert design["engineering_cycle"]["strand_count_parameterization"] == {
        "family": "independent-per-group",
        "outward_non_decreasing": False,
        "stress_priority": True,
        "displacement_rebalanced_next_cycle": True,
    }
    assert checkpoint["schema"] == "bridgezoo.engineering_cable_cycle_3d.v11"
    assert np.asarray(
        checkpoint["state"]["feedback_step_scale_per_phase_group"]
    ).shape == (2, 2)
    assert np.asarray(
        checkpoint["state"]["strand_step_scale_per_group"]
    ).shape == (2,)
    assert design["engineering_cycle"]["tension_step_memory"][
        "nominal_equivalent_stress_MPa"
    ] == pytest.approx(30.0)
    assert len(design["engineering_cycle"]["strand_step_memory"]["next"]) == 2
    assert all(
        "next_strand_step_scale" in group for group in design["cable_groups"]
    )
    assert checkpoint["completed_cycles"] == 1
    assert checkpoint["cumulative_FEM_cases"] == design["search"][
        "OpenSees_FEM_cases_this_cycle"
    ]
    assert len(checkpoint["history"]) == 1

    replay_path = tmp_path / "replayed.json"
    assert single_staged_cli(
        [
            "--bridge",
            "omo3d",
            "--backend",
            "opensees",
            "--design",
            str(design_path),
            "--render",
            "none",
            "--output",
            str(replay_path),
        ]
    ) == 0
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    for group in design["cable_groups"]:
        for cable_id, stress_mpa in group["physical_final_stress_MPa"].items():
            assert replay["final"]["cable_stress_Pa"][cable_id] / 1.0e6 == pytest.approx(
                stress_mpa,
                rel=1.0e-10,
                abs=1.0e-10,
            )

    # Existing v4 engineering state remains restartable after the two-round
    # architecture upgrade; only the valid numerical state is migrated.
    checkpoint["schema"] = "bridgezoo.engineering_cable_cycle_3d.v4"
    checkpoint["problem"]["optimizer_architecture"] = (
        "restartable_current_stage_actual_displacement_feedback_"
        "without_influence_matrices"
    )
    checkpoint["settings"].pop("strand_max_change_per_cycle")
    checkpoint["settings"]["feedback_relaxation"] = 0.5
    checkpoint["settings"]["tension_step_fraction"] = 0.08
    checkpoint["state"].pop("feedback_step_scale_per_phase_group")
    checkpoint["state"].pop("strand_step_scale_per_group")
    checkpoint["state"]["feedback_step_scale"] = 0.0625
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2), encoding="utf-8"
    )
    assert engineering_cli([*common, "--resume"]) == 0
    resumed = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert resumed["completed_cycles"] == 2
    assert resumed["cumulative_FEM_cases"] == 6
    assert len(resumed["history"]) == 2


def test_engineering_cycle_resume_rejects_changed_targets(tmp_path):
    pytest.importorskip("openseespy.opensees")
    out_dir = tmp_path / "engineering"
    common = [
        "--bridge",
        "omo3d",
        "--n",
        "1",
        "--quiet",
        "--out",
        str(out_dir),
    ]
    assert engineering_cli(common) == 0

    with pytest.raises(ValueError, match="targets or update settings differ"):
        engineering_cli([*common, "--resume", "--target-stress-mpa", "550"])


def test_engineering_cycle_uses_activation_a_and_final_zero_target(tmp_path):
    pytest.importorskip("openseespy.opensees")
    out_dir = tmp_path / "engineering_n2"
    assert engineering_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "2",
            "--quiet",
            "--out",
            str(out_dir),
        ]
    ) == 0
    design = json.loads((out_dir / "best_design.json").read_text(encoding="utf-8"))
    controls = design["engineering_cycle"]["substage_controls"]
    stage2_a = next(
        item for item in controls if item["stage"] == 2 and item["phase"] == "A"
    )
    stage1_b = next(
        item for item in controls if item["stage"] == 1 and item["phase"] == "B"
    )

    assert "tangent birth" in stage2_a["displacement_basis"]
    assert "final secondary_load" in stage1_b["displacement_basis"]
    assert math.isfinite(stage2_a["response_before"]["deck_uz_mm"])
    assert stage2_a["normalized_local_residual_after"] < (
        stage2_a["normalized_local_residual_before"]
    )
    assert stage1_b["deck_target_is_soft"] is False
    assert stage1_b["target_deck_uz_mm"] == pytest.approx(0.0)
    assert stage1_b["target_tower_dx_mm"] == pytest.approx(0.0)


def test_engineering_cycle_uses_separate_tension_and_sizing_replays():
    optimizer = _direct_optimizer(n_seg=2)
    result = optimizer.run_cycle(
        np.full(4, 100, dtype=int),
        cycle_index=1,
        pretension_a=np.full(4, 3.0e6),
        pretension_b=np.full(4, 3.0e6),
    )

    assert result.fem_cases in (3, 4)
    assert result.tension_update_accepted is True
    assert result.strand_update_attempted is True
    if result.strand_update_accepted:
        assert result.strand_score_after < result.strand_score_before
    assert np.array_equal(
        result.evaluation_before_sizing.design.strands,
        result.strands_before_sizing,
    )
    assert np.array_equal(
        result.evaluation_after_sizing.design.strands,
        result.strands_after_sizing,
    )
    assert np.max(
        np.abs(result.strands_after_sizing - result.strands_before_sizing)
    ) <= optimizer.options.strand_max_change_per_cycle
    assert result.local_score_after_tension < result.local_score_before
    assert result.stress_after_sizing_mpa.shape == (4,)

    next_result = optimizer.run_cycle(
        result.strands_after_sizing,
        cycle_index=2,
        pretension_a=result.pretension_a,
        pretension_b=result.pretension_b,
        strand_step_scale=result.next_strand_step_scales,
        baseline_evaluation=result.evaluation_after_sizing,
    )
    assert next_result.fem_cases in (2, 3)
    assert next_result.local_score_before == pytest.approx(result.local_score_after)
    assert next_result.local_score_after_tension <= next_result.local_score_before


def test_engineering_b_response_is_final_secondary_load_displacement():
    optimizer = _direct_optimizer(n_seg=2)
    result = optimizer.run_cycle(
        np.full(4, 100, dtype=int),
        cycle_index=1,
        pretension_a=np.full(4, 3.0e6),
        pretension_b=np.full(4, 3.0e6),
    )
    plan = optimizer.evaluator.build_plan(
        result.evaluation_before_sizing.design.strands,
        result.evaluation_before_sizing.design.pretension,
        result.evaluation_before_sizing.design.pretension_a_ratio,
    )
    stage2_main_stays = [
        cable
        for cable in plan.model.cables.values()
        if cable.construction_stage == 2 and cable.group == "main_stay"
    ]
    deck_nodes = {cable.j for cable in stage2_main_stays}
    final_record = result.evaluation_before_sizing.staged_result.final
    assert final_record.stage_label == "secondary_load"
    actual_uz = np.mean(
        [final_record.displacement[node_id][2] for node_id in deck_nodes]
    )
    stage2_b_control = next(
        control
        for control in result.controls
        if control.construction_stage == 2 and control.phase == "B"
    )

    assert (
        stage2_b_control.response_after_tension.main_stay_deck_uz_m
        == pytest.approx(actual_uz, abs=1.0e-12)
    )
    assert stage2_b_control.response_after_tension.stage_index == final_record.stage_index

    responses, final_stress = optimizer._local_stage_responses(
        result.evaluation_before_sizing
    )
    assert all(
        responses[stage, "B"].stage_index == final_record.stage_index
        for stage in range(1, 3)
    )
    for stage in range(1, 3):
        for local_index, group in enumerate(("backstay", "main_stay")):
            cables = [
                cable
                for cable in plan.model.cables.values()
                if cable.construction_stage == stage and cable.group == group
            ]
            expected = np.mean(
                [final_record.cable_stress[cable.id] for cable in cables]
            ) / 1.0e6
            assert final_stress[2 * (stage - 1) + local_index] == pytest.approx(
                expected, abs=1.0e-12
            )


def test_engineering_cycle_prioritizes_stress_and_bounds_accelerated_sizing():
    optimizer = _direct_optimizer(n_seg=4)
    strands = np.full(8, 100, dtype=int)
    pretension_a = np.full(8, 3.0e6)
    pretension_b = np.full(8, 3.0e6)
    baseline = None
    step_scale = 1.0
    strand_step_scale = 1.0
    results = []
    for cycle in range(1, 6):
        result = optimizer.run_cycle(
            strands,
            cycle_index=cycle,
            pretension_a=pretension_a,
            pretension_b=pretension_b,
            step_scale=step_scale,
            strand_step_scale=strand_step_scale,
            baseline_evaluation=baseline,
        )
        results.append(result)
        strands = result.strands_after_sizing
        pretension_a = result.pretension_a
        pretension_b = result.pretension_b
        baseline = result.evaluation_after_sizing
        step_scale = result.next_tension_step_scales
        strand_step_scale = result.next_strand_step_scales

    assert all(
        result.local_score_after_tension
        <= result.local_score_before + 1.0e-12
        for result in results
    )
    assert all(
        result.strand_score_after
        <= result.strand_score_before + 1.0e-12
        for result in results
    )
    assert all(
        np.max(
            np.abs(result.strands_after_sizing - result.strands_before_sizing)
        ) <= optimizer.options.strand_max_change_per_cycle
        for result in results
    )
    assert any(result.tension_update_accepted for result in results)
    assert any(result.strand_update_accepted for result in results)
    assert any(
        result.strand_update_accepted
        and result.local_score_after
        > result.local_score_after_tension * 1.01
        for result in results
    )


def test_engineering_tension_feedback_uses_each_final_zero_target_residual():
    optimizer = _direct_optimizer(n_seg=2)
    response = {
        (1, "A"): StageAControlResponse3D(1, 1, 0.0, 0.0),
        (1, "B"): StageAControlResponse3D(1, 2, 0.0, 0.5),
        (2, "A"): StageAControlResponse3D(2, 4, 0.0, 0.0),
        (2, "B"): StageAControlResponse3D(2, 5, 0.0, 0.5),
    }
    before_a = np.full(4, 3.0e6)
    before_b = np.full(4, 3.0e6)

    _, after_b = optimizer._update_tensions(
        np.full(4, 100, dtype=int),
        before_a,
        before_b,
        response,
        1.0,
    )

    assert after_b[1] < before_b[1]
    assert after_b[3] < before_b[3]


def test_engineering_tension_feedback_residuals_match_activation_and_final_targets():
    optimizer = _direct_optimizer(n_seg=2)
    responses = {
        (1, "A"): StageAControlResponse3D(1, 1, 0.01, 0.02),
        (1, "B"): StageAControlResponse3D(1, 2, 0.03, 0.10),
        (2, "A"): StageAControlResponse3D(2, 4, -0.04, -0.05),
        (2, "B"): StageAControlResponse3D(2, 5, -0.06, -0.20),
    }

    residuals = optimizer._tension_feedback_residuals(responses)

    assert residuals[0] == pytest.approx([0.01, 0.02, -0.04, -0.05])
    assert residuals[1] == pytest.approx([0.03, 0.10, -0.06, -0.20])


def test_engineering_partial_tension_retry_isolates_worst_control_group():
    optimizer = _direct_optimizer(n_seg=2)
    baseline_a = np.full(4, 10.0)
    baseline_b = np.full(4, 20.0)
    proposed_a = np.asarray([11.0, 12.0, 13.0, 14.0])
    proposed_b = np.asarray([21.0, 22.0, 23.0, 24.0])
    residual_before = np.asarray(
        [
            [0.001, 0.02, 0.003, 0.04],
            [0.005, -0.20, 0.006, -0.50],
        ]
    )

    partial_a, partial_b, keep = (
        optimizer._isolated_tension_candidate(
            baseline_a,
            baseline_b,
            proposed_a,
            proposed_b,
            residual_before,
        )
    )

    assert keep[0].tolist() == [False, False, False, False]
    assert keep[1].tolist() == [False, False, False, True]
    assert partial_a == pytest.approx(baseline_a)
    assert partial_b == pytest.approx([20.0, 20.0, 20.0, 24.0])


def test_engineering_tension_step_memory_adapts_each_component_independently():
    optimizer = _direct_optimizer(n_seg=2)
    scales = np.ones((2, 4))
    before = np.ones((2, 4))
    after = np.asarray(
        [
            [0.5, 1.2, -0.2, 0.9],
            [0.6, 1.0, 0.8, 1.5],
        ]
    )
    changed = np.ones((2, 4), dtype=bool)

    adapted = optimizer._adapt_tension_step_scales(
        scales, before, after, changed
    )

    assert optimizer.nominal_tension_step_stress_mpa() == pytest.approx(30.0)
    assert adapted[0] == pytest.approx([1.15, 0.5, 0.5, 1.0])
    assert adapted[1] == pytest.approx([1.15, 1.0, 1.0, 0.5])


def test_engineering_n3_strand_steps_accelerate_and_cache_each_group_independently():
    optimizer = _direct_optimizer(n_seg=3)
    strands = np.full(6, 100, dtype=int)
    step_scales = np.asarray([0.5, 0.5, 1.0, 1.0, 2.0, 2.0])

    resized = optimizer._resize_strands(
        strands,
        np.zeros(6),
        np.full(6, 750.0),
        step_scales,
    )

    assert resized.tolist() == [103, 103, 106, 106, 112, 112]
    assert np.max(np.abs(resized - strands)) > 1

    adapted = optimizer._adapt_strand_step_scales(
        np.ones(6),
        np.full(6, 700.0),
        np.asarray([600.0, 750.0, 450.0, 640.0, 700.0, 560.0]),
        np.asarray([True, True, True, True, False, True]),
        optimizer.options.target_stress_mpa,
    )

    assert adapted == pytest.approx([1.15, 0.5, 0.5, 1.15, 1.0, 1.15])


def test_engineering_n3_strand_step_preserves_proportional_stress_differences():
    optimizer = _direct_optimizer(n_seg=3)
    strands = np.full(6, 100, dtype=int)

    resized = optimizer._resize_strands(
        strands,
        np.zeros(6),
        np.repeat([600.0, 650.0, 700.0], 2),
        np.ones(6),
    )

    assert resized.tolist() == [102, 102, 104, 104, 105, 105]

    bounded = optimizer._resize_strands(
        strands,
        np.zeros(6),
        np.full(6, 2000.0),
        np.full(6, 2.0),
    )
    assert np.max(np.abs(bounded - strands)) <= 25


def test_engineering_strand_counts_are_independent_without_curve_projection():
    optimizer = _direct_optimizer(n_seg=3)

    resized = optimizer._resize_strands(
        np.full(6, 100, dtype=int),
        np.zeros(6),
        np.asarray([800.0, 200.0, 700.0, 300.0, 600.0, 400.0]),
        np.ones(6),
    )

    assert resized.tolist() == [108, 92, 105, 95, 102, 98]


def test_engineering_final_stress_target_is_signed():
    optimizer = _direct_optimizer(n_seg=1)

    assert optimizer._strand_score(np.asarray([500.0, 500.0])) == pytest.approx(0.0)
    assert optimizer._strand_score(np.asarray([-500.0, 500.0])) == pytest.approx(
        np.sqrt(2.0)
    )
    resized = optimizer._resize_strands(
        np.asarray([100, 100]),
        np.zeros(2),
        np.asarray([-500.0, 500.0]),
        np.ones(2),
    )
    assert resized.tolist() == [88, 100]


class _TTYBuffer(StringIO):
    def isatty(self):
        return True


def test_engineering_progress_refreshes_one_24_group_dashboard(tmp_path):
    stream = _TTYBuffer()
    rows = tuple(
        EngineeringStageStatus3D(
            construction_stage=stage,
            backstay_strands=90 + stage,
            main_stay_strands=100 + stage,
            target_final_deck_uz_m=0.0,
            stage_a_tower_dx_m=stage / 100_000.0,
            stage_a_deck_uz_m=0.0,
            final_tower_dx_m=stage / 50_000.0,
            final_deck_uz_m=0.5,
            backstay_final_stress_mpa=490.0,
            main_stay_final_stress_mpa=510.0,
        )
        for stage in range(1, 25)
    )
    display = _EngineeringProgressDisplay(
        first_cycle=1,
        cycles_this_run=2,
        target_stress_mpa=500.0,
        nominal_tension_step_stress_mpa=30.0,
        output_dir=tmp_path,
        stream=stream,
        clock=lambda: 10.0,
    )
    display.update(
        EngineeringProgress3D(
            cycle_index=1,
            phase="读取A激活与最终状态",
            fem_cases_completed=0,
            fem_cases_total=3,
            elapsed_seconds=0.0,
            eta_seconds=10.0,
            stage_status=rows,
        )
    )
    display.update(
        EngineeringProgress3D(
            cycle_index=1,
            phase="本轮完成",
            fem_cases_completed=3,
            fem_cases_total=3,
            elapsed_seconds=10.0,
            eta_seconds=0.0,
            stage_status=rows,
            local_score=0.25,
            score_change_percent=12.5,
            tension_update_accepted=True,
            strand_update_accepted=True,
        )
    )
    display.close()

    rendered = stream.getvalue()
    assert "根数(背/中)" in rendered
    assert "梁端 z [mm] A激活 / 最终 / 目标" in rendered
    assert "塔端 x [mm] A激活 / 最终" in rendered
    assert "最终应力 [MPa] 背 / 中" in rendered
    assert "基准调索步长≈30.0 MPa" in rendered
    assert "24" in rendered
    assert "\x1b[" in rendered
    assert "print" not in rendered
