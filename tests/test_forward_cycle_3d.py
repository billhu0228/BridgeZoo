import json
from dataclasses import asdict, replace

import numpy as np
import pytest
import scripts.optimize_cables_3d_forward as forward_script

from bridgezoo.fem.single_staged import (
    SingleStagedDirectSolver3D,
    build_single_staged_3d,
)
from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator3D,
    CableOptimizationProblem,
    ForwardCableCycleOptimizer3D,
    ForwardCycleOptions3D,
)
from bridgezoo.optim.problem import ObjectiveWeights
from scripts.bridge_config import load_single_staged_3d_config
from scripts.optimize_cables_3d_forward import main as forward_cli
from scripts.single_staged_3d import load_optimized_design_3d


def _direct_optimizer(n_seg=1, options=None, *, progress=None, milestone=None):
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
            strand_min=1,
            strand_max=500,
            stress_lower_mpa=400.0,
            stress_upper_mpa=600.0,
            tension_bound_stress_mpa=1600.0,
        ),
        weights=ObjectiveWeights(),
        strand_area=config.strand_area,
        backend="direct",
        model_family="single_staged_3d",
    )
    return (
        ForwardCableCycleOptimizer3D(
            CableDesignEvaluator3D(problem, config),
            options or ForwardCycleOptions3D(),
            progress=progress,
            milestone=milestone,
        ),
        config,
    )


def test_direct_3d_birth_displacement_is_the_pre_increment_tangent_snapshot():
    common = dict(
        n_seg=1,
        strands_per_cable=((100, 100),),
        pretension_a_ratio=((1.0, 1.0),),
    )
    unloaded_plan = build_single_staged_3d(
        **common,
        pretension_per_cable=((0.0, 0.0),),
    )
    tensioned_plan = build_single_staged_3d(
        **common,
        pretension_per_cable=((1.0e6, 1.0e6),),
    )
    solver = SingleStagedDirectSolver3D()
    unloaded = solver.solve_stage(unloaded_plan, 1)
    tensioned = solver.solve_stage(tensioned_plan, 1)
    deck_nodes = {
        cable.j
        for cable in tensioned_plan.model.cables.values()
        if cable.group == "main_stay"
    }

    assert deck_nodes.issubset(unloaded.birth_displacement)
    assert {
        node: tensioned.birth_displacement[node] for node in deck_nodes
    } == pytest.approx(
        {node: unloaded.birth_displacement[node] for node in deck_nodes},
        abs=1.0e-12,
    )
    assert any(
        not np.allclose(
            tensioned.displacement[node],
            tensioned.birth_displacement[node],
            rtol=0.0,
            atol=1.0e-8,
        )
        for node in deck_nodes
    )


def test_forward_cycle_tunes_A_then_wet_weight_B_and_locks_prior_groups():
    optimizer, config = _direct_optimizer(n_seg=2)
    result = optimizer.run_cycle(np.full(4, 100, dtype=int))

    assert [(item.construction_stage, item.phase) for item in result.controls] == [
        (1, "A"),
        (1, "B"),
        (2, "A"),
        (2, "B"),
    ]
    assert [item.stage_index for item in result.controls] == [1, 2, 4, 5]
    assert all(item.target_reached for item in result.controls)
    assert all(
        item.response_after.max_abs_m
        <= optimizer.options.displacement_tolerance_m
        for item in result.controls
    )
    assert all(
        "tangent-birth" in item.displacement_basis for item in result.controls
    )
    assert result.controls[0].backstay_tension_N == pytest.approx(
        result.pretension_a[0]
    )
    assert result.controls[0].main_stay_tension_N == pytest.approx(
        result.pretension_a[1]
    )
    assert result.controls[1].backstay_tension_N == pytest.approx(
        result.pretension_b[0]
    )
    assert result.controls[1].main_stay_tension_N == pytest.approx(
        result.pretension_b[1]
    )
    assert result.final_evaluation.staged_result.final.stage_label == "secondary_load"
    expected_total = result.pretension_a + result.pretension_b
    assert result.final_evaluation.design.pretension == pytest.approx(expected_total)
    assert np.all(result.pretension_a >= 0.0)
    assert np.all(result.pretension_b >= 0.0)
    assert np.all(
        expected_total
        <= 1600.0e6
        * config.strand_area
        * result.strands_before_sizing
        * (1.0 + 1.0e-12)
    )


def test_forward_strand_sizing_is_force_conserving_math_without_fem():
    optimizer, config = _direct_optimizer(n_seg=1)
    cases_before = optimizer.total_fem_replays
    resized, raw, clipped, force_N = optimizer.resize_strands(
        np.asarray([50, 50]),
        np.asarray([300.0, 300.0]),
    )

    assert resized.tolist() == [30, 30]
    assert raw.tolist() == [30, 30]
    assert not np.any(clipped)
    assert force_N == pytest.approx(
        np.full(2, 300.0e6 * 50 * config.strand_area)
    )
    assert optimizer.total_fem_replays == cases_before


def test_forward_cycle_resumes_at_next_group_without_reoptimizing_milestones():
    class StopAfterMilestone(RuntimeError):
        pass

    saved = []

    def stop_after_first(milestone):
        saved.append(milestone)
        raise StopAfterMilestone

    interrupted, _ = _direct_optimizer(n_seg=2, milestone=stop_after_first)
    with pytest.raises(StopAfterMilestone):
        interrupted.run_cycle(np.full(4, 100, dtype=int))
    milestone = saved[0]
    assert milestone.construction_stage == 1
    assert milestone.completed_phase == "A"
    assert len(milestone.controls) == 1
    assert np.all(milestone.pretension_a[2:] == 0.0)
    assert np.all(milestone.pretension_b == 0.0)

    progress = []
    resumed, _ = _direct_optimizer(n_seg=2, progress=progress.append)
    result = resumed.run_cycle(
        milestone.strands,
        cycle_index=milestone.cycle_index,
        start_construction_stage=1,
        start_phase="B",
        pretension_a=milestone.pretension_a,
        pretension_b=milestone.pretension_b,
        completed_controls=milestone.controls,
        pending_birth_uz_m=milestone.birth_uz_m,
    )

    assert not any("group 1/2 A" in message for message in progress)
    assert any("group 1/2 B" in message for message in progress)
    assert any("group 2/2 A" in message for message in progress)
    assert result.pretension_a[:2] == pytest.approx(milestone.pretension_a[:2])
    assert np.all(result.pretension_b[:2] > milestone.pretension_b[:2])
    assert result.controls[:1] == milestone.controls
    assert result.final_evaluation.staged_result.final.stage_label == "secondary_load"


def test_forward_cli_saves_verified_design_and_resumes_from_resized_counts(tmp_path):
    out_dir = tmp_path / "forward"
    common = [
        "--bridge",
        "omo3d",
        "--n",
        "1",
        "--backend",
        "direct",
        "--quiet",
        "--out",
        str(out_dir),
    ]

    assert forward_cli(common) == 0
    design = json.loads((out_dir / "best_design.json").read_text(encoding="utf-8"))
    strands = json.loads(
        (out_dir / "strand_configuration.json").read_text(encoding="utf-8")
    )
    checkpoint = json.loads(
        (out_dir / "forward_checkpoint.json").read_text(encoding="utf-8")
    )

    assert design["algorithm"] == "sequential_forward_stage_balance"
    assert design["forward_cycle"]["A_state"] == "steel_and_A"
    assert design["forward_cycle"]["B_state"].startswith("deck_weight_and_B")
    assert design["forward_cycle"]["final_sizing_state"] == "secondary_load"
    assert design["forward_cycle"]["strands_used_by_verified_FEM"] == [100, 100]
    next_counts = design["forward_cycle"]["strands_saved_for_next_cycle"]
    assert next_counts == checkpoint["state"]["strands_for_next_cycle"]
    assert next_counts == strands["strands_stage_major_backstay_main_stay"]
    assert next_counts != [100, 100]
    assert all(
        item["FEM_replays_from_stage_1"] >= 4
        for item in design["forward_cycle"]["substage_controls"]
    )
    assert checkpoint["state"]["cycle_in_progress"] is False

    replay_config, _ = load_optimized_design_3d(
        out_dir / "best_design.json",
        replace(load_single_staged_3d_config("omo3d"), n_seg=1),
    )
    assert replay_config.strands_per_cable == ((100, 100),)

    assert forward_cli([*common, "--resume"]) == 0
    resumed_design = json.loads(
        (out_dir / "best_design.json").read_text(encoding="utf-8")
    )
    resumed_checkpoint = json.loads(
        (out_dir / "forward_checkpoint.json").read_text(encoding="utf-8")
    )
    assert resumed_checkpoint["completed_cycles"] == 2
    assert len(resumed_checkpoint["history"]) == 2
    assert resumed_design["forward_cycle"]["strands_used_by_verified_FEM"] == (
        next_counts
    )


def test_forward_cli_checkpoint_resumes_after_each_completed_group(
    tmp_path, monkeypatch
):
    out_dir = tmp_path / "milestone"
    common = [
        "--bridge",
        "omo3d",
        "--n",
        "2",
        "--backend",
        "direct",
        "--quiet",
        "--out",
        str(out_dir),
    ]
    original_writer = forward_script._write_stage_milestone

    class SimulatedInterruption(RuntimeError):
        pass

    def save_then_interrupt(*args, **kwargs):
        original_writer(*args, **kwargs)
        raise SimulatedInterruption

    monkeypatch.setattr(
        forward_script, "_write_stage_milestone", save_then_interrupt
    )
    with pytest.raises(SimulatedInterruption):
        forward_script.main(common)

    checkpoint_path = out_dir / "forward_checkpoint.json"
    partial = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert partial["completed_cycles"] == 0
    assert partial["state"]["cycle_in_progress"] is True
    assert partial["state"]["milestone_stage"] == 1
    assert partial["state"]["milestone_phase"] == "A"
    assert partial["state"]["completed_construction_stage"] == 0
    assert partial["state"]["next_construction_stage"] == 1
    assert partial["state"]["next_phase"] == "B"
    locked_a = partial["state"]["locked_pretension_A_per_group_N"]
    locked_b = partial["state"]["locked_pretension_B_per_group_N"]
    assert len(partial["state"]["locked_controls"]) == 1

    monkeypatch.setattr(
        forward_script, "_write_stage_milestone", original_writer
    )
    assert forward_script.main([*common, "--resume"]) == 0
    complete = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    design = json.loads((out_dir / "best_design.json").read_text(encoding="utf-8"))
    assert complete["completed_cycles"] == 1
    assert complete["state"]["cycle_in_progress"] is False
    assert len(design["forward_cycle"]["substage_controls"]) == 4
    assert design["cable_groups"][0][
        "pretension_A_per_physical_cable_N"
    ] == pytest.approx(locked_a[0])
    assert design["cable_groups"][1][
        "pretension_A_per_physical_cable_N"
    ] == pytest.approx(locked_a[1])
    assert design["cable_groups"][0][
        "pretension_B_per_physical_cable_N"
    ] > locked_b[0]
    assert design["cable_groups"][1][
        "pretension_B_per_physical_cable_N"
    ] > locked_b[1]
