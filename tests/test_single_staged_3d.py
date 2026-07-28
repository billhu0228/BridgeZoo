import json
from dataclasses import replace

import numpy as np
import pytest

from bridgezoo.fem.single_staged import (
    HSection3D,
    HollowBoxSection3D,
    RectangularSection3D,
    SingleStaged3DConfig,
    SingleStagedDirectBatchSolver3D,
    SingleStagedDirectSolver3D,
    SingleStagedOpenSeesSolver3D,
    build_single_staged_3d,
)
from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator3D,
    CableOptimizationProblem,
    build_affine_model,
)
from bridgezoo.render.staged3d import render_staged_3d
from scripts.bridge_config import load_single_staged_3d_config, resolve_bridge_config
from scripts.optimize_cables_3d import main as optimize_3d_cli
from scripts.single_staged_3d import main as run_cli


def test_physical_h_and_hollow_box_sections_derive_properties_from_dimensions():
    h_section = HSection3D("test H", 2.0, 1.0, 0.04, 0.05)
    expected_area = 2.0 * 1.0 * 0.05 + (2.0 - 0.10) * 0.04
    assert h_section.A == pytest.approx(expected_area)
    assert h_section.Iy > h_section.Iz > 0.0
    assert h_section.J > 0.0

    box = HollowBoxSection3D("test box", 6.0, 5.0, 0.6)
    expected_box_area = 6.0 * 5.0 - 4.8 * 3.8
    assert box.A == pytest.approx(expected_box_area)
    assert box.Iy > 0.0
    assert box.Iz > box.Iy
    assert box.J > 0.0

    with pytest.raises(ValueError, match="hollow core"):
        HollowBoxSection3D("invalid", 1.0, 1.0, 0.5)


def test_bundled_omo_3d_yaml_supplies_complete_physical_input():
    config = load_single_staged_3d_config("omo3d")

    assert resolve_bridge_config("omo3d").name == "omo_bridge_3d.yaml"
    assert config.n_seg == 24
    assert config.anchor_base_height == pytest.approx(48.75)
    assert config.left_span == pytest.approx(40.0)
    assert config.girder_spacing == pytest.approx(12.5)
    assert config.cross_girder_spacing == pytest.approx(4.0)
    assert config.deck_offset == pytest.approx(1.90)
    assert config.main_girder_section.shape == "H"
    assert config.cross_girder_section.shape == "H"
    assert config.tower_section.shape == "hollow_box"
    assert config.steel.E == pytest.approx(206.0e9)
    assert config.concrete.E == pytest.approx(3.3238e10)
    assert config.cable_material.E == pytest.approx(1.95e11)
    assert config.superimposed_dead_load == pytest.approx(1750.0)

    # The full 24-cable OMO grid has more than 100 longitudinal stations;
    # node/element namespaces must remain disjoint at production scale.
    plan = build_single_staged_3d(config)
    assert len(plan.metadata["station_x"]) > 100
    assert len(plan.model.nodes) > 500
    assert not (set(plan.model.frames) & set(plan.model.cables))


def test_3d_builder_creates_twin_girder_grid_eccentric_slab_and_box_tower():
    plan = build_single_staged_3d(n_seg=2, left_span=10.0)
    model = plan.model
    station_count = len(plan.metadata["station_x"])
    cross_girder_x = np.asarray(plan.metadata["cross_girder_x"])

    main_girders = [frame for frame in model.frames.values() if frame.group == "main_girder"]
    cross_girders = [frame for frame in model.frames.values() if frame.group == "cross_girder"]
    slab_frames = [frame for frame in model.frames.values() if frame.group.startswith("deck_")]
    tower_frames = [frame for frame in model.frames.values() if frame.group == "tower"]

    assert len(main_girders) == 2 * (station_count - 1)
    assert len(cross_girders) == len(cross_girder_x)
    assert len(model.rigid_links) == 2 * station_count
    assert len(model.cables) == 4 * 2  # paired main stays + backstays in two cable planes
    assert all(isinstance(frame.section, HSection3D) for frame in main_girders + cross_girders)
    assert all(isinstance(frame.section, RectangularSection3D) for frame in slab_frames)
    assert all(isinstance(frame.section, HollowBoxSection3D) for frame in tower_frames)

    for link in model.rigid_links.values():
        master = model.nodes[link.master]
        slave = model.nodes[link.slave]
        assert (slave.x, slave.y) == pytest.approx((master.x, master.y))
        assert slave.z - master.z == pytest.approx(1.54)

    main_node_ids = {frame.i for frame in main_girders} | {frame.j for frame in main_girders}
    assert all(frame.i in main_node_ids and frame.j in main_node_ids for frame in cross_girders)
    assert np.diff(cross_girder_x) == pytest.approx(
        np.full(len(cross_girder_x) - 1, plan.metadata["actual_cross_girder_spacing"]),
        abs=1.0e-12,
    )
    assert all(load.load_case != "self_weight" for load in model.frame_loads if model.frames[load.member].group == "deck_transverse")
    assert [stage.label for stage in plan.stages] == ["cable1", "cable2", "tip", "left_span"]


def test_3d_builder_accepts_independent_backstay_and_main_stay_group_designs():
    plan = build_single_staged_3d(
        n_seg=2,
        strands_per_cable=((10, 20), (30, 40)),
        pretension_per_cable=((1.0e6, 2.0e6), (3.0e6, 4.0e6)),
    )

    for stage, expected in enumerate(((10, 20, 1.0e6, 2.0e6), (30, 40, 3.0e6, 4.0e6)), start=1):
        back_strands, main_strands, back_tension, main_tension = expected
        backstays = [
            cable
            for cable in plan.model.cables.values()
            if cable.activation_stage == stage and cable.group == "backstay"
        ]
        main_stays = [
            cable
            for cable in plan.model.cables.values()
            if cable.activation_stage == stage and cable.group == "main_stay"
        ]
        assert len(backstays) == len(main_stays) == 2
        assert {cable.area for cable in backstays} == {back_strands * 1.4e-4}
        assert {cable.area for cable in main_stays} == {main_strands * 1.4e-4}
        assert {cable.pretension for cable in backstays} == {back_tension}
        assert {cable.pretension for cable in main_stays} == {main_tension}


def test_3d_optimization_evaluator_reports_groups_and_physical_material_quantity():
    config = SingleStaged3DConfig(
        n_seg=1,
        left_span=5.0,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
    )
    problem = CableOptimizationProblem(
        n_seg=1,
        bounds=CableBounds(strand_min=1, strand_max=100),
        strand_area=config.strand_area,
        backend="direct",
        model_family="single_staged_3d",
    )

    result = CableDesignEvaluator3D(problem, config).evaluate(
        [10, 20],
        [1.0e6, 2.0e6],
        keep_result=True,
    )

    assert result.cable_ids == (1001, 2001)
    assert result.metrics.total_strands == 2 * (10 + 20)
    assert set(result.cable_group_members) == {1001, 2001}
    assert all(len(members) == 2 for members in result.cable_group_members.values())
    assert set(result.physical_cable_stress_mpa) == {
        member
        for members in result.cable_group_members.values()
        for member in members
    }
    assert len(result.staged_result.records) == 1


def test_3d_optimization_opensees_backend_matches_direct_group_metrics():
    pytest.importorskip("openseespy.opensees")
    config = SingleStaged3DConfig(
        n_seg=1,
        left_span=5.0,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
    )
    problem = CableOptimizationProblem(
        n_seg=1,
        bounds=CableBounds(strand_min=1, strand_max=100),
        strand_area=config.strand_area,
        backend="direct",
        model_family="single_staged_3d",
    )
    design = ([55, 65], [3.0e6, 4.0e6])

    direct = CableDesignEvaluator3D(problem, config).evaluate(*design)
    reference = CableDesignEvaluator3D(replace(problem, backend="opensees"), config).evaluate(
        *design
    )

    assert reference.metrics.shape_rmse_m == pytest.approx(
        direct.metrics.shape_rmse_m,
        rel=2.0e-6,
        abs=2.0e-9,
    )
    assert reference.metrics.tower_top_dx_m == pytest.approx(
        direct.metrics.tower_top_dx_m,
        rel=2.0e-6,
        abs=2.0e-9,
    )
    assert reference.cable_stress_mpa == pytest.approx(
        direct.cable_stress_mpa,
        rel=2.0e-6,
        abs=2.0e-5,
    )


def test_direct_3d_solver_runs_all_stages_and_balances_physical_dead_load():
    plan = build_single_staged_3d(
        n_seg=2,
        left_span=10.0,
        pretension_per_cable=0.0,
    )
    result = SingleStagedDirectSolver3D().run(plan)

    assert [record.stage_label for record in result.records] == [
        "cable1",
        "cable2",
        "tip",
        "left_span",
    ]
    assert all(record.converged for record in result.records)
    final = result.final
    deck_displacements = [
        values[2]
        for node_id, values in final.displacement.items()
        if plan.model.nodes[node_id].role.startswith("main_girder")
    ]
    assert min(deck_displacements) < -1.0e-6
    assert np.isfinite(np.asarray(list(final.displacement.values()))).all()
    assert sum(reaction[2] for reaction in final.support_reaction.values()) == pytest.approx(
        -final.applied_load[2],
        rel=1.0e-10,
        abs=1.0e-5,
    )


def test_direct_3d_batch_solver_matches_scalar_and_uses_one_multi_rhs_solve(monkeypatch):
    common = {
        "n_seg": 1,
        "left_span": 5.0,
        "cross_girder_spacing": 10.0,
        "tower_element_size": 10.0,
        "strands_per_cable": ((55, 65),),
    }
    plans = [
        build_single_staged_3d(
            **common,
            pretension_per_cable=(tensions,),
        )
        for tensions in ((0.0, 0.0), (1.0e6, 2.0e6), (3.0e6, 4.0e6))
    ]
    stage = plans[0].final_stage
    scalar = [
        SingleStagedDirectSolver3D().solve_stage(plan, stage.index, stage.label)
        for plan in plans
    ]

    solve_calls = []
    original_solve = np.linalg.solve

    def counted_solve(matrix, right_hand_side):
        solve_calls.append((matrix.shape, right_hand_side.shape))
        return original_solve(matrix, right_hand_side)

    monkeypatch.setattr(np.linalg, "solve", counted_solve)
    batch = SingleStagedDirectBatchSolver3D().solve_stage_batch(
        plans,
        stage.index,
        stage.label,
    )

    assert len(solve_calls) == 1
    assert solve_calls[0][1][1] == len(plans)
    for batched, separate in zip(batch, scalar):
        assert batched.converged == separate.converged
        assert batched.applied_load == pytest.approx(separate.applied_load, abs=1.0e-10)
        for field in (
            "displacement",
            "frame_force",
            "cable_force",
            "cable_stress",
            "support_reaction",
        ):
            batched_values = getattr(batched, field)
            separate_values = getattr(separate, field)
            assert set(batched_values) == set(separate_values)
            for object_id in batched_values:
                assert batched_values[object_id] == pytest.approx(
                    separate_values[object_id],
                    rel=1.0e-11,
                    abs=1.0e-8,
                )

    different_area = build_single_staged_3d(
        **{**common, "strands_per_cable": ((56, 65),)},
        pretension_per_cable=((1.0e6, 2.0e6),),
    )
    with pytest.raises(ValueError, match="cable structure"):
        SingleStagedDirectBatchSolver3D().solve_stage_batch(
            [plans[0], different_area],
            stage.index,
            stage.label,
        )


def test_direct_3d_affine_model_uses_batch_kernel_and_matches_real_solve(monkeypatch):
    config = SingleStaged3DConfig(
        n_seg=1,
        left_span=5.0,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
    )
    problem = CableOptimizationProblem(
        n_seg=1,
        bounds=CableBounds(strand_min=1, strand_max=100),
        strand_area=config.strand_area,
        backend="direct",
        model_family="single_staged_3d",
    )
    evaluator = CableDesignEvaluator3D(problem, config)
    strands = np.asarray([55, 65])

    solve_rhs_counts = []
    original_solve = np.linalg.solve

    def counted_solve(matrix, right_hand_side):
        solve_rhs_counts.append(right_hand_side.shape[1])
        return original_solve(matrix, right_hand_side)

    monkeypatch.setattr(np.linalg, "solve", counted_solve)
    affine = build_affine_model(evaluator, strands)

    assert solve_rhs_counts == [evaluator.layout.size + 1]
    tension = np.asarray([2.2e6, 3.4e6])
    actual = evaluator.evaluate(strands, tension)
    assert solve_rhs_counts == [evaluator.layout.size + 1, 1]
    assert affine.stress_mpa(tension) == pytest.approx(
        [actual.cable_stress_mpa[cable_id] for cable_id in actual.cable_ids],
        rel=1.0e-10,
        abs=1.0e-8,
    )
    assert affine.deck_err_m(tension) == pytest.approx(
        list(actual.deck_errors_m.values()),
        rel=1.0e-10,
        abs=1.0e-10,
    )
    assert affine.tower_dx_m(tension) == pytest.approx(
        actual.metrics.tower_top_dx_m,
        rel=1.0e-10,
        abs=1.0e-10,
    )


def test_3d_opensees_backend_matches_direct_linear_model():
    pytest.importorskip("openseespy.opensees")
    plan = build_single_staged_3d(
        n_seg=1,
        left_span=5.0,
        pretension_per_cable=0.0,
    )
    direct = SingleStagedDirectSolver3D().run(plan).final
    reference = SingleStagedOpenSeesSolver3D().run(plan).final

    assert direct.converged and reference.converged
    beam_nodes = plan.metadata["beam_grid_node_ids"]
    for node_id in beam_nodes:
        assert reference.displacement[node_id] == pytest.approx(
            direct.displacement[node_id],
            rel=2.0e-6,
            abs=2.0e-9,
        )


def test_opensees_3d_result_renders_stage_frames_and_gif(tmp_path):
    pytest.importorskip("openseespy.opensees")
    plan = build_single_staged_3d(
        n_seg=1,
        left_span=None,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
        pretension_per_cable=0.0,
    )
    result = SingleStagedOpenSeesSolver3D().run(plan)
    output = tmp_path / "staged3d.gif"
    frames = tmp_path / "frames"

    artifacts = render_staged_3d(
        plan,
        result,
        scale=5.0,
        out=output,
        frames_dir=frames,
        fps=2,
    )

    assert artifacts["output"] == output
    assert output.stat().st_size > 1000
    assert len(artifacts["frames"]) == len(result.records)
    assert all(path.stat().st_size > 1000 for path in artifacts["frames"])


def test_minimum_3d_cli_writes_machine_readable_json(tmp_path):
    output = tmp_path / "single_staged_3d.json"
    assert run_cli(["--bridge", "omo3d", "--n", "1", "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == "bridgezoo.single_staged_3d.result.v1"
    assert payload["backend"] == "direct3d"
    assert payload["model"]["coordinate_system"] == "x longitudinal, y transverse, z vertical"
    assert payload["final"]["converged"] is True
    assert payload["input"]["n_seg"] == 1
    assert payload["input"]["deck_width"] == pytest.approx(16.0)
    assert "3D rendering" not in payload["todo"]


def test_independent_3d_optimization_cli_writes_outputs_resumes_and_solves_design(
    tmp_path,
    capsys,
):
    out_dir = tmp_path / "opt3d"
    base_args = [
        "--bridge",
        "omo3d",
        "--n",
        "1",
        "--backend",
        "direct",
        "--outer-iterations",
        "0",
        "--continuous-maxiter",
        "20",
        "--initial-strands",
        "91",
        "--quiet",
        "--out",
        str(out_dir),
    ]

    assert optimize_3d_cli(base_args) == 0
    assert optimize_3d_cli([*base_args, "--resume"]) == 0
    payload = json.loads((out_dir / "best_design.json").read_text(encoding="utf-8"))

    assert payload["schema"] == "bridgezoo.cable_optimization_3d.v1"
    assert payload["model_family"] == "single_staged_3d"
    assert payload["search"]["run_index"] == 2
    assert [item["group"] for item in payload["cable_groups"]] == [
        "backstay",
        "main_stay",
    ]
    assert all(len(item["physical_cable_ids"]) == 2 for item in payload["cable_groups"])
    assert (out_dir / "history.csv").is_file()
    assert (out_dir / "summary.txt").is_file()

    analysis_output = tmp_path / "optimized_analysis.json"
    assert run_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "1",
            "--design",
            str(out_dir / "best_design.json"),
            "--backend",
            "direct",
            "--render",
            "none",
            "--output",
            str(analysis_output),
        ]
    ) == 0
    analysis = json.loads(analysis_output.read_text(encoding="utf-8"))
    assert analysis["optimized_design"]["objective"] == pytest.approx(payload["objective"])
    assert analysis["input"]["strands_per_cable"] == [
        [
            payload["cable_groups"][0]["strands_per_physical_cable"],
            payload["cable_groups"][1]["strands_per_physical_cable"],
        ]
    ]
    for group in payload["cable_groups"]:
        for member_id, stress_mpa in group["physical_final_stress_MPa"].items():
            assert analysis["final"]["cable_stress_Pa"][member_id] / 1.0e6 == pytest.approx(
                stress_mpa,
                rel=1.0e-9,
                abs=1.0e-9,
            )

    with pytest.raises(ValueError, match="geometry/materials differ"):
        run_cli(
            [
                "--bridge",
                "omo3d",
                "--design",
                str(out_dir / "best_design.json"),
                "--render",
                "none",
            ]
        )

    capsys.readouterr()
    progress_out = tmp_path / "progress"
    assert optimize_3d_cli(
        [
            item
            for item in [*base_args[:-2], "--progress-refresh", "10", "--out", str(progress_out)]
            if item != "--quiet"
        ]
    ) == 0
    output_lines = capsys.readouterr().out.splitlines()
    assert not any("optimize tensions:" in line for line in output_lines)
    assert not any("candidate cable=" in line for line in output_lines)
    assert len(output_lines) <= 15
