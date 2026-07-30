import json
from dataclasses import replace

import numpy as np
import pytest

from bridgezoo.fem.single_staged.birth3d import TangentDisplacementHistory3D
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
    SecondaryTensionOptions3D,
    StageAControlOptions,
    Staged3DOptimizationOptions,
    StagedCableOptimizer3D,
    build_affine_model,
    build_smooth_curve_basis,
    build_stage_major_curve_basis,
    project_strands_to_smooth_curve,
)
from bridgezoo.optim.single_staged3d import StageAControlResponse3D
from bridgezoo.optim.variables import CableLayout
from bridgezoo.render.staged3d import export_final_3d_dxf, render_staged_3d
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
    assert config.resolved_right_fix == pytest.approx(0.0)
    assert config.flexible_birth_correction_factor == pytest.approx(1.0)
    assert config.left_span == pytest.approx(40.0)
    assert config.girder_spacing == pytest.approx(11.4)
    assert config.cross_girder_spacing == pytest.approx(4.0)
    assert config.deck_offset == pytest.approx(1.4)
    assert config.main_girder_section.shape == "H"
    assert config.cross_girder_section.shape == "H"
    assert config.tower_section.shape == "hollow_box"
    assert config.steel.E == pytest.approx(206.0e9)
    assert config.concrete.E == pytest.approx(3.3238e10)
    assert config.cable_material.E == pytest.approx(1.95e11)
    assert config.secondary_main_girder_line_load == pytest.approx(0.0)
    assert config.secondary_deck_pressure == pytest.approx(1750.0)
    assert config.pretension_a_ratio == pytest.approx(0.5)
    assert config.superimposed_dead_load is None

    # The full 24-cable OMO grid has more than 100 longitudinal stations;
    # node/element namespaces must remain disjoint at production scale.
    plan = build_single_staged_3d(config)
    assert len(plan.metadata["station_x"]) > 100
    assert len(plan.model.nodes) > 500
    assert not (set(plan.model.frames) & set(plan.model.cables))
    assert plan.final_stage.label == "secondary_load"
    assert len(plan.stages) == 3 * (config.n_seg + 2) + 1

    with pytest.raises(ValueError, match="flexible_birth_correction_factor"):
        replace(config, flexible_birth_correction_factor=float("inf"))
    with pytest.raises(ValueError, match="retired and must equal 1.0"):
        replace(config, flexible_birth_correction_factor=0.5)


def test_3d_zero_right_fix_reuses_tower_axis_and_fully_fixes_both_girders():
    plan = build_single_staged_3d(n_seg=1, right_fix=0.0)
    model = plan.model

    assert "right_bearing" not in plan.metadata["station_x"]
    assert list(plan.metadata["station_x"].values()).count(0.0) == 1
    fixed_girder_supports = [
        support
        for support in model.supports.values()
        if model.nodes[support.node].role.startswith("main_girder")
        and model.nodes[support.node].x == pytest.approx(0.0)
    ]
    assert len(fixed_girder_supports) == 2
    assert all(support.restraints == (True,) * 6 for support in fixed_girder_supports)


def test_3d_pretension_b_target_points_are_normalized_and_interpolated():
    config = SingleStaged3DConfig(
        n_seg=10,
        pretension_b_target_points=((0.0, 0.0), (-132.0, 0.01)),
    )

    assert config.pretension_b_target_points == ((-132.0, 0.01), (0.0, 0.0))
    assert config.cable_station_x(10) == pytest.approx(-78.0)
    assert config.pretension_b_target_uz_m(0.0) == pytest.approx(0.0)
    assert config.pretension_b_target_uz_m(-66.0) == pytest.approx(0.005)
    assert config.pretension_b_target_uz_m(-132.0) == pytest.approx(0.01)
    assert config.pretension_b_target_uz_m(-200.0) == pytest.approx(0.01)
    assert config.pretension_b_target_uz_m(20.0) == pytest.approx(0.0)

    omo = load_single_staged_3d_config("omo3d")
    assert omo.pretension_b_target_points == ((-132.0, 0.01), (0.0, 0.0))
    omo_n10 = replace(omo, n_seg=10)
    assert omo_n10.cable_station_x(10) == pytest.approx(-132.0)
    assert omo_n10.pretension_b_target_uz_m(
        omo_n10.cable_station_x(10)
    ) == pytest.approx(0.01)

    with pytest.raises(ValueError, match="must contain"):
        replace(config, pretension_b_target_points=((0.0,),))
    with pytest.raises(ValueError, match="must be unique"):
        replace(config, pretension_b_target_points=((0.0, 0.0), (0.0, 0.01)))


def test_legacy_3d_yaml_deck_load_is_migrated_without_inventing_line_load(tmp_path):
    source = resolve_bridge_config("omo3d")
    lines = [
        line
        for line in source.read_text(encoding="utf-8").splitlines()
        if not line.startswith(("secondary_main_girder_line_load:", "secondary_deck_pressure:"))
    ]
    lines.append("superimposed_dead_load: 1750.0")
    legacy_path = tmp_path / "legacy_omo3d.yaml"
    legacy_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    config = load_single_staged_3d_config(legacy_path)

    assert config.secondary_main_girder_line_load == 0.0
    assert config.secondary_deck_pressure == 0.0
    assert config.superimposed_dead_load == pytest.approx(1750.0)
    assert config.resolved_secondary_deck_pressure == pytest.approx(1750.0)


def test_3d_builder_creates_twin_girder_grid_eccentric_slab_and_box_tower():
    plan = build_single_staged_3d(n_seg=2, left_span=10.0)
    model = plan.model
    station_count = len(plan.metadata["station_x"])
    cross_girder_x = np.asarray(plan.metadata["cross_girder_x"])

    main_girders = [frame for frame in model.frames.values() if frame.group == "main_girder"]
    cross_girders = [frame for frame in model.frames.values() if frame.group == "cross_girder"]
    slab_longitudinal = [
        frame for frame in model.frames.values() if frame.group == "deck_longitudinal"
    ]
    slab_transverse = [
        frame for frame in model.frames.values() if frame.group == "deck_transverse"
    ]
    slab_frames = slab_longitudinal + slab_transverse
    tower_frames = [frame for frame in model.frames.values() if frame.group == "tower"]

    assert len(main_girders) == 2 * (station_count - 1)
    assert len(cross_girders) == len(cross_girder_x)
    assert len(slab_longitudinal) == 4 * (station_count - 1)
    assert len(slab_transverse) == 3 * len(cross_girder_x)
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
    assert plan.metadata["deck_grid_y"] == pytest.approx(
        (-6.25, -5.25, 5.25, 6.25)
    )
    assert sum(plan.metadata["deck_tributary_widths"]) == pytest.approx(12.5)
    assert sorted(plan.metadata["deck_tributary_widths"]) == pytest.approx(
        [0.5, 0.5, 5.75, 5.75]
    )
    for station_x in cross_girder_x:
        station_members = [
            frame
            for frame in slab_transverse
            if model.nodes[frame.i].x == pytest.approx(station_x)
        ]
        transverse_y = {
            model.nodes[node_id].y
            for frame in station_members
            for node_id in (frame.i, frame.j)
        }
        assert len(station_members) == 3
        assert min(transverse_y) == pytest.approx(-6.25)
        assert max(transverse_y) == pytest.approx(6.25)
    assert np.diff(cross_girder_x) == pytest.approx(
        np.full(len(cross_girder_x) - 1, plan.metadata["actual_cross_girder_spacing"]),
        abs=1.0e-12,
    )
    assert all(
        load.load_case != "self_weight"
        for load in model.frame_loads
        if model.frames[load.member].group.startswith("deck_")
    )
    assert [stage.label for stage in plan.stages] == [
        "cable1_steel_A",
        "cable1_deck_weight_B",
        "cable1_composite",
        "cable2_steel_A",
        "cable2_deck_weight_B",
        "cable2_composite",
        "tip_steel_A",
        "tip_deck_weight_B",
        "tip_composite",
        "left_span_steel_A",
        "left_span_deck_weight_B",
        "left_span_composite",
    ]


def test_3d_builder_accepts_independent_backstay_and_main_stay_group_designs():
    plan = build_single_staged_3d(
        n_seg=2,
        strands_per_cable=((10, 20), (30, 40)),
        pretension_per_cable=((1.0e6, 2.0e6), (3.0e6, 4.0e6)),
        pretension_a_ratio=((0.25, 0.50), (0.75, 1.0)),
    )

    for stage, expected in enumerate(((10, 20, 1.0e6, 2.0e6), (30, 40, 3.0e6, 4.0e6)), start=1):
        back_strands, main_strands, back_tension, main_tension = expected
        backstays = [
            cable
            for cable in plan.model.cables.values()
            if cable.construction_stage == stage and cable.group == "backstay"
        ]
        main_stays = [
            cable
            for cable in plan.model.cables.values()
            if cable.construction_stage == stage and cable.group == "main_stay"
        ]
        assert len(backstays) == len(main_stays) == 2
        assert {cable.area for cable in backstays} == {back_strands * 1.4e-4}
        assert {cable.area for cable in main_stays} == {main_strands * 1.4e-4}
        assert {cable.pretension for cable in backstays} == {back_tension}
        assert {cable.pretension for cable in main_stays} == {main_tension}
        expected_ratios = ((0.25, 0.50), (0.75, 1.0))[stage - 1]
        assert {cable.pretension_a for cable in backstays} == {
            back_tension * expected_ratios[0]
        }
        assert {cable.pretension_b for cable in main_stays} == {
            main_tension * (1.0 - expected_ratios[1])
        }


def test_3d_builder_lumps_each_physical_cable_self_weight_at_its_end_nodes():
    plan = build_single_staged_3d(
        n_seg=2,
        left_span=None,
        strands_per_cable=((10, 20), (30, 40)),
        pretension_per_cable=0.0,
    )
    model = plan.model
    loads = [
        load for load in model.nodal_loads if load.load_case == "cable_self_weight"
    ]

    assert len(loads) == 2 * len(model.cables)
    assert all(load.values[:2] == (0.0, 0.0) for load in loads)
    assert all(load.values[3:] == (0.0, 0.0, 0.0) for load in loads)

    actual_by_node_stage: dict[tuple[int, int], float] = {}
    for load in loads:
        key = (load.node, load.activation_stage)
        actual_by_node_stage[key] = actual_by_node_stage.get(key, 0.0) + load.values[2]

    expected_by_node_stage: dict[tuple[int, int], float] = {}
    expected_total = 0.0
    config = plan.metadata["config"]
    for cable in model.cables.values():
        length = np.linalg.norm(
            np.asarray(model.nodes[cable.j].xyz) - np.asarray(model.nodes[cable.i].xyz)
        )
        end_weight = -0.5 * cable.material.density * cable.area * length * config.gravity
        expected_total += 2.0 * end_weight
        for node_id in (cable.i, cable.j):
            key = (node_id, cable.activation_stage)
            expected_by_node_stage[key] = expected_by_node_stage.get(key, 0.0) + end_weight

    assert set(actual_by_node_stage) == set(expected_by_node_stage)
    assert actual_by_node_stage == pytest.approx(expected_by_node_stage, rel=1.0e-12)
    assert sum(load.values[2] for load in loads) == pytest.approx(
        expected_total,
        rel=1.0e-12,
    )


def test_detailed_3d_stage_commits_wet_deck_load_before_composite_activation():
    common = dict(
        n_seg=1,
        left_span=None,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
        pretension_per_cable=((1.0e6, 2.0e6),),
    )
    split_plan = build_single_staged_3d(
        **common,
        pretension_a_ratio=((0.25, 0.75),),
    )
    model = split_plan.model

    assert [stage.phase for stage in split_plan.stages[:3]] == [
        "steel_and_A",
        "deck_weight_and_B",
        "composite",
    ]
    first_main = [
        frame
        for frame in model.frames.values()
        if frame.group == "main_girder" and frame.activation_stage == 1
    ]
    first_slab = [
        frame
        for frame in model.frames.values()
        if frame.group.startswith("deck_") and frame.activation_stage == 3
    ]
    first_temporary_loads = [
        load
        for load in model.frame_loads
        if load.load_case == "temporary_deck_self_weight"
        and load.activation_stage == 2
    ]
    assert first_main and first_slab
    assert {load.member for load in first_temporary_loads} == {
        frame.id for frame in first_main
    }
    assert all(load.is_defined_at(2) and not load.is_defined_at(3) for load in first_temporary_loads)
    expected_line_load = (
        split_plan.metadata["config"].concrete.density
        * split_plan.metadata["deck_width"]
        * split_plan.metadata["config"].deck_thickness
        * split_plan.metadata["config"].gravity
        / 2.0
    )
    assert {load.qz for load in first_temporary_loads} == {-expected_line_load}

    split = SingleStagedDirectSolver3D().run(split_plan)
    steel_a, deck_weight_b, composite = split.records[:3]
    common_nodes = set(deck_weight_b.displacement) & set(composite.displacement)
    assert composite.applied_load == pytest.approx(deck_weight_b.applied_load, abs=1.0e-9)
    assert {
        node_id: composite.displacement[node_id] for node_id in common_nodes
    } == pytest.approx(
        {node_id: deck_weight_b.displacement[node_id] for node_id in common_nodes},
        rel=1.0e-12,
        abs=1.0e-12,
    )
    assert all(
        np.max(np.abs(composite.frame_force[frame.id])) < 1.0e-6
        for frame in first_slab
    )

    all_a = SingleStagedDirectSolver3D().run(
        build_single_staged_3d(**common, pretension_a_ratio=1.0)
    )
    cable_id = min(steel_a.cable_force)
    assert steel_a.cable_force[cable_id] != pytest.approx(
        all_a.records[0].cable_force[cable_id]
    )
    assert deck_weight_b.cable_force[cable_id] == pytest.approx(
        all_a.records[1].cable_force[cable_id],
        rel=1.0e-11,
        abs=1.0e-6,
    )

    with pytest.raises(ValueError, match="between zero and one"):
        build_single_staged_3d(n_seg=1, pretension_a_ratio=1.01)


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
    assert len(result.staged_result.records) == len(
        CableDesignEvaluator3D(problem, config).build_plan(
            [10, 20],
            [1.0e6, 2.0e6],
        ).stages
    )

    ratios = np.asarray([0.25, 0.75])
    staged = CableDesignEvaluator3D(problem, config).evaluate_stage_a(
        [10, 20],
        [1.0e6, 2.0e6],
        ratios,
        1,
    )
    assert staged.construction_stage == 1
    assert staged.stage_index == 1
    assert np.isfinite(staged.backstay_tower_dx_m)
    assert np.isfinite(staged.main_stay_deck_uz_m)
    plan = CableDesignEvaluator3D(problem, config).build_plan(
        [10, 20],
        [1.0e6, 2.0e6],
        ratios,
    )
    backstay = next(cable for cable in plan.model.cables.values() if cable.group == "backstay")
    main_stay = next(cable for cable in plan.model.cables.values() if cable.group == "main_stay")
    assert backstay.pretension_a == pytest.approx(0.25e6)
    assert main_stay.pretension_a == pytest.approx(1.5e6)


def test_efficient_3d_stage_a_controller_solves_balance_directly_without_final_effects():
    problem = CableOptimizationProblem(
        n_seg=2,
        bounds=CableBounds(strand_min=1, strand_max=100),
        backend="opensees",
        model_family="single_staged_3d",
    )

    class LocalControlEvaluator:
        def __init__(self):
            self.problem = problem
            self.layout = CableLayout(problem.n_seg)
            self.final_evaluations = 0

        def default_pretension_a_ratio(self):
            return np.full(self.layout.size, 0.5)

        def evaluate_stage_a(self, strands, pretension, ratios, construction_stage):
            offset = 2 * (construction_stage - 1)
            targets = ((0.2, 0.8), (0.7, 0.3))[construction_stage - 1]
            upper = (
                problem.bounds.tension_bound_stress_mpa
                * 1.0e6
                * problem.strand_area
                * np.asarray(strands, dtype=float)
            )
            applied_a = np.asarray(pretension) * np.asarray(ratios)
            return StageAControlResponse3D(
                construction_stage=construction_stage,
                stage_index=3 * construction_stage - 2,
                backstay_tower_dx_m=float(applied_a[offset] / upper[offset] - targets[0]),
                main_stay_deck_uz_m=float(
                    applied_a[offset + 1] / upper[offset + 1] - targets[1]
                ),
            )

        def evaluate(self, *args, **kwargs):
            self.final_evaluations += 1
            raise AssertionError("stage-A coefficient control must not evaluate final effects")

    evaluator = LocalControlEvaluator()
    progress = []
    optimizer = StagedCableOptimizer3D(
        evaluator,
        Staged3DOptimizationOptions(stage_a=StageAControlOptions()),
        progress=progress.append,
    )
    strands = np.asarray([20, 20, 20, 20])
    pretension_a, controls = optimizer.calculate_initial_tension(strands)
    upper = optimizer.tension_upper_bounds(strands)

    assert pretension_a / upper == pytest.approx([0.2, 0.8, 0.7, 0.3], abs=1.0e-10)
    assert evaluator.final_evaluations == 0
    assert all(item.feasible and item.stable and item.matrix_rank == 2 for item in controls)
    assert all(abs(item.response.backstay_tower_dx_m) < 1.0e-10 for item in controls)
    assert all(abs(item.response.main_stay_deck_uz_m) < 1.0e-10 for item in controls)
    assert any("ETA≈" in message for message in progress)


def test_3d_smooth_curves_are_bounded_and_strand_counts_increase_outward():
    bernstein = build_smooth_curve_basis(24, 4, "bernstein")
    piecewise = build_smooth_curve_basis(24, 4, "piecewise-linear")
    stage_major = build_stage_major_curve_basis(24, 4, "bernstein")

    assert bernstein.shape == (24, 4)
    assert piecewise.shape == (24, 4)
    assert stage_major.shape == (48, 8)
    assert np.all(bernstein >= 0.0)
    assert np.all(piecewise >= 0.0)
    assert np.sum(bernstein, axis=1) == pytest.approx(np.ones(24))
    assert np.sum(piecewise, axis=1) == pytest.approx(np.ones(24))
    assert bernstein[0] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert bernstein[-1] == pytest.approx([0.0, 0.0, 0.0, 1.0])
    assert np.all(stage_major[0::2, 4:] == 0.0)
    assert np.all(stage_major[1::2, :4] == 0.0)

    raw = np.asarray([80, 90, 70, 82, 95, 76, 88, 110, 105, 96, 100, 120])
    curve = project_strands_to_smooth_curve(
        raw,
        n_seg=6,
        control_points=4,
        family="bernstein",
        lower=5,
        upper=500,
    )
    assert curve.interpolated_strands.dtype.kind == "i"
    assert np.all(np.diff(curve.interpolated_strands[0::2]) >= 0)
    assert np.all(np.diff(curve.interpolated_strands[1::2]) >= 0)
    assert np.all((curve.interpolated_strands >= 5) & (curve.interpolated_strands <= 500))


def test_3d_low_dimensional_curve_budget_does_not_grow_with_cable_count():
    problem = CableOptimizationProblem(
        n_seg=24,
        backend="opensees",
        model_family="single_staged_3d",
    )

    class LayoutOnlyEvaluator:
        def __init__(self):
            self.problem = problem
            self.layout = CableLayout(problem.n_seg)

    bernstein = StagedCableOptimizer3D(
        LayoutOnlyEvaluator(),
        Staged3DOptimizationOptions(
            secondary=SecondaryTensionOptions3D(
                curve_family="bernstein",
                control_points_per_group=4,
            )
        ),
    )
    automatic = StagedCableOptimizer3D(
        LayoutOnlyEvaluator(),
        Staged3DOptimizationOptions(
            secondary=SecondaryTensionOptions3D(
                curve_family="auto",
                control_points_per_group=4,
            )
        ),
    )

    assert bernstein.secondary_fem_cases_per_cycle() == 10
    assert automatic.secondary_fem_cases_per_cycle() == 19


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
        stage.label for stage in plan.stages
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


def test_3d_tangent_history_accumulates_every_pre_activation_stage():
    config = replace(
        load_single_staged_3d_config("omo3d"),
        n_seg=3,
        pretension_per_cable=0.0,
    )
    plan = build_single_staged_3d(config)
    model = plan.model
    history = TangentDisplacementHistory3D(model)
    displacement = {}

    first = next(
        node
        for node in model.nodes.values()
        if node.role == "main_girder_0" and node.x == pytest.approx(-24.0)
    )
    second = next(
        node
        for node in model.nodes.values()
        if node.role == "main_girder_0" and node.x == pytest.approx(-36.0)
    )
    third = next(
        node
        for node in model.nodes.values()
        if node.role == "main_girder_0" and node.x == pytest.approx(-48.0)
    )

    history.activate(1, displacement)
    stage1 = {node_id: np.zeros(6) for node_id in displacement}
    stage1[first.id][2] = -0.10
    stage1[first.id][4] = 0.01
    history.accumulate(1, stage1)

    stage2_birth = history.activate(4, displacement)
    assert stage2_birth[second.id][2] == pytest.approx(0.02, abs=1.0e-14)
    assert history.virtual[third.id][2, 0] == pytest.approx(0.14, abs=1.0e-14)

    stage2 = {node_id: np.zeros(6) for node_id in displacement}
    stage2[second.id][2] = -0.02
    stage2[second.id][4] = -0.005
    history.accumulate(4, stage2)

    stage3_birth = history.activate(7, displacement)
    assert stage3_birth[third.id][2] == pytest.approx(0.06, abs=1.0e-14)
    assert stage3_birth[third.id][4] == pytest.approx(0.005, abs=1.0e-14)


def test_3d_new_cable_force_does_not_change_its_beam_birth_reference():
    common = dict(
        n_seg=2,
        strands_per_cable=((100, 100), (100, 100)),
        pretension_a_ratio=((1.0, 1.0), (1.0, 1.0)),
    )
    unloaded_second = build_single_staged_3d(
        **common,
        pretension_per_cable=((1.0e6, 1.0e6), (0.0, 0.0)),
    )
    tensioned_second = build_single_staged_3d(
        **common,
        pretension_per_cable=((1.0e6, 1.0e6), (4.0e6, 4.0e6)),
    )
    solver = SingleStagedDirectSolver3D()
    unloaded = solver.solve_stage(unloaded_second, 4)
    tensioned = solver.solve_stage(tensioned_second, 4)
    deck_nodes = {
        cable.j
        for cable in tensioned_second.model.cables.values()
        if cable.group == "main_stay" and cable.construction_stage == 2
    }

    assert deck_nodes.issubset(unloaded.birth_displacement)
    assert {
        node_id: tensioned.birth_displacement[node_id]
        for node_id in deck_nodes
    } == pytest.approx(
        {
            node_id: unloaded.birth_displacement[node_id]
            for node_id in deck_nodes
        },
        abs=1.0e-12,
    )


def test_3d_backends_report_historical_birth_displacements_for_new_nodes():
    pytest.importorskip("openseespy.opensees")
    config = replace(
        load_single_staged_3d_config("omo3d"),
        n_seg=2,
        pretension_per_cable=0.0,
    )
    plan = build_single_staged_3d(config)

    direct = SingleStagedDirectSolver3D().solve_stage(plan, 4)
    reference = SingleStagedOpenSeesSolver3D().solve_stage(plan, 4)

    assert direct.birth_displacement
    assert set(reference.birth_displacement) == set(direct.birth_displacement)
    for record in (direct, reference):
        assert set(record.birth_displacement).issubset(record.displacement)
        assert all(
            plan.model.nodes[node_id].activation_stage == 4
            for node_id in record.birth_displacement
        )
        assert np.isfinite(np.asarray(list(record.birth_displacement.values()))).all()


def test_3d_secondary_line_load_and_deck_pressure_activate_in_separate_final_stage():
    line_load = 4.0e3
    deck_pressure = 2.5e3
    plan = build_single_staged_3d(
        n_seg=1,
        left_span=5.0,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
        pretension_per_cable=0.0,
        secondary_main_girder_line_load=line_load,
        secondary_deck_pressure=deck_pressure,
    )
    model = plan.model

    assert [stage.label for stage in plan.stages] == [
        "cable1_steel_A",
        "cable1_deck_weight_B",
        "cable1_composite",
        "tip_steel_A",
        "tip_deck_weight_B",
        "tip_composite",
        "left_span_steel_A",
        "left_span_deck_weight_B",
        "left_span_composite",
        "secondary_load",
    ]
    secondary_stage = plan.final_stage.index
    assert plan.metadata["secondary_load_stage"] == secondary_stage
    main_loads = [
        load
        for load in model.frame_loads
        if load.load_case == "secondary_main_girder_line"
    ]
    pressure_loads = [
        load
        for load in model.frame_loads
        if load.load_case == "secondary_deck_pressure"
    ]
    main_frames = [frame for frame in model.frames.values() if frame.group == "main_girder"]
    deck_frames = [
        frame for frame in model.frames.values() if frame.group == "deck_longitudinal"
    ]
    assert {load.member for load in main_loads} == {frame.id for frame in main_frames}
    assert {load.member for load in pressure_loads} == {frame.id for frame in deck_frames}
    assert {load.qz for load in main_loads} == {-line_load}
    assert all(
        load.qz
        == pytest.approx(-deck_pressure * model.frames[load.member].section.width_y)
        for load in pressure_loads
    )
    assert sum(plan.metadata["deck_tributary_widths"]) == pytest.approx(
        plan.metadata["deck_width"]
    )
    assert all(load.activation_stage == secondary_stage for load in main_loads + pressure_loads)

    solver = SingleStagedDirectSolver3D()
    before = solver.solve_stage(plan, secondary_stage - 1)
    final = solver.solve_stage(plan, secondary_stage)

    def loaded_length(load):
        frame = model.frames[load.member]
        start = np.asarray(model.nodes[frame.i].xyz)
        end = np.asarray(model.nodes[frame.j].xyz)
        return np.linalg.norm(end - start)

    expected_increment = sum(
        load.qz * loaded_length(load) for load in main_loads + pressure_loads
    )
    assert final.applied_load[2] - before.applied_load[2] == pytest.approx(
        expected_increment,
        rel=1.0e-12,
        abs=1.0e-6,
    )
    assert final.displacement != before.displacement

    legacy = build_single_staged_3d(n_seg=1, superimposed_dead_load=deck_pressure)
    assert legacy.final_stage.label == "secondary_load"
    with pytest.raises(ValueError, match="not both"):
        build_single_staged_3d(
            n_seg=1,
            secondary_deck_pressure=deck_pressure,
            superimposed_dead_load=deck_pressure,
        )


def test_direct_3d_batch_solver_matches_scalar_and_uses_one_multi_rhs_solve(monkeypatch):
    common = {
        "n_seg": 2,
        "left_span": 5.0,
        "cross_girder_spacing": 10.0,
        "tower_element_size": 10.0,
        "strands_per_cable": ((55, 65), (55, 65)),
    }
    plans = [
        build_single_staged_3d(
            **common,
            pretension_per_cable=tensions,
        )
        for tensions in (
            ((0.0, 0.0), (0.0, 0.0)),
            ((1.0e6, 2.0e6), (1.5e6, 2.5e6)),
            ((3.0e6, 4.0e6), (3.5e6, 4.5e6)),
        )
    ]
    stage = plans[0].final_stage
    scalar = [
        SingleStagedDirectSolver3D().solve_stage(plan, stage.index, stage.label)
        for plan in plans
    ]

    from scipy.linalg import cho_factor as original_factor
    from scipy.linalg import cho_solve as original_solve

    factor_calls = []
    solve_rhs_shapes = []

    def counted_factor(matrix, *args, **kwargs):
        factor_calls.append(matrix.shape)
        return original_factor(matrix, *args, **kwargs)

    def counted_solve(factor, right_hand_side, *args, **kwargs):
        solve_rhs_shapes.append(right_hand_side.shape)
        return original_solve(factor, right_hand_side, *args, **kwargs)

    monkeypatch.setattr("scipy.linalg.cho_factor", counted_factor)
    monkeypatch.setattr("scipy.linalg.cho_solve", counted_solve)
    batch = SingleStagedDirectBatchSolver3D().solve_stage_batch(
        plans,
        stage.index,
        stage.label,
    )

    assert len(factor_calls) == len(plans[0].stages)
    assert len(solve_rhs_shapes) == len(plans[0].stages)
    assert all(shape[1] == len(plans) for shape in solve_rhs_shapes)
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
            # Multi-RHS BLAS changes the summation order relative to a
            # single-column solve.  Only recovered frame end forces amplify
            # that machine-level displacement difference across the complete
            # construction history; 1e-5 N/Nm remains negligible, while
            # displacement/cable/reaction guards stay much tighter.
            rel_tol, abs_tol = (
                (1.0e-9, 1.0e-5)
                if field == "frame_force"
                else (1.0e-11, 1.0e-8)
            )
            for object_id in batched_values:
                assert batched_values[object_id] == pytest.approx(
                    separate_values[object_id],
                    rel=rel_tol,
                    abs=abs_tol,
                )

    different_area = build_single_staged_3d(
        **{**common, "strands_per_cable": ((56, 65), (55, 65))},
        pretension_per_cable=((1.0e6, 2.0e6), (1.5e6, 2.5e6)),
    )
    with pytest.raises(ValueError, match="cable structure"):
        SingleStagedDirectBatchSolver3D().solve_stage_batch(
            [plans[0], different_area],
            stage.index,
            stage.label,
        )

    with pytest.raises(ValueError, match="retired and must equal 1.0"):
        build_single_staged_3d(
            **common,
            pretension_per_cable=((1.0e6, 2.0e6), (1.5e6, 2.5e6)),
            flexible_birth_correction_factor=0.5,
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

    from scipy.linalg import cho_factor as original_factor
    from scipy.linalg import cho_solve as original_solve

    factor_calls = []
    solve_calls = []

    def counted_factor(matrix, *args, **kwargs):
        factor_calls.append(matrix.shape)
        return original_factor(matrix, *args, **kwargs)

    def counted_solve(factor, right_hand_side, *args, **kwargs):
        solve_calls.append(right_hand_side.shape)
        return original_solve(factor, right_hand_side, *args, **kwargs)

    monkeypatch.setattr("scipy.linalg.cho_factor", counted_factor)
    monkeypatch.setattr("scipy.linalg.cho_solve", counted_solve)
    affine = build_affine_model(evaluator, strands)

    stage_count = len(evaluator.build_plan(strands, np.zeros(2)).stages)
    assert len(factor_calls) == stage_count
    assert len(solve_calls) == stage_count
    assert all(shape[1] == evaluator.layout.size + 1 for shape in solve_calls)
    tension = np.asarray([2.2e6, 3.4e6])
    actual = evaluator.evaluate(strands, tension)
    assert len(factor_calls) == 2 * stage_count
    assert len(solve_calls) == 2 * stage_count
    assert all(shape[1] == 1 for shape in solve_calls[stage_count:])
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
        n_seg=2,
        left_span=5.0,
        pretension_per_cable=0.0,
        secondary_main_girder_line_load=4.0e3,
        secondary_deck_pressure=2.5e3,
    )
    direct = SingleStagedDirectSolver3D().run(plan).final
    reference = SingleStagedOpenSeesSolver3D().run(plan).final

    assert direct.converged and reference.converged
    assert reference.applied_load == pytest.approx(
        direct.applied_load,
        rel=1.0e-12,
        abs=1.0e-6,
    )
    beam_nodes = plan.metadata["beam_grid_node_ids"]
    for node_id in beam_nodes:
        assert reference.displacement[node_id] == pytest.approx(
            direct.displacement[node_id],
            rel=2.0e-6,
            abs=2.0e-9,
        )


def test_opensees_3d_result_renders_stage_frames_and_gif_without_implicit_dxf(tmp_path):
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
    assert artifacts["dxf"] is None
    assert not output.with_suffix(".dxf").exists()


def test_final_3d_dxf_uses_true_model_coordinates_and_named_layers(tmp_path):
    import ezdxf
    from ezdxf import units

    plan = build_single_staged_3d(
        n_seg=1,
        left_span=5.0,
        cross_girder_spacing=10.0,
        tower_element_size=10.0,
        pretension_per_cable=0.0,
    )
    result = SingleStagedDirectSolver3D().run(plan)
    output = tmp_path / "final_geometry.dxf"

    assert export_final_3d_dxf(plan, result, output) == output
    document = ezdxf.readfile(output)
    assert document.units == units.M
    entities = list(document.modelspace())
    final = result.final
    model = plan.model
    active_nodes = set(final.displacement)
    active_frames = [
        frame
        for frame in model.frames.values()
        if frame.activation_stage <= final.stage_index
        and frame.i in active_nodes
        and frame.j in active_nodes
    ]
    active_links = [
        link
        for link in model.rigid_links.values()
        if link.activation_stage <= final.stage_index
        and link.master in active_nodes
        and link.slave in active_nodes
    ]

    points = [entity for entity in entities if entity.dxftype() == "POINT"]
    lines = [entity for entity in entities if entity.dxftype() == "LINE"]
    faces = [entity for entity in entities if entity.dxftype() == "3DFACE"]
    assert len(points) == len(active_nodes)
    assert len(lines) == len(active_frames) + len(final.cable_force) + len(active_links)
    assert faces
    face_y = [
        float(getattr(face.dxf, f"vtx{vertex}").y)
        for face in faces
        for vertex in range(4)
    ]
    assert min(face_y) == pytest.approx(-plan.metadata["deck_width"] / 2.0)
    assert max(face_y) == pytest.approx(plan.metadata["deck_width"] / 2.0)
    assert {
        "BZ_MAIN_GIRDER",
        "BZ_CROSS_GIRDER",
        "BZ_DECK_PANEL",
        "BZ_TOWER",
        "BZ_MAIN_STAY",
        "BZ_BACKSTAY",
        "BZ_RIGID_LINK",
        "BZ_NODE",
    }.issubset({entity.dxf.layer for entity in entities})

    frame = next(item for item in active_frames if item.group == "main_girder")
    start = model.nodes[frame.i].xyz
    end = model.nodes[frame.j].xyz
    main_lines = [entity for entity in lines if entity.dxf.layer == "BZ_MAIN_GIRDER"]
    assert any(
        tuple(entity.dxf.start) == pytest.approx(start)
        and tuple(entity.dxf.end) == pytest.approx(end)
        for entity in main_lines
    )


def test_minimum_3d_cli_writes_machine_readable_json_and_detailed_text(
    tmp_path, capsys
):
    output = tmp_path / "single_staged_3d.json"
    render_output = tmp_path / "text_mode_must_not_write.gif"
    forbidden_dxf = render_output.with_suffix(".dxf")
    assert run_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "1",
            "--backend",
            "direct",
            "--render",
            "text",
            "--out",
            str(render_output),
            "--output",
            str(output),
        ]
    ) == 0
    rendered = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema"] == "bridgezoo.single_staged_3d.result.v1"
    assert payload["backend"] == "direct3d"
    assert payload["model"]["coordinate_system"] == "x longitudinal, y transverse, z vertical"
    assert payload["final"]["converged"] is True
    assert payload["input"]["n_seg"] == 1
    assert payload["input"]["flexible_birth_correction_factor"] == pytest.approx(
        1.0
    )
    assert payload["input"]["deck_width"] == pytest.approx(13.4)
    assert "3D rendering" not in payload["todo"]
    assert payload["deformation_convention"]["translation_unit"] == "m"
    assert payload["deformation_convention"]["rotation_unit"] == "rad"
    assert all(
        stage["deformation"]["node_count"]
        == len(stage["deformation"]["nodes"])
        for stage in payload["stages"]
    )
    assert not forbidden_dxf.exists()
    lines = rendered.splitlines()
    assert "主梁控制点位移（当前施工组横桥向控制节点平均" in rendered
    assert sum(line.startswith("阶段 ") for line in lines) == 3
    assert sum(line.startswith("  梁点 S03 ") for line in lines) == 3
    assert ": 未激活" not in rendered
    assert "最终阶段拉索应力" in rendered
    assert sum(line.startswith("  拉索 id=") for line in lines) == 4
    assert "最终阶段塔顶位移" in rendered
    assert "ux=" in rendered
    assert "uy=" in rendered
    assert "uz=" in rendered

    text_dir = tmp_path / "text_mode_must_not_write_text"
    total_files = sorted((text_dir / "main_girder_actual_total").glob("*.txt"))
    delta_files = sorted((text_dir / "main_girder_stage_delta").glob("*.txt"))
    assert len(total_files) == len(delta_files) == len(payload["stages"])
    assert total_files[0].name == "stage_001_cable1_steel_A.txt"
    assert delta_files[0].name == "stage_001_cable1_steel_A.txt"

    def displacement_rows(path):
        rows = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("node_id "):
                continue
            fields = line.split()
            rows[int(fields[0])] = fields
        return rows

    total_rows = displacement_rows(total_files[0])
    delta_rows = displacement_rows(delta_files[0])
    node_id = payload["stages"][0]["main_girder_control_node_ids"][0]
    first_main_node = next(
        node
        for node in payload["stages"][0]["deformation"]["nodes"]
        if node["node_id"] == node_id
    )
    assert set(total_rows) == set(delta_rows)
    assert float(total_rows[node_id][7]) == pytest.approx(
        first_main_node["actual_total_deformation"]["translation_m"]["uz"]
        * 1000.0,
        abs=1.0e-9,
    )
    assert float(delta_rows[node_id][7]) == pytest.approx(
        first_main_node["step_deformation"]["translation_m"]["uz"] * 1000.0,
        abs=1.0e-9,
    )
    assert "本阶段增量变形 delta" in delta_files[0].read_text(encoding="utf-8")

    z_summary = text_dir / "main_girder_z_summary.txt"
    summary_lines = [
        line
        for line in z_summary.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert summary_lines[0] == (
        "stage_index stage_label node_id delta_uz_mm actual_total_uz_mm"
    )
    expected_row_count = sum(
        len(stage["main_girder_control_node_ids"])
        for stage in payload["stages"]
    )
    assert len(summary_lines) - 1 == expected_row_count
    first_summary = summary_lines[1].split()
    assert first_summary[:3] == [
        str(payload["stages"][0]["index"]),
        payload["stages"][0]["label"],
        str(node_id),
    ]
    assert float(first_summary[3]) == pytest.approx(
        first_main_node["step_deformation"]["translation_m"]["uz"] * 1000.0,
        abs=1.0e-9,
    )
    assert float(first_summary[4]) == pytest.approx(
        first_main_node["actual_total_deformation"]["translation_m"]["uz"]
        * 1000.0,
        abs=1.0e-9,
    )


def test_3d_json_deformation_separates_activation_step_and_actual_total(
    tmp_path,
):
    output = tmp_path / "deformation_history.json"
    assert run_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "1",
            "--backend",
            "direct",
            "--render",
            "none",
            "--output",
            str(output),
        ]
    ) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    first, second = payload["stages"][:2]
    first_nodes = {
        item["node_id"]: item for item in first["deformation"]["nodes"]
    }
    second_nodes = {
        item["node_id"]: item for item in second["deformation"]["nodes"]
    }

    def translation(item, field):
        values = item[field]["translation_m"]
        return np.asarray([values["ux"], values["uy"], values["uz"]])

    def rotation(item, field):
        values = item[field]["rotation_rad"]
        return np.asarray([values["rx"], values["ry"], values["rz"]])

    newly_activated = next(
        item
        for item in first_nodes.values()
        if item["role"].startswith("main_girder")
    )
    activation_reference = newly_activated["activation"]["reference_deformation"]
    reference_translation = np.asarray(
        list(activation_reference["translation_m"].values())
    )
    reference_rotation = np.asarray(
        list(activation_reference["rotation_rad"].values())
    )
    assert (
        reference_translation
        + translation(newly_activated, "step_deformation")
    ) == pytest.approx(
        translation(newly_activated, "actual_total_deformation"),
        abs=1.0e-12,
    )
    assert (
        reference_rotation + rotation(newly_activated, "step_deformation")
    ) == pytest.approx(
        rotation(newly_activated, "actual_total_deformation"),
        abs=1.0e-12,
    )
    design_position = np.asarray(
        list(newly_activated["design_position_m"].values())
    )
    activation_position = np.asarray(
        list(newly_activated["activation"]["position_m"].values())
    )
    actual_position = np.asarray(
        list(newly_activated["actual_position_m"].values())
    )
    assert activation_position == pytest.approx(
        design_position + reference_translation,
        abs=1.0e-12,
    )
    assert actual_position == pytest.approx(
        design_position
        + translation(newly_activated, "actual_total_deformation"),
        abs=1.0e-12,
    )

    existing_id = next(
        node_id
        for node_id, item in first_nodes.items()
        if item["role"] in {"tower", "tower_anchor"}
        and node_id in second_nodes
    )
    assert (
        translation(first_nodes[existing_id], "actual_total_deformation")
        + translation(second_nodes[existing_id], "step_deformation")
    ) == pytest.approx(
        translation(second_nodes[existing_id], "actual_total_deformation"),
        abs=1.0e-12,
    )
    assert second_nodes[existing_id]["activation"] == first_nodes[existing_id][
        "activation"
    ]


def test_3d_cli_text_all_stages_prints_every_analysis_stage(tmp_path, capsys):
    output = tmp_path / "all_stages.json"
    text_dir = tmp_path / "all_stages_text"
    assert run_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "1",
            "--backend",
            "direct",
            "--render",
            "text",
            "--text-all-stages",
            "--text-dir",
            str(text_dir),
            "--output",
            str(output),
        ]
    ) == 0
    rendered = capsys.readouterr().out
    payload = json.loads(output.read_text(encoding="utf-8"))
    stage_lines = [
        line for line in rendered.splitlines() if line.startswith("阶段 ")
    ]

    assert "全部10个分析阶段" in rendered
    assert len(stage_lines) == len(payload["stages"]) == 10
    assert stage_lines[0].endswith("cable1_steel_A")
    assert stage_lines[-1].endswith("secondary_load")
    assert ": 未激活" not in rendered
    control_sets = [
        tuple(stage["main_girder_control_node_ids"])
        for stage in payload["stages"]
    ]
    assert control_sets[0] == control_sets[1] == control_sets[2]
    assert control_sets[3] == control_sets[4] == control_sets[5]
    assert control_sets[6] == control_sets[7] == control_sets[8]
    assert len({control_sets[0], control_sets[3], control_sets[6]}) == 3
    assert set(control_sets[-1]) == set().union(*map(set, control_sets[:-1]))
    for stage, control_ids in zip(payload["stages"], control_sets):
        active_ids = {
            node["node_id"] for node in stage["deformation"]["nodes"]
        }
        assert set(control_ids).issubset(active_ids)
    assert len(list((text_dir / "main_girder_actual_total").glob("*.txt"))) == 10
    assert len(list((text_dir / "main_girder_stage_delta").glob("*.txt"))) == 10


def test_3d_cli_exports_dxf_independently_without_plot_rendering(tmp_path):
    output = tmp_path / "standalone_geometry.dxf"

    assert run_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "1",
            "--backend",
            "direct",
            "--render",
            "none",
            "--dxf",
            "--dxf-out",
            str(output),
        ]
    ) == 0
    assert output.stat().st_size > 1000


def test_3d_cli_rejects_dxf_path_without_dxf_switch(tmp_path):
    with pytest.raises(ValueError, match="--dxf-out requires --dxf"):
        run_cli(["--dxf-out", str(tmp_path / "unused.dxf")])


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
        "opensees",
        "--strand-iterations",
        "0",
        "--secondary-max-nfev",
        "4",
        "--initial-strands",
        "91",
        "--quiet",
        "--out",
        str(out_dir),
    ]

    assert optimize_3d_cli(base_args) == 0
    assert optimize_3d_cli([*base_args, "--resume"]) == 0
    payload = json.loads((out_dir / "best_design.json").read_text(encoding="utf-8"))

    assert payload["schema"] == "bridgezoo.cable_optimization_3d.v3"
    assert payload["model_family"] == "single_staged_3d"
    assert payload["search"]["run_index"] == 2
    assert [item["group"] for item in payload["cable_groups"]] == [
        "backstay",
        "main_stay",
    ]
    assert all(len(item["physical_cable_ids"]) == 2 for item in payload["cable_groups"])
    assert all(0.0 <= item["pretension_a_ratio"] <= 1.0 for item in payload["cable_groups"])
    assert all(
        item["pretension_A_per_physical_cable_N"]
        + item["pretension_B_per_physical_cable_N"]
        == pytest.approx(item["pretension_per_physical_cable_N"])
        for item in payload["cable_groups"]
    )
    assert len(payload["stage_a_controls"]) == 1
    assert payload["stage_a_controls"][0]["influence_matrix_rank"] == 2
    assert "final_schedule_backstay_tower_dx_mm" in payload["stage_a_controls"][0]
    assert payload["stage_b_response_model"]["validated"] is True
    assert payload["stage_b_response_model"]["selected_curve_family"] == "bernstein"
    assert payload["stage_b_response_model"]["control_points_per_group"] == 1
    assert payload["smooth_curves"]["strand_count"][
        "monotone_non_decreasing_outward"
    ] is True
    assert len(
        payload["smooth_curves"]["secondary_tension_B"][
            "interpolated_stage_major_tension_N"
        ]
    ) == 2
    assert payload["search"]["OpenSees_FEM_cases_this_run"] == 8
    assert (out_dir / "history.csv").is_file()
    assert (out_dir / "summary.txt").is_file()

    analysis_output = tmp_path / "optimized_analysis.json"
    assert run_cli(
        [
            "--bridge",
            "omo3d",
            "--design",
            str(out_dir / "best_design.json"),
            "--backend",
            "opensees",
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
    assert analysis["input"]["pretension_a_ratio"] == [
        [
            payload["cable_groups"][0]["pretension_a_ratio"],
            payload["cable_groups"][1]["pretension_a_ratio"],
        ]
    ]
    for group in payload["cable_groups"]:
        for member_id, stress_mpa in group["physical_final_stress_MPa"].items():
            assert analysis["final"]["cable_stress_Pa"][member_id] / 1.0e6 == pytest.approx(
                stress_mpa,
                rel=1.0e-9,
                abs=1.0e-9,
            )

    tampered = json.loads(json.dumps(payload))
    tampered["cable_groups"][0]["pretension_A_per_physical_cable_N"] += 10.0
    tampered_path = tmp_path / "tampered_design.json"
    tampered_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="do not match T and coefficient"):
        run_cli(
            [
                "--bridge",
                "omo3d",
                "--n",
                "1",
                "--design",
                str(tampered_path),
                "--render",
                "none",
            ]
        )

    with pytest.raises(ValueError, match="geometry/materials differ"):
        run_cli(
            [
                "--bridge",
                "omo3d",
                "--n",
                "2",
                "--design",
                str(out_dir / "best_design.json"),
                "--render",
                "none",
            ]
        )

    capsys.readouterr()
    with pytest.raises(SystemExit):
        optimize_3d_cli(["--bridge", "omo3d", "--backend", "direct"])


def test_3d_cli_interpolates_low_dimensional_curve_to_every_group(tmp_path):
    pytest.importorskip("openseespy.opensees")
    out_dir = tmp_path / "smooth_opt3d"

    assert optimize_3d_cli(
        [
            "--bridge",
            "omo3d",
            "--n",
            "3",
            "--curve-family",
            "bernstein",
            "--curve-control-points",
            "2",
            "--secondary-max-nfev",
            "4",
            "--strand-iterations",
            "0",
            "--initial-strands",
            "91",
            "--quiet",
            "--out",
            str(out_dir),
        ]
    ) == 0
    payload = json.loads((out_dir / "best_design.json").read_text(encoding="utf-8"))

    assert len(payload["cable_groups"]) == 6
    assert payload["search"]["OpenSees_FEM_cases_this_run"] == 18
    assert payload["stage_b_response_model"]["control_points_per_group"] == 2
    assert payload["stage_b_response_model"]["validated"] is True
    assert len(
        payload["smooth_curves"]["secondary_tension_B"][
            "interpolated_stage_major_tension_N"
        ]
    ) == 6
    strands = np.asarray(
        [item["strands_per_physical_cable"] for item in payload["cable_groups"]]
    )
    assert np.all(np.diff(strands[0::2]) >= 0)
    assert np.all(np.diff(strands[1::2]) >= 0)
