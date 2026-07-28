import json

import numpy as np
import pytest

from bridgezoo.fem.single_staged import (
    HSection3D,
    HollowBoxSection3D,
    RectangularSection3D,
    SingleStagedDirectSolver3D,
    SingleStagedOpenSeesSolver3D,
    build_single_staged_3d,
)
from bridgezoo.render.staged3d import render_staged_3d
from scripts.bridge_config import load_single_staged_3d_config, resolve_bridge_config
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
