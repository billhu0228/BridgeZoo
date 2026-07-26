from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import optimize_cables
from scripts.bridge_config import (
    load_bridge_config,
    model_family_for_bridge_type,
    resolve_bridge_config,
    staged_api_for_bridge_type,
)
from scripts.staged_analysis import parse_args as parse_staged_args
from scripts.validate_staged import parse_args as parse_validate_args


def test_bundled_bridge_configs_keep_previous_values():
    model = load_bridge_config("model")
    p4b = load_bridge_config("p4b")
    omo = load_bridge_config("omo")

    assert model["bridge_type"] == "normal"
    assert model["n"] == 6
    assert model["beam_Iz"] == pytest.approx(10.0 / 12.0)
    assert model["dw"] == 0.0
    assert model["tower_stiffness"] == [[0.0, 1.0e18], [52.0, 1.0e18]]
    assert model["tower_element_size"] == 2.0
    assert p4b["bridge_type"] == "normal"
    assert p4b["n"] == 19
    assert p4b["wg"] == 3.2e5
    assert p4b["beam_E"] == 200e9
    assert p4b["tower_stiffness"][-1][0] == 110.0
    assert omo["bridge_type"] == "single"
    assert omo["right_fix"] == pytest.approx(3.0)
    assert omo["left_span"] == pytest.approx(25.0)


def test_bridge_config_alias_resolves_to_yaml():
    assert resolve_bridge_config("p4b").name == "p4b_defaults.yaml"
    assert resolve_bridge_config("model_defaults").name == "model_defaults.yaml"
    assert resolve_bridge_config("omo").name == "omo_bridge.yaml"
    assert resolve_bridge_config("omo_bridge").name == "omo_bridge.yaml"


def test_bridge_types_dispatch_to_matching_model_modules():
    normal_builder, normal_direct, _ = staged_api_for_bridge_type("normal")
    single_builder, single_direct, _ = staged_api_for_bridge_type("single")

    assert normal_builder.__module__ == "bridgezoo.fem.staged.builder"
    assert normal_direct.__module__ == "bridgezoo.fem.staged.direct"
    assert single_builder.__module__ == "bridgezoo.fem.single_staged.builder"
    assert single_direct.__module__ == "bridgezoo.fem.single_staged.direct"
    assert model_family_for_bridge_type("normal") == "staged"
    assert model_family_for_bridge_type("single") == "single_staged"


def test_staged_analysis_uses_selected_bridge_and_allows_cli_override():
    args = parse_staged_args(["--bridge", "model", "--wg", "12345", "--render", "text"])

    assert args.bridge == "model"
    assert args.bridge_defaults["bridge_type"] == "normal"
    assert args.bridge_defaults["n"] == 6
    assert args.anchor_base == 32.0
    assert args.wg == 12345.0


def test_optimization_requires_explicit_bridge_yaml():
    with pytest.raises(SystemExit):
        optimize_cables.parse_args([])

    args = optimize_cables.parse_args(["--bridge", "model"])
    assert args.bridge == "model"
    assert args.bridge_defaults["n"] == 6


def test_best_design_payload_records_canonical_bridge_yaml():
    ev = SimpleNamespace(
        objective=1.0,
        components=SimpleNamespace(
            shape=0.1,
            total_strands=0.2,
            stress_uniform=0.3,
            stress_violation=0.4,
        ),
        metrics=SimpleNamespace(
            shape_rmse_m=0.001,
            shape_max_abs_m=0.002,
            total_strands=10,
            stress_mean_mpa=500.0,
            stress_std_mpa=1.0,
            stress_min_mpa=499.0,
            stress_max_mpa=501.0,
            stress_violation_rms_mpa=0.0,
            stress_violation_max_mpa=0.0,
        ),
        cable_ids=[1001],
        design=SimpleNamespace(strands=[10], pretension=[1.0e6]),
        cable_stress_mpa={1001: 500.0},
        deck_errors_m={},
    )
    bridge_yaml = optimize_cables._bridge_yaml_reference("model")

    payload = optimize_cables._evaluation_payload(ev, bridge_yaml)

    assert payload["bridge_yaml"] == "scripts/bridges/model_defaults.yaml"


def test_staged_analysis_restores_bridge_yaml_from_design(tmp_path: Path):
    design = tmp_path / "best_design.json"
    design.write_text(
        """
        {
          "bridge_yaml": "scripts/bridges/model_defaults.yaml",
          "cables": [{"cable_id": 1001, "strands": 10, "pretension_N": 1000000.0}]
        }
        """,
        encoding="utf-8",
    )

    args = parse_staged_args(["--design", str(design), "--render", "text"])

    assert args.bridge == "scripts/bridges/model_defaults.yaml"
    assert args.bridge_defaults["n"] == 6


def test_staged_analysis_rejects_missing_or_conflicting_design_yaml(tmp_path: Path):
    legacy = tmp_path / "legacy_design.json"
    legacy.write_text(
        '{"cables": [{"cable_id": 1001, "strands": 10, "pretension_N": 1000000.0}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="bridge_yaml"):
        parse_staged_args(["--design", str(legacy)])

    design = tmp_path / "best_design.json"
    design.write_text(
        """
        {
          "bridge_yaml": "scripts/bridges/model_defaults.yaml",
          "cables": [{"cable_id": 1001, "strands": 10, "pretension_N": 1000000.0}]
        }
        """,
        encoding="utf-8",
    )
    with pytest.raises(SystemExit):
        parse_staged_args(["--design", str(design), "--bridge", "p4b"])


def test_validate_staged_can_use_p4b_defaults():
    args = parse_validate_args(["--bridge", "p4b"])

    assert args.n == 19
    assert args.bridge_defaults["bridge_type"] == "normal"
    assert args.anchor_base == 60.0
    assert args.right_end == 8.0


def test_bridge_config_rejects_missing_keys(tmp_path: Path):
    path = tmp_path / "incomplete.yaml"
    path.write_text("n: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing bridge config keys"):
        load_bridge_config(path)


def test_bridge_config_rejects_unknown_bridge_type(tmp_path: Path):
    source = resolve_bridge_config("model").read_text(encoding="utf-8")
    path = tmp_path / "unknown_type.yaml"
    path.write_text(source.replace("bridge_type: normal", "bridge_type: other"), encoding="utf-8")

    with pytest.raises(ValueError, match="bridge_type"):
        load_bridge_config(path)


def test_single_only_geometry_is_required_for_single_and_rejected_for_normal(tmp_path: Path):
    single_source = resolve_bridge_config("omo").read_text(encoding="utf-8")
    missing = tmp_path / "missing_right_fix.yaml"
    missing.write_text(
        "\n".join(line for line in single_source.splitlines() if not line.startswith("right_fix:")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires 'right_fix'"):
        load_bridge_config(missing)

    missing_span = tmp_path / "missing_left_span.yaml"
    missing_span.write_text(
        "\n".join(line for line in single_source.splitlines() if not line.startswith("left_span:")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="requires 'left_span'"):
        load_bridge_config(missing_span)

    normal_source = resolve_bridge_config("model").read_text(encoding="utf-8")
    normal = tmp_path / "normal_right_fix.yaml"
    normal.write_text(normal_source + "\nright_fix: 3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only valid for 'single'"):
        load_bridge_config(normal)

    normal_span = tmp_path / "normal_left_span.yaml"
    normal_span.write_text(normal_source + "\nleft_span: 25.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="only valid for 'single'"):
        load_bridge_config(normal_span)
