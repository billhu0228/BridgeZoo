from pathlib import Path

import pytest

from scripts.bridge_config import load_bridge_config, resolve_bridge_config
from scripts.staged_analysis import parse_args as parse_staged_args
from scripts.validate_staged import parse_args as parse_validate_args


def test_bundled_bridge_configs_keep_previous_values():
    model = load_bridge_config("model")
    p4b = load_bridge_config("p4b")

    assert model["n"] == 6
    assert model["beam_Iz"] == pytest.approx(10.0 / 12.0)
    assert model["dw"] == 0.0
    assert model["tower_stiffness"] == [[0.0, 1.0e18], [52.0, 1.0e18]]
    assert model["tower_element_size"] == 2.0
    assert p4b["n"] == 19
    assert p4b["wg"] == 3.2e5
    assert p4b["beam_E"] == 200e9
    assert p4b["tower_stiffness"][-1][0] == 110.0


def test_bridge_config_alias_resolves_to_yaml():
    assert resolve_bridge_config("p4b").name == "p4b_defaults.yaml"
    assert resolve_bridge_config("model_defaults").name == "model_defaults.yaml"


def test_staged_analysis_uses_selected_bridge_and_allows_cli_override():
    args = parse_staged_args(["--bridge", "model", "--wg", "12345", "--render", "text"])

    assert args.bridge == "model"
    assert args.bridge_defaults["n"] == 6
    assert args.anchor_base == 32.0
    assert args.wg == 12345.0


def test_validate_staged_can_use_p4b_defaults():
    args = parse_validate_args(["--bridge", "p4b"])

    assert args.n == 19
    assert args.anchor_base == 60.0
    assert args.right_end == 8.0


def test_bridge_config_rejects_missing_keys(tmp_path: Path):
    path = tmp_path / "incomplete.yaml"
    path.write_text("n: 3\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing bridge config keys"):
        load_bridge_config(path)
