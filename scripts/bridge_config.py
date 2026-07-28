"""Load staged-bridge defaults from ``scripts/bridges/*.yaml``.

The dependency-free parser supports the project's top-level numeric values and
the ``tower_stiffness`` sequence of ``[z, EI]`` pairs.  It accepts either an
inline sequence or the usual indented YAML list form.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path


BRIDGES_DIR = Path(__file__).with_name("bridges")

_ALIASES = {
    "model": "model_defaults.yaml",
    "model_defaults": "model_defaults.yaml",
    "p4b": "p4b_defaults.yaml",
    "p4b_defaults": "p4b_defaults.yaml",
    "omo": "omo_bridge.yaml",
    "omo_bridge": "omo_bridge.yaml",
    "omo3d": "omo_bridge_3d.yaml",
    "omo_bridge_3d": "omo_bridge_3d.yaml",
}
_REQUIRED_KEYS = {
    "bridge_type",
    "n",
    "anchor_base",
    "anchor_spacing",
    "anchor_free",
    "left_start",
    "left_spacing",
    "left_end",
    "right_start",
    "right_spacing",
    "right_end",
    "wg",
    "dw",
    "beam_E",
    "beam_A",
    "beam_Iz",
    "tower_stiffness",
    "tower_element_size",
    "tower_axial_rigidity",
}
_SINGLE_ONLY_KEYS = {"right_fix", "left_span"}
_KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

_3D_CONFIG_KEYS = {
    "n_seg",
    "anchor_base_height",
    "anchor_spacing",
    "anchor_top_free",
    "left_start",
    "left_spacing",
    "left_end",
    "right_start",
    "right_spacing",
    "right_end",
    "right_fix",
    "left_span",
    "girder_spacing",
    "cross_girder_spacing",
    "deck_width",
    "deck_thickness",
    "deck_offset",
    "tower_element_size",
    "main_girder_depth",
    "main_girder_flange_width",
    "main_girder_web_thickness",
    "main_girder_flange_thickness",
    "cross_girder_depth",
    "cross_girder_flange_width",
    "cross_girder_web_thickness",
    "cross_girder_flange_thickness",
    "tower_outer_width",
    "tower_outer_depth",
    "tower_wall_thickness",
    "strand_area",
    "strands_per_cable",
    "pretension_per_cable",
    "gravity",
    "secondary_main_girder_line_load",
    "secondary_deck_pressure",
}
_OPTIONAL_3D_CONFIG_KEYS = {"superimposed_dead_load"}
_3D_MATERIAL_KEYS = {
    "steel_name",
    "steel_E",
    "steel_poisson",
    "steel_density",
    "concrete_name",
    "concrete_E",
    "concrete_poisson",
    "concrete_density",
    "cable_material_name",
    "cable_material_E",
    "cable_material_poisson",
    "cable_material_density",
}
_REQUIRED_3D_KEYS = {"bridge_type", *_3D_CONFIG_KEYS, *_3D_MATERIAL_KEYS}


def resolve_bridge_config(source: str | Path) -> Path:
    """Resolve a bundled bridge name or a YAML file path."""

    raw = str(source)
    bundled = _ALIASES.get(raw, raw if raw.endswith((".yaml", ".yml")) and "/" not in raw else None)
    if bundled is not None:
        candidate = BRIDGES_DIR / bundled
        if candidate.is_file():
            return candidate

    path = Path(source).expanduser()
    if path.is_file():
        return path.resolve()
    raise FileNotFoundError(f"bridge config not found: {source}")


def _parse_value(path: Path, line_number: int, key: str, raw_value: str):
    try:
        return ast.literal_eval(raw_value)
    except (SyntaxError, ValueError) as exc:
        if _KEY_PATTERN.fullmatch(raw_value):
            return raw_value
        raise ValueError(f"{path}:{line_number}: invalid value for {key!r}") from exc


def _parse_flat_yaml(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    sequence_key: str | None = None
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        content = raw_line.split("#", 1)[0].rstrip()
        if not content:
            continue
        if content[0].isspace():
            item = content.strip()
            if sequence_key is None or not item.startswith("-"):
                raise ValueError(f"{path}:{line_number}: unexpected nested YAML content")
            raw_value = item[1:].strip()
            values[sequence_key].append(_parse_value(path, line_number, sequence_key, raw_value))
            continue
        sequence_key = None
        if ":" not in content:
            raise ValueError(f"{path}:{line_number}: expected a top-level 'key: value' entry")
        key, raw_value = (part.strip() for part in content.split(":", 1))
        if not _KEY_PATTERN.fullmatch(key):
            raise ValueError(f"{path}:{line_number}: invalid key {key!r}")
        if key in values:
            raise ValueError(f"{path}:{line_number}: duplicate key {key!r}")
        if raw_value:
            values[key] = _parse_value(path, line_number, key, raw_value)
        else:
            values[key] = []
            sequence_key = key
    return values


def load_bridge_config(source: str | Path) -> dict[str, object]:
    """Load and validate one staged-bridge defaults file."""

    path = resolve_bridge_config(source)
    config = _parse_flat_yaml(path)
    missing = sorted(_REQUIRED_KEYS - config.keys())
    unknown = sorted(config.keys() - _REQUIRED_KEYS - _SINGLE_ONLY_KEYS)
    if missing:
        raise ValueError(f"{path}: missing bridge config keys: {missing}")
    if unknown:
        raise ValueError(f"{path}: unknown bridge config keys: {unknown}")
    if isinstance(config["n"], bool) or not isinstance(config["n"], int) or config["n"] <= 0:
        raise ValueError(f"{path}: 'n' must be a positive integer")
    bridge_type = config["bridge_type"]
    if bridge_type not in {"normal", "single"}:
        raise ValueError(f"{path}: 'bridge_type' must be 'normal' or 'single'")
    if bridge_type == "single":
        for key in sorted(_SINGLE_ONLY_KEYS):
            if key not in config:
                raise ValueError(f"{path}: 'single' bridge config requires {key!r}")
    else:
        for key in sorted(_SINGLE_ONLY_KEYS & config.keys()):
            raise ValueError(f"{path}: {key!r} is only valid for 'single' bridges")
    numeric_keys = (
        _REQUIRED_KEYS | (_SINGLE_ONLY_KEYS & config.keys())
    ) - {"bridge_type", "n", "tower_stiffness"}
    for key in numeric_keys:
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: {key!r} must be numeric")
    stiffness = config["tower_stiffness"]
    if not isinstance(stiffness, (list, tuple)) or not stiffness:
        raise ValueError(f"{path}: 'tower_stiffness' must contain (z, EI) pairs")
    normalized = []
    previous_z = None
    for index, pair in enumerate(stiffness):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"{path}: tower_stiffness[{index}] must be a (z, EI) pair")
        z, ei = pair
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in pair):
            raise ValueError(f"{path}: tower_stiffness[{index}] values must be numeric")
        z, ei = float(z), float(ei)
        if z < 0.0 or ei <= 0.0 or (previous_z is not None and z <= previous_z):
            raise ValueError(f"{path}: tower_stiffness requires increasing z >= 0 and EI > 0")
        normalized.append([z, ei])
        previous_z = z
    config["tower_stiffness"] = normalized
    if config["tower_element_size"] <= 0.0:
        raise ValueError(f"{path}: 'tower_element_size' must be positive")
    if config["tower_axial_rigidity"] <= 0.0:
        raise ValueError(f"{path}: 'tower_axial_rigidity' must be positive")
    for key in sorted(_SINGLE_ONLY_KEYS & config.keys()):
        if not math.isfinite(config[key]) or config[key] <= 0.0:
            raise ValueError(f"{path}: {key!r} must be finite and positive")
    return config


def load_single_staged_3d_config(source: str | Path):
    """Load a complete 3D single-tower YAML into ``SingleStaged3DConfig``.

    The YAML remains flat and dependency-free like the established 2D bridge
    files.  Material scalars are converted to the solver-neutral physical
    material objects expected by the 3D builder.
    """

    from bridgezoo.fem.single_staged.builder3d import SingleStaged3DConfig
    from bridgezoo.fem.single_staged.sections3d import ElasticMaterial3D

    path = resolve_bridge_config(source)
    values = _parse_flat_yaml(path)
    if "superimposed_dead_load" in values:
        # Backward compatibility for first-round 3D YAML files.  The legacy
        # value is interpreted as deck pressure; no main-girder line load is
        # invented during migration.
        values.setdefault("secondary_main_girder_line_load", 0.0)
        values.setdefault("secondary_deck_pressure", 0.0)
    missing = sorted(_REQUIRED_3D_KEYS - values.keys())
    unknown = sorted(values.keys() - _REQUIRED_3D_KEYS - _OPTIONAL_3D_CONFIG_KEYS)
    if missing:
        raise ValueError(f"{path}: missing 3D bridge config keys: {missing}")
    if unknown:
        raise ValueError(f"{path}: unknown 3D bridge config keys: {unknown}")
    if values["bridge_type"] != "single_3d":
        raise ValueError(f"{path}: 'bridge_type' must be 'single_3d'")

    for name_key in ("steel_name", "concrete_name", "cable_material_name"):
        if not isinstance(values[name_key], str) or not values[name_key].strip():
            raise ValueError(f"{path}: {name_key!r} must be a nonempty string")
    if (
        isinstance(values["n_seg"], bool)
        or not isinstance(values["n_seg"], int)
        or values["n_seg"] <= 0
    ):
        raise ValueError(f"{path}: 'n_seg' must be a positive integer")

    scalar_keys = (
        _3D_CONFIG_KEYS - {"n_seg", "strands_per_cable", "pretension_per_cable"}
    ) | (_3D_MATERIAL_KEYS - {"steel_name", "concrete_name", "cable_material_name"}) | (
        _OPTIONAL_3D_CONFIG_KEYS & values.keys()
    )
    for key in scalar_keys:
        value = values[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{path}: {key!r} must be numeric")

    steel = ElasticMaterial3D(
        str(values["steel_name"]),
        float(values["steel_E"]),
        float(values["steel_poisson"]),
        float(values["steel_density"]),
    )
    concrete = ElasticMaterial3D(
        str(values["concrete_name"]),
        float(values["concrete_E"]),
        float(values["concrete_poisson"]),
        float(values["concrete_density"]),
    )
    cable_material = ElasticMaterial3D(
        str(values["cable_material_name"]),
        float(values["cable_material_E"]),
        float(values["cable_material_poisson"]),
        float(values["cable_material_density"]),
    )
    config_values = {key: values[key] for key in _3D_CONFIG_KEYS}
    config_values.update(
        {key: values[key] for key in _OPTIONAL_3D_CONFIG_KEYS if key in values}
    )
    return SingleStaged3DConfig(
        **config_values,
        steel=steel,
        concrete=concrete,
        cable_material=cable_material,
    )


def model_family_for_bridge_type(bridge_type: str) -> str:
    """Map YAML bridge type to the optimization model-family identifier."""

    if bridge_type == "normal":
        return "staged"
    if bridge_type == "single":
        return "single_staged"
    raise ValueError(f"unknown bridge_type: {bridge_type!r}")


def staged_api_for_bridge_type(bridge_type: str):
    """Return ``(builder, direct solver, OpenSees solver)`` for a bridge type."""

    if bridge_type == "normal":
        from bridgezoo.fem.staged import (
            StagedDirectSolver,
            StagedOpenSeesSolver,
            build_staged_cantilever,
        )

        return build_staged_cantilever, StagedDirectSolver, StagedOpenSeesSolver
    if bridge_type == "single":
        from bridgezoo.fem.single_staged import (
            StagedDirectSolver,
            StagedOpenSeesSolver,
            build_staged_cantilever,
        )

        return build_staged_cantilever, StagedDirectSolver, StagedOpenSeesSolver
    raise ValueError(f"unknown bridge_type: {bridge_type!r}")
