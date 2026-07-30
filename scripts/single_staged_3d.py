"""YAML-driven analysis and staged rendering for the 3D single-tower model.

Examples
--------
``python -m scripts.single_staged_3d --bridge omo3d --n 3``

``python -m scripts.single_staged_3d --bridge omo3d --backend direct --render text``

``python -m scripts.single_staged_3d --bridge omo3d --backend opensees --render both --dxf``

  python -m scripts.single_staged_3d --bridge omo3d --design results/cable_opt_3d_forward/best_design.json \
  --render text

"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from bridgezoo.fem.single_staged import (
    SingleStaged3DConfig,
    SingleStagedDirectSolver3D,
    SingleStagedOpenSeesSolver3D,
    ElasticMaterial3D,
    build_single_staged_3d,
)
from bridgezoo.optim.variables import CableLayout


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run and render the 3D single-tower grillage staged analysis."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--bridge",
        help="bundled 3D bridge alias (for example omo3d) or YAML path",
    )
    source.add_argument(
        "--input",
        type=Path,
        help="legacy JSON config or a 3D bridge YAML path",
    )
    parser.add_argument("--n", type=int, dest="n_seg", help="number of cable/deck activation stages")
    parser.add_argument("--right-fix", type=float, help="right bearing station measured from tower (m)")
    parser.add_argument("--left-span", type=float, help="optional auxiliary span beyond the free tip (m)")
    parser.add_argument(
        "--design",
        type=Path,
        help="3D optimization best_design.json to apply before solving",
    )
    parser.add_argument(
        "--backend",
        choices=("direct", "opensees"),
        default="opensees",
        help="solver backend (default: opensees)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--render",
        choices=("none", "plot", "text", "both"),
        default="both",
        help="plot writes staged 3D images, text prints the summary, both does both",
    )
    parser.add_argument(
        "--text-all-stages",
        action="store_true",
        help="with text/both, print control-point displacement for every analysis stage",
    )
    parser.add_argument(
        "--text-dir",
        type=Path,
        help="directory for all-stage main-girder total and stage-delta TXT files",
    )
    parser.add_argument("--out", type=Path, default=Path("results/single_staged_3d.gif"))
    parser.add_argument(
        "--dxf",
        action="store_true",
        help="independently export the final undeformed 3D topology to DXF",
    )
    parser.add_argument(
        "--dxf-out",
        type=Path,
        help="DXF path used with --dxf (default: --out with a .dxf suffix)",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("results/single_staged_3d_frames"),
        help="directory for per-stage PNG files",
    )
    parser.add_argument("--scale", type=float, default=10.0, help="displacement display scale")
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--elev", type=float, default=22.0, help="3D camera elevation in degrees")
    parser.add_argument("--azim", type=float, default=-62.0, help="3D camera azimuth in degrees")
    return parser.parse_args(argv)


def _material_from_json(value, field_name: str):
    if isinstance(value, ElasticMaterial3D):
        return value
    if not isinstance(value, Mapping):
        raise ValueError(f"JSON field {field_name!r} must be a material object")
    return ElasticMaterial3D(
        str(value["name"]),
        float(value["E"]),
        float(value["poisson"]),
        float(value["density"]),
    )


def _load_config(args: argparse.Namespace) -> SingleStaged3DConfig:
    if args.bridge is not None:
        from scripts.bridge_config import load_single_staged_3d_config

        config = load_single_staged_3d_config(args.bridge)
    elif args.input is not None and args.input.suffix.lower() in {".yaml", ".yml"}:
        from scripts.bridge_config import load_single_staged_3d_config

        config = load_single_staged_3d_config(args.input)
    elif args.input is not None:
        values = json.loads(args.input.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError("3D input JSON must contain one object")
        values = dict(values)
        for material_name in ("steel", "concrete", "cable_material"):
            if material_name in values:
                values[material_name] = _material_from_json(values[material_name], material_name)
        config = SingleStaged3DConfig(**values)
    else:
        config = SingleStaged3DConfig()

    overrides = {}
    for name in ("n_seg", "right_fix", "left_span"):
        value = getattr(args, name)
        if value is not None:
            overrides[name] = value
    return replace(config, **overrides) if overrides else config


def _project_output_path(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_optimized_design_3d(
    path: str | Path,
    config: SingleStaged3DConfig,
    *,
    adopt_saved_n_seg: bool = False,
) -> tuple[SingleStaged3DConfig, dict[str, object]]:
    """Validate and apply a 3D optimization result to a physical bridge config."""

    design_path = Path(path).expanduser()
    if not design_path.is_absolute():
        design_path = PROJECT_ROOT / design_path
    if not design_path.is_file():
        raise FileNotFoundError(f"3D optimized design not found: {design_path}")
    try:
        payload = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid 3D optimized design JSON: {design_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("3D optimized design root must be an object")
    schema = payload.get("schema")
    if schema not in {
        "bridgezoo.cable_optimization_3d.v1",
        "bridgezoo.cable_optimization_3d.v2",
        "bridgezoo.cable_optimization_3d.v3",
    }:
        raise ValueError("unsupported 3D optimized design schema")
    if payload.get("model_family") != "single_staged_3d":
        raise ValueError("optimized design is not for the single_staged_3d model")

    problem = payload.get("problem")
    if not isinstance(problem, dict):
        raise ValueError("3D optimized design is missing problem metadata")
    saved_config = problem.get("bridge_config")
    if adopt_saved_n_seg:
        if not isinstance(saved_config, dict):
            raise ValueError("3D optimized design is missing bridge configuration")
        try:
            saved_n_seg = int(saved_config["n_seg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("3D optimized design n_seg is invalid") from exc
        if saved_n_seg <= 0 or saved_n_seg != saved_config["n_seg"]:
            raise ValueError("3D optimized design n_seg is invalid")
        config = replace(config, n_seg=saved_n_seg)
    current_config = asdict(config)
    current_config.pop("strands_per_cable", None)
    current_config.pop("pretension_per_cable", None)
    if schema in {
        "bridgezoo.cable_optimization_3d.v2",
        "bridgezoo.cable_optimization_3d.v3",
    }:
        # In v2/v3 the coefficient is a design variable, not immutable bridge
        # geometry.  It is validated and applied from each cable-group entry.
        current_config.pop("pretension_a_ratio", None)
    normalized_current = json.loads(json.dumps(current_config))
    if saved_config != normalized_current:
        raise ValueError(
            "optimized design bridge geometry/materials differ from the calculation model"
        )
    if problem.get("n_seg") != config.n_seg:
        raise ValueError("optimized design n_seg differs from the calculation model")
    try:
        saved_strand_area = float(problem["strand_area"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("optimized design strand_area is invalid") from exc
    if not math.isfinite(saved_strand_area) or not math.isclose(
        saved_strand_area,
        config.strand_area,
    ):
        raise ValueError("optimized design strand_area differs from the calculation model")

    entries = payload.get("cable_groups")
    if not isinstance(entries, list):
        raise ValueError("3D optimized design is missing cable_groups")
    by_id: dict[int, dict] = {}
    for item in entries:
        if not isinstance(item, dict) or isinstance(item.get("group_id"), bool):
            raise ValueError("malformed cable group in 3D optimized design")
        try:
            raw_group_id = float(item["group_id"])
            group_id = int(raw_group_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("invalid cable group id in 3D optimized design") from exc
        if not math.isfinite(raw_group_id) or raw_group_id != group_id:
            raise ValueError("invalid cable group id in 3D optimized design")
        if group_id in by_id:
            raise ValueError(f"duplicate cable group {group_id} in 3D optimized design")
        by_id[group_id] = item

    layout = CableLayout(config.n_seg)
    if set(by_id) != set(layout.cable_ids) or len(entries) != layout.size:
        raise ValueError("optimized cable-group ids differ from the calculation model")
    bounds = problem.get("bounds")
    if not isinstance(bounds, dict):
        raise ValueError("3D optimized design is missing cable bounds")
    try:
        strand_min = int(bounds["strand_min"])
        strand_max = int(bounds["strand_max"])
        tension_bound_stress_mpa = float(bounds["tension_bound_stress_mpa"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("3D optimized design cable bounds are invalid") from exc
    if strand_min <= 0 or strand_max < strand_min or tension_bound_stress_mpa <= 0.0:
        raise ValueError("3D optimized design cable bounds are invalid")

    strand_pairs: list[tuple[int, int]] = []
    tension_pairs: list[tuple[float, float]] = []
    ratio_pairs: list[tuple[float, float]] = []
    stage_strands: list[int] = []
    stage_tensions: list[float] = []
    stage_ratios: list[float] = []
    for index, group_id in enumerate(layout.cable_ids):
        item = by_id[group_id]
        expected_stage = index // 2 + 1
        expected_group = "backstay" if index % 2 == 0 else "main_stay"
        if item.get("stage") != expected_stage or item.get("group") != expected_group:
            raise ValueError(f"optimized cable group {group_id} metadata is inconsistent")
        physical_ids = item.get("physical_cable_ids")
        if not isinstance(physical_ids, list) or len(physical_ids) != 2:
            raise ValueError(f"optimized cable group {group_id} must contain two physical cables")
        raw_strands = item.get("strands_per_physical_cable")
        if isinstance(raw_strands, bool):
            raise ValueError(f"optimized cable group {group_id} strands must be an integer")
        try:
            strands_float = float(raw_strands)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"optimized cable group {group_id} strands must be an integer"
            ) from exc
        strands = int(round(strands_float))
        if (
            not math.isfinite(strands_float)
            or strands_float != strands
            or not strand_min <= strands <= strand_max
        ):
            raise ValueError(f"optimized cable group {group_id} strands are outside saved bounds")
        try:
            tension = float(item["pretension_per_physical_cable_N"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"optimized cable group {group_id} pretension is invalid") from exc
        tension_limit = tension_bound_stress_mpa * 1.0e6 * config.strand_area * strands
        if not math.isfinite(tension) or tension < 0.0 or tension > tension_limit * (1.0 + 1.0e-12):
            raise ValueError(f"optimized cable group {group_id} pretension is outside saved bounds")
        if schema in {
            "bridgezoo.cable_optimization_3d.v2",
            "bridgezoo.cable_optimization_3d.v3",
        }:
            try:
                ratio = float(item["pretension_a_ratio"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"optimized cable group {group_id} A/B coefficient is invalid"
                ) from exc
            if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
                raise ValueError(
                    f"optimized cable group {group_id} A/B coefficient is outside [0, 1]"
                )
            try:
                saved_a = float(item["pretension_A_per_physical_cable_N"])
                saved_b = float(item["pretension_B_per_physical_cable_N"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"optimized cable group {group_id} A/B pretensions are invalid"
                ) from exc
            if not (
                math.isclose(saved_a, tension * ratio, rel_tol=1.0e-10, abs_tol=1.0e-6)
                and math.isclose(
                    saved_b,
                    tension * (1.0 - ratio),
                    rel_tol=1.0e-10,
                    abs_tol=1.0e-6,
                )
            ):
                raise ValueError(
                    f"optimized cable group {group_id} A/B pretensions do not match T and coefficient"
                )
            stage_ratios.append(ratio)
        stage_strands.append(strands)
        stage_tensions.append(tension)
        if len(stage_strands) == 2:
            strand_pairs.append((stage_strands[0], stage_strands[1]))
            tension_pairs.append((stage_tensions[0], stage_tensions[1]))
            if schema in {
                "bridgezoo.cable_optimization_3d.v2",
                "bridgezoo.cable_optimization_3d.v3",
            }:
                ratio_pairs.append((stage_ratios[0], stage_ratios[1]))
            stage_strands.clear()
            stage_tensions.clear()
            stage_ratios.clear()

    try:
        objective = float(payload["objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("optimized design objective must be finite") from exc
    if not math.isfinite(objective):
        raise ValueError("optimized design objective must be finite")
    design_values = {
        "strands_per_cable": tuple(strand_pairs),
        "pretension_per_cable": tuple(tension_pairs),
    }
    if schema in {
        "bridgezoo.cable_optimization_3d.v2",
        "bridgezoo.cable_optimization_3d.v3",
    }:
        design_values["pretension_a_ratio"] = tuple(ratio_pairs)
    applied = replace(config, **design_values)
    metadata = {
        "source": str(design_path.resolve()),
        "schema": payload["schema"],
        "objective": objective,
        "cable_group_count": layout.size,
        "includes_optimized_pretension_a_ratio": schema.endswith((".v2", ".v3")),
    }
    return applied, metadata


def _position_payload(values) -> dict[str, float]:
    return {
        "x": float(values[0]),
        "y": float(values[1]),
        "z": float(values[2]),
    }


def _deformation_payload(values) -> dict[str, dict[str, float]]:
    return {
        "translation_m": {
            "ux": float(values[0]),
            "uy": float(values[1]),
            "uz": float(values[2]),
        },
        "rotation_rad": {
            "rx": float(values[3]),
            "ry": float(values[4]),
            "rz": float(values[5]),
        },
    }


def _stage_deformation_payloads(plan, result) -> dict[int, dict[str, object]]:
    """Build explicit MIDAS-oriented activation/increment/total histories."""

    model = plan.model
    activation: dict[int, dict[str, object]] = {}
    previous_displacement: dict[int, np.ndarray] = {}
    by_stage: dict[int, dict[str, object]] = {}
    for record in result.records:
        for node_id, values in record.birth_displacement.items():
            if node_id in activation:
                raise RuntimeError(f"3D node {node_id} has multiple activation records")
            node = model.nodes[node_id]
            reference = np.asarray(values, dtype=float)
            design_position = np.asarray(node.xyz, dtype=float)
            activation[node_id] = {
                "model_stage_index": node.activation_stage,
                "analysis_stage_index": record.stage_index,
                "analysis_stage_label": record.stage_label,
                "reference": reference,
                "position": design_position + reference[:3],
            }

        nodes = []
        for node_id in sorted(record.displacement):
            if node_id not in activation:
                raise RuntimeError(
                    f"3D node {node_id} is active without an activation reference"
                )
            node = model.nodes[node_id]
            total = np.asarray(record.displacement[node_id], dtype=float)
            reference = np.asarray(activation[node_id]["reference"], dtype=float)
            previous = previous_displacement.get(node_id, reference)
            step = total - previous
            design_position = np.asarray(node.xyz, dtype=float)
            nodes.append(
                {
                    "node_id": node_id,
                    "role": node.role,
                    "design_position_m": _position_payload(design_position),
                    "activation": {
                        "model_stage_index": activation[node_id][
                            "model_stage_index"
                        ],
                        "analysis_stage_index": activation[node_id][
                            "analysis_stage_index"
                        ],
                        "analysis_stage_label": activation[node_id][
                            "analysis_stage_label"
                        ],
                        "position_m": _position_payload(
                            activation[node_id]["position"]
                        ),
                        "reference_deformation": _deformation_payload(reference),
                    },
                    "step_deformation": _deformation_payload(step),
                    "actual_total_deformation": _deformation_payload(total),
                    "actual_position_m": _position_payload(
                        design_position + total[:3]
                    ),
                }
            )
        previous_displacement = {
            node_id: np.asarray(values, dtype=float).copy()
            for node_id, values in record.displacement.items()
        }
        by_stage[record.stage_index] = {
            "node_count": len(nodes),
            "nodes": nodes,
        }
    return by_stage


def _main_girder_controls_by_construction_stage(
    plan,
) -> dict[int, tuple[int, ...]]:
    """Return the two current girder control nodes for each erection group."""

    model = plan.model
    controls: dict[int, tuple[int, ...]] = {}
    for construction_stage in sorted(
        {
            stage.construction_stage
            for stage in plan.stages
            if stage.phase != "secondary_load"
        }
    ):
        cable_nodes = {
            cable.j
            for cable in model.cables.values()
            if cable.group == "main_stay"
            and cable.construction_stage == construction_stage
        }
        if cable_nodes:
            controls[construction_stage] = tuple(sorted(cable_nodes))
            continue

        steel_stage = next(
            stage.index
            for stage in plan.stages
            if stage.construction_stage == construction_stage
            and stage.phase == "steel_and_A"
        )
        new_girder_nodes = [
            node
            for node in model.nodes.values()
            if node.role.startswith("main_girder")
            and node.activation_stage == steel_stage
        ]
        if not new_girder_nodes:
            controls[construction_stage] = ()
            continue
        control_x = min(node.x for node in new_girder_nodes)
        controls[construction_stage] = tuple(
            sorted(
                node.id
                for node in new_girder_nodes
                if math.isclose(node.x, control_x, abs_tol=1.0e-10)
            )
        )
    return controls


def _result_payload(config, plan, result) -> dict[str, object]:
    model = plan.model
    final = result.final
    deformation_by_stage = _stage_deformation_payloads(plan, result)
    stage_definition = {stage.index: stage for stage in plan.stages}
    controls_by_construction = _main_girder_controls_by_construction_stage(plan)
    all_control_nodes = tuple(
        sorted(
            {
                node_id
                for node_ids in controls_by_construction.values()
                for node_id in node_ids
            }
        )
    )
    stages = []
    for record in result.records:
        definition = stage_definition[record.stage_index]
        control_nodes = (
            all_control_nodes
            if definition.phase == "secondary_load"
            else controls_by_construction.get(definition.construction_stage, ())
        )
        translations = [values[:3] for values in record.displacement.values()]
        stages.append(
            {
                "index": record.stage_index,
                "label": record.stage_label,
                "construction_stage": definition.construction_stage,
                "phase": definition.phase,
                "main_girder_control_node_ids": list(control_nodes),
                "converged": record.converged,
                "node_count": len(record.displacement),
                "max_abs_translation_m": max(
                    abs(component) for vector in translations for component in vector
                ),
                "applied_load_N": record.applied_load,
                "deformation": deformation_by_stage[record.stage_index],
            }
        )
    return {
        "schema": "bridgezoo.single_staged_3d.result.v1",
        "backend": result.backend,
        "input": asdict(config),
        "model": {
            "name": model.name,
            "nodes": len(model.nodes),
            "frames": len(model.frames),
            "cables": len(model.cables),
            "rigid_links": len(model.rigid_links),
            "supports": len(model.supports),
            "coordinate_system": plan.metadata["coordinate_system"],
            "analysis_scope": plan.metadata["analysis_scope"],
        },
        "deformation_convention": {
            "coordinate_system": "x longitudinal, y transverse, z vertical",
            "translation_unit": "m",
            "rotation_unit": "rad",
            "activation_position_m": (
                "design_position plus the stress-free tangent-birth translation"
            ),
            "activation_reference_deformation": (
                "stress-free tangent-birth displacement and rotation relative "
                "to the design model"
            ),
            "step_deformation": (
                "actual total deformation at this stage minus the preceding "
                "analysis stage; for a newly activated node, minus its "
                "activation reference deformation"
            ),
            "actual_total_deformation": (
                "cumulative deformation relative to the design model after "
                "this analysis stage"
            ),
            "actual_position_m": (
                "design position plus actual total translation after this stage"
            ),
        },
        "stages": stages,
        "final": {
            "stage": final.stage_label,
            "converged": final.converged,
            "displacement": {str(key): value for key, value in final.displacement.items()},
            "cable_force_N": {str(key): value for key, value in final.cable_force.items()},
            "cable_stress_Pa": {str(key): value for key, value in final.cable_stress.items()},
            "support_reaction_N_Nm": {
                str(key): value for key, value in final.support_reaction.items()
            },
        },
        "todo": [
            "geometric nonlinearity and cable sag",
        ],
    }


def _main_girder_deformation_table(
    stage: Mapping[str, object],
    deformation_field: str,
) -> str:
    deformation = stage["deformation"]
    control_node_ids = set(stage["main_girder_control_node_ids"])
    nodes = [
        node
        for node in deformation["nodes"]
        if node["node_id"] in control_node_ids
    ]
    description = {
        "actual_total_deformation": "实际总变形（相对设计模型）",
        "step_deformation": "本阶段增量变形 delta（仅计本分析阶段）",
    }[deformation_field]
    lines = [
        f"# stage_index={stage['index']}",
        f"# stage_label={stage['label']}",
        f"# result={description}",
        "# coordinates: design x/y/z in m",
        "# displacement: ux/uy/uz in mm; rotation: rx/ry/rz in rad",
        "node_id role x_m y_m z_m ux_mm uy_mm uz_mm rx_rad ry_rad rz_rad",
    ]
    for node in nodes:
        position = node["design_position_m"]
        values = node[deformation_field]
        translation = values["translation_m"]
        rotation = values["rotation_rad"]
        lines.append(
            f"{node['node_id']} {node['role']} "
            f"{position['x']:+.6f} {position['y']:+.6f} {position['z']:+.6f} "
            f"{translation['ux'] * 1000.0:+.9f} "
            f"{translation['uy'] * 1000.0:+.9f} "
            f"{translation['uz'] * 1000.0:+.9f} "
            f"{rotation['rx']:+.12e} {rotation['ry']:+.12e} "
            f"{rotation['rz']:+.12e}"
        )
    return "\n".join(lines) + "\n"


def _write_main_girder_deformation_texts(
    payload: Mapping[str, object],
    output_dir: Path,
) -> dict[str, tuple[Path, ...]]:
    """Write all-stage main-girder total and incremental deformation tables."""

    categories = {
        "actual_total": "actual_total_deformation",
        "stage_delta": "step_deformation",
    }
    written: dict[str, tuple[Path, ...]] = {}
    for category, field in categories.items():
        category_dir = output_dir / f"main_girder_{category}"
        category_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for stage in payload["stages"]:
            label = "".join(
                character if character.isalnum() or character in {"-", "_"} else "_"
                for character in str(stage["label"])
            )
            path = category_dir / f"stage_{stage['index']:03d}_{label}.txt"
            path.write_text(
                _main_girder_deformation_table(stage, field),
                encoding="utf-8",
            )
            paths.append(path)
        written[category] = tuple(paths)

    summary_path = output_dir / "main_girder_z_summary.txt"
    summary_lines = [
        "# 全阶段主梁节点竖向变形汇总",
        "# delta_uz_mm: 仅由本分析阶段产生的竖向增量",
        "# actual_total_uz_mm: 相对设计模型的真实竖向总变形",
        "stage_index stage_label node_id delta_uz_mm actual_total_uz_mm",
    ]
    for stage in payload["stages"]:
        control_node_ids = set(stage["main_girder_control_node_ids"])
        for node in stage["deformation"]["nodes"]:
            if node["node_id"] not in control_node_ids:
                continue
            delta_uz = node["step_deformation"]["translation_m"]["uz"]
            total_uz = node["actual_total_deformation"]["translation_m"]["uz"]
            summary_lines.append(
                f"{stage['index']} {stage['label']} {node['node_id']} "
                f"{delta_uz * 1000.0:+.9f} {total_uz * 1000.0:+.9f}"
            )
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    written["z_summary"] = (summary_path,)
    return written


def _text_summary(
    plan,
    result,
    design_metadata=None,
    *,
    all_stages: bool = False,
) -> str:
    """Build the console report for ``--render text`` and ``both``."""

    model = plan.model
    final = result.final
    max_translation = max(
        abs(component)
        for values in final.displacement.values()
        for component in values[:3]
    )
    lines = []
    if design_metadata is not None:
        lines.append(
            f"design={design_metadata['source']} "
            f"objective={design_metadata['objective']:.6g}"
        )
    lines.extend(
        (
            model.summary(),
            (
                f"backend={result.backend} stages={len(result.records)} "
                f"converged={final.converged}"
            ),
            (
                f"final_stage={final.stage_label} "
                f"max_abs_translation={max_translation:.6e} m"
            ),
        )
    )

    controls_by_construction = _main_girder_controls_by_construction_stage(plan)
    stage_definition = {stage.index: stage for stage in plan.stages}
    displayed_records = result.records if all_stages else result.records[-3:]
    stage_scope = (
        f"全部{len(displayed_records)}个分析阶段"
        if all_stages
        else f"最后{len(displayed_records)}个分析阶段"
    )
    lines.append(
        "主梁控制点位移（当前施工组横桥向控制节点平均，单位 mm；"
        f"{stage_scope}）"
    )
    for record in displayed_records:
        lines.append(f"阶段 {record.stage_index}: {record.stage_label}")
        definition = stage_definition[record.stage_index]
        control_points = (
            sorted(controls_by_construction.items())
            if definition.phase == "secondary_load"
            else [
                (
                    definition.construction_stage,
                    controls_by_construction.get(definition.construction_stage, ()),
                )
            ]
        )
        for construction_stage, raw_node_ids in control_points:
            node_ids = tuple(
                sorted(raw_node_ids, key=lambda node_id: model.nodes[node_id].y)
            )
            active = [
                record.displacement[node_id]
                for node_id in node_ids
                if node_id in record.displacement
            ]
            if not active:
                continue
            mean = tuple(
                1000.0 * sum(values[index] for values in active) / len(active)
                for index in range(3)
            )
            x_m = sum(model.nodes[node_id].x for node_id in node_ids) / len(
                node_ids
            )
            lines.append(
                f"  梁点 S{construction_stage:02d} nodes={node_ids} x={x_m:+.3f} m: "
                f"ux={mean[0]:+.3f}, uy={mean[1]:+.3f}, uz={mean[2]:+.3f}"
            )

    lines.append(f"最终阶段拉索应力（{final.stage_label}，单位 MPa）")
    group_order = {"backstay": 0, "main_stay": 1}
    group_label = {"backstay": "背索", "main_stay": "主跨索"}
    cables = sorted(
        model.cables.values(),
        key=lambda cable: (
            cable.construction_stage,
            group_order.get(cable.group, 99),
            model.nodes[cable.j].y,
            cable.id,
        ),
    )
    for cable in cables:
        stress_mpa = final.cable_stress.get(cable.id)
        stress_text = "未激活" if stress_mpa is None else f"{stress_mpa / 1.0e6:+.3f}"
        lines.append(
            f"  拉索 id={cable.id} S{cable.construction_stage:02d} "
            f"{group_label.get(cable.group, cable.group)} "
            f"y={model.nodes[cable.j].y:+.3f} m: stress={stress_text}"
        )

    tower_nodes = [
        model.nodes[node_id]
        for node_id in plan.metadata.get("tower_node_ids", ())
        if node_id in final.displacement
    ]
    if not tower_nodes:
        tower_nodes = [
            node
            for node in model.nodes.values()
            if node.role in {"tower", "tower_anchor"}
            and node.id in final.displacement
        ]
    if tower_nodes:
        tower_top = max(tower_nodes, key=lambda node: (node.z, node.id))
        displacement = final.displacement[tower_top.id]
        lines.append(
            f"最终阶段塔顶位移（{final.stage_label}，单位 mm）: "
            f"node={tower_top.id} z={tower_top.z:+.3f} m, "
            f"ux={displacement[0] * 1000.0:+.3f}, "
            f"uy={displacement[1] * 1000.0:+.3f}, "
            f"uz={displacement[2] * 1000.0:+.3f}"
        )
    else:
        lines.append(f"最终阶段塔顶位移（{final.stage_label}）: 无可用塔顶节点")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.dxf_out is not None and not args.dxf:
        raise ValueError("--dxf-out requires --dxf")
    config = _load_config(args)
    design_metadata = None
    if args.design is not None:
        config, design_metadata = load_optimized_design_3d(
            args.design,
            config,
            adopt_saved_n_seg=args.n_seg is None,
        )
    plan = build_single_staged_3d(config)
    solver = (
        SingleStagedDirectSolver3D()
        if args.backend == "direct"
        else SingleStagedOpenSeesSolver3D()
    )
    result = solver.run(plan)
    payload = _result_payload(config, plan, result)
    if design_metadata is not None:
        payload["optimized_design"] = design_metadata
    if args.render in {"text", "both"}:
        print(
            _text_summary(
                plan,
                result,
                design_metadata,
                all_stages=args.text_all_stages,
            )
        )
        default_text_dir = args.out.with_name(f"{args.out.stem}_text")
        text_dir = _project_output_path(args.text_dir or default_text_dir)
        text_artifacts = _write_main_girder_deformation_texts(payload, text_dir)
        print(
            f"text_actual_total={len(text_artifacts['actual_total'])} files in "
            f"{text_dir / 'main_girder_actual_total'}"
        )
        print(
            f"text_stage_delta={len(text_artifacts['stage_delta'])} files in "
            f"{text_dir / 'main_girder_stage_delta'}"
        )
        print(f"text_z_summary={text_artifacts['z_summary'][0]}")
    if args.output is not None:
        output_path = _project_output_path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"json={output_path}")
    if args.render in {"plot", "both"}:
        from bridgezoo.render.staged3d import render_staged_3d

        render_output = _project_output_path(args.out)
        render_frames = _project_output_path(args.frames_dir)
        artifacts = render_staged_3d(
            plan,
            result,
            scale=args.scale,
            out=render_output,
            frames_dir=render_frames,
            fps=args.fps,
            elevation=args.elev,
            azimuth=args.azim,
        )
        print(f"render={artifacts['output']}")
        print(f"frames={len(artifacts['frames'])} in {render_frames}")
    if args.dxf:
        from bridgezoo.render.staged3d import export_final_3d_dxf

        dxf_output = _project_output_path(args.dxf_out or args.out.with_suffix(".dxf"))
        written_dxf = export_final_3d_dxf(plan, result, dxf_output)
        print(f"dxf={written_dxf}")
    return 0 if all(record.converged for record in result.records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
