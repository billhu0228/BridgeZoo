"""YAML-driven analysis and staged rendering for the 3D single-tower model.

Examples
--------
``python -m scripts.single_staged_3d --bridge omo3d --n 3``

``python -m scripts.single_staged_3d --bridge omo3d --backend opensees --render both``

  python -m scripts.single_staged_3d --bridge omo3d --design results/cable_opt_3d/best_design.json

"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path

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
        help="solver backend (default: direct)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--render",
        choices=("none", "plot", "text", "both"),
        default="both",
        help="plot writes staged 3D images, text prints the summary, both does both",
    )
    parser.add_argument("--out", type=Path, default=Path("results/single_staged_3d.gif"))
    parser.add_argument(
        "--dxf-out",
        type=Path,
        help="final-state DXF path (default: --out with a .dxf suffix)",
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
    if payload.get("schema") != "bridgezoo.cable_optimization_3d.v1":
        raise ValueError("unsupported 3D optimized design schema")
    if payload.get("model_family") != "single_staged_3d":
        raise ValueError("optimized design is not for the single_staged_3d model")

    problem = payload.get("problem")
    if not isinstance(problem, dict):
        raise ValueError("3D optimized design is missing problem metadata")
    saved_config = problem.get("bridge_config")
    current_config = asdict(config)
    current_config.pop("strands_per_cable", None)
    current_config.pop("pretension_per_cable", None)
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
    stage_strands: list[int] = []
    stage_tensions: list[float] = []
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
        stage_strands.append(strands)
        stage_tensions.append(tension)
        if len(stage_strands) == 2:
            strand_pairs.append((stage_strands[0], stage_strands[1]))
            tension_pairs.append((stage_tensions[0], stage_tensions[1]))
            stage_strands.clear()
            stage_tensions.clear()

    try:
        objective = float(payload["objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("optimized design objective must be finite") from exc
    if not math.isfinite(objective):
        raise ValueError("optimized design objective must be finite")
    applied = replace(
        config,
        strands_per_cable=tuple(strand_pairs),
        pretension_per_cable=tuple(tension_pairs),
    )
    metadata = {
        "source": str(design_path.resolve()),
        "schema": payload["schema"],
        "objective": objective,
        "cable_group_count": layout.size,
    }
    return applied, metadata


def _result_payload(config, plan, result) -> dict[str, object]:
    model = plan.model
    final = result.final
    stages = []
    for record in result.records:
        translations = [values[:3] for values in record.displacement.values()]
        stages.append(
            {
                "index": record.stage_index,
                "label": record.stage_label,
                "converged": record.converged,
                "node_count": len(record.displacement),
                "max_abs_translation_m": max(
                    abs(component) for vector in translations for component in vector
                ),
                "applied_load_N": record.applied_load,
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
            "path-dependent stress-free activation and displacement lock-in",
            "geometric nonlinearity and cable sag",
        ],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args)
    design_metadata = None
    if args.design is not None:
        config, design_metadata = load_optimized_design_3d(args.design, config)
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
    final = result.final
    max_translation = max(
        abs(component)
        for values in final.displacement.values()
        for component in values[:3]
    )
    if args.render in {"text", "both"}:
        if design_metadata is not None:
            print(
                f"design={design_metadata['source']} "
                f"objective={design_metadata['objective']:.6g}"
            )
        print(plan.model.summary())
        print(f"backend={result.backend} stages={len(result.records)} converged={final.converged}")
        print(f"final_stage={final.stage_label} max_abs_translation={max_translation:.6e} m")
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
        render_dxf = _project_output_path(args.dxf_out)
        artifacts = render_staged_3d(
            plan,
            result,
            scale=args.scale,
            out=render_output,
            frames_dir=render_frames,
            fps=args.fps,
            elevation=args.elev,
            azimuth=args.azim,
            dxf_out=render_dxf,
        )
        print(f"render={artifacts['output']}")
        print(f"frames={len(artifacts['frames'])} in {render_frames}")
        print(f"dxf={artifacts['dxf']}")
    return 0 if all(record.converged for record in result.records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
