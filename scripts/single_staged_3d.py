"""YAML-driven analysis and staged rendering for the 3D single-tower model.

Examples
--------
``python -m scripts.single_staged_3d --bridge omo3d --n 3``

``python -m scripts.single_staged_3d --bridge omo3d --backend opensees --render both``
"""

from __future__ import annotations

import argparse
import json
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
        "--backend",
        choices=("direct", "opensees"),
        default="direct",
        help="solver backend (default: direct)",
    )
    parser.add_argument("--output", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--render",
        choices=("none", "plot", "text", "both"),
        default="text",
        help="plot writes staged 3D images, text prints the summary, both does both",
    )
    parser.add_argument("--out", type=Path, default=Path("results/single_staged_3d.gif"))
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
    plan = build_single_staged_3d(config)
    solver = (
        SingleStagedDirectSolver3D()
        if args.backend == "direct"
        else SingleStagedOpenSeesSolver3D()
    )
    result = solver.run(plan)
    payload = _result_payload(config, plan, result)
    final = result.final
    max_translation = max(
        abs(component)
        for values in final.displacement.values()
        for component in values[:3]
    )
    if args.render in {"text", "both"}:
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
    return 0 if all(record.converged for record in result.records) else 2


if __name__ == "__main__":
    raise SystemExit(main())
