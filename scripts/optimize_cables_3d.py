"""Optimize symmetric cable groups for the 3D single-tower bridge.

Example
-------
``python -m scripts.optimize_cables_3d --bridge omo3d --backend opensees``
  python -m scripts.optimize_cables_3d \
    --bridge omo3d \
    --backend direct \
    --progress-refresh 2 \
    --out results/cable_opt_3d

The design vector is stage-major ``(backstay, main_stay)``.  Each variable
controls the two physical cables in the transverse cable planes.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator3D,
    CableHybridOptimizer,
    CableOptimizationProblem,
    ContinuousOptions,
    HybridOptions,
    IntegerSearchOptions,
    ObjectiveWeights,
)
from bridgezoo.optim.variables import CableLayout
from scripts.bridge_config import load_single_staged_3d_config, resolve_bridge_config
from scripts.optimize_cables import _OptimizationProgressDisplay


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_HISTORY_HEADER = [
    "evaluation",
    "objective",
    "shape_rmse_mm",
    "shape_max_abs_mm",
    "tower_top_dx_mm",
    "tower_anchor_dx_rmse_mm",
    "total_physical_strands",
    "stress_mean_mpa",
    "stress_std_mpa",
    "stress_min_mpa",
    "stress_max_mpa",
    "stress_violation_rms_mpa",
]


@dataclass(frozen=True)
class _ResumeState:
    design_path: Path
    strands: np.ndarray
    pretension: np.ndarray
    previous_objective: float
    history_rows: list[list[str]]
    run_index: int
    tracked_outer_iterations: int
    tracked_random_trials: int


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _bridge_yaml_reference(source: str | Path) -> str:
    path = resolve_bridge_config(source).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _group_values(value, n_seg: int, *, integer: bool) -> np.ndarray:
    """Flatten config values into stage-major ``(backstay, main_stay)``."""

    if isinstance(value, (int, float)):
        values = [value, value] * n_seg
    else:
        raw = list(value)
        if len(raw) == n_seg:
            values = []
            for item in raw:
                if isinstance(item, (list, tuple)):
                    if len(item) != 2:
                        raise ValueError("3D cable stage pairs must contain two values")
                    values.extend(item)
                else:
                    values.extend((item, item))
        elif len(raw) == 2 * n_seg:
            values = raw
        else:
            raise ValueError("3D cable values do not match n_seg")
    array = np.asarray(values, dtype=float)
    if integer:
        rounded = np.rint(array).astype(int)
        if not np.allclose(array, rounded):
            raise ValueError("initial strand counts must be integers")
        return rounded
    return array


def _problem_metadata(problem: CableOptimizationProblem, config) -> dict:
    config_data = asdict(config)
    config_data.pop("strands_per_cable", None)
    config_data.pop("pretension_per_cable", None)
    metadata = {
        "n_seg": problem.n_seg,
        "model_family": problem.model_family,
        "backend": problem.backend,
        "strand_area": problem.strand_area,
        "grouping": "stage-major (backstay, main_stay), two symmetric physical cables per group",
        "bridge_config": config_data,
        "bounds": asdict(problem.bounds),
        "weights": asdict(problem.weights),
    }
    return json.loads(json.dumps(metadata))


def _history_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        return []
    if rows[0] != _HISTORY_HEADER:
        raise ValueError(f"cannot resume: incompatible history header in {path}")
    return rows[1:]


def _resume_design_path(args, out_dir: Path) -> Path | None:
    if args.resume:
        source = out_dir
    elif args.resume_from is not None:
        source = _project_path(args.resume_from)
    else:
        return None
    if source.is_dir() or source.suffix.lower() != ".json":
        source = source / "best_design.json"
    return source.resolve()


def _load_resume_state(
    design_path: Path,
    *,
    problem: CableOptimizationProblem,
    bridge_yaml: str,
    problem_metadata: dict,
) -> _ResumeState:
    if not design_path.is_file():
        raise FileNotFoundError(f"resume design not found: {design_path}")
    payload = json.loads(design_path.read_text(encoding="utf-8"))
    if payload.get("schema") != "bridgezoo.cable_optimization_3d.v1":
        raise ValueError(f"cannot resume: incompatible 3D optimization schema in {design_path}")
    if payload.get("bridge_yaml") != bridge_yaml:
        raise ValueError("cannot resume: bridge YAML differs from the saved run")
    if payload.get("problem") != problem_metadata:
        raise ValueError(
            "cannot resume: backend, model, bounds, or objective weights differ from the saved run"
        )

    entries = payload.get("cable_groups")
    if not isinstance(entries, list):
        raise ValueError(f"cannot resume: missing cable_groups in {design_path}")
    by_id = {int(item["group_id"]): item for item in entries}
    layout = CableLayout(problem.n_seg)
    if set(by_id) != set(layout.cable_ids) or len(entries) != layout.size:
        raise ValueError("cannot resume: 3D cable-group ids differ from the current model")

    strands = []
    pretension = []
    for index, group_id in enumerate(layout.cable_ids):
        item = by_id[group_id]
        expected_group = "backstay" if index % 2 == 0 else "main_stay"
        if item.get("stage") != index // 2 + 1 or item.get("group") != expected_group:
            raise ValueError(f"cannot resume: group {group_id} metadata is inconsistent")
        strand_value = float(item["strands_per_physical_cable"])
        strand_int = int(round(strand_value))
        if strand_value != strand_int or not (
            problem.bounds.strand_min <= strand_int <= problem.bounds.strand_max
        ):
            raise ValueError(f"cannot resume: invalid strands for group {group_id}")
        tension = float(item["pretension_per_physical_cable_N"])
        tension_limit = (
            problem.bounds.tension_bound_stress_mpa
            * 1.0e6
            * problem.strand_area
            * strand_int
        )
        if not np.isfinite(tension) or tension < 0.0 or tension > tension_limit * (1.0 + 1e-12):
            raise ValueError(f"cannot resume: invalid pretension for group {group_id}")
        strands.append(strand_int)
        pretension.append(tension)

    search = payload.get("search", {})
    previous_objective = float(payload["objective"])
    if not np.isfinite(previous_objective):
        raise ValueError("cannot resume: saved objective is not finite")
    return _ResumeState(
        design_path=design_path,
        strands=np.asarray(strands, dtype=int),
        pretension=np.asarray(pretension, dtype=float),
        previous_objective=previous_objective,
        history_rows=_history_rows(design_path.with_name("history.csv")),
        run_index=int(search.get("run_index", 1)),
        tracked_outer_iterations=int(search.get("outer_iterations_tracked_total", 0)),
        tracked_random_trials=int(search.get("random_trials_tracked_total", 0)),
    )


def _band_verdict(best, result, lower: float, upper: float) -> str:
    violation = max(
        0.0,
        lower - best.metrics.stress_min_mpa,
        best.metrics.stress_max_mpa - upper,
    )
    verdict = "WITHIN TARGET" if violation <= 1.0e-6 else "OUTSIDE TARGET"
    lp_note = (
        ""
        if result.feasibility_violation_mpa is None
        else f", target-band LP s*={result.feasibility_violation_mpa:.3f} MPa"
    )
    return (
        f"target stress band [{lower:g}, {upper:g}] MPa: {verdict} "
        f"(max departure {violation:.3f} MPa{lp_note})"
    )


def _evaluation_payload(best, bridge_yaml: str, problem_metadata: dict, search: dict) -> dict:
    return {
        "schema": "bridgezoo.cable_optimization_3d.v1",
        "bridge_yaml": bridge_yaml,
        "model_family": "single_staged_3d",
        "problem": problem_metadata,
        "search": search,
        "objective": best.objective,
        "components": asdict(best.components),
        "metrics": {
            "shape_rmse_mm": best.metrics.shape_rmse_m * 1000.0,
            "shape_max_abs_mm": best.metrics.shape_max_abs_m * 1000.0,
            "tower_top_dx_mm": best.metrics.tower_top_dx_m * 1000.0,
            "tower_anchor_dx_rmse_mm": best.metrics.tower_anchor_dx_rmse_m * 1000.0,
            "total_physical_strands": best.metrics.total_strands,
            "stress_mean_mpa": best.metrics.stress_mean_mpa,
            "stress_std_mpa": best.metrics.stress_std_mpa,
            "stress_min_mpa": best.metrics.stress_min_mpa,
            "stress_max_mpa": best.metrics.stress_max_mpa,
            "stress_violation_rms_mpa": best.metrics.stress_violation_rms_mpa,
            "stress_violation_max_mpa": best.metrics.stress_violation_max_mpa,
        },
        "cable_groups": [
            {
                "group_id": group_id,
                "stage": index // 2 + 1,
                "group": "backstay" if index % 2 == 0 else "main_stay",
                "physical_cable_ids": list(best.cable_group_members[group_id]),
                "strands_per_physical_cable": int(best.design.strands[index]),
                "pretension_per_physical_cable_N": float(best.design.pretension[index]),
                "mean_final_stress_MPa": best.cable_stress_mpa[group_id],
                "physical_final_stress_MPa": {
                    str(member): best.physical_cable_stress_mpa[member]
                    for member in best.cable_group_members[group_id]
                },
            }
            for index, group_id in enumerate(best.cable_ids)
        ],
        "deck_errors_mm": {
            str(node): error * 1000.0 for node, error in best.deck_errors_m.items()
        },
        "tower_anchor_dx_mm": {
            str(node): dx * 1000.0 for node, dx in best.tower_anchor_dx_m.items()
        },
    }


def _write_outputs(
    out_dir: Path,
    result,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    search_metadata: dict,
    history_prefix: list[list[str]] | None,
    band_line: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    best = result.best
    payload = _evaluation_payload(best, bridge_yaml, problem_metadata, search_metadata)
    (out_dir / "best_design.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(_HISTORY_HEADER)
        prefix = history_prefix or []
        for index, row in enumerate(prefix):
            writer.writerow([index, *row[1:]])
        for index, evaluation in enumerate(result.history, start=len(prefix)):
            metrics = evaluation.metrics
            writer.writerow(
                [
                    index,
                    evaluation.objective,
                    metrics.shape_rmse_m * 1000.0,
                    metrics.shape_max_abs_m * 1000.0,
                    metrics.tower_top_dx_m * 1000.0,
                    metrics.tower_anchor_dx_rmse_m * 1000.0,
                    metrics.total_strands,
                    metrics.stress_mean_mpa,
                    metrics.stress_std_mpa,
                    metrics.stress_min_mpa,
                    metrics.stress_max_mpa,
                    metrics.stress_violation_rms_mpa,
                ]
            )

    group_lines = []
    for index, group_id in enumerate(best.cable_ids):
        group = "backstay" if index % 2 == 0 else "main_stay"
        group_lines.append(
            f"stage {index // 2 + 1:02d} {group}: "
            f"strands={int(best.design.strands[index])} per cable, "
            f"pretension={best.design.pretension[index]:.2f} N per cable, "
            f"stress={best.cable_stress_mpa[group_id]:.3f} MPa, "
            f"members={best.cable_group_members[group_id]}"
        )
    summary = [
        "model family: single_staged_3d",
        f"backend: {problem_metadata['backend']}",
        f"objective: {best.objective:.6g}",
        f"shape rmse: {best.metrics.shape_rmse_m * 1000.0:.6f} mm",
        f"shape max abs: {best.metrics.shape_max_abs_m * 1000.0:.6f} mm",
        f"tower top dx: {best.metrics.tower_top_dx_m * 1000.0:.6f} mm",
        f"tower anchor dx rmse: {best.metrics.tower_anchor_dx_rmse_m * 1000.0:.6f} mm",
        f"total physical strands: {best.metrics.total_strands}",
        (
            "stress MPa: "
            f"mean={best.metrics.stress_mean_mpa:.3f}, "
            f"std={best.metrics.stress_std_mpa:.3f}, "
            f"min={best.metrics.stress_min_mpa:.3f}, "
            f"max={best.metrics.stress_max_mpa:.3f}"
        ),
        band_line,
        (
            f"search run: {search_metadata['run_index']} "
            f"({'resumed' if search_metadata['resumed'] else 'fresh'})"
        ),
        "",
        "cable groups (each controls two symmetric physical cables):",
        *group_lines,
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def run(args):
    config = load_single_staged_3d_config(args.bridge)
    if args.n is not None:
        config = replace(config, n_seg=args.n)
    bridge_yaml = _bridge_yaml_reference(args.bridge)
    geometry_metadata = asdict(config)
    geometry_metadata.pop("strands_per_cable", None)
    geometry_metadata.pop("pretension_per_cable", None)
    problem = CableOptimizationProblem(
        n_seg=config.n_seg,
        model_kwargs=geometry_metadata,
        bounds=CableBounds(
            strand_min=args.strand_min,
            strand_max=args.strand_max,
            stress_lower_mpa=args.stress_lower,
            stress_upper_mpa=args.stress_upper,
            tension_bound_stress_mpa=args.tension_bound_stress,
        ),
        weights=ObjectiveWeights(
            shape=args.weight_shape,
            total_strands=args.weight_strands,
            stress_uniform=args.weight_stress_uniform,
            stress_violation=args.weight_stress_violation,
            shape_scale_m=args.shape_scale_mm / 1000.0,
            stress_scale_mpa=args.stress_scale,
            strand_scale=args.strand_scale,
            tower_displacement=args.weight_tower_displacement,
            tower_anchor_displacement=args.weight_tower_anchor_displacement,
        ),
        strand_area=config.strand_area,
        backend=args.backend,
        model_family="single_staged_3d",
    )
    problem_metadata = _problem_metadata(problem, config)
    out_dir = _project_path(args.out)
    resume_path = _resume_design_path(args, out_dir)
    resume_state = (
        None
        if resume_path is None
        else _load_resume_state(
            resume_path,
            problem=problem,
            bridge_yaml=bridge_yaml,
            problem_metadata=problem_metadata,
        )
    )

    options = HybridOptions(
        continuous=ContinuousOptions(
            maxiter=args.continuous_maxiter,
            ftol=args.continuous_ftol,
            progress_every=0 if args.quiet else args.progress_every,
            method=args.continuous_method,
        ),
        integer=IntegerSearchOptions(
            outer_iterations=args.outer_iterations,
            coordinate_step=args.coordinate_step,
            random_trials=args.random_trials,
            seed=args.seed,
            stress_guided=not args.no_stress_guided_strands,
            resize=not args.no_strand_resize,
            band_priority=args.band_priority,
        ),
    )
    evaluator = CableDesignEvaluator3D(problem, config)
    progress_display = None
    if not args.quiet:
        progress_display = _OptimizationProgressDisplay(
            total_outer=args.outer_iterations,
            total_cables=2 * config.n_seg,
            refresh_interval=args.progress_refresh,
        )
    optimizer = CableHybridOptimizer(
        problem,
        options,
        progress=None if progress_display is None else progress_display.update,
        evaluator=evaluator,
    )
    if resume_state is None:
        initial_strands = (
            np.full(2 * config.n_seg, args.initial_strands, dtype=int)
            if args.initial_strands is not None
            else _group_values(config.strands_per_cable, config.n_seg, integer=True)
        )
        initial_pretension = _group_values(
            config.pretension_per_cable,
            config.n_seg,
            integer=False,
        )
    else:
        initial_strands = resume_state.strands
        initial_pretension = resume_state.pretension
        if not args.quiet:
            print(
                f"resuming {resume_state.design_path}: "
                f"objective={resume_state.previous_objective:.6g}",
                flush=True,
            )

    try:
        result = optimizer.optimize(
            initial_strands=initial_strands,
            initial_pretension=initial_pretension,
        )
    finally:
        if progress_display is not None:
            progress_display.close()
    previous_outer = 0 if resume_state is None else resume_state.tracked_outer_iterations
    previous_random = 0 if resume_state is None else resume_state.tracked_random_trials
    search_metadata = {
        "run_index": 1 if resume_state is None else resume_state.run_index + 1,
        "resumed": resume_state is not None,
        "resume_source": None if resume_state is None else str(resume_state.design_path),
        "outer_iterations_requested": args.outer_iterations,
        "outer_iterations_completed": result.outer_iterations_completed,
        "outer_iterations_tracked_total": previous_outer + result.outer_iterations_completed,
        "random_trials_completed": result.random_trials_completed,
        "random_trials_tracked_total": previous_random + result.random_trials_completed,
        "seed": args.seed,
        "history_evaluations_previous": (
            0 if resume_state is None else len(resume_state.history_rows)
        ),
        "history_evaluations_this_run": len(result.history),
    }
    band_line = _band_verdict(
        result.best,
        result,
        args.stress_lower,
        args.stress_upper,
    )
    _write_outputs(
        out_dir,
        result,
        bridge_yaml=bridge_yaml,
        problem_metadata=problem_metadata,
        search_metadata=search_metadata,
        history_prefix=None if resume_state is None else resume_state.history_rows,
        band_line=band_line,
    )

    print("3D cable optimization complete")
    print(f"  backend: {args.backend}")
    print(f"  objective: {result.best.objective:.6g}")
    print(f"  shape rmse: {result.best.metrics.shape_rmse_m * 1000.0:.6f} mm")
    print(f"  tower top dx: {result.best.metrics.tower_top_dx_m * 1000.0:.6f} mm")
    print(f"  total physical strands: {result.best.metrics.total_strands}")
    print(
        "  stress MPa: "
        f"mean={result.best.metrics.stress_mean_mpa:.3f}, "
        f"std={result.best.metrics.stress_std_mpa:.3f}, "
        f"min={result.best.metrics.stress_min_mpa:.3f}, "
        f"max={result.best.metrics.stress_max_mpa:.3f}"
    )
    print(f"  {band_line}")
    print(f"  outputs: {out_dir}")

    if args.verify_opensees and args.backend != "opensees":
        verify_problem = replace(problem, backend="opensees")
        verify = CableDesignEvaluator3D(verify_problem, config).evaluate(
            result.best.design.strands,
            result.best.design.pretension,
        )
        print("OpenSees 3D verification")
        print(f"  shape rmse: {verify.metrics.shape_rmse_m * 1000.0:.6f} mm")
        print(
            "  stress MPa: "
            f"mean={verify.metrics.stress_mean_mpa:.3f}, "
            f"std={verify.metrics.stress_std_mpa:.3f}, "
            f"min={verify.metrics.stress_min_mpa:.3f}, "
            f"max={verify.metrics.stress_max_mpa:.3f}"
        )
    return result


def build_parser(config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optimize 3D single-tower backstay/main-stay cable groups."
    )
    parser.add_argument(
        "--bridge",
        required=True,
        help="3D bridge YAML alias (for example omo3d) or YAML path",
    )
    parser.add_argument("--n", type=int, default=config.n_seg)
    parser.add_argument("--backend", choices=("opensees", "direct"), default="opensees")
    parser.add_argument("--out", default="results/cable_opt_3d")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-from", metavar="PATH")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--strand-min", type=int, default=20)
    parser.add_argument("--strand-max", type=int, default=500)
    parser.add_argument("--initial-strands", type=int)
    parser.add_argument("--stress-lower", type=float, default=600.0)
    parser.add_argument("--stress-upper", type=float, default=700.0)
    parser.add_argument("--tension-bound-stress", type=float, default=1600.0)

    parser.add_argument("--weight-shape", type=float, default=1.0)
    parser.add_argument("--weight-tower-displacement", type=float, default=1.0)
    parser.add_argument("--weight-tower-anchor-displacement", type=float, default=1.0)
    parser.add_argument("--weight-strands", type=float, default=0.02)
    parser.add_argument("--weight-stress-uniform", type=float, default=0.2)
    parser.add_argument("--weight-stress-violation", type=float, default=100.0)
    parser.add_argument("--shape-scale-mm", type=_positive_float, default=100.0)
    parser.add_argument("--stress-scale", type=_positive_float, default=100.0)
    parser.add_argument("--strand-scale", type=_positive_float, default=8200.0)

    parser.add_argument(
        "--continuous-method",
        choices=("linear", "slsqp"),
        default="linear",
    )
    parser.add_argument("--continuous-maxiter", type=int, default=80)
    parser.add_argument("--continuous-ftol", type=float, default=1.0e-7)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument(
        "--progress-refresh",
        type=_positive_float,
        default=1.0,
        help="Minimum seconds between live dashboard refreshes (default: 1.0).",
    )
    parser.add_argument("--outer-iterations", type=int, default=4)
    parser.add_argument("--coordinate-step", type=int, default=1)
    parser.add_argument("--random-trials", type=int, default=0)
    parser.add_argument("--no-stress-guided-strands", action="store_true")
    parser.add_argument("--no-strand-resize", action="store_true")
    parser.add_argument("--band-priority", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--verify-opensees", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--bridge")
    known, _ = bootstrap.parse_known_args(argv)
    if known.bridge is None:
        bootstrap.error("--bridge is required; use --bridge omo3d for the bundled model")
    return build_parser(load_single_staged_3d_config(known.bridge)).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
