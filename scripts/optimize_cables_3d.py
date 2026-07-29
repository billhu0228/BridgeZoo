"""Efficiently calculate A tension and optimize B tension for 3D cable groups.

Example
-------
python -m scripts.optimize_cables_3d --bridge omo3d --out results/cable_opt_3d

The OpenSees-only optimizer calculates A tension from local balance influence
matrices and optimizes a low-dimensional smooth B-tension curve before
interpolating it to every physical cable group.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator3D,
    CableOptimizationProblem,
    ObjectiveWeights,
    SecondaryTensionOptions3D,
    StageAControlOptions,
    Staged3DOptimizationOptions,
    StagedCableOptimizer3D,
    StrandSearchOptions3D,
)
from bridgezoo.optim.variables import CableLayout
from scripts.bridge_config import load_single_staged_3d_config, resolve_bridge_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SCHEMA = "bridgezoo.cable_optimization_3d.v3"
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
    pretension_a_ratio: np.ndarray
    previous_objective: float
    history_rows: list[list[str]]
    run_index: int
    tracked_strand_iterations: int
    tracked_continuous_solves: int
    tracked_fem_cases: int
    tracked_fem_seconds: float


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a nonnegative finite number")
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
    for variable in (
        "strands_per_cable",
        "pretension_per_cable",
        "pretension_a_ratio",
    ):
        config_data.pop(variable, None)
    metadata = {
        "n_seg": problem.n_seg,
        "model_family": problem.model_family,
        "backend": problem.backend,
        "optimizer_architecture": (
            "direct_stage_a_balance_then_smooth_low_dimensional_stage_b_"
            "then_monotone_strand_curve_search"
        ),
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
    if payload.get("schema") != _SCHEMA:
        raise ValueError(
            "cannot resume: the efficient 3D optimizer requires a v3 design"
        )
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
    ratios = []
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
        ratio = float(item["pretension_a_ratio"])
        tension_limit = (
            problem.bounds.tension_bound_stress_mpa
            * 1.0e6
            * problem.strand_area
            * strand_int
        )
        if not math.isfinite(tension) or not 0.0 <= tension <= tension_limit * (1.0 + 1e-12):
            raise ValueError(f"cannot resume: invalid pretension for group {group_id}")
        if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
            raise ValueError(f"cannot resume: invalid A/B coefficient for group {group_id}")
        strands.append(strand_int)
        pretension.append(tension)
        ratios.append(ratio)

    search = payload.get("search", {})
    previous_objective = float(payload["objective"])
    if not math.isfinite(previous_objective):
        raise ValueError("cannot resume: saved objective is not finite")
    return _ResumeState(
        design_path=design_path,
        strands=np.asarray(strands, dtype=int),
        pretension=np.asarray(pretension, dtype=float),
        pretension_a_ratio=np.asarray(ratios, dtype=float),
        previous_objective=previous_objective,
        history_rows=_history_rows(design_path.with_name("history.csv")),
        run_index=int(search.get("run_index", 1)),
        tracked_strand_iterations=int(search.get("strand_iterations_tracked_total", 0)),
        tracked_continuous_solves=int(search.get("continuous_solves_tracked_total", 0)),
        tracked_fem_cases=int(search.get("OpenSees_FEM_cases_tracked_total", 0)),
        tracked_fem_seconds=float(search.get("OpenSees_FEM_seconds_tracked_total", 0.0)),
    )


def _band_verdict(best, lower: float, upper: float) -> str:
    violation = max(
        0.0,
        lower - best.metrics.stress_min_mpa,
        best.metrics.stress_max_mpa - upper,
    )
    verdict = "WITHIN TARGET" if violation <= 1.0e-6 else "OUTSIDE TARGET"
    return (
        f"target stress band [{lower:g}, {upper:g}] MPa: {verdict} "
        f"(max departure {violation:.3f} MPa)"
    )


def _evaluation_payload(
    best,
    controls,
    secondary,
    strand_curve,
    bridge_yaml: str,
    problem_metadata: dict,
    search: dict,
) -> dict:
    control_by_stage = {
        item.response.construction_stage: item for item in controls
    }
    return {
        "schema": _SCHEMA,
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
        "stage_a_controls": [
            {
                "stage": stage,
                "stage_index": control.response.stage_index,
                "backstay_tower_dx_mm": control.response.backstay_tower_dx_m * 1000.0,
                "main_stay_deck_uz_mm": control.response.main_stay_deck_uz_m * 1000.0,
                "calculation_basis": control.balance_basis,
                "final_schedule_backstay_tower_dx_mm": (
                    control.final_schedule_response.backstay_tower_dx_m * 1000.0
                ),
                "final_schedule_main_stay_deck_uz_mm": (
                    control.final_schedule_response.main_stay_deck_uz_m * 1000.0
                ),
                "final_schedule_feasible": control.final_schedule_feasible,
                "backstay_initial_tension_A_N": control.backstay_a_N,
                "main_stay_initial_tension_A_N": control.main_stay_a_N,
                "feasible": control.feasible,
                "stable": control.stable,
                "influence_matrix_rank": control.matrix_rank,
                "influence_condition_number": control.condition_number,
                "active_bounds": list(control.active_bounds),
                "success": control.success,
                "FEM_cases": control.nfev,
            }
            for stage, control in sorted(control_by_stage.items())
        ],
        "stage_b_response_model": {
            "method": "OpenSees low-dimensional smooth-curve affine response",
            "selected_curve_family": secondary.curve_family,
            "control_points_per_group": int(
                secondary.curve_control_coordinates.size
            ),
            "control_coordinates_inner_to_outer": (
                secondary.curve_control_coordinates.tolist()
            ),
            "backstay_normalized_B_controls": secondary.curve_control_values[
                : secondary.curve_control_coordinates.size
            ].tolist(),
            "main_stay_normalized_B_controls": secondary.curve_control_values[
                secondary.curve_control_coordinates.size :
            ].tolist(),
            "curve_trials": [asdict(item) for item in secondary.curve_trials],
            "optimization_nfev_without_FEM": secondary.nfev,
            "FEM_cases": secondary.fem_cases,
            "validated": secondary.validated,
            "max_displacement_prediction_error_m": (
                secondary.max_displacement_prediction_error_m
            ),
            "max_stress_prediction_error_MPa": (
                secondary.max_stress_prediction_error_mpa
            ),
            "message": secondary.message,
        },
        "smooth_curves": {
            "stage_coordinate_definition": "0=innermost, 1=outermost",
            "strand_count": {
                "family": strand_curve.family,
                "monotone_non_decreasing_outward": True,
                "control_coordinates": strand_curve.control_coordinates.tolist(),
                "backstay_control_strands": (
                    strand_curve.backstay_control_strands.tolist()
                ),
                "main_stay_control_strands": (
                    strand_curve.main_stay_control_strands.tolist()
                ),
                "interpolated_stage_major_integer_strands": (
                    strand_curve.interpolated_strands.tolist()
                ),
            },
            "secondary_tension_B": {
                "family": secondary.curve_family,
                "parameterization": "normalized available B-tension capacity",
                "control_coordinates": (
                    secondary.curve_control_coordinates.tolist()
                ),
                "backstay_controls": secondary.curve_control_values[
                    : secondary.curve_control_coordinates.size
                ].tolist(),
                "main_stay_controls": secondary.curve_control_values[
                    secondary.curve_control_coordinates.size :
                ].tolist(),
                "interpolated_stage_major_tension_N": (
                    secondary.pretension_b.tolist()
                ),
            },
        },
        "cable_groups": [
            {
                "group_id": group_id,
                "stage": index // 2 + 1,
                "group": "backstay" if index % 2 == 0 else "main_stay",
                "physical_cable_ids": list(best.cable_group_members[group_id]),
                "strands_per_physical_cable": int(best.design.strands[index]),
                "pretension_per_physical_cable_N": float(best.design.pretension[index]),
                "pretension_a_ratio": float(best.design.pretension_a_ratio[index]),
                "pretension_A_per_physical_cable_N": float(
                    best.design.pretension[index] * best.design.pretension_a_ratio[index]
                ),
                "pretension_B_per_physical_cable_N": float(
                    best.design.pretension[index]
                    * (1.0 - best.design.pretension_a_ratio[index])
                ),
                "stage_A_control_target": (
                    "tower_end_horizontal_displacement_zero"
                    if index % 2 == 0
                    else "girder_end_vertical_displacement_zero"
                ),
                "stage_A_control_displacement_mm": float(
                    control_by_stage[index // 2 + 1].response.backstay_tower_dx_m
                    * 1000.0
                    if index % 2 == 0
                    else control_by_stage[index // 2 + 1].response.main_stay_deck_uz_m
                    * 1000.0
                ),
                "stage_A_final_schedule_displacement_mm": float(
                    control_by_stage[index // 2 + 1]
                    .final_schedule_response.backstay_tower_dx_m
                    * 1000.0
                    if index % 2 == 0
                    else control_by_stage[index // 2 + 1]
                    .final_schedule_response.main_stay_deck_uz_m
                    * 1000.0
                ),
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
    payload = _evaluation_payload(
        best,
        result.controls,
        result.secondary,
        result.strand_curve,
        bridge_yaml,
        problem_metadata,
        search_metadata,
    )
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

    control_by_stage = {
        item.response.construction_stage: item for item in result.controls
    }
    group_lines = []
    for index, group_id in enumerate(best.cable_ids):
        group = "backstay" if index % 2 == 0 else "main_stay"
        ratio = best.design.pretension_a_ratio[index]
        control = control_by_stage[index // 2 + 1]
        residual_mm = (
            control.response.backstay_tower_dx_m * 1000.0
            if group == "backstay"
            else control.response.main_stay_deck_uz_m * 1000.0
        )
        final_residual_mm = (
            control.final_schedule_response.backstay_tower_dx_m * 1000.0
            if group == "backstay"
            else control.final_schedule_response.main_stay_deck_uz_m * 1000.0
        )
        group_lines.append(
            f"stage {index // 2 + 1:02d} {group}: "
            f"strands={int(best.design.strands[index])} per cable, "
            f"T={best.design.pretension[index]:.2f} N, "
            f"A_ratio={ratio:.8f}, "
            f"A={best.design.pretension[index] * ratio:.2f} N, "
            f"B={best.design.pretension[index] * (1.0 - ratio):.2f} N, "
            f"stage_A_preliminary_residual={residual_mm:.6f} mm, "
            f"stage_A_final_schedule_residual={final_residual_mm:.6f} mm, "
            f"final_stress={best.cable_stress_mpa[group_id]:.3f} MPa"
        )
    summary = [
        "model family: single_staged_3d",
        "optimizer: direct stage-A balance / smooth low-dimensional stage-B response / monotone strand curve",
        (
            "smooth curves: "
            f"B={result.secondary.curve_family}, "
            f"controls/group={result.secondary.curve_control_coordinates.size}, "
            f"strand={result.strand_curve.family} outward-monotone"
        ),
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
            "A-stage feasibility: "
            f"direct_balance={sum(item.success for item in result.controls)}/"
            f"{len(result.controls)}, "
            f"final_schedule={sum(bool(item.final_schedule_feasible) for item in result.controls)}/"
            f"{len(result.controls)} stages"
        ),
        (
            "B-response validation: "
            f"{'PASS' if result.secondary.validated else 'FAIL'}, "
            f"disp_error={result.secondary.max_displacement_prediction_error_m:.3e} m, "
            f"stress_error={result.secondary.max_stress_prediction_error_mpa:.3e} MPa"
        ),
        (
            f"OpenSees FEM: cases={result.fem_cases}, "
            f"time={result.fem_seconds:.3f} s, "
            f"average={result.fem_seconds / max(result.fem_cases, 1):.3f} s/case"
        ),
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
    for variable in (
        "strands_per_cable",
        "pretension_per_cable",
        "pretension_a_ratio",
    ):
        geometry_metadata.pop(variable, None)
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
        backend="opensees",
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

    options = Staged3DOptimizationOptions(
        ab_correction_passes=args.ab_correction_passes,
        stage_a=StageAControlOptions(
            probe_fraction=args.stage_a_probe_fraction,
            feasibility_tolerance_m=args.stage_a_feasibility_mm / 1000.0,
            condition_limit=args.stage_a_condition_limit,
        ),
        secondary=SecondaryTensionOptions3D(
            curve_family=args.curve_family,
            control_points_per_group=args.curve_control_points,
            max_nfev=args.secondary_max_nfev,
            ftol=args.secondary_ftol,
            xtol=args.secondary_ftol,
            gtol=args.secondary_ftol,
            displacement_validation_tolerance_m=(
                args.secondary_validation_displacement_mm / 1000.0
            ),
            stress_validation_tolerance_mpa=args.secondary_validation_stress_mpa,
        ),
        strands=StrandSearchOptions3D(
            iterations=args.strand_iterations,
            step=args.strand_step,
        ),
    )
    evaluator = CableDesignEvaluator3D(problem, config)
    optimizer = StagedCableOptimizer3D(
        evaluator,
        options,
        progress=None if args.quiet else lambda message: print(message, flush=True),
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
        initial_ratio = _group_values(
            config.pretension_a_ratio,
            config.n_seg,
            integer=False,
        )
    else:
        initial_strands = resume_state.strands
        initial_pretension = resume_state.pretension
        initial_ratio = resume_state.pretension_a_ratio
        if not args.quiet:
            historical_average = resume_state.tracked_fem_seconds / max(
                resume_state.tracked_fem_cases, 1
            )
            expected_cases = (1 + args.ab_correction_passes) * (
                4 * config.n_seg + optimizer.secondary_fem_cases_per_cycle()
            )
            print(
                f"resuming {resume_state.design_path}: "
                f"objective={resume_state.previous_objective:.6g}; "
                f"historical FEM average={historical_average:.3f} s/case; "
                f"next fixed-strand design≈{expected_cases} cases / "
                f"{historical_average * expected_cases:.1f} s",
                flush=True,
            )

    result = optimizer.optimize(
        initial_strands=initial_strands,
        initial_pretension=initial_pretension,
        initial_ratio=initial_ratio,
    )
    previous_strand = 0 if resume_state is None else resume_state.tracked_strand_iterations
    previous_continuous = 0 if resume_state is None else resume_state.tracked_continuous_solves
    previous_fem_cases = 0 if resume_state is None else resume_state.tracked_fem_cases
    previous_fem_seconds = 0.0 if resume_state is None else resume_state.tracked_fem_seconds
    search_metadata = {
        "run_index": 1 if resume_state is None else resume_state.run_index + 1,
        "resumed": resume_state is not None,
        "resume_source": None if resume_state is None else str(resume_state.design_path),
        "stage_a_method": "direct bounded 2x2 influence solve with B=0",
        "stage_a_probe_fraction": args.stage_a_probe_fraction,
        "stage_a_feasibility_mm": args.stage_a_feasibility_mm,
        "stage_a_condition_limit": args.stage_a_condition_limit,
        "stage_b_method": "OpenSees low-dimensional smooth-curve affine response",
        "curve_family_requested": args.curve_family,
        "curve_control_points_requested_per_group": args.curve_control_points,
        "curve_family_selected": result.secondary.curve_family,
        "curve_control_points_effective_per_group": int(
            result.secondary.curve_control_coordinates.size
        ),
        "strand_curve_monotone_non_decreasing_outward": True,
        "A_B_correction_passes": args.ab_correction_passes,
        "secondary_max_nfev_without_FEM": args.secondary_max_nfev,
        "secondary_ftol": args.secondary_ftol,
        "secondary_validation_displacement_mm": (
            args.secondary_validation_displacement_mm
        ),
        "secondary_validation_stress_mpa": args.secondary_validation_stress_mpa,
        "strand_iterations_requested": args.strand_iterations,
        "strand_step": args.strand_step,
        "strand_iterations_completed": result.strand_iterations_completed,
        "strand_iterations_tracked_total": previous_strand + result.strand_iterations_completed,
        "continuous_solves_this_run": result.continuous_solves,
        "continuous_solves_tracked_total": previous_continuous + result.continuous_solves,
        "OpenSees_FEM_cases_this_run": result.fem_cases,
        "OpenSees_FEM_seconds_this_run": result.fem_seconds,
        "OpenSees_FEM_cases_tracked_total": previous_fem_cases + result.fem_cases,
        "OpenSees_FEM_seconds_tracked_total": previous_fem_seconds + result.fem_seconds,
        "OpenSees_FEM_average_seconds_per_case": (
            result.fem_seconds / max(result.fem_cases, 1)
        ),
        "history_evaluations_previous": (
            0 if resume_state is None else len(resume_state.history_rows)
        ),
        "history_evaluations_this_run": len(result.history),
    }
    band_line = _band_verdict(result.best, args.stress_lower, args.stress_upper)
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
    print("  backend: opensees")
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
    print(
        f"  A-stage feasible: direct_balance="
        f"{sum(item.success for item in result.controls)}/{len(result.controls)}, "
        f"final_schedule="
        f"{sum(bool(item.final_schedule_feasible) for item in result.controls)}/"
        f"{len(result.controls)}"
    )
    max_final_a_residual_mm = max(
        max(
            abs(item.final_schedule_response.backstay_tower_dx_m),
            abs(item.final_schedule_response.main_stay_deck_uz_m),
        )
        * 1000.0
        for item in result.controls
    )
    print(f"  A-stage final-schedule max residual: {max_final_a_residual_mm:.6f} mm")
    if not all(bool(item.final_schedule_feasible) for item in result.controls):
        if args.ab_correction_passes == 0:
            print(
                "  recommendation: B sequence disturbs later A balance; rerun with "
                "--ab-correction-passes 1 before accepting the design"
            )
        else:
            print(
                "  recommendation: A/B correction still misses the balance tolerance; "
                "review active tension bounds, strand counts, or B-stage targets"
            )
    print(
        "  B-response validation: "
        f"{'PASS' if result.secondary.validated else 'FAIL'} "
        f"(disp={result.secondary.max_displacement_prediction_error_m:.3e} m, "
        f"stress={result.secondary.max_stress_prediction_error_mpa:.3e} MPa)"
    )
    print(
        f"  smooth curves: B={result.secondary.curve_family}, "
        f"controls/group={result.secondary.curve_control_coordinates.size}; "
        f"strand={result.strand_curve.family}, outward-monotone"
    )
    print(
        f"  OpenSees FEM: {result.fem_cases} cases, "
        f"{result.fem_seconds:.2f} s total, "
        f"{result.fem_seconds / max(result.fem_cases, 1):.2f} s/case"
    )
    print(f"  outputs: {out_dir}")
    return result


def build_parser(config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate local-balance stage-A tension and optimize completed-bridge "
            "stage-B tension on a smooth low-dimensional OpenSees response matrix."
        )
    )
    parser.add_argument(
        "--bridge",
        required=True,
        help="3D bridge YAML alias (for example omo3d) or YAML path",
    )
    parser.add_argument("--n", type=int, default=config.n_seg)
    parser.add_argument(
        "--backend",
        choices=("opensees",),
        default="opensees",
        help="optimization backend (the efficient architecture is currently OpenSees-only)",
    )
    parser.add_argument("--out", default="results/cable_opt_3d")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-from", metavar="PATH")

    parser.add_argument("--strand-min", type=int, default=5)
    parser.add_argument("--strand-max", type=int, default=500)
    parser.add_argument("--initial-strands", type=int)
    parser.add_argument("--stress-lower", type=float, default=400.0)
    parser.add_argument("--stress-upper", type=float, default=600.0)
    parser.add_argument("--tension-bound-stress", type=float, default=1600.0)

    parser.add_argument("--weight-shape", type=_nonnegative_float, default=1.0)
    parser.add_argument(
        "--weight-tower-displacement", type=_nonnegative_float, default=1.0
    )
    parser.add_argument(
        "--weight-tower-anchor-displacement", type=_nonnegative_float, default=1.0
    )
    parser.add_argument("--weight-strands", type=_nonnegative_float, default=0.02)
    parser.add_argument("--weight-stress-uniform", type=_nonnegative_float, default=0.2)
    parser.add_argument(
        "--weight-stress-violation", type=_nonnegative_float, default=100.0
    )
    parser.add_argument("--shape-scale-mm", type=_positive_float, default=100.0)
    parser.add_argument("--stress-scale", type=_positive_float, default=100.0)
    parser.add_argument("--strand-scale", type=_positive_float, default=8200.0)

    parser.add_argument("--stage-a-probe-fraction", type=_positive_float, default=1.0)
    parser.add_argument("--stage-a-feasibility-mm", type=_positive_float, default=0.1)
    parser.add_argument("--stage-a-condition-limit", type=_positive_float, default=1.0e10)
    parser.add_argument(
        "--ab-correction-passes",
        type=_nonnegative_int,
        default=0,
        help=(
            "recalculate A with the previous optimized B schedule, then rebuild B; "
            "each pass repeats the configured smooth-curve FEM budget"
        ),
    )
    parser.add_argument(
        "--curve-family",
        choices=("bernstein", "piecewise-linear", "auto"),
        default="bernstein",
        help=(
            "B-tension curve; auto evaluates both families and keeps the better "
            "validated full-model result"
        ),
    )
    parser.add_argument(
        "--curve-control-points",
        type=_positive_int,
        default=4,
        help="control points per cable type before interpolation (default: 4)",
    )
    parser.add_argument("--secondary-max-nfev", type=_positive_int, default=80)
    parser.add_argument("--secondary-ftol", type=_positive_float, default=1.0e-9)
    parser.add_argument(
        "--secondary-validation-displacement-mm",
        type=_positive_float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--secondary-validation-stress-mpa",
        type=_positive_float,
        default=1.0e-3,
    )
    parser.add_argument("--strand-iterations", type=_nonnegative_int, default=0)
    parser.add_argument("--strand-step", type=_positive_int, default=1)
    parser.add_argument("--quiet", action="store_true")
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
