"""Sequential forward A/B tuning and 500 MPa strand sizing for 3D bridges.

Fresh run::

    python -m scripts.optimize_cables_3d_forward --bridge omo3d --cycles 1

Continue from the strand configuration saved by the last complete cycle::

    python -m scripts.optimize_cables_3d_forward --bridge omo3d --resume

``best_design.json`` is the FEM-verified design from the forward pass.
``strand_configuration.json`` is the FEM-free resized configuration that will
be used at the beginning of the next cycle.  ``forward_checkpoint.json`` is
updated after every accepted A or B operation, so an interrupted run resumes
at the next operation without re-optimizing locked forces.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator3D,
    CableOptimizationProblem,
    ForwardCableCycleOptimizer3D,
    ForwardCycleOptions3D,
    ForwardLocalResponse3D,
    ForwardSubstageResult3D,
)
from bridgezoo.optim.problem import ObjectiveWeights
from bridgezoo.optim.variables import CableLayout, validate_strand_vector
from scripts.bridge_config import load_single_staged_3d_config, resolve_bridge_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_SCHEMA = "bridgezoo.forward_cable_cycle_3d.v1"
_DESIGN_SCHEMA = "bridgezoo.cable_optimization_3d.v3"
_STRAND_SCHEMA = "bridgezoo.cable_strand_configuration_3d.v1"


@dataclass(frozen=True)
class _ResumeState:
    path: Path
    completed_cycles: int
    strands: np.ndarray
    history: list[dict]
    cumulative_fem_replays: int
    cumulative_fem_seconds: float
    cycle_in_progress: bool = False
    active_cycle: int | None = None
    next_construction_stage: int = 1
    next_phase: str = "A"
    pretension_a: np.ndarray | None = None
    pretension_b: np.ndarray | None = None
    controls: tuple[ForwardSubstageResult3D, ...] = ()
    pending_birth_uz_m: dict[int, float] | None = None
    active_cycle_fem_replays: int = 0
    active_cycle_fem_seconds: float = 0.0


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def _unit_fraction(value: str) -> float:
    parsed = _positive_float(value)
    if parsed > 1.0:
        raise argparse.ArgumentTypeError("must not exceed one")
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


def _problem_metadata(problem: CableOptimizationProblem, config) -> dict:
    config_data = asdict(config)
    for variable in (
        "strands_per_cable",
        "pretension_per_cable",
        "pretension_a_ratio",
    ):
        config_data.pop(variable, None)
    return json.loads(
        json.dumps(
            {
                "n_seg": problem.n_seg,
                "model_family": problem.model_family,
                "backend": problem.backend,
                "optimizer_architecture": (
                    "sequential_forward_stage_A_then_B_with_locked_prior_groups_"
                    "followed_by_force_conserving_strand_sizing"
                ),
                "strand_area": problem.strand_area,
                "grouping": (
                    "stage-major (backstay, main_stay), two symmetric physical "
                    "cables per group"
                ),
                "bridge_config": config_data,
                "bounds": asdict(problem.bounds),
                "weights": asdict(problem.weights),
            }
        )
    )


def _resume_path(args, out_dir: Path) -> Path | None:
    if args.resume:
        source = out_dir
    elif args.resume_from is not None:
        source = _project_path(args.resume_from)
    else:
        return None
    if source.is_dir() or source.suffix.lower() != ".json":
        source = source / "forward_checkpoint.json"
    return source.resolve()


def _response_from_payload(payload: dict) -> ForwardLocalResponse3D:
    return ForwardLocalResponse3D(
        tower_anchor_dx_m=float(payload["tower_anchor_dx_mm"]) / 1000.0,
        deck_anchor_relative_uz_m=(
            float(payload["deck_anchor_relative_uz_mm"]) / 1000.0
        ),
    )


def _control_from_payload(payload: dict) -> ForwardSubstageResult3D:
    condition = payload.get("influence_condition")
    return ForwardSubstageResult3D(
        construction_stage=int(payload["stage"]),
        phase=str(payload["phase"]),
        stage_index=int(payload["stage_index"]),
        stage_label=str(payload["stage_label"]),
        displacement_basis=str(payload["displacement_basis"]),
        target=_response_from_payload(payload["target"]),
        response_before=_response_from_payload(payload["response_before"]),
        predicted_response=_response_from_payload(payload["predicted_response"]),
        response_after=_response_from_payload(payload["response_after"]),
        backstay_tension_N=float(payload["backstay_tension_N"]),
        main_stay_tension_N=float(payload["main_stay_tension_N"]),
        target_reached=bool(payload["target_reached"]),
        best_feasible=bool(payload["best_feasible"]),
        influence_rank=int(payload["influence_rank"]),
        influence_condition=(math.inf if condition is None else float(condition)),
        active_bounds=tuple(str(item) for item in payload["active_bounds"]),
        correction_passes=int(payload["correction_passes"]),
        fem_replays=int(payload["FEM_replays_from_stage_1"]),
    )


def _load_resume(
    path: Path,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    settings: dict,
    layout: CableLayout,
) -> _ResumeState:
    if not path.is_file():
        raise FileNotFoundError(f"forward checkpoint not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != _CHECKPOINT_SCHEMA:
        raise ValueError("unsupported forward-cycle checkpoint schema")
    if payload.get("bridge_yaml") != bridge_yaml:
        raise ValueError("cannot resume: bridge YAML differs")
    if payload.get("problem") != problem_metadata:
        raise ValueError("cannot resume: bridge model, backend, or bounds differ")
    if payload.get("settings") != settings:
        raise ValueError("cannot resume: forward-cycle settings differ")
    state = payload.get("state", {})
    cycle_in_progress = bool(state.get("cycle_in_progress", False))
    strand_key = (
        "strands_current_cycle"
        if cycle_in_progress
        else "strands_for_next_cycle"
    )
    strands = validate_strand_vector(
        state.get(strand_key, []),
        layout,
        int(problem_metadata["bounds"]["strand_min"]),
        int(problem_metadata["bounds"]["strand_max"]),
    )
    completed_cycles = int(payload.get("completed_cycles", 0))
    if completed_cycles < 0:
        raise ValueError("cannot resume: completed cycle count is invalid")
    if cycle_in_progress:
        active_cycle = int(state.get("active_cycle", 0))
        milestone_stage = int(state.get("milestone_stage", 0))
        milestone_phase = str(state.get("milestone_phase", ""))
        next_stage = int(state.get("next_construction_stage", 0))
        next_phase = str(state.get("next_phase", ""))
        if active_cycle != completed_cycles + 1:
            raise ValueError("cannot resume: active cycle is inconsistent")
        if not 1 <= milestone_stage <= layout.n_seg or milestone_phase not in {
            "A",
            "B",
        }:
            raise ValueError("cannot resume: construction milestone is invalid")
        expected_next = (
            (milestone_stage, "B")
            if milestone_phase == "A"
            else (milestone_stage + 1, "A")
        )
        if (next_stage, next_phase) != expected_next:
            raise ValueError("cannot resume: next construction operation is invalid")
        if not 1 <= next_stage <= layout.n_seg + 1 or (
            next_stage == layout.n_seg + 1 and next_phase != "A"
        ):
            raise ValueError("cannot resume: next construction operation is invalid")
        pretension_a = np.asarray(
            state.get("locked_pretension_A_per_group_N", []), dtype=float
        )
        pretension_b = np.asarray(
            state.get("locked_pretension_B_per_group_N", []), dtype=float
        )
        if pretension_a.shape != (layout.size,) or pretension_b.shape != (
            layout.size,
        ):
            raise ValueError("cannot resume: locked A/B vectors differ from model")
        if (
            not np.all(np.isfinite(pretension_a))
            or not np.all(np.isfinite(pretension_b))
            or np.any(pretension_a < 0.0)
            or np.any(pretension_b < 0.0)
        ):
            raise ValueError("cannot resume: locked A/B vectors are invalid")
        controls = tuple(
            _control_from_payload(item)
            for item in state.get("locked_controls", [])
        )
        expected_control_count = 2 * (milestone_stage - 1) + (
            1 if milestone_phase == "A" else 2
        )
        if len(controls) != expected_control_count:
            raise ValueError("cannot resume: locked controls differ from milestone")
        raw_birth = state.get("pending_tangent_birth_uz_m", {})
        pending_birth_uz_m = (
            {int(node): float(value) for node, value in raw_birth.items()}
            if next_phase == "B"
            else None
        )
        if next_phase == "B" and not pending_birth_uz_m:
            raise ValueError("cannot resume: phase B tangent birth is missing")
    else:
        active_cycle = None
        next_stage = 1
        next_phase = "A"
        pretension_a = None
        pretension_b = None
        controls = ()
        pending_birth_uz_m = None
    return _ResumeState(
        path=path,
        completed_cycles=completed_cycles,
        strands=strands,
        history=list(payload.get("history", [])),
        cumulative_fem_replays=int(payload.get("cumulative_FEM_replays", 0)),
        cumulative_fem_seconds=float(payload.get("cumulative_FEM_seconds", 0.0)),
        cycle_in_progress=cycle_in_progress,
        active_cycle=active_cycle,
        next_construction_stage=next_stage,
        next_phase=next_phase,
        pretension_a=pretension_a,
        pretension_b=pretension_b,
        controls=controls,
        pending_birth_uz_m=pending_birth_uz_m,
        active_cycle_fem_replays=int(
            state.get("FEM_replays_in_active_cycle", 0)
        ),
        active_cycle_fem_seconds=float(
            state.get("FEM_seconds_in_active_cycle", 0.0)
        ),
    )


def _response_payload(response) -> dict:
    return {
        "tower_anchor_dx_mm": response.tower_anchor_dx_m * 1000.0,
        "deck_anchor_relative_uz_mm": (
            response.deck_anchor_relative_uz_m * 1000.0
        ),
        "max_abs_mm": response.max_abs_m * 1000.0,
    }


def _response_residual(response, target) -> ForwardLocalResponse3D:
    return ForwardLocalResponse3D(
        tower_anchor_dx_m=(
            response.tower_anchor_dx_m - target.tower_anchor_dx_m
        ),
        deck_anchor_relative_uz_m=(
            response.deck_anchor_relative_uz_m
            - target.deck_anchor_relative_uz_m
        ),
    )


def _control_payload(control) -> dict:
    return {
        "stage": control.construction_stage,
        "phase": control.phase,
        "stage_index": control.stage_index,
        "stage_label": control.stage_label,
        "displacement_basis": control.displacement_basis,
        "target": _response_payload(control.target),
        "response_before": _response_payload(control.response_before),
        "residual_before": _response_payload(
            _response_residual(control.response_before, control.target)
        ),
        "predicted_response": _response_payload(control.predicted_response),
        "response_after": _response_payload(control.response_after),
        "residual_after": _response_payload(
            _response_residual(control.response_after, control.target)
        ),
        "backstay_tension_N": control.backstay_tension_N,
        "main_stay_tension_N": control.main_stay_tension_N,
        "target_reached": control.target_reached,
        "best_feasible": control.best_feasible,
        "influence_rank": control.influence_rank,
        "influence_condition": (
            control.influence_condition
            if math.isfinite(control.influence_condition)
            else None
        ),
        "active_bounds": list(control.active_bounds),
        "correction_passes": control.correction_passes,
        "FEM_replays_from_stage_1": control.fem_replays,
    }


def _design_payload(
    result,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    search: dict,
    target_stress_mpa: float,
) -> dict:
    evaluation = result.final_evaluation
    total = result.pretension_a + result.pretension_b
    ratio = np.divide(
        result.pretension_a,
        total,
        out=np.zeros_like(total),
        where=total > 0.0,
    )
    return {
        "schema": _DESIGN_SCHEMA,
        "bridge_yaml": bridge_yaml,
        "model_family": "single_staged_3d",
        "algorithm": "sequential_forward_stage_balance",
        "problem": problem_metadata,
        "search": search,
        "objective": evaluation.objective,
        "components": asdict(evaluation.components),
        "metrics": asdict(evaluation.metrics),
        "forward_cycle": {
            "cycle_index": result.cycle_index,
            "construction_order": "group 1..n; A then B; accepted prior groups locked",
            "A_state": "steel_and_A",
            "B_state": "deck_weight_and_B after wet-deck weight",
            "final_sizing_state": evaluation.staged_result.final.stage_label,
            "strands_used_by_verified_FEM": result.strands_before_sizing.tolist(),
            "strands_saved_for_next_cycle": result.strands_after_sizing.tolist(),
            "target_stress_MPa": target_stress_mpa,
            "resulting_stress_after_integer_sizing_MPa": (
                result.final_balanced_force_N
                / (
                    result.strands_after_sizing
                    * problem_metadata["strand_area"]
                )
                / 1.0e6
            ).tolist(),
            "sizing_equation": (
                "ceil(final_stress * current_strands / target_stress); "
                "then apply explicit strand bounds"
            ),
            "substage_controls": [
                _control_payload(control) for control in result.controls
            ],
        },
        "cable_groups": [
            {
                "group_id": group_id,
                "stage": index // 2 + 1,
                "group": "backstay" if index % 2 == 0 else "main_stay",
                "physical_cable_ids": list(
                    evaluation.cable_group_members[group_id]
                ),
                "strands_per_physical_cable": int(
                    result.strands_before_sizing[index]
                ),
                "strands_for_next_cycle": int(
                    result.strands_after_sizing[index]
                ),
                "unclipped_target_strands": int(
                    result.unclipped_target_strands[index]
                ),
                "sizing_was_clipped": bool(result.sizing_clipped[index]),
                "pretension_per_physical_cable_N": float(total[index]),
                "pretension_a_ratio": float(ratio[index]),
                "pretension_A_per_physical_cable_N": float(
                    result.pretension_a[index]
                ),
                "pretension_B_per_physical_cable_N": float(
                    result.pretension_b[index]
                ),
                "final_balanced_force_per_physical_cable_N": float(
                    result.final_balanced_force_N[index]
                ),
                "mean_final_stress_MPa": float(result.final_stress_mpa[index]),
                "physical_final_stress_MPa": {
                    str(member): evaluation.physical_cable_stress_mpa[member]
                    for member in evaluation.cable_group_members[group_id]
                },
            }
            for index, group_id in enumerate(evaluation.cable_ids)
        ],
        "final_state_controls": {
            "stage_label": evaluation.staged_result.final.stage_label,
            "deck_uz_mm": {
                str(node): evaluation.staged_result.final.displacement[node][2]
                * 1000.0
                for node in evaluation.deck_errors_m
            },
            "tower_anchor_dx_mm": {
                str(node): value * 1000.0
                for node, value in evaluation.tower_anchor_dx_m.items()
            },
        },
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_stage_milestone(
    out_dir: Path,
    milestone,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    settings: dict,
    completed_cycles: int,
    history: list[dict],
    prior_active_cycle_fem_replays: int,
    prior_active_cycle_fem_seconds: float,
    cumulative_fem_replays_before_segment: int,
    cumulative_fem_seconds_before_segment: float,
) -> None:
    """Atomically persist one accepted A or B construction milestone."""

    out_dir.mkdir(parents=True, exist_ok=True)
    if milestone.completed_phase == "A":
        completed_stage = milestone.construction_stage - 1
        next_stage = milestone.construction_stage
        next_phase = "B"
    else:
        completed_stage = milestone.construction_stage
        next_stage = milestone.construction_stage + 1
        next_phase = "A"
    checkpoint = {
        "schema": _CHECKPOINT_SCHEMA,
        "bridge_yaml": bridge_yaml,
        "problem": problem_metadata,
        "settings": settings,
        "completed_cycles": completed_cycles,
        "state": {
            "cycle_in_progress": True,
            "active_cycle": milestone.cycle_index,
            "milestone_stage": milestone.construction_stage,
            "milestone_phase": milestone.completed_phase,
            "completed_construction_stage": completed_stage,
            "next_construction_stage": next_stage,
            "next_phase": next_phase,
            "strands_current_cycle": milestone.strands.tolist(),
            "locked_pretension_A_per_group_N": (
                milestone.pretension_a.tolist()
            ),
            "locked_pretension_B_per_group_N": (
                milestone.pretension_b.tolist()
            ),
            "locked_controls": [
                _control_payload(control) for control in milestone.controls
            ],
            "pending_tangent_birth_uz_m": {
                str(node): value for node, value in milestone.birth_uz_m.items()
            },
            "FEM_replays_in_active_cycle": (
                prior_active_cycle_fem_replays + milestone.fem_replays
            ),
            "FEM_seconds_in_active_cycle": (
                prior_active_cycle_fem_seconds + milestone.fem_seconds
            ),
        },
        "history": history,
        "cumulative_FEM_replays": (
            cumulative_fem_replays_before_segment + milestone.fem_replays
        ),
        "cumulative_FEM_seconds": (
            cumulative_fem_seconds_before_segment + milestone.fem_seconds
        ),
    }
    _write_json_atomic(out_dir / "forward_checkpoint.json", checkpoint)


def _write_cycle_outputs(
    out_dir: Path,
    result,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    settings: dict,
    history: list[dict],
    cycle_fem_replays: int,
    cycle_fem_seconds: float,
    cumulative_fem_replays: int,
    cumulative_fem_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    search = {
        "completed_cycles": result.cycle_index,
        "FEM_replays_this_cycle": cycle_fem_replays,
        "FEM_seconds_this_cycle": cycle_fem_seconds,
        "FEM_replays_cumulative": cumulative_fem_replays,
        "FEM_seconds_cumulative": cumulative_fem_seconds,
        "restartable_checkpoint": "forward_checkpoint.json",
        "next_strand_configuration": "strand_configuration.json",
    }
    design = _design_payload(
        result,
        bridge_yaml=bridge_yaml,
        problem_metadata=problem_metadata,
        search=search,
        target_stress_mpa=settings["target_stress_mpa"],
    )
    strand_configuration = {
        "schema": _STRAND_SCHEMA,
        "bridge_yaml": bridge_yaml,
        "cycle_completed": result.cycle_index,
        "target_stress_MPa": settings["target_stress_mpa"],
        "note": (
            "This is the FEM-free sizing result and the input to the next "
            "forward cycle; best_design.json retains the FEM-verified counts."
        ),
        "strands_stage_major_backstay_main_stay": (
            result.strands_after_sizing.tolist()
        ),
        "cable_groups": [
            {
                "group_id": group_id,
                "stage": index // 2 + 1,
                "group": "backstay" if index % 2 == 0 else "main_stay",
                "strands_per_physical_cable": int(
                    result.strands_after_sizing[index]
                ),
            }
            for index, group_id in enumerate(result.final_evaluation.cable_ids)
        ],
    }
    checkpoint = {
        "schema": _CHECKPOINT_SCHEMA,
        "bridge_yaml": bridge_yaml,
        "problem": problem_metadata,
        "settings": settings,
        "completed_cycles": result.cycle_index,
        "state": {
            "cycle_in_progress": False,
            "strands_for_next_cycle": result.strands_after_sizing.tolist(),
        },
        "history": history,
        "cumulative_FEM_replays": cumulative_fem_replays,
        "cumulative_FEM_seconds": cumulative_fem_seconds,
    }
    _write_json_atomic(out_dir / "best_design.json", design)
    _write_json_atomic(out_dir / "strand_configuration.json", strand_configuration)
    _write_json_atomic(out_dir / "forward_checkpoint.json", checkpoint)
    summary = [
        "algorithm: sequential forward A/B tuning with locked prior groups",
        f"completed cycle: {result.cycle_index}",
        (
            "verified FEM strands: "
            + ",".join(str(value) for value in result.strands_before_sizing)
        ),
        (
            "next-cycle strands: "
            + ",".join(str(value) for value in result.strands_after_sizing)
        ),
        (
            "final secondary-load stress MPa: "
            f"min={np.min(result.final_stress_mpa):.3f}, "
            f"mean={np.mean(result.final_stress_mpa):.3f}, "
            f"max={np.max(result.final_stress_mpa):.3f}"
        ),
        (
            f"FEM replays this cycle: {cycle_fem_replays}, "
            f"{cycle_fem_seconds:.3f} s"
        ),
        "verified design: best_design.json",
        "saved next configuration: strand_configuration.json",
    ]
    (out_dir / "summary.txt").write_text(
        "\n".join(summary) + "\n", encoding="utf-8"
    )


def run(args):
    config = load_single_staged_3d_config(args.bridge)
    if args.n is not None:
        config = replace(config, n_seg=args.n)
    bridge_yaml = _bridge_yaml_reference(args.bridge)
    model_kwargs = asdict(config)
    for variable in (
        "strands_per_cable",
        "pretension_per_cable",
        "pretension_a_ratio",
    ):
        model_kwargs.pop(variable, None)
    problem = CableOptimizationProblem(
        n_seg=config.n_seg,
        model_kwargs=model_kwargs,
        bounds=CableBounds(
            strand_min=args.strand_min,
            strand_max=args.strand_max,
            stress_lower_mpa=0.8 * args.target_stress_mpa,
            stress_upper_mpa=1.2 * args.target_stress_mpa,
            tension_bound_stress_mpa=args.tension_bound_stress,
        ),
        weights=ObjectiveWeights(),
        strand_area=config.strand_area,
        backend=args.backend,
        model_family="single_staged_3d",
    )
    options = ForwardCycleOptions3D(
        target_stress_mpa=args.target_stress_mpa,
        displacement_tolerance_m=args.displacement_tolerance_mm / 1000.0,
        linearity_tolerance_m=args.linearity_tolerance_mm / 1000.0,
        probe_fraction=args.probe_fraction,
        max_corrections=args.max_corrections,
        require_secondary_load=not args.allow_no_secondary_load,
        require_target=not args.allow_best_feasible,
    )
    settings = json.loads(json.dumps(asdict(options)))
    problem_metadata = _problem_metadata(problem, config)
    out_dir = _project_path(args.out)
    layout = CableLayout(config.n_seg)
    resume_path = _resume_path(args, out_dir)
    state = (
        None
        if resume_path is None
        else _load_resume(
            resume_path,
            bridge_yaml=bridge_yaml,
            problem_metadata=problem_metadata,
            settings=settings,
            layout=layout,
        )
    )
    if state is None:
        strands = np.full(layout.size, args.initial_strands, dtype=int)
        strands = validate_strand_vector(
            strands,
            layout,
            problem.bounds.strand_min,
            problem.bounds.strand_max,
        )
        completed = 0
        history: list[dict] = []
        cumulative_replays = 0
        cumulative_seconds = 0.0
        cycle_in_progress = False
        active_cycle = None
        start_construction_stage = 1
        start_phase = "A"
        locked_a = None
        locked_b = None
        locked_controls = ()
        pending_birth_uz_m = None
        prior_active_replays = 0
        prior_active_seconds = 0.0
    else:
        strands = state.strands
        completed = state.completed_cycles
        history = state.history
        cumulative_replays = state.cumulative_fem_replays
        cumulative_seconds = state.cumulative_fem_seconds
        cycle_in_progress = state.cycle_in_progress
        active_cycle = state.active_cycle
        start_construction_stage = state.next_construction_stage
        start_phase = state.next_phase
        locked_a = state.pretension_a
        locked_b = state.pretension_b
        locked_controls = state.controls
        pending_birth_uz_m = state.pending_birth_uz_m
        prior_active_replays = state.active_cycle_fem_replays
        prior_active_seconds = state.active_cycle_fem_seconds

    progress = None if args.quiet else print
    latest = None
    for run_index in range(args.cycles):
        if run_index == 0 and cycle_in_progress:
            cycle_index = active_cycle
        else:
            cycle_index = completed + 1
            start_construction_stage = 1
            start_phase = "A"
            locked_a = None
            locked_b = None
            locked_controls = ()
            pending_birth_uz_m = None
            prior_active_replays = 0
            prior_active_seconds = 0.0
        if cycle_index is None:
            raise ValueError("active forward cycle is missing from checkpoint")

        cumulative_before_segment = cumulative_replays
        seconds_before_segment = cumulative_seconds

        def save_milestone(milestone) -> None:
            _write_stage_milestone(
                out_dir,
                milestone,
                bridge_yaml=bridge_yaml,
                problem_metadata=problem_metadata,
                settings=settings,
                completed_cycles=completed,
                history=history,
                prior_active_cycle_fem_replays=prior_active_replays,
                prior_active_cycle_fem_seconds=prior_active_seconds,
                cumulative_fem_replays_before_segment=(
                    cumulative_before_segment
                ),
                cumulative_fem_seconds_before_segment=seconds_before_segment,
            )

        optimizer = ForwardCableCycleOptimizer3D(
            CableDesignEvaluator3D(problem, config),
            options,
            progress=progress,
            milestone=save_milestone,
        )
        latest = optimizer.run_cycle(
            strands,
            cycle_index=cycle_index,
            start_construction_stage=start_construction_stage,
            start_phase=start_phase,
            pretension_a=locked_a,
            pretension_b=locked_b,
            completed_controls=locked_controls,
            pending_birth_uz_m=pending_birth_uz_m,
        )
        cycle_replays = prior_active_replays + latest.fem_replays
        cycle_seconds = prior_active_seconds + latest.fem_seconds
        strands = latest.strands_after_sizing
        cumulative_replays += latest.fem_replays
        cumulative_seconds += latest.fem_seconds
        history.append(
            {
                "cycle": cycle_index,
                "strands_before_sizing": latest.strands_before_sizing.tolist(),
                "strands_after_sizing": latest.strands_after_sizing.tolist(),
                "final_stress_min_MPa": float(np.min(latest.final_stress_mpa)),
                "final_stress_mean_MPa": float(np.mean(latest.final_stress_mpa)),
                "final_stress_max_MPa": float(np.max(latest.final_stress_mpa)),
                "FEM_replays": cycle_replays,
                "FEM_seconds": cycle_seconds,
            }
        )
        completed = cycle_index
        _write_cycle_outputs(
            out_dir,
            latest,
            bridge_yaml=bridge_yaml,
            problem_metadata=problem_metadata,
            settings=settings,
            history=history,
            cycle_fem_replays=cycle_replays,
            cycle_fem_seconds=cycle_seconds,
            cumulative_fem_replays=cumulative_replays,
            cumulative_fem_seconds=cumulative_seconds,
        )
        cycle_in_progress = False
    return latest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Tune each 3D cable group in forward construction order, lock its "
            "A/B forces, then resize strands at 500 MPa without another FEM run."
        )
    )
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--n", type=_positive_int)
    parser.add_argument("--backend", choices=("opensees", "direct"), default="opensees")
    parser.add_argument("--out", default="results/cable_opt_3d_forward")
    parser.add_argument("--cycles", type=_positive_int, default=1)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-from", metavar="PATH")
    parser.add_argument("--initial-strands", type=_positive_int, default=100)
    parser.add_argument("--strand-min", type=_positive_int, default=1)
    parser.add_argument("--strand-max", type=_positive_int, default=500)
    parser.add_argument(
        "--tension-bound-stress", type=_positive_float, default=1600.0
    )
    parser.add_argument("--target-stress-mpa", type=_positive_float, default=500.0)
    parser.add_argument(
        "--displacement-tolerance-mm", type=_positive_float, default=0.1
    )
    parser.add_argument(
        "--linearity-tolerance-mm", type=_positive_float, default=1.0e-5
    )
    parser.add_argument("--probe-fraction", type=_unit_fraction, default=0.10)
    parser.add_argument("--max-corrections", type=_positive_int, default=2)
    parser.add_argument(
        "--allow-best-feasible",
        action="store_true",
        help="continue when bounds prevent the requested displacement tolerance",
    )
    parser.add_argument(
        "--allow-no-secondary-load",
        action="store_true",
        help="size from the final composite state when no secondary-load stage exists",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.strand_max < args.strand_min:
        raise ValueError("strand maximum must not be below strand minimum")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
