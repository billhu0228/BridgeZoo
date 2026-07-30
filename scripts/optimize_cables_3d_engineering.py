"""Restartable three-round engineering feedback cycles for the OMO 3D bridge.

首次运行（最终 secondary_load 主梁/桥塔位移目标均为0、连续2轮）：python -m scripts.optimize_cables_3d_engineering --bridge omo3d --cycles 2 --out results/cable_opt_3d_engineering
中断后续算2轮：python -m scripts.optimize_cables_3d_engineering --bridge omo3d --cycles 2 --resume --out results/cable_opt_3d_engineering
结果重放验证：python -m scripts.single_staged_3d --bridge omo3d --backend opensees --design results/cable_opt_3d_engineering/best_design.json --render none --output results/cable_opt_3d_engineering/replayed.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from bridgezoo.optim import CableBounds, CableDesignEvaluator3D, CableOptimizationProblem
from bridgezoo.optim.engineering_cycle3d import (
    EngineeringCableCycleOptimizer3D,
    EngineeringCycleOptions3D,
    EngineeringProgress3D,
)
from bridgezoo.optim.problem import ObjectiveWeights
from scripts.bridge_config import load_single_staged_3d_config, resolve_bridge_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
_CHECKPOINT_SCHEMA = "bridgezoo.engineering_cable_cycle_3d.v13"
_LEGACY_CHECKPOINT_SCHEMAS = {
    "bridgezoo.engineering_cable_cycle_3d.v4",
    "bridgezoo.engineering_cable_cycle_3d.v5",
    "bridgezoo.engineering_cable_cycle_3d.v6",
    "bridgezoo.engineering_cable_cycle_3d.v7",
    "bridgezoo.engineering_cable_cycle_3d.v8",
    "bridgezoo.engineering_cable_cycle_3d.v9",
    "bridgezoo.engineering_cable_cycle_3d.v10",
    "bridgezoo.engineering_cable_cycle_3d.v11",
    "bridgezoo.engineering_cable_cycle_3d.v12",
}
_DESIGN_SCHEMA = "bridgezoo.cable_optimization_3d.v3"


@dataclass(frozen=True)
class _ResumeState:
    path: Path
    completed_cycles: int
    strands: np.ndarray
    pretension_a: np.ndarray
    pretension_b: np.ndarray
    step_scales: np.ndarray
    strand_step_scales: np.ndarray
    history: list[dict]
    cumulative_fem_cases: int
    cumulative_fem_seconds: float


class _EngineeringProgressDisplay:
    """Render all 24 construction groups in one refreshing terminal region."""

    _BAR_WIDTH = 30

    def __init__(
        self,
        *,
        first_cycle: int,
        cycles_this_run: int,
        target_stress_mpa: float,
        nominal_tension_step_stress_mpa: float,
        output_dir: Path,
        resumed_from: Path | None = None,
        refresh_interval: float = 0.2,
        stream=None,
        clock=None,
    ) -> None:
        if refresh_interval <= 0.0:
            raise ValueError("progress refresh interval must be positive")
        self.first_cycle = first_cycle
        self.last_cycle = first_cycle + cycles_this_run - 1
        self.cycles_this_run = cycles_this_run
        self.target_stress_mpa = target_stress_mpa
        self.nominal_tension_step_stress_mpa = nominal_tension_step_stress_mpa
        self.output_dir = output_dir
        self.resumed_from = resumed_from
        self.refresh_interval = refresh_interval
        self.stream = sys.stdout if stream is None else stream
        self.clock = time.perf_counter if clock is None else clock
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started = self.clock()
        self.last_rendered = float("-inf")
        self.rendered_lines = 0
        self.latest: EngineeringProgress3D | None = None
        self.closed = False

    @staticmethod
    def _pair(first: float | None, second: float | None) -> str:
        if first is None or second is None:
            return "      — /       —"
        return f"{first * 1000.0:7.2f} / {second * 1000.0:7.2f}"

    @staticmethod
    def _deck_values(
        stage_a: float | None,
        final: float | None,
        target: float,
    ) -> str:
        if stage_a is None or final is None:
            return f"      — /       — / {target * 1000.0:7.2f}"
        return (
            f"{stage_a * 1000.0:7.2f} / {final * 1000.0:7.2f} / "
            f"{target * 1000.0:7.2f}"
        )

    @staticmethod
    def _stress_pair(first: float | None, second: float | None) -> str:
        if first is None or second is None:
            return "      — /       —"
        return f"{first:7.1f} / {second:7.1f}"

    @staticmethod
    def _mn_pair(first: float | None, second: float | None) -> str:
        if first is None or second is None:
            return "      — /       —"
        return f"{first:7.3f} / {second:7.3f}"

    def _overall_fraction(self, event: EngineeringProgress3D) -> float:
        within = event.fem_cases_completed / max(1, event.fem_cases_total)
        completed_before = event.cycle_index - self.first_cycle
        return min(1.0, (completed_before + within) / self.cycles_this_run)

    def _dashboard_lines(self, event: EngineeringProgress3D) -> list[str]:
        fraction = self._overall_fraction(event)
        filled = min(self._BAR_WIDTH, int(round(self._BAR_WIDTH * fraction)))
        bar = "█" * filled + "·" * (self._BAR_WIDTH - filled)
        score = "—" if event.local_score is None else f"{event.local_score:.5f}"
        change = (
            "—"
            if event.score_change_percent is None
            else f"{event.score_change_percent:+.2f}%"
        )
        proposal = (
            "—" if event.proposal_score is None else f"{event.proposal_score:.5f}"
        )
        decision = (
            "—"
            if event.update_accepted is None
            else ("接受" if event.update_accepted else "拒绝")
        )
        if (
            event.tension_update_accepted is not None
            or event.strand_update_accepted is not None
            or event.repair_update_accepted is not None
        ):
            tension_decision = (
                "—"
                if event.tension_update_accepted is None
                else ("接受" if event.tension_update_accepted else "保留")
            )
            strand_decision = (
                "—"
                if event.strand_update_accepted is None
                else ("接受" if event.strand_update_accepted else "保留")
            )
            repair_decision = (
                "—"
                if event.repair_update_accepted is None
                else ("接受" if event.repair_update_accepted else "保留")
            )
            decision = (
                f"索力{tension_decision}/根数{strand_decision}/"
                f"修复{repair_decision}"
            )
        step_min = event.step_scale if event.step_scale_min is None else event.step_scale_min
        step_max = event.step_scale if event.step_scale_max is None else event.step_scale_max
        elapsed = self.clock() - self.started
        eta = (
            elapsed * (1.0 - fraction) / fraction
            if fraction > 0.0
            else event.eta_seconds
        )
        lines = [
            (
                "3D 工程调索（索力轮 → 根数轮 → 线形修复轮）  "
                f"cycle {event.cycle_index}/{self.last_cycle}  "
                f"FEM {event.fem_cases_completed}/{event.fem_cases_total}"
            ),
            f"总进度 [{bar}] {fraction * 100.0:5.1f}%  elapsed {elapsed:7.1f}s  ETA≈{eta:7.1f}s",
            (
                f"当前：{event.phase}  位移控制指标 {score}  候选 {proposal}  "
                f"决策 {decision}  改善 {change}  步长记忆×{step_min:.3f}…{step_max:.3f}"
            ),
            (
                "最终 secondary_load 目标：主梁 z=0.0 mm、塔锚 x=0.0 mm；"
                f"基准调索步长≈{self.nominal_tension_step_stress_mpa:.1f} MPa；"
                f"最终应力≈{self.target_stress_mpa:.1f} MPa。A仅校核激活切线零位移。"
            ),
            "",
            "组   根数(背/中)   A+B索力 [MN] 背 / 中   梁端 z [mm] A激活 / 最终 / 目标   塔端 x [mm] A激活 / 最终   最终应力 [MPa] 背 / 中",
            "──   ──────────   ────────────────────   ───────────────────────────────   ───────────────────────   ─────────────────────",
        ]
        for row in event.stage_status:
            lines.append(
                f"{row.construction_stage:02d}   "
                f"{row.backstay_strands:4d}/{row.main_stay_strands:<4d}   "
                f"{self._mn_pair(row.backstay_total_tension_mn, row.main_stay_total_tension_mn)}   "
                f"{self._deck_values(row.stage_a_deck_uz_m, row.final_deck_uz_m, row.target_final_deck_uz_m)}   "
                f"{self._pair(row.stage_a_tower_dx_m, row.final_tower_dx_m)}   "
                f"{self._stress_pair(row.backstay_final_stress_mpa, row.main_stay_final_stress_mpa)}"
            )
        source = "fresh run" if self.resumed_from is None else f"resume {self.resumed_from}"
        lines.extend(
            (
                "",
                f"状态：{source}",
                f"结果：{self.output_dir / 'best_design.json'}",
            )
        )
        return lines

    def update(self, event: EngineeringProgress3D) -> None:
        self.latest = event
        force = event.phase.startswith(("本轮完成", "本循环完成", "第1轮完成")) or event.phase in {
            "读取A激活与最终状态",
            "复用上轮修正后回放",
        }
        now = self.clock()
        if self.live:
            if not force and now - self.last_rendered < self.refresh_interval:
                return
            lines = self._dashboard_lines(event)
            if self.rendered_lines:
                self.stream.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                self.stream.write(f"\x1b[2K{line}\n")
            self.stream.flush()
            self.rendered_lines = len(lines)
            self.last_rendered = now

    def close(self) -> None:
        if self.closed:
            return
        # Redirected output cannot be refreshed.  Emit one final snapshot only,
        # avoiding the hundreds of append-only progress lines produced before.
        if not self.live and self.latest is not None:
            self.stream.write("\n".join(self._dashboard_lines(self.latest)) + "\n")
            self.stream.flush()
        elif self.live:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True


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


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
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


def _stage_group_vector(value, n_seg: int, *, dtype=float) -> np.ndarray:
    """Flatten scalar, per-stage, or paired configuration values."""

    if isinstance(value, (int, float)):
        flat = [value] * (2 * n_seg)
    else:
        raw = list(value)
        if len(raw) == 2 * n_seg:
            flat = raw
        elif len(raw) == n_seg:
            flat = []
            for item in raw:
                if isinstance(item, (tuple, list)):
                    if len(item) != 2:
                        raise ValueError("stage-pair configuration values need two entries")
                    flat.extend(item)
                else:
                    flat.extend((item, item))
        else:
            raise ValueError("configuration vector does not match n_seg")
    return np.asarray(flat, dtype=dtype)


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
                    "restartable_final_state_stress_priority_independent_count_"
                    "feedback_with_main_stay_b_shape_repair_without_"
                    "influence_matrices"
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
        source = source / "engineering_checkpoint.json"
    return source.resolve()


def _load_resume(
    path: Path,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    settings: dict,
    size: int,
) -> _ResumeState:
    if not path.is_file():
        raise FileNotFoundError(f"engineering checkpoint not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema not in {_CHECKPOINT_SCHEMA, *_LEGACY_CHECKPOINT_SCHEMAS}:
        raise ValueError(
            "unsupported engineering-cycle checkpoint schema; start a fresh "
            "final-state feedback run"
        )
    if payload.get("bridge_yaml") != bridge_yaml:
        raise ValueError("cannot resume: bridge YAML differs")
    saved_problem = dict(payload.get("problem", {}))
    current_problem = dict(problem_metadata)
    if schema in _LEGACY_CHECKPOINT_SCHEMAS:
        saved_problem.pop("optimizer_architecture", None)
        current_problem.pop("optimizer_architecture", None)
    if saved_problem != current_problem:
        raise ValueError("cannot resume: bridge model or bounds differ")
    saved_settings = dict(payload.get("settings", {}))
    current_settings = dict(settings)
    if schema in _LEGACY_CHECKPOINT_SCHEMAS and schema not in {
        "bridgezoo.engineering_cable_cycle_3d.v11",
        "bridgezoo.engineering_cable_cycle_3d.v12",
    }:
        # Older schemas used construction-stage B displacement/stress and a
        # configurable camber target.  Preserve their verified design vectors,
        # discard those obsolete target settings, and continue using the final
        # secondary-load state and independent-count semantics.
        target_keys = {
            "target_stress_mpa",
            "displacement_tolerance_m",
            "feedback_deck_scale_m",
            "feedback_tower_scale_m",
        }
        saved_settings = {
            key: value for key, value in saved_settings.items() if key in target_keys
        }
        current_settings = {
            key: value for key, value in current_settings.items() if key in target_keys
        }
    if saved_settings != current_settings:
        raise ValueError("cannot resume: engineering-cycle targets or update settings differ")
    state = payload.get("state", {})
    strands = np.asarray(state.get("strands_per_group", []), dtype=int)
    pretension_a = np.asarray(state.get("pretension_A_per_group_N", []), dtype=float)
    pretension_b = np.asarray(state.get("pretension_B_per_group_N", []), dtype=float)
    if any(value.size != size for value in (strands, pretension_a, pretension_b)):
        raise ValueError("cannot resume: checkpoint cable vectors differ from model")
    stored_step_scales = state.get("feedback_step_scale_per_phase_group")
    if stored_step_scales is not None:
        step_scales = np.asarray(stored_step_scales, dtype=float)
        if step_scales.shape != (2, size):
            raise ValueError("cannot resume: independent tension step memory differs from model")
    elif schema == "bridgezoo.engineering_cable_cycle_3d.v4":
        # v4's scalar was repeatedly reduced by the now-removed coupled
        # rejection mechanism; start the new independent memories neutrally.
        step_scales = np.ones((2, size), dtype=float)
    else:
        step_scales = np.full(
            (2, size), float(state.get("feedback_step_scale", 1.0)), dtype=float
        )
    if not np.all(np.isfinite(step_scales)) or np.any(step_scales <= 0.0):
        raise ValueError("cannot resume: invalid independent tension step memory")
    stored_strand_scales = state.get("strand_step_scale_per_group")
    strand_step_scales = (
        np.ones(size, dtype=float)
        if stored_strand_scales is None
        else np.asarray(stored_strand_scales, dtype=float)
    )
    if strand_step_scales.shape != (size,):
        raise ValueError("cannot resume: independent strand step memory differs from model")
    if (
        not np.all(np.isfinite(strand_step_scales))
        or np.any(strand_step_scales <= 0.0)
        or np.any(strand_step_scales > 2.0)
    ):
        raise ValueError("cannot resume: invalid independent strand step memory")
    return _ResumeState(
        path=path,
        completed_cycles=int(payload.get("completed_cycles", 0)),
        strands=strands,
        pretension_a=pretension_a,
        pretension_b=pretension_b,
        step_scales=step_scales,
        strand_step_scales=strand_step_scales,
        history=list(payload.get("history", [])),
        cumulative_fem_cases=int(payload.get("cumulative_FEM_cases", 0)),
        cumulative_fem_seconds=float(payload.get("cumulative_FEM_seconds", 0.0)),
    )


def _response_payload(response) -> dict:
    return {
        "stage_index": response.stage_index,
        "tower_dx_mm": response.backstay_tower_dx_m * 1000.0,
        "deck_uz_mm": response.main_stay_deck_uz_m * 1000.0,
    }


def _metrics_payload(evaluation) -> dict:
    metrics = evaluation.metrics
    return {
        "shape_rmse_mm": metrics.shape_rmse_m * 1000.0,
        "shape_max_abs_mm": metrics.shape_max_abs_m * 1000.0,
        "tower_top_dx_mm": metrics.tower_top_dx_m * 1000.0,
        "tower_anchor_dx_rmse_mm": metrics.tower_anchor_dx_rmse_m * 1000.0,
        "total_physical_strands": metrics.total_strands,
        "stress_mean_mpa": metrics.stress_mean_mpa,
        "stress_std_mpa": metrics.stress_std_mpa,
        "stress_min_mpa": metrics.stress_min_mpa,
        "stress_max_mpa": metrics.stress_max_mpa,
        "stress_violation_rms_mpa": metrics.stress_violation_rms_mpa,
        "stress_violation_max_mpa": metrics.stress_violation_max_mpa,
    }


def _design_payload(
    result,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    settings: dict,
    search: dict,
) -> dict:
    evaluation = result.evaluation_after_sizing
    total = evaluation.design.pretension
    ratios = evaluation.design.pretension_a_ratio
    nominal_step_mpa = (
        settings["feedback_relaxation"]
        * settings["tension_step_fraction"]
        * problem_metadata["bounds"]["tension_bound_stress_mpa"]
    )
    return {
        "schema": _DESIGN_SCHEMA,
        "bridge_yaml": bridge_yaml,
        "model_family": "single_staged_3d",
        "algorithm": "engineering_final_state_feedback_cycle",
        "problem": problem_metadata,
        "search": search,
        # Retained for compatibility and replay diagnostics.  The engineering
        # cycle uses the explicit final-state controls documented below.
        "objective": evaluation.objective,
        "components": asdict(evaluation.components),
        "metrics": _metrics_payload(evaluation),
        "engineering_cycle": {
            "cycle_index": result.cycle_index,
            "optimization_scope": (
                "A controls activation tangent displacement; B controls final "
                "secondary-load deck/tower displacement; round 2 adjusts independent "
                "strand counts from final cable stress and gives the 500 MPa target "
                "priority over temporary displacement disturbance; round 3 freezes "
                "A/counts/backstay B and repairs maximum final deck displacement "
                "with main-stay B only"
            ),
            "coordinate_sign": "+z is upward; +x follows the bridge model",
            "update_method": (
                "bounded activation-A/final-state-B tension feedback with independent "
                "A/B memory per cable group and one FEM-verified worst-control "
                "isolated retry after a rejected full proposal, followed by bounded "
                "proportional sizing with independent counts and adaptive memory per "
                "cable group; negative final stress forces a one-third count "
                "reduction outside the ordinary relaxation/change cap but inside "
                "hard capacity bounds; a final main-stay-B-only candidate is accepted "
                "only when max(abs(final deck z)) falls, with one verified worst-deck "
                "group retry; no strand curve and no influence matrix"
            ),
            "model_limit_warning": (
                "current model is linear small-displacement; the final-state zero "
                "displacement target is not a geometric-nonlinear verification"
            ),
            "settings": settings,
            "final_state_target": {
                "stage_label": "secondary_load",
                "deck_uz_mm": 0.0,
                "tower_anchor_dx_mm": 0.0,
                "cable_stress_mpa": settings["target_stress_mpa"],
            },
            "FEM_cases": result.fem_cases,
            "FEM_seconds": result.fem_seconds,
            "local_balance_score_before": result.local_score_before,
            "local_balance_score_after_tension": result.local_score_after_tension,
            "local_balance_score_after_strand_round": (
                result.local_score_after_strand
            ),
            "local_balance_score_after": result.local_score_after,
            "tension_proposal_local_balance_score": result.proposal_local_score,
            "tension_first_proposal_local_balance_score": (
                result.first_tension_proposal_local_score
            ),
            "tension_update_accepted": result.tension_update_accepted,
            "tension_partial_retry_attempted": (
                result.tension_partial_retry_attempted
            ),
            "tension_partial_retry_accepted": (
                result.tension_partial_retry_accepted
            ),
            "strand_update_attempted": result.strand_update_attempted,
            "strand_update_accepted": result.strand_update_accepted,
            "compression_strand_reduction_attempted": (
                result.compression_strand_reduction_attempted
            ),
            "compression_strand_reduction_accepted": (
                result.compression_strand_reduction_accepted
            ),
            "compression_strand_reduction_groups": [
                evaluation.cable_ids[index]
                for index in np.flatnonzero(
                    result.compression_strand_reduction_mask
                )
            ],
            "strand_stress_score_before": result.strand_score_before,
            "strand_proposal_stress_score": result.strand_proposal_score,
            "strand_round_stress_score_after": (
                result.strand_round_score_after
            ),
            "strand_stress_score_after": result.strand_score_after,
            "shape_repair_max_abs_deck_mm_before": (
                result.repair_max_deck_before_m * 1000.0
            ),
            "shape_repair_first_proposal_max_abs_deck_mm": (
                result.repair_first_proposal_max_deck_m * 1000.0
            ),
            "shape_repair_proposal_max_abs_deck_mm": (
                result.repair_proposal_max_deck_m * 1000.0
            ),
            "shape_repair_max_abs_deck_mm_after": (
                result.repair_max_deck_after_m * 1000.0
            ),
            "shape_repair_update_attempted": result.repair_update_attempted,
            "shape_repair_update_accepted": result.repair_update_accepted,
            "shape_repair_partial_retry_attempted": (
                result.repair_partial_retry_attempted
            ),
            "shape_repair_partial_retry_accepted": (
                result.repair_partial_retry_accepted
            ),
            # Compatibility fields used by earlier result readers.
            "proposal_local_balance_score": result.proposal_local_score,
            "update_accepted": result.update_accepted,
            "feedback_step_scale_used": result.step_scale_used,
            "feedback_step_scale_next": result.next_step_scale,
            "tension_step_memory": {
                "nominal_equivalent_stress_MPa": nominal_step_mpa,
                "used_A": result.tension_step_scales_used[0].tolist(),
                "used_B": result.tension_step_scales_used[1].tolist(),
                "next_A": result.next_tension_step_scales[0].tolist(),
                "next_B": result.next_tension_step_scales[1].tolist(),
                "next_equivalent_stress_A_MPa": (
                    nominal_step_mpa * result.next_tension_step_scales[0]
                ).tolist(),
                "next_equivalent_stress_B_MPa": (
                    nominal_step_mpa * result.next_tension_step_scales[1]
                ).tolist(),
            },
            "strand_step_memory": {
                "used": result.strand_step_scales_used.tolist(),
                "next": result.next_strand_step_scales.tolist(),
            },
            "strands_before_sizing": result.strands_before_sizing.tolist(),
            "strands_after_sizing": result.strands_after_sizing.tolist(),
            "final_stress_before_sizing_MPa": (
                result.stress_before_sizing_mpa.tolist()
            ),
            "final_stress_after_strand_round_MPa": (
                result.stress_after_strand_round_mpa.tolist()
            ),
            "final_stress_after_sizing_MPa": (
                result.stress_after_sizing_mpa.tolist()
            ),
            "strand_count_parameterization": {
                "family": "independent-per-group",
                "outward_non_decreasing": False,
                "stress_priority": True,
                "displacement_repaired_same_cycle": True,
            },
            "substage_controls": [
                {
                    "stage": control.construction_stage,
                    "phase": control.phase,
                    "displacement_basis": (
                        "deck z relative to this group's tangent birth; tower x is "
                        "actual cumulative displacement"
                        if control.phase == "A"
                        else "deck z and tower x are actual displacements in the "
                        "final secondary_load state"
                    ),
                    "target_tower_dx_mm": control.target_tower_dx_m * 1000.0,
                    "target_deck_uz_mm": control.target_deck_uz_m * 1000.0,
                    "deck_target_is_soft": control.deck_target_is_soft,
                    "deck_target_weight": control.deck_target_weight,
                    "deck_target_tolerance_mm": (
                        control.deck_target_tolerance_m * 1000.0
                    ),
                    "response_before": _response_payload(control.response_before),
                    "response_after_tension": _response_payload(
                        control.response_after_tension
                    ),
                    "response_after_sizing": _response_payload(
                        control.response_after
                    ),
                    "response_after_shape_repair": _response_payload(
                        control.response_after
                    ),
                    # Compatibility key: always the final saved design response.
                    "response_after": _response_payload(control.response_after),
                    "backstay_tension_before_N": control.backstay_tension_before_N,
                    "main_stay_tension_before_N": control.main_stay_tension_before_N,
                    "backstay_tension_after_N": control.backstay_tension_after_N,
                    "main_stay_tension_after_N": control.main_stay_tension_after_N,
                    "normalized_local_residual_before": control.residual_norm_before,
                    "normalized_local_residual_after": control.residual_norm_after,
                    "improved_this_cycle": control.improved,
                    "active_bounds": list(control.active_bounds),
                    "FEM_cases_for_influence_matrix": 0,
                }
                for control in result.controls
            ],
        },
        "cable_groups": [
            {
                "group_id": group_id,
                "stage": index // 2 + 1,
                "group": "backstay" if index % 2 == 0 else "main_stay",
                "physical_cable_ids": list(evaluation.cable_group_members[group_id]),
                "strands_per_physical_cable": int(evaluation.design.strands[index]),
                "pretension_per_physical_cable_N": float(total[index]),
                "pretension_a_ratio": float(ratios[index]),
                "pretension_A_per_physical_cable_N": float(result.pretension_a[index]),
                "pretension_B_per_physical_cable_N": float(result.pretension_b[index]),
                "pretension_A_plus_B_MN": float(total[index] / 1.0e6),
                "next_tension_step_A_MPa": float(
                    nominal_step_mpa * result.next_tension_step_scales[0, index]
                ),
                "next_tension_step_B_MPa": float(
                    nominal_step_mpa * result.next_tension_step_scales[1, index]
                ),
                "next_strand_step_scale": float(
                    result.next_strand_step_scales[index]
                ),
                "final_stress_used_for_sizing_MPa": float(
                    result.stress_after_sizing_mpa[index]
                ),
                "mean_final_stress_MPa": evaluation.cable_stress_mpa[group_id],
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


def _write_cycle_outputs(
    out_dir: Path,
    result,
    *,
    bridge_yaml: str,
    problem_metadata: dict,
    settings: dict,
    history: list[dict],
    cumulative_fem_cases: int,
    cumulative_fem_seconds: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    search = {
        "completed_cycles": result.cycle_index,
        "OpenSees_FEM_cases_this_cycle": result.fem_cases,
        "OpenSees_FEM_seconds_this_cycle": result.fem_seconds,
        "OpenSees_FEM_cases_cumulative": cumulative_fem_cases,
        "OpenSees_FEM_seconds_cumulative": cumulative_fem_seconds,
        "restartable_checkpoint": "engineering_checkpoint.json",
    }
    design = _design_payload(
        result,
        bridge_yaml=bridge_yaml,
        problem_metadata=problem_metadata,
        settings=settings,
        search=search,
    )
    checkpoint = {
        "schema": _CHECKPOINT_SCHEMA,
        "bridge_yaml": bridge_yaml,
        "problem": problem_metadata,
        "settings": settings,
        "completed_cycles": result.cycle_index,
        "state": {
            "strands_per_group": result.strands_after_sizing.tolist(),
            "pretension_A_per_group_N": result.pretension_a.tolist(),
            "pretension_B_per_group_N": result.pretension_b.tolist(),
            "feedback_step_scale": result.next_step_scale,
            "feedback_step_scale_per_phase_group": (
                result.next_tension_step_scales.tolist()
            ),
            "strand_step_scale_per_group": (
                result.next_strand_step_scales.tolist()
            ),
        },
        "history": history,
        "cumulative_FEM_cases": cumulative_fem_cases,
        "cumulative_FEM_seconds": cumulative_fem_seconds,
    }
    _write_json_atomic(out_dir / "best_design.json", design)
    _write_json_atomic(out_dir / "engineering_checkpoint.json", checkpoint)
    with (out_dir / "engineering_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = list(dict.fromkeys(key for row in history for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    stage_stress = result.stress_after_sizing_mpa
    improvement = (
        0.0
        if result.local_score_before == 0.0
        else 100.0
        * (result.local_score_before - result.local_score_after)
        / result.local_score_before
    )
    evaluation = result.evaluation_after_sizing
    nominal_step_mpa = (
        settings["feedback_relaxation"]
        * settings["tension_step_fraction"]
        * problem_metadata["bounds"]["tension_bound_stress_mpa"]
    )
    next_step_mpa = nominal_step_mpa * result.next_tension_step_scales
    lines = [
        "algorithm: final-state stress-priority independent-count feedback plus main-stay-B shape repair (no influence matrix)",
        f"completed cycle: {result.cycle_index}",
        "final secondary_load target: deck z=0.000 mm, tower-anchor x=0.000 mm",
        "activation A target: tangent-relative deck z=0.000 mm, tower-anchor x=0.000 mm",
        (
            f"local balance score: {result.local_score_before:.6g} -> "
            f"{result.local_score_after:.6g} ({improvement:+.3f}%)"
        ),
        "strand round may disturb displacement; the same cycle then runs a main-stay-B-only shape repair",
        (
            "negative-final-stress forced one-third strand reduction: "
            f"{'accepted' if result.compression_strand_reduction_accepted else ('attempted' if result.compression_strand_reduction_attempted else 'not needed')}"
        ),
        (
            f"tension round: {'accepted' if result.tension_update_accepted else 'kept'}, "
            f"score={result.proposal_local_score:.6g}; partial retry="
            f"{'accepted' if result.tension_partial_retry_accepted else ('rejected' if result.tension_partial_retry_attempted else 'not needed')}; "
            f"strand round: "
            f"{'accepted' if result.strand_update_accepted else 'kept'}, "
            f"round_stress_score={result.strand_round_score_after:.6g}, "
            f"final_stress_score={result.strand_score_after:.6g}; "
            f"next independent step={np.min(next_step_mpa):.3f}.."
            f"{np.max(next_step_mpa):.3f} MPa; next strand scale="
            f"{np.min(result.next_strand_step_scales):.3f}.."
            f"{np.max(result.next_strand_step_scales):.3f}"
        ),
        (
            "final-shape repair: "
            f"{'accepted' if result.repair_update_accepted else ('rejected' if result.repair_update_attempted else 'not needed')}; "
            f"max |deck z|={result.repair_max_deck_before_m * 1000.0:.3f} -> "
            f"{result.repair_max_deck_after_m * 1000.0:.3f} mm; partial retry="
            f"{'accepted' if result.repair_partial_retry_accepted else ('rejected' if result.repair_partial_retry_attempted else 'not needed')}"
        ),
        (
            "final secondary_load stress MPa: "
            f"mean={np.mean(stage_stress):.3f}, min={np.min(stage_stress):.3f}, "
            f"max={np.max(stage_stress):.3f}"
        ),
        f"physical strands: {evaluation.metrics.total_strands}",
        "final deck/tower displacement and cable stress drive tuning; other metrics are diagnostics",
        (
            f"completed-bridge diagnostic shape RMSE: "
            f"{evaluation.metrics.shape_rmse_m * 1000.0:.3f} mm"
        ),
        f"cycle FEM: {result.fem_cases} full replays, {result.fem_seconds:.3f} s",
        (
            "resume: python -m scripts.optimize_cables_3d_engineering "
            f"--bridge {bridge_yaml} --n {problem_metadata['n_seg']} "
            f"--resume --out {out_dir}"
        ),
    ]
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
            shape=1.0,
            total_strands=0.02,
            stress_uniform=0.2,
            stress_violation=100.0,
            shape_scale_m=0.1,
            stress_scale_mpa=100.0,
            strand_scale=8200.0,
            tower_displacement=1.0,
            tower_anchor_displacement=1.0,
        ),
        strand_area=config.strand_area,
        backend="opensees",
        model_family="single_staged_3d",
    )
    options = EngineeringCycleOptions3D(
        target_stress_mpa=args.target_stress_mpa,
        displacement_tolerance_m=args.displacement_tolerance_mm / 1000.0,
        feedback_relaxation=args.feedback_relaxation,
        tension_step_fraction=args.tension_step_fraction,
        feedback_deck_scale_m=args.feedback_deck_scale_mm / 1000.0,
        feedback_tower_scale_m=args.feedback_tower_scale_mm / 1000.0,
        strand_relaxation=args.strand_relaxation,
        strand_max_change_fraction=args.strand_max_change_fraction,
        strand_max_change_per_cycle=args.strand_max_change_per_cycle,
    )
    settings = json.loads(json.dumps(asdict(options)))
    problem_metadata = _problem_metadata(problem, config)
    out_dir = _project_path(args.out)
    resume_path = _resume_path(args, out_dir)
    state = (
        None
        if resume_path is None
        else _load_resume(
            resume_path,
            bridge_yaml=bridge_yaml,
            problem_metadata=problem_metadata,
            settings=settings,
            size=2 * config.n_seg,
        )
    )
    size = 2 * config.n_seg
    if state is None:
        strands = np.full(size, args.initial_strands, dtype=int)
        initial_total = _stage_group_vector(
            config.pretension_per_cable, config.n_seg
        )
        initial_ratio = _stage_group_vector(
            config.pretension_a_ratio, config.n_seg
        )
        pretension_a = initial_total * initial_ratio
        pretension_b = initial_total - pretension_a
        step_scales = np.ones((2, size), dtype=float)
        strand_step_scales = np.ones(size, dtype=float)
    else:
        strands = state.strands
        pretension_a = state.pretension_a
        pretension_b = state.pretension_b
        step_scales = state.step_scales
        strand_step_scales = state.strand_step_scales
    completed = 0 if state is None else state.completed_cycles
    history = [] if state is None else state.history
    cumulative_cases = 0 if state is None else state.cumulative_fem_cases
    cumulative_seconds = 0.0 if state is None else state.cumulative_fem_seconds

    display = None
    if not args.quiet:
        display = _EngineeringProgressDisplay(
            first_cycle=completed + 1,
            cycles_this_run=args.cycles,
            target_stress_mpa=args.target_stress_mpa,
            nominal_tension_step_stress_mpa=(
                args.feedback_relaxation
                * args.tension_step_fraction
                * args.tension_bound_stress
            ),
            output_dir=out_dir,
            resumed_from=None if state is None else state.path,
            refresh_interval=args.progress_refresh,
        )
    evaluator = CableDesignEvaluator3D(problem, config)
    optimizer = EngineeringCableCycleOptimizer3D(
        evaluator,
        options,
        progress=None if display is None else display.update,
    )
    latest = None
    baseline_evaluation = None
    try:
        for cycle_index in range(completed + 1, completed + args.cycles + 1):
            latest = optimizer.run_cycle(
                strands,
                cycle_index=cycle_index,
                pretension_a=pretension_a,
                pretension_b=pretension_b,
                step_scale=step_scales,
                strand_step_scale=strand_step_scales,
                baseline_evaluation=baseline_evaluation,
            )
            strands = latest.strands_after_sizing
            pretension_a = latest.pretension_a
            pretension_b = latest.pretension_b
            step_scales = latest.next_tension_step_scales
            strand_step_scales = latest.next_strand_step_scales
            baseline_evaluation = latest.evaluation_after_sizing
            cumulative_cases += latest.fem_cases
            cumulative_seconds += latest.fem_seconds
            evaluation = latest.evaluation_after_sizing
            local_stress = latest.stress_after_sizing_mpa
            improvement = (
                0.0
                if latest.local_score_before == 0.0
                else 100.0
                * (latest.local_score_before - latest.local_score_after)
                / latest.local_score_before
            )
            history.append(
                {
                    "cycle": cycle_index,
                    "local_balance_score_before": latest.local_score_before,
                    "local_balance_score_after_tension": (
                        latest.local_score_after_tension
                    ),
                    "local_balance_score_after_strand_round": (
                        latest.local_score_after_strand
                    ),
                    "local_balance_score_after": latest.local_score_after,
                    "local_improvement_percent": improvement,
                    "proposal_local_balance_score": latest.proposal_local_score,
                    "first_tension_proposal_local_balance_score": (
                        latest.first_tension_proposal_local_score
                    ),
                    "update_accepted": latest.update_accepted,
                    "tension_update_accepted": latest.tension_update_accepted,
                    "tension_partial_retry_attempted": (
                        latest.tension_partial_retry_attempted
                    ),
                    "tension_partial_retry_accepted": (
                        latest.tension_partial_retry_accepted
                    ),
                    "strand_update_attempted": latest.strand_update_attempted,
                    "strand_update_accepted": latest.strand_update_accepted,
                    "compression_strand_reduction_attempted": (
                        latest.compression_strand_reduction_attempted
                    ),
                    "compression_strand_reduction_accepted": (
                        latest.compression_strand_reduction_accepted
                    ),
                    "compression_strand_reduction_groups": ";".join(
                        str(evaluation.cable_ids[index])
                        for index in np.flatnonzero(
                            latest.compression_strand_reduction_mask
                        )
                    ),
                    "strand_stress_score_before": latest.strand_score_before,
                    "strand_proposal_stress_score": latest.strand_proposal_score,
                    "strand_round_stress_score_after": (
                        latest.strand_round_score_after
                    ),
                    "strand_stress_score_after": latest.strand_score_after,
                    "shape_repair_max_abs_deck_mm_before": (
                        latest.repair_max_deck_before_m * 1000.0
                    ),
                    "shape_repair_first_proposal_max_abs_deck_mm": (
                        latest.repair_first_proposal_max_deck_m * 1000.0
                    ),
                    "shape_repair_proposal_max_abs_deck_mm": (
                        latest.repair_proposal_max_deck_m * 1000.0
                    ),
                    "shape_repair_max_abs_deck_mm_after": (
                        latest.repair_max_deck_after_m * 1000.0
                    ),
                    "shape_repair_update_attempted": (
                        latest.repair_update_attempted
                    ),
                    "shape_repair_update_accepted": (
                        latest.repair_update_accepted
                    ),
                    "shape_repair_partial_retry_attempted": (
                        latest.repair_partial_retry_attempted
                    ),
                    "shape_repair_partial_retry_accepted": (
                        latest.repair_partial_retry_accepted
                    ),
                    "feedback_step_scale_used": latest.step_scale_used,
                    "feedback_step_scale_next": latest.next_step_scale,
                    "feedback_step_scale_min_next": float(
                        np.min(latest.next_tension_step_scales)
                    ),
                    "feedback_step_scale_max_next": float(
                        np.max(latest.next_tension_step_scales)
                    ),
                    "strand_step_scale_min_next": float(
                        np.min(latest.next_strand_step_scales)
                    ),
                    "strand_step_scale_max_next": float(
                        np.max(latest.next_strand_step_scales)
                    ),
                    "improved_substages": sum(
                        control.improved for control in latest.controls
                    ),
                    "physical_strands": evaluation.metrics.total_strands,
                    "final_stress_mean_mpa": float(np.mean(local_stress)),
                    "final_stress_min_mpa": float(np.min(local_stress)),
                    "final_stress_max_mpa": float(np.max(local_stress)),
                    "completed_shape_rmse_mm_diagnostic": (
                        evaluation.metrics.shape_rmse_m * 1000.0
                    ),
                    "completed_tower_top_dx_mm_diagnostic": (
                        evaluation.metrics.tower_top_dx_m * 1000.0
                    ),
                    "FEM_full_replays": latest.fem_cases,
                    "FEM_seconds": latest.fem_seconds,
                }
            )
            _write_cycle_outputs(
                out_dir,
                latest,
                bridge_yaml=bridge_yaml,
                problem_metadata=problem_metadata,
                settings=settings,
                history=history,
                cumulative_fem_cases=cumulative_cases,
                cumulative_fem_seconds=cumulative_seconds,
            )
    finally:
        if display is not None:
            display.close()
    return latest


def build_parser(config) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run restartable tension, strand, and final-shape-repair engineering "
            "cycles without per-stage influence matrices."
        ),
        epilog=(
            "首次运行（最终 secondary_load 主梁/桥塔位移目标均为0、连续2轮）：\n"
            "  python -m scripts.optimize_cables_3d_engineering --bridge omo3d --cycles 2 --out results/cable_opt_3d_engineering\n"
            "中断后续算2轮：\n"
            "  python -m scripts.optimize_cables_3d_engineering --bridge omo3d --cycles 2 --resume --out results/cable_opt_3d_engineering\n"
            "结果重放验证：\n"
            "  python -m scripts.single_staged_3d --bridge omo3d --backend opensees --design results/cable_opt_3d_engineering/best_design.json --render none --output results/cable_opt_3d_engineering/replayed.json"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--n", type=int, default=config.n_seg)
    parser.add_argument("--out", default="results/cable_opt_3d_engineering")
    parser.add_argument("--cycles", type=_positive_int, default=1)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", action="store_true")
    resume.add_argument("--resume-from", metavar="PATH")
    parser.add_argument("--initial-strands", type=_positive_int, default=100)
    parser.add_argument("--strand-min", type=_positive_int, default=5)
    parser.add_argument("--strand-max", type=_positive_int, default=500)
    parser.add_argument("--stress-lower", type=_positive_float, default=400.0)
    parser.add_argument("--stress-upper", type=_positive_float, default=600.0)
    parser.add_argument("--tension-bound-stress", type=_positive_float, default=1600.0)
    parser.add_argument("--target-stress-mpa", type=_positive_float, default=500.0)
    parser.add_argument(
        "--displacement-tolerance-mm", type=_positive_float, default=1.0
    )
    parser.add_argument(
        "--feedback-relaxation",
        type=_unit_fraction,
        default=0.25,
        help="multiplier in the nominal equivalent-stress tension step",
    )
    parser.add_argument(
        "--tension-step-fraction",
        type=_unit_fraction,
        default=0.075,
        help=(
            "fraction of the tension-bound stress before relaxation; defaults "
            "to an approximately 30 MPa independent step"
        ),
    )
    parser.add_argument(
        "--feedback-deck-scale-mm", type=_positive_float, default=100.0
    )
    parser.add_argument(
        "--feedback-tower-scale-mm", type=_positive_float, default=10.0
    )
    parser.add_argument(
        "--strand-relaxation",
        type=_unit_fraction,
        default=0.125,
        help=(
            "fraction of the bounded strand step before applying each group's "
            "adaptive memory"
        ),
    )
    parser.add_argument(
        "--strand-max-change-fraction", type=_unit_fraction, default=0.25
    )
    parser.add_argument(
        "--strand-max-change-per-cycle",
        type=_positive_int,
        default=25,
        help=(
            "absolute safety cap for one cable group's strand-count change in a "
            "sizing round; its independent adaptive memory controls the actual step"
        ),
    )
    parser.add_argument("--progress-refresh", type=_positive_float, default=0.2)
    parser.add_argument("--quiet", action="store_true")
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--bridge")
    known, _ = bootstrap.parse_known_args(argv)
    if known.bridge is None:
        bootstrap.error("--bridge is required; use --bridge omo3d")
    return build_parser(load_single_staged_3d_config(known.bridge)).parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        run(parse_args(argv))
    except KeyboardInterrupt:
        sys.stderr.write(
            "\r\x1b[2K工程调索已中止；最后一个完整循环的检查点可以继续。\n"
        )
        sys.stderr.flush()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
