"""Restartable engineering feedback cycles for staged 3D cable tuning.

This optimizer intentionally avoids a global mathematical solve and does not
build per-stage influence matrices.  One cycle instead:

1. replays the current design once and reads the activation-A and final responses;
2. changes only A/B tension and independently accepts that verified replay;
3. changes every cable group's strand count independently and accepts the
   second verified replay whenever the 500 MPa stress indicator improves,
   even if displacement is temporarily disturbed.

Only initial tension A uses an intermediate construction record: its tower
response and the new girder's tangent-birth-relative response are controlled to
zero at activation.  Tension B, strand sizing, deck displacement, tower
displacement, and cable stress all use the final ``secondary_load`` state.
Final deck and tower displacement targets are both exactly zero.  The following
cycle retunes A/B forces from an accepted count state and therefore seeks a new
final-state balance after every stiffness change.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np

from bridgezoo.optim.single_staged3d import (
    CableDesignEvaluator3D,
    EvaluationResult3D,
    StageAControlResponse3D,
)
from bridgezoo.optim.variables import (
    validate_strand_vector,
    validate_tension_vector,
)


@dataclass(frozen=True)
class EngineeringCycleOptions3D:
    """Numerical controls for deliberately relaxed engineering iterations."""

    target_stress_mpa: float = 500.0
    displacement_tolerance_m: float = 1.0e-3
    feedback_relaxation: float = 0.25
    tension_step_fraction: float = 0.075
    feedback_deck_scale_m: float = 0.10
    feedback_tower_scale_m: float = 0.010
    strand_relaxation: float = 0.125
    strand_max_change_fraction: float = 0.25
    strand_max_change_per_cycle: int = 25


@dataclass(frozen=True)
class EngineeringStageStatus3D:
    """One row in the live engineering-cycle dashboard."""

    construction_stage: int
    backstay_strands: int
    main_stay_strands: int
    target_final_deck_uz_m: float
    stage_a_tower_dx_m: float | None = None
    stage_a_deck_uz_m: float | None = None
    final_tower_dx_m: float | None = None
    final_deck_uz_m: float | None = None
    backstay_final_stress_mpa: float | None = None
    main_stay_final_stress_mpa: float | None = None


@dataclass(frozen=True)
class EngineeringProgress3D:
    """Structured progress event consumed by a refreshing terminal display."""

    cycle_index: int
    phase: str
    fem_cases_completed: int
    fem_cases_total: int
    elapsed_seconds: float
    eta_seconds: float
    stage_status: tuple[EngineeringStageStatus3D, ...]
    local_score: float | None = None
    score_change_percent: float | None = None
    proposal_score: float | None = None
    update_accepted: bool | None = None
    step_scale: float = 1.0
    tension_update_accepted: bool | None = None
    strand_update_accepted: bool | None = None
    step_scale_min: float | None = None
    step_scale_max: float | None = None


@dataclass(frozen=True)
class EngineeringSubstageControl3D:
    """Actual before/after response for one relaxed A or B correction."""

    construction_stage: int
    phase: str
    target_tower_dx_m: float
    target_deck_uz_m: float
    deck_target_is_soft: bool
    deck_target_weight: float
    deck_target_tolerance_m: float
    response_before: StageAControlResponse3D
    response_after_tension: StageAControlResponse3D
    response_after: StageAControlResponse3D
    backstay_tension_before_N: float
    main_stay_tension_before_N: float
    backstay_tension_after_N: float
    main_stay_tension_after_N: float
    residual_norm_before: float
    residual_norm_after: float
    improved: bool
    active_bounds: tuple[str, ...]
    message: str
    fem_cases: int = 0

    # Compatibility aliases retained for existing result readers.
    @property
    def predicted_response(self) -> StageAControlResponse3D:
        return self.response_after

    @property
    def first_round_response(self) -> StageAControlResponse3D:
        return self.response_before

    @property
    def post_sizing_response(self) -> StageAControlResponse3D:
        return self.response_after

    @property
    def backstay_tension_N(self) -> float:
        return self.backstay_tension_after_N

    @property
    def main_stay_tension_N(self) -> float:
        return self.main_stay_tension_after_N

    @property
    def feasible(self) -> bool:
        return self.residual_norm_after <= 1.0

    @property
    def stable(self) -> bool:
        return self.improved


@dataclass(frozen=True)
class EngineeringCycleResult3D:
    cycle_index: int
    evaluation_before_sizing: EvaluationResult3D
    evaluation_after_sizing: EvaluationResult3D
    controls: tuple[EngineeringSubstageControl3D, ...]
    pretension_a_before: np.ndarray
    pretension_b_before: np.ndarray
    pretension_a: np.ndarray
    pretension_b: np.ndarray
    strands_before_sizing: np.ndarray
    strands_after_sizing: np.ndarray
    stress_before_sizing_mpa: np.ndarray
    stress_after_sizing_mpa: np.ndarray
    local_score_before: float
    local_score_after_tension: float
    local_score_after: float
    proposal_local_score: float
    first_tension_proposal_local_score: float
    update_accepted: bool
    tension_update_accepted: bool
    tension_partial_retry_attempted: bool
    tension_partial_retry_accepted: bool
    strand_update_attempted: bool
    strand_update_accepted: bool
    strand_score_before: float
    strand_proposal_score: float | None
    strand_score_after: float
    step_scale_used: float
    next_step_scale: float
    tension_step_scales_used: np.ndarray
    next_tension_step_scales: np.ndarray
    strand_step_scales_used: np.ndarray
    next_strand_step_scales: np.ndarray
    fem_cases: int
    fem_seconds: float


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class EngineeringCableCycleOptimizer3D:
    """Perform one local-response engineering feedback cycle."""

    def __init__(
        self,
        evaluator: CableDesignEvaluator3D,
        options: EngineeringCycleOptions3D | None = None,
        progress: Callable[[EngineeringProgress3D], None] | None = None,
    ) -> None:
        if evaluator.problem.backend not in {"opensees", "direct"}:
            raise ValueError("engineering 3D cable cycles require a 3D FEM backend")
        self.evaluator = evaluator
        self.problem = evaluator.problem
        self.layout = evaluator.layout
        self.options = options or EngineeringCycleOptions3D()
        self.progress = progress
        self.total_fem_cases = 0
        self.total_fem_seconds = 0.0
        self._validate_options()

    def _validate_options(self) -> None:
        options = self.options
        positive = {
            "target stress": options.target_stress_mpa,
            "displacement tolerance": options.displacement_tolerance_m,
            "feedback deck scale": options.feedback_deck_scale_m,
            "feedback tower scale": options.feedback_tower_scale_m,
        }
        invalid = [name for name, value in positive.items() if value <= 0.0]
        if invalid:
            raise ValueError(f"positive values required for: {', '.join(invalid)}")
        for name, value in (
            ("feedback relaxation", options.feedback_relaxation),
            ("tension step fraction", options.tension_step_fraction),
            ("strand relaxation", options.strand_relaxation),
            ("strand maximum change fraction", options.strand_max_change_fraction),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if (
            isinstance(options.strand_max_change_per_cycle, bool)
            or options.strand_max_change_per_cycle < 1
        ):
            raise ValueError("strand maximum change per cycle must be positive")

    def tension_upper_bounds(self, strands: np.ndarray) -> np.ndarray:
        return (
            self.problem.bounds.tension_bound_stress_mpa
            * 1.0e6
            * self.problem.strand_area
            * strands.astype(float)
        )

    def nominal_tension_step_stress_mpa(self) -> float:
        """Return the saturated tension step as an equivalent cable stress."""

        return float(
            self.options.feedback_relaxation
            * self.options.tension_step_fraction
            * self.problem.bounds.tension_bound_stress_mpa
        )

    def _tension_step_scales(self, values) -> np.ndarray:
        """Broadcast legacy scalar memory to independent A/B group memory."""

        raw = np.asarray(values, dtype=float)
        shape = (2, self.layout.size)
        if raw.ndim == 0:
            scales = np.full(shape, float(raw), dtype=float)
        elif raw.shape == shape:
            scales = raw.copy()
        elif raw.size == 2 * self.layout.size:
            scales = raw.reshape(shape).copy()
        else:
            raise ValueError(
                "engineering tension step memory must be scalar or have "
                f"shape {shape}"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("engineering tension step memory must be positive and finite")
        if np.any(scales > 2.0):
            raise ValueError("engineering tension step memory must not exceed 2")
        return scales

    def _strand_step_scales(self, values) -> np.ndarray:
        """Broadcast a scalar to independent strand-count group memory."""

        raw = np.asarray(values, dtype=float)
        shape = (self.layout.size,)
        if raw.ndim == 0:
            scales = np.full(shape, float(raw), dtype=float)
        elif raw.shape == shape:
            scales = raw.copy()
        elif raw.size == self.layout.size:
            scales = raw.reshape(shape).copy()
        else:
            raise ValueError(
                "engineering strand step memory must be scalar or have "
                f"shape {shape}"
            )
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0.0):
            raise ValueError("engineering strand step memory must be positive and finite")
        if np.any(scales > 2.0):
            raise ValueError("engineering strand step memory must not exceed 2")
        return scales

    @staticmethod
    def _total_and_ratio(
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        total = pretension_a + pretension_b
        ratio = np.divide(
            pretension_a,
            total,
            out=np.zeros_like(total),
            where=total > 0.0,
        )
        return total, ratio

    def _emit(self, event: EngineeringProgress3D) -> None:
        if self.progress is not None:
            self.progress(event)

    def _evaluate(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
    ) -> EvaluationResult3D:
        total, ratio = self._total_and_ratio(pretension_a, pretension_b)
        started = time.perf_counter()
        result = self.evaluator.evaluate(
            strands, total, ratio, keep_result=True
        )
        duration = time.perf_counter() - started
        self.total_fem_cases += 1
        self.total_fem_seconds += duration
        return result

    def _local_stage_responses(
        self,
        evaluation: EvaluationResult3D,
    ) -> tuple[
        dict[tuple[int, str], StageAControlResponse3D],
        np.ndarray,
    ]:
        """Extract activation-A tangent response and final-state controls."""

        total = evaluation.design.pretension
        ratio = evaluation.design.pretension_a_ratio
        plan = self.evaluator.build_plan(evaluation.design.strands, total, ratio)
        records = {
            record.stage_index: record for record in evaluation.staged_result.records
        }
        final_record = evaluation.staged_result.final
        if final_record.stage_label != "secondary_load":
            raise RuntimeError(
                "engineering cable tuning requires a final secondary_load stage"
            )
        responses: dict[tuple[int, str], StageAControlResponse3D] = {}
        stress = np.zeros(self.layout.size, dtype=float)
        for construction_stage in range(1, self.problem.n_seg + 1):
            offset = 2 * (construction_stage - 1)
            cables = [
                cable
                for cable in plan.model.cables.values()
                if cable.construction_stage == construction_stage
            ]
            backstays = [cable for cable in cables if cable.group == "backstay"]
            main_stays = [cable for cable in cables if cable.group == "main_stay"]
            if len(backstays) != 2 or len(main_stays) != 2:
                raise RuntimeError(
                    f"3D stage {construction_stage} must contain two cables per group"
                )
            tower_nodes = {cable.i for cable in backstays}
            deck_nodes = {cable.j for cable in main_stays}
            a_record = records[3 * construction_stage - 2]
            missing_birth = deck_nodes.difference(a_record.birth_displacement)
            if missing_birth:
                raise RuntimeError(
                    f"stage {construction_stage} is missing tangent-birth displacement "
                    f"for deck nodes {sorted(missing_birth)}"
                )
            stage_a_deck_reference = {
                node_id: float(a_record.birth_displacement[node_id][2])
                for node_id in deck_nodes
            }
            for phase, record in (("A", a_record), ("B", final_record)):
                responses[construction_stage, phase] = StageAControlResponse3D(
                    construction_stage=construction_stage,
                    stage_index=record.stage_index,
                    backstay_tower_dx_m=float(
                        np.mean(
                            [
                                record.displacement[node_id][0]
                                for node_id in tower_nodes
                            ]
                        )
                    ),
                    main_stay_deck_uz_m=float(
                        np.mean(
                            [
                                record.displacement[node_id][2]
                                - (
                                    stage_a_deck_reference[node_id]
                                    if phase == "A"
                                    else 0.0
                                )
                                for node_id in deck_nodes
                            ]
                        )
                    ),
                )
            stress[offset] = float(
                np.mean([final_record.cable_stress[cable.id] for cable in backstays])
                / 1.0e6
            )
            stress[offset + 1] = float(
                np.mean([final_record.cable_stress[cable.id] for cable in main_stays])
                / 1.0e6
            )
        return responses, stress

    def _stage_status(
        self,
        strands: np.ndarray,
        responses: dict[tuple[int, str], StageAControlResponse3D] | None,
        stress_mpa: np.ndarray | None,
    ) -> tuple[EngineeringStageStatus3D, ...]:
        rows = []
        for stage in range(1, self.problem.n_seg + 1):
            offset = 2 * (stage - 1)
            response_a = None if responses is None else responses[stage, "A"]
            response_b = None if responses is None else responses[stage, "B"]
            rows.append(
                EngineeringStageStatus3D(
                    construction_stage=stage,
                    backstay_strands=int(strands[offset]),
                    main_stay_strands=int(strands[offset + 1]),
                    target_final_deck_uz_m=0.0,
                    stage_a_tower_dx_m=(
                        None if response_a is None else response_a.backstay_tower_dx_m
                    ),
                    stage_a_deck_uz_m=(
                        None if response_a is None else response_a.main_stay_deck_uz_m
                    ),
                    final_tower_dx_m=(
                        None if response_b is None else response_b.backstay_tower_dx_m
                    ),
                    final_deck_uz_m=(
                        None if response_b is None else response_b.main_stay_deck_uz_m
                    ),
                    backstay_final_stress_mpa=(
                        None if stress_mpa is None else float(stress_mpa[offset])
                    ),
                    main_stay_final_stress_mpa=(
                        None if stress_mpa is None else float(stress_mpa[offset + 1])
                    ),
                )
            )
        return tuple(rows)

    def _local_score(
        self,
        responses: dict[tuple[int, str], StageAControlResponse3D],
        stress_mpa: np.ndarray | None = None,
    ) -> float:
        """Return the displacement-balance indicator used by the tension round."""

        options = self.options
        residuals = []
        for stage in range(1, self.problem.n_seg + 1):
            response_a = responses[stage, "A"]
            response_b = responses[stage, "B"]
            residuals.extend(
                (
                    response_a.backstay_tower_dx_m
                    / options.feedback_tower_scale_m,
                    response_a.main_stay_deck_uz_m
                    / options.feedback_deck_scale_m,
                    response_b.backstay_tower_dx_m
                    / options.feedback_tower_scale_m,
                    response_b.main_stay_deck_uz_m
                    / options.feedback_deck_scale_m,
                )
            )
        values = np.asarray(residuals, dtype=float)
        return float(np.sqrt(np.mean(values * values)))

    def _strand_score(self, stress_mpa: np.ndarray) -> float:
        """Return the stress-uniformity indicator used only by the sizing round."""

        residual = (
            np.asarray(stress_mpa, dtype=float) - self.options.target_stress_mpa
        ) / self.options.target_stress_mpa
        return float(np.sqrt(np.mean(residual * residual)))

    def _feedback_correction(
        self,
        *,
        error: float,
        scale: float,
        upper: float,
        step_scale: float,
    ) -> float:
        options = self.options
        cap = upper * options.tension_step_fraction
        # Final responses are coupled to every cable group.  A scalar secant
        # inferred while all groups change is therefore
        # not a valid derivative for one cable and can reverse convergence at
        # outer stages.  Use deliberately slow bounded feedback instead.
        correction = (
            options.feedback_relaxation
            * cap
            * step_scale
            * float(np.clip(error / scale, -1.0, 1.0))
        )
        return float(np.clip(correction, -cap, cap))

    def _tension_feedback_residuals(
        self,
        responses: dict[tuple[int, str], StageAControlResponse3D],
    ) -> np.ndarray:
        """Return A activation and B final-state residuals to zero."""

        residuals = np.zeros((2, self.layout.size), dtype=float)
        for stage in range(1, self.problem.n_seg + 1):
            offset = 2 * (stage - 1)
            response_a = responses[stage, "A"]
            response_b = responses[stage, "B"]
            # A is the only intermediate-stage control.  B changes are judged
            # by the common final secondary-load state for every cable group.
            residuals[0, offset] = response_a.backstay_tower_dx_m
            residuals[0, offset + 1] = response_a.main_stay_deck_uz_m
            residuals[1, offset] = response_b.backstay_tower_dx_m
            residuals[1, offset + 1] = response_b.main_stay_deck_uz_m
        return residuals

    def _isolated_tension_candidate(
        self,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
        proposed_a: np.ndarray,
        proposed_b: np.ndarray,
        residual_before: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Isolate the worst changed control component for a verified retry."""

        changed = np.vstack(
            (
                np.abs(proposed_a - pretension_a) > 1.0e-9,
                np.abs(proposed_b - pretension_b) > 1.0e-9,
            )
        )
        normalized = np.zeros_like(residual_before)
        backstay = np.arange(self.layout.size) % 2 == 0
        main_stay = ~backstay
        normalized[:, backstay] = (
            residual_before[:, backstay]
            / self.options.feedback_tower_scale_m
        )
        normalized[0, main_stay] = (
            residual_before[0, main_stay]
            / self.options.feedback_deck_scale_m
        )
        normalized[1, main_stay] = (
            residual_before[1, main_stay]
            / self.options.feedback_deck_scale_m
        )
        priority = np.where(changed, np.abs(normalized), -np.inf)
        keep = np.zeros_like(changed)
        if np.any(changed):
            keep.flat[int(np.argmax(priority))] = True
        partial_a = np.where(keep[0], proposed_a, pretension_a)
        partial_b = np.where(keep[1], proposed_b, pretension_b)
        return partial_a, partial_b, keep

    @staticmethod
    def _adapt_tension_step_scales(
        scales: np.ndarray,
        residual_before: np.ndarray,
        residual_after: np.ndarray,
        changed: np.ndarray,
    ) -> np.ndarray:
        """Adapt every A/B cable-group memory without additional FEM cases."""

        updated = scales.copy()
        before_abs = np.abs(residual_before)
        after_abs = np.abs(residual_after)
        sign_flip = residual_before * residual_after < 0.0
        worsened = after_abs > before_abs * (1.0 + 1.0e-6)
        strong_improvement = after_abs < 0.70 * before_abs
        shrink = changed & (worsened | sign_flip)
        grow = changed & strong_improvement & ~sign_flip
        updated[shrink] *= 0.5
        updated[grow] *= 1.15
        return np.clip(updated, 1.0 / 32.0, 2.0)

    @staticmethod
    def _adapt_strand_step_scales(
        scales: np.ndarray,
        stress_before_mpa: np.ndarray,
        stress_after_mpa: np.ndarray,
        changed: np.ndarray,
        target_stress_mpa: float,
    ) -> np.ndarray:
        """Adapt every cable group's sizing memory from its own stress response."""

        residual_before = stress_before_mpa - target_stress_mpa
        residual_after = stress_after_mpa - target_stress_mpa
        updated = scales.copy()
        before_abs = np.abs(residual_before)
        after_abs = np.abs(residual_after)
        sign_flip = residual_before * residual_after < 0.0
        worsened = after_abs > before_abs * (1.0 + 1.0e-6)
        improved = after_abs < before_abs * (1.0 - 1.0e-6)
        shrink = changed & (worsened | sign_flip)
        grow = changed & improved & ~sign_flip
        updated[shrink] *= 0.5
        updated[grow] *= 1.15
        return np.clip(updated, 1.0 / 32.0, 2.0)

    def _update_tensions(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
        responses: dict[tuple[int, str], StageAControlResponse3D],
        step_scales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        options = self.options
        step_scales = self._tension_step_scales(step_scales)
        upper = self.tension_upper_bounds(strands)
        updated_a = pretension_a.copy()
        updated_b = pretension_b.copy()
        residuals = self._tension_feedback_residuals(responses)
        for stage in range(1, self.problem.n_seg + 1):
            offset = 2 * (stage - 1)
            for phase_index, (phase, values) in enumerate(
                (("A", updated_a), ("B", updated_b))
            ):
                response = responses[stage, phase]
                errors = -residuals[phase_index, offset : offset + 2]
                scales = (
                    options.feedback_tower_scale_m,
                    options.feedback_deck_scale_m,
                )
                for local_index in range(2):
                    index = offset + local_index
                    correction = self._feedback_correction(
                        error=errors[local_index],
                        scale=scales[local_index],
                        upper=float(upper[index]),
                        step_scale=float(step_scales[phase_index, index]),
                    )
                    # Tangent activation can make A-stage deck displacement
                    # exactly insensitive to main-stay A tension.  In that
                    # under-determined case the minimal A force is preferred.
                    if (
                        phase == "A"
                        and local_index == 1
                        and abs(errors[local_index])
                        <= options.displacement_tolerance_m
                    ):
                        correction = -min(
                            float(values[index]),
                            0.5
                            * upper[index]
                            * options.tension_step_fraction
                            * float(step_scales[phase_index, index]),
                        )
                    other = updated_b[index] if phase == "A" else updated_a[index]
                    values[index] = float(
                        np.clip(values[index] + correction, 0.0, upper[index] - other)
                    )
        return updated_a, updated_b

    def _resize_strands(
        self,
        strands: np.ndarray,
        total_tension: np.ndarray,
        stress_mpa: np.ndarray,
        step_scales: np.ndarray,
    ) -> np.ndarray:
        options = self.options
        step_scales = self._strand_step_scales(step_scales)
        target = np.ceil(
            strands.astype(float)
            * np.maximum(stress_mpa, 0.0)
            / options.target_stress_mpa
        )
        minimum_for_tension = np.ceil(
            np.divide(
                total_tension,
                self.problem.bounds.tension_bound_stress_mpa
                * 1.0e6
                * self.problem.strand_area,
            )
        )
        target = np.maximum(target, minimum_for_tension)
        max_change = np.minimum(
            float(options.strand_max_change_per_cycle),
            np.maximum(
                1.0,
                np.ceil(
                    options.strand_max_change_fraction * strands
                ),
            ),
        )
        target_delta = target - strands
        # Preserve group-to-group utilization differences even while every
        # cable is far from the target stress.  The previous normalized/clipped
        # residual reduced all sufficiently large errors to the same +/-1 and
        # therefore kept initially uniform strand counts artificially uniform.
        # The small relaxation and independent cached scale make this direct
        # proportional correction conservative; max_change remains the hard
        # per-cycle safety bound.
        relaxed = strands + (
            options.strand_relaxation
            * step_scales
            * target_delta
        )
        limited = np.clip(relaxed, strands - max_change, strands + max_change)
        proposed = np.rint(limited).astype(int)
        proposed = np.clip(
            proposed,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        required = np.clip(
            minimum_for_tension.astype(int),
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        # Counts are true per-group design variables.  Do not fit a Bernstein
        # curve or impose outward monotonicity: either operation couples an
        # otherwise improving group to unrelated groups and can prevent low-
        # stress cables from shedding strands.
        return np.maximum(proposed, required)

    def _substage_residual_norm(
        self,
        phase: str,
        response: StageAControlResponse3D,
    ) -> float:
        options = self.options
        return float(
            math.hypot(
                response.backstay_tower_dx_m / options.feedback_tower_scale_m,
                response.main_stay_deck_uz_m / options.feedback_deck_scale_m,
            )
        )

    def _controls(
        self,
        before: dict[tuple[int, str], StageAControlResponse3D],
        after_tension: dict[tuple[int, str], StageAControlResponse3D],
        after_sizing: dict[tuple[int, str], StageAControlResponse3D],
        pretension_a_before: np.ndarray,
        pretension_b_before: np.ndarray,
        pretension_a_after: np.ndarray,
        pretension_b_after: np.ndarray,
        upper_after: np.ndarray,
    ) -> tuple[EngineeringSubstageControl3D, ...]:
        controls = []
        for stage in range(1, self.problem.n_seg + 1):
            offset = 2 * (stage - 1)
            for phase, tension_before, tension_after in (
                ("A", pretension_a_before, pretension_a_after),
                ("B", pretension_b_before, pretension_b_after),
            ):
                response_before = before[stage, phase]
                response_after_tension = after_tension[stage, phase]
                response_after = after_sizing[stage, phase]
                norm_before = self._substage_residual_norm(phase, response_before)
                norm_after = self._substage_residual_norm(
                    phase, response_after_tension
                )
                active = []
                for local, name in enumerate(("backstay", "main_stay")):
                    index = offset + local
                    total = pretension_a_after[index] + pretension_b_after[index]
                    if tension_after[index] <= 1.0:
                        active.append(f"{name}:lower")
                    if total >= upper_after[index] - 1.0:
                        active.append(f"{name}:total_upper")
                controls.append(
                    EngineeringSubstageControl3D(
                        construction_stage=stage,
                        phase=phase,
                        target_tower_dx_m=0.0,
                        target_deck_uz_m=0.0,
                        deck_target_is_soft=False,
                        deck_target_weight=1.0,
                        deck_target_tolerance_m=self.options.feedback_deck_scale_m,
                        response_before=response_before,
                        response_after_tension=response_after_tension,
                        response_after=response_after,
                        backstay_tension_before_N=float(tension_before[offset]),
                        main_stay_tension_before_N=float(tension_before[offset + 1]),
                        backstay_tension_after_N=float(tension_after[offset]),
                        main_stay_tension_after_N=float(tension_after[offset + 1]),
                        residual_norm_before=norm_before,
                        residual_norm_after=norm_after,
                        improved=bool(norm_after < norm_before),
                        active_bounds=tuple(active),
                        message=(
                            (
                                "activation-A tangent feedback"
                                if phase == "A"
                                else "final secondary-load feedback"
                            )
                            + "; no influence matrix; "
                            f"local residual {norm_before:.6g}->{norm_after:.6g}"
                        ),
                    )
                )
        return tuple(controls)

    def run_cycle(
        self,
        strands,
        *,
        cycle_index: int,
        pretension_a=None,
        pretension_b=None,
        step_scale=1.0,
        strand_step_scale=1.0,
        baseline_evaluation: EvaluationResult3D | None = None,
    ) -> EngineeringCycleResult3D:
        """Run independently verified tension and strand-count subrounds."""

        started = time.perf_counter()
        start_cases = self.total_fem_cases
        start_seconds = self.total_fem_seconds
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension_a = validate_tension_vector(
            np.zeros(self.layout.size) if pretension_a is None else pretension_a,
            self.layout,
        )
        pretension_b = validate_tension_vector(
            np.zeros(self.layout.size) if pretension_b is None else pretension_b,
            self.layout,
        )
        if np.any(pretension_a + pretension_b > self.tension_upper_bounds(strands)):
            raise ValueError("initial A+B pretension exceeds the strand tension bound")
        tension_step_scales = self._tension_step_scales(step_scale)
        strand_step_scales = self._strand_step_scales(strand_step_scale)
        displayed_step_scale = float(np.median(tension_step_scales))
        total_before, ratio_before = self._total_and_ratio(
            pretension_a, pretension_b
        )
        if baseline_evaluation is not None:
            baseline_design = baseline_evaluation.design
            if (
                baseline_evaluation.staged_result is None
                or not np.array_equal(baseline_design.strands, strands)
                or not np.allclose(baseline_design.pretension, total_before)
                or not np.allclose(
                    baseline_design.pretension_a_ratio, ratio_before
                )
            ):
                raise ValueError(
                    "cached engineering baseline does not match the current design"
                )
        # Steady-state cost is normally two full replays: one tension-only
        # candidate and one strand-only candidate.  A rejected tension vector
        # may add one verified partial-candidate replay.  A fresh process also
        # needs one baseline replay; an in-process cycle reuses the prior one.
        fem_cases_total = 3 if baseline_evaluation is None else 2

        self._emit(
            EngineeringProgress3D(
                cycle_index=cycle_index,
                phase=(
                    "读取A激活与最终状态"
                    if baseline_evaluation is None
                    else "复用上轮修正后回放"
                ),
                fem_cases_completed=0,
                fem_cases_total=fem_cases_total,
                elapsed_seconds=0.0,
                eta_seconds=0.0,
                stage_status=self._stage_status(strands, None, None),
                step_scale=displayed_step_scale,
                step_scale_min=float(np.min(tension_step_scales)),
                step_scale_max=float(np.max(tension_step_scales)),
            )
        )
        before_evaluation = (
            self._evaluate(strands, pretension_a, pretension_b)
            if baseline_evaluation is None
            else baseline_evaluation
        )
        cases_before_update = 1 if baseline_evaluation is None else 0
        before_responses, baseline_stress = self._local_stage_responses(
            before_evaluation
        )
        score_before = self._local_score(before_responses)
        first_elapsed = time.perf_counter() - started
        self._emit(
            EngineeringProgress3D(
                cycle_index=cycle_index,
                phase="第1轮：只调整A/B索力",
                fem_cases_completed=cases_before_update,
                fem_cases_total=fem_cases_total,
                elapsed_seconds=first_elapsed,
                eta_seconds=max(first_elapsed, self.total_fem_seconds / max(1, self.total_fem_cases)) * 2.0,
                stage_status=self._stage_status(
                    strands, before_responses, baseline_stress
                ),
                local_score=score_before,
                step_scale=displayed_step_scale,
                step_scale_min=float(np.min(tension_step_scales)),
                step_scale_max=float(np.max(tension_step_scales)),
            )
        )

        updated_a, updated_b = self._update_tensions(
            strands,
            pretension_a,
            pretension_b,
            before_responses,
            tension_step_scales,
        )
        tension_changed = not (
            np.array_equal(updated_a, pretension_a)
            and np.array_equal(updated_b, pretension_b)
        )
        self._emit(
            EngineeringProgress3D(
                cycle_index=cycle_index,
                phase="第1轮：回放索力候选（根数固定）",
                fem_cases_completed=cases_before_update,
                fem_cases_total=fem_cases_total,
                elapsed_seconds=time.perf_counter() - started,
                eta_seconds=max(first_elapsed, self.total_fem_seconds / max(1, self.total_fem_cases)) * 2.0,
                stage_status=self._stage_status(
                    strands, before_responses, baseline_stress
                ),
                local_score=score_before,
                step_scale=displayed_step_scale,
                step_scale_min=float(np.min(tension_step_scales)),
                step_scale_max=float(np.max(tension_step_scales)),
            )
        )
        residual_before_tension = self._tension_feedback_residuals(
            before_responses
        )
        tension_partial_retry_attempted = False
        tension_partial_retry_accepted = False
        if tension_changed:
            tension_proposal_evaluation = self._evaluate(
                strands, updated_a, updated_b
            )
            tension_proposal_responses, tension_proposal_stress = (
                self._local_stage_responses(tension_proposal_evaluation)
            )
            tension_proposal_score = self._local_score(
                tension_proposal_responses
            )
            tension_update_accepted = bool(
                tension_proposal_score <= score_before + 1.0e-12
            )
        else:
            tension_proposal_evaluation = before_evaluation
            tension_proposal_responses = before_responses
            tension_proposal_stress = baseline_stress
            tension_proposal_score = score_before
            tension_update_accepted = False

        first_tension_proposal_score = tension_proposal_score
        full_changed_components = np.vstack(
            (
                np.abs(updated_a - pretension_a) > 1.0e-9,
                np.abs(updated_b - pretension_b) > 1.0e-9,
            )
        )
        isolated_retry_components = np.zeros_like(full_changed_components)
        if tension_changed and not tension_update_accepted:
            partial_a, partial_b, partial_components = (
                self._isolated_tension_candidate(
                    pretension_a,
                    pretension_b,
                    updated_a,
                    updated_b,
                    residual_before_tension,
                )
            )
            partial_differs_from_full = not (
                np.array_equal(partial_a, updated_a)
                and np.array_equal(partial_b, updated_b)
            )
            if np.any(partial_components) and partial_differs_from_full:
                tension_partial_retry_attempted = True
                isolated_retry_components = partial_components
                fem_cases_total += 1
                self._emit(
                    EngineeringProgress3D(
                        cycle_index=cycle_index,
                        phase="第1轮：回放局部改善索组",
                        fem_cases_completed=self.total_fem_cases - start_cases,
                        fem_cases_total=fem_cases_total,
                        elapsed_seconds=time.perf_counter() - started,
                        eta_seconds=(
                            self.total_fem_seconds
                            / max(1, self.total_fem_cases)
                        ),
                        stage_status=self._stage_status(
                            strands,
                            tension_proposal_responses,
                            tension_proposal_stress,
                        ),
                        local_score=score_before,
                        proposal_score=first_tension_proposal_score,
                        update_accepted=False,
                        step_scale=displayed_step_scale,
                        step_scale_min=float(np.min(tension_step_scales)),
                        step_scale_max=float(np.max(tension_step_scales)),
                    )
                )
                tension_proposal_evaluation = self._evaluate(
                    strands, partial_a, partial_b
                )
                tension_proposal_responses, tension_proposal_stress = (
                    self._local_stage_responses(tension_proposal_evaluation)
                )
                tension_proposal_score = self._local_score(
                    tension_proposal_responses
                )
                tension_update_accepted = bool(
                    tension_proposal_score <= score_before + 1.0e-12
                )
                tension_partial_retry_accepted = tension_update_accepted
                updated_a = partial_a
                updated_b = partial_b

        changed_components = np.vstack(
            (
                np.abs(updated_a - pretension_a) > 1.0e-9,
                np.abs(updated_b - pretension_b) > 1.0e-9,
            )
        )
        if tension_update_accepted:
            next_tension_step_scales = self._adapt_tension_step_scales(
                tension_step_scales,
                residual_before_tension,
                self._tension_feedback_residuals(tension_proposal_responses),
                changed_components,
            )
        else:
            next_tension_step_scales = tension_step_scales.copy()
            # Only shrink the independently verified component.  The full
            # vector cannot identify which of its coupled changes caused the
            # rejection, so untouched groups retain their own memories.
            rejected_components = (
                isolated_retry_components
                if tension_partial_retry_attempted
                else full_changed_components
            )
            next_tension_step_scales[rejected_components] = np.maximum(
                1.0 / 32.0,
                0.5 * tension_step_scales[rejected_components],
            )
        next_step_scale = float(np.median(next_tension_step_scales))

        if tension_update_accepted:
            tension_evaluation = tension_proposal_evaluation
            tension_responses = tension_proposal_responses
            stress_before_sizing = tension_proposal_stress
            score_after_tension = tension_proposal_score
            accepted_a = updated_a
            accepted_b = updated_b
        else:
            tension_evaluation = before_evaluation
            tension_responses = before_responses
            stress_before_sizing = baseline_stress
            score_after_tension = score_before
            accepted_a = pretension_a
            accepted_b = pretension_b

        cases_after_tension = self.total_fem_cases - start_cases
        self._emit(
            EngineeringProgress3D(
                cycle_index=cycle_index,
                phase=(
                    "第1轮完成：接受局部改善索组"
                    if tension_partial_retry_accepted
                    else (
                        "第1轮完成：接受索力更新"
                        if tension_update_accepted
                        else "第1轮完成：保留原索力并缩小步长"
                    )
                ),
                fem_cases_completed=cases_after_tension,
                fem_cases_total=fem_cases_total,
                elapsed_seconds=time.perf_counter() - started,
                eta_seconds=self.total_fem_seconds / max(1, self.total_fem_cases),
                stage_status=self._stage_status(
                    strands, tension_responses, stress_before_sizing
                ),
                local_score=score_after_tension,
                proposal_score=tension_proposal_score,
                update_accepted=tension_update_accepted,
                step_scale=next_step_scale,
                step_scale_min=float(np.min(next_tension_step_scales)),
                step_scale_max=float(np.max(next_tension_step_scales)),
                tension_update_accepted=tension_update_accepted,
            )
        )

        accepted_total, _ = self._total_and_ratio(accepted_a, accepted_b)
        resized = self._resize_strands(
            strands, accepted_total, stress_before_sizing, strand_step_scales
        )
        strand_update_attempted = not np.array_equal(resized, strands)
        strand_score_before = self._strand_score(stress_before_sizing)
        self._emit(
            EngineeringProgress3D(
                cycle_index=cycle_index,
                phase="第2轮：只调整根数并回放候选",
                fem_cases_completed=cases_after_tension,
                fem_cases_total=fem_cases_total,
                elapsed_seconds=time.perf_counter() - started,
                eta_seconds=self.total_fem_seconds / max(1, self.total_fem_cases),
                stage_status=self._stage_status(
                    resized, tension_responses, stress_before_sizing
                ),
                local_score=score_after_tension,
                step_scale=next_step_scale,
                step_scale_min=float(np.min(next_tension_step_scales)),
                step_scale_max=float(np.max(next_tension_step_scales)),
                tension_update_accepted=tension_update_accepted,
            )
        )
        if strand_update_attempted:
            strand_proposal_evaluation = self._evaluate(
                resized, accepted_a, accepted_b
            )
            strand_proposal_responses, strand_proposal_stress = (
                self._local_stage_responses(strand_proposal_evaluation)
            )
            strand_proposal_score = self._strand_score(strand_proposal_stress)
            # A strand-count change alters cable EA and may temporarily move
            # the bridge away from its displacement target.  Material
            # utilization has priority in this round; the next cycle starts
            # from this verified state and retunes A/B forces to find the new
            # displacement balance position.
            strand_proposal_balance = self._local_score(
                strand_proposal_responses
            )
            strand_update_accepted = bool(
                strand_proposal_score < strand_score_before - 1.0e-12
            )
        else:
            strand_proposal_evaluation = tension_evaluation
            strand_proposal_responses = tension_responses
            strand_proposal_stress = stress_before_sizing
            strand_proposal_score = None
            strand_proposal_balance = score_after_tension
            strand_update_accepted = False

        next_strand_step_scales = self._adapt_strand_step_scales(
            strand_step_scales,
            stress_before_sizing,
            strand_proposal_stress,
            resized != strands,
            self.options.target_stress_mpa,
        )
        if strand_update_accepted:
            after_evaluation = strand_proposal_evaluation
            after_responses = strand_proposal_responses
            stress_after = strand_proposal_stress
            score_after = strand_proposal_balance
            strand_score_after = float(strand_proposal_score)
            accepted_strands = resized
        else:
            after_evaluation = tension_evaluation
            after_responses = tension_responses
            stress_after = stress_before_sizing
            score_after = score_after_tension
            strand_score_after = strand_score_before
            accepted_strands = strands

        update_accepted = tension_update_accepted or strand_update_accepted
        controls = self._controls(
            before_responses,
            tension_responses,
            after_responses,
            pretension_a,
            pretension_b,
            accepted_a,
            accepted_b,
            self.tension_upper_bounds(accepted_strands),
        )
        fem_cases = self.total_fem_cases - start_cases
        fem_seconds = self.total_fem_seconds - start_seconds
        score_change = (
            0.0
            if score_before == 0.0
            else 100.0 * (score_before - score_after) / score_before
        )
        self._emit(
            EngineeringProgress3D(
                cycle_index=cycle_index,
                phase=(
                    "本循环完成：索力与根数独立验收"
                ),
                fem_cases_completed=fem_cases,
                fem_cases_total=fem_cases_total,
                elapsed_seconds=time.perf_counter() - started,
                eta_seconds=0.0,
                stage_status=self._stage_status(
                    accepted_strands, after_responses, stress_after
                ),
                local_score=score_after,
                score_change_percent=score_change,
                proposal_score=tension_proposal_score,
                update_accepted=update_accepted,
                step_scale=next_step_scale,
                step_scale_min=float(np.min(next_tension_step_scales)),
                step_scale_max=float(np.max(next_tension_step_scales)),
                tension_update_accepted=tension_update_accepted,
                strand_update_accepted=strand_update_accepted,
            )
        )
        return EngineeringCycleResult3D(
            cycle_index=cycle_index,
            evaluation_before_sizing=tension_evaluation,
            evaluation_after_sizing=after_evaluation,
            controls=controls,
            pretension_a_before=pretension_a,
            pretension_b_before=pretension_b,
            pretension_a=accepted_a,
            pretension_b=accepted_b,
            strands_before_sizing=strands,
            strands_after_sizing=accepted_strands,
            stress_before_sizing_mpa=stress_before_sizing,
            stress_after_sizing_mpa=stress_after,
            local_score_before=score_before,
            local_score_after_tension=score_after_tension,
            local_score_after=score_after,
            proposal_local_score=tension_proposal_score,
            first_tension_proposal_local_score=(
                first_tension_proposal_score
            ),
            update_accepted=update_accepted,
            tension_update_accepted=tension_update_accepted,
            tension_partial_retry_attempted=(
                tension_partial_retry_attempted
            ),
            tension_partial_retry_accepted=tension_partial_retry_accepted,
            strand_update_attempted=strand_update_attempted,
            strand_update_accepted=strand_update_accepted,
            strand_score_before=strand_score_before,
            strand_proposal_score=strand_proposal_score,
            strand_score_after=strand_score_after,
            step_scale_used=displayed_step_scale,
            next_step_scale=next_step_scale,
            tension_step_scales_used=tension_step_scales,
            next_tension_step_scales=next_tension_step_scales,
            strand_step_scales_used=strand_step_scales,
            next_strand_step_scales=next_strand_step_scales,
            fem_cases=fem_cases,
            fem_seconds=fem_seconds,
        )


__all__ = [
    "EngineeringCableCycleOptimizer3D",
    "EngineeringCycleOptions3D",
    "EngineeringCycleResult3D",
    "EngineeringProgress3D",
    "EngineeringStageStatus3D",
    "EngineeringSubstageControl3D",
]
