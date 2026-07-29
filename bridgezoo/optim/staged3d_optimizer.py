"""Efficient OpenSees-only optimization for staged 3D cable tensioning.

The implementation separates the two physical tensioning operations:

* first tension ``T_A`` is calculated once, stage by stage, from a 2x2 local
  displacement influence matrix (tower ux and girder-end uz);
* second tension ``T_B`` is optimized through a low-dimensional smooth curve
  on an OpenSees-built affine response surface, then interpolated to all groups;
* integer strand candidates are projected onto outward non-decreasing smooth
  curves before rebuilding responses because cable area changes stiffness.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from bridgezoo.optim.objectives import stress_violation_mpa
from bridgezoo.optim.single_staged3d import (
    CableDesignEvaluator3D,
    EvaluationResult3D,
    StageAControlResponse3D,
)
from bridgezoo.optim.smooth_curves import (
    CURVE_FAMILIES,
    SmoothStrandCurve3D,
    build_smooth_curve_basis,
    build_stage_major_curve_basis,
    project_strands_to_smooth_curve,
)
from bridgezoo.optim.variables import (
    validate_ratio_vector,
    validate_strand_vector,
    validate_tension_vector,
)


@dataclass(frozen=True)
class StageAControlOptions:
    probe_fraction: float = 1.0
    feasibility_tolerance_m: float = 1.0e-4
    condition_limit: float = 1.0e10


@dataclass(frozen=True)
class SecondaryTensionOptions3D:
    curve_family: str = "bernstein"
    control_points_per_group: int = 4
    max_nfev: int = 80
    ftol: float = 1.0e-9
    xtol: float = 1.0e-9
    gtol: float = 1.0e-9
    displacement_validation_tolerance_m: float = 1.0e-7
    stress_validation_tolerance_mpa: float = 1.0e-3


@dataclass(frozen=True)
class StrandSearchOptions3D:
    iterations: int = 0
    step: int = 1
    improvement_tol: float = 1.0e-8


@dataclass(frozen=True)
class Staged3DOptimizationOptions:
    ab_correction_passes: int = 0
    stage_a: StageAControlOptions = field(default_factory=StageAControlOptions)
    secondary: SecondaryTensionOptions3D = field(
        default_factory=SecondaryTensionOptions3D
    )
    strands: StrandSearchOptions3D = field(default_factory=StrandSearchOptions3D)


@dataclass(frozen=True)
class StageAControlResult3D:
    response: StageAControlResponse3D
    backstay_a_N: float
    main_stay_a_N: float
    backstay_ratio: float
    main_stay_ratio: float
    feasible: bool
    stable: bool
    condition_number: float
    matrix_rank: int
    active_bounds: tuple[str, ...]
    success: bool
    message: str
    nfev: int
    final_schedule_response: StageAControlResponse3D | None = None
    final_schedule_feasible: bool | None = None
    balance_basis: str = "all second-tension forces B=0"


@dataclass(frozen=True)
class SecondaryTensionResult3D:
    evaluation: EvaluationResult3D
    pretension_b: np.ndarray
    success: bool
    validated: bool
    message: str
    nfev: int
    fem_cases: int
    max_displacement_prediction_error_m: float
    max_stress_prediction_error_mpa: float
    curve_family: str
    curve_control_coordinates: np.ndarray
    curve_control_values: np.ndarray
    curve_trials: tuple["SmoothCurveTrial3D", ...]


@dataclass(frozen=True)
class SmoothCurveTrial3D:
    family: str
    objective: float
    success: bool
    validated: bool
    nfev: int
    fem_cases: int


@dataclass(frozen=True)
class _SecondaryCurveCandidate3D:
    model: "SecondaryAffineModel3D"
    evaluation: EvaluationResult3D
    pretension_b: np.ndarray
    controls: np.ndarray
    optimizer_success: bool
    validated: bool
    message: str
    nfev: int
    max_displacement_error_m: float
    max_stress_error_mpa: float


@dataclass(frozen=True)
class ContinuousDesignResult3D:
    evaluation: EvaluationResult3D
    controls: tuple[StageAControlResult3D, ...]
    secondary: SecondaryTensionResult3D
    success: bool
    message: str
    stage_a_fem_cases: int
    secondary_matrix_nfev: int
    fem_cases: int
    fem_seconds: float


@dataclass(frozen=True)
class Staged3DOptimizationResult:
    best: EvaluationResult3D
    controls: tuple[StageAControlResult3D, ...]
    secondary: SecondaryTensionResult3D
    history: list[EvaluationResult3D]
    strand_iterations_completed: int
    continuous_solves: int
    fem_cases: int
    fem_seconds: float
    strand_curve: SmoothStrandCurve3D


@dataclass(frozen=True)
class SecondaryAffineModel3D:
    """Completed response driven by low-dimensional normalized-B controls."""

    upper_b_N: np.ndarray
    curve_family: str
    curve_basis: np.ndarray
    curve_control_coordinates: np.ndarray
    deck_node_ids: tuple[int, ...]
    tower_anchor_ids: tuple[int, ...]
    cable_ids: tuple[int, ...]
    baseline_deck_m: np.ndarray
    deck_delta_m: np.ndarray
    baseline_tower_dx_m: float
    tower_delta_m: np.ndarray
    baseline_anchor_dx_m: np.ndarray
    anchor_delta_m: np.ndarray
    baseline_stress_mpa: np.ndarray
    stress_delta_mpa: np.ndarray

    def expand_controls(self, controls) -> np.ndarray:
        """Interpolate curve controls to all stage-major cable groups."""

        return self.curve_basis @ np.asarray(controls, dtype=float)

    def predict(self, controls) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        value = np.asarray(controls, dtype=float)
        return (
            self.baseline_deck_m + self.deck_delta_m @ value,
            float(self.baseline_tower_dx_m + self.tower_delta_m @ value),
            self.baseline_anchor_dx_m + self.anchor_delta_m @ value,
            self.baseline_stress_mpa + self.stress_delta_mpa @ value,
        )


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m{remainder:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class _FEMProgress:
    """Weighted progress estimate based on solved construction increments."""

    def __init__(
        self,
        optimizer: "StagedCableOptimizer3D",
        *,
        label: str,
        total_cases: int,
        total_work: float,
    ) -> None:
        self.optimizer = optimizer
        self.label = label
        self.total_cases = total_cases
        self.total_work = max(float(total_work), 1.0)
        self.completed_cases = 0
        self.completed_work = 0.0
        self.started = time.perf_counter()

    def run(self, description: str, work: float, callback):
        started = time.perf_counter()
        result = callback()
        duration = time.perf_counter() - started
        self.optimizer._total_fem_cases += 1
        self.optimizer._total_fem_seconds += duration
        self.completed_cases += 1
        self.completed_work += float(work)
        elapsed = time.perf_counter() - self.started
        rate = elapsed / max(self.completed_work, 1.0e-12)
        eta = rate * max(0.0, self.total_work - self.completed_work)
        percent = min(100.0, 100.0 * self.completed_work / self.total_work)
        self.optimizer._emit(
            f"[{self.label} FEM {self.completed_cases}/{self.total_cases} | "
            f"{percent:5.1f}%] {description} | solve={_format_duration(duration)} | "
            f"elapsed={_format_duration(elapsed)} | ETA≈{_format_duration(eta)}"
        )
        return result


class StagedCableOptimizer3D:
    """Calculate A directly and optimize B on an OpenSees response surface."""

    def __init__(
        self,
        evaluator: CableDesignEvaluator3D,
        options: Staged3DOptimizationOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if evaluator.problem.backend != "opensees":
            raise ValueError(
                "the efficient 3D optimizer currently supports only the OpenSees backend"
            )
        self.evaluator = evaluator
        self.problem = evaluator.problem
        self.layout = evaluator.layout
        self.options = options or Staged3DOptimizationOptions()
        self.progress = progress
        self._total_fem_cases = 0
        self._total_fem_seconds = 0.0
        self._validate_options()

    def _validate_options(self) -> None:
        stage_a = self.options.stage_a
        secondary = self.options.secondary
        if not 0.0 < stage_a.probe_fraction <= 1.0:
            raise ValueError("stage-A probe_fraction must be in (0, 1]")
        if stage_a.feasibility_tolerance_m <= 0.0:
            raise ValueError("stage-A feasibility tolerance must be positive")
        if stage_a.condition_limit <= 1.0:
            raise ValueError("stage-A condition limit must exceed one")
        if secondary.curve_family not in {*CURVE_FAMILIES, "auto"}:
            raise ValueError(
                "secondary curve_family must be bernstein, piecewise-linear, or auto"
            )
        if secondary.control_points_per_group < 1:
            raise ValueError("secondary curve control points must be positive")
        if secondary.max_nfev < 1:
            raise ValueError("secondary max_nfev must be positive")
        if (
            secondary.displacement_validation_tolerance_m <= 0.0
            or secondary.stress_validation_tolerance_mpa <= 0.0
        ):
            raise ValueError("secondary response validation tolerances must be positive")
        if self.options.strands.iterations < 0:
            raise ValueError("strand search iterations must be nonnegative")
        if self.options.ab_correction_passes < 0:
            raise ValueError("A/B correction passes must be nonnegative")
        if self.options.strands.step < 1:
            raise ValueError("strand search step must be positive")
        weights = self.problem.weights
        weighted_terms = (
            weights.shape,
            weights.total_strands,
            weights.stress_uniform,
            weights.stress_violation,
            weights.tower_displacement,
            weights.tower_anchor_displacement,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in weighted_terms):
            raise ValueError("3D objective weights must be finite and nonnegative")
        scales = (weights.shape_scale_m, weights.stress_scale_mpa, weights.strand_scale)
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise ValueError("3D objective scales must be finite and positive")

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def tension_upper_bounds(self, strands: np.ndarray) -> np.ndarray:
        return (
            self.problem.bounds.tension_bound_stress_mpa
            * 1.0e6
            * self.problem.strand_area
            * strands.astype(float)
        )

    def default_tension(self, strands: np.ndarray) -> np.ndarray:
        target_stress = 0.5 * (
            self.problem.bounds.stress_lower_mpa
            + self.problem.bounds.stress_upper_mpa
        )
        return target_stress * 1.0e6 * self.problem.strand_area * strands.astype(float)

    def _curve_families_and_bases(self) -> list[tuple[str, np.ndarray]]:
        requested = self.options.secondary.curve_family
        families = CURVE_FAMILIES if requested == "auto" else (requested,)
        unique: list[tuple[str, np.ndarray]] = []
        for family in families:
            basis = build_stage_major_curve_basis(
                self.problem.n_seg,
                self.options.secondary.control_points_per_group,
                family,
            )
            if not any(np.allclose(basis, existing) for _, existing in unique):
                unique.append((family, basis))
        return unique

    def secondary_fem_cases_per_cycle(self) -> int:
        """Return baseline + curve probes + one validation per unique family."""

        return 1 + sum(
            basis.shape[1] + 1 for _, basis in self._curve_families_and_bases()
        )

    def project_strands(self, strands) -> SmoothStrandCurve3D:
        family = self.options.secondary.curve_family
        if family == "auto":
            family = "bernstein"
        return project_strands_to_smooth_curve(
            strands,
            n_seg=self.problem.n_seg,
            control_points=self.options.secondary.control_points_per_group,
            family=family,
            lower=self.problem.bounds.strand_min,
            upper=self.problem.bounds.strand_max,
        )

    @staticmethod
    def _total_and_ratio(
        pretension_a: np.ndarray, pretension_b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        total = np.asarray(pretension_a, dtype=float) + np.asarray(
            pretension_b, dtype=float
        )
        ratio = np.divide(
            pretension_a,
            total,
            out=np.zeros_like(total),
            where=total > 0.0,
        )
        return total, ratio

    def _evaluate_ab(
        self,
        strands,
        pretension_a,
        pretension_b,
        *,
        keep_result: bool = False,
    ) -> EvaluationResult3D:
        total, ratio = self._total_and_ratio(pretension_a, pretension_b)
        return self.evaluator.evaluate(
            strands,
            total,
            ratio,
            keep_result=keep_result,
        )

    def _evaluate_stage_a_ab(
        self,
        strands,
        pretension_a,
        pretension_b,
        construction_stage: int,
    ) -> StageAControlResponse3D:
        total, ratio = self._total_and_ratio(pretension_a, pretension_b)
        return self.evaluator.evaluate_stage_a(
            strands,
            total,
            ratio,
            construction_stage,
        )

    def calculate_initial_tension(
        self,
        strands,
        *,
        fixed_b=None,
        tracker: _FEMProgress | None = None,
    ) -> tuple[np.ndarray, tuple[StageAControlResult3D, ...]]:
        """Calculate all ``T_A`` values directly from local influence matrices.

        In the preliminary pass all second-tension values are zero.  Optional
        correction passes supply a fixed B schedule, so preceding B increments
        participate in later A-stage balance.  Earlier stages retain their
        solved A force, while the current pair is probed at
        zero/backstay/main-stay values.  A fourth solve validates the bounded
        linear solution on the real staged model.
        """

        try:
            from scipy.optimize import lsq_linear
        except ImportError as exc:
            raise RuntimeError("scipy is required for 3D cable optimization") from exc

        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        upper = self.tension_upper_bounds(strands)
        pretension_a = np.zeros(self.layout.size, dtype=float)
        pretension_b = (
            np.zeros(self.layout.size, dtype=float)
            if fixed_b is None
            else validate_tension_vector(fixed_b, self.layout)
        )
        balance_basis = (
            "all second-tension forces B=0"
            if fixed_b is None or not np.any(pretension_b)
            else "fixed previous optimized B schedule"
        )
        controls: list[StageAControlResult3D] = []
        standalone = tracker is None
        if tracker is None:
            total_work = 4.0 * sum(
                3 * stage - 2
                for stage in range(1, self.problem.n_seg + 1)
            )
            tracker = _FEMProgress(
                self,
                label="A direct",
                total_cases=4 * self.problem.n_seg,
                total_work=total_work,
            )
            self._emit(
                f"A initial balance: {self.problem.n_seg} stages, "
                f"{4 * self.problem.n_seg} partial OpenSees cases"
            )

        for construction_stage in range(1, self.problem.n_seg + 1):
            offset = 2 * (construction_stage - 1)
            stage_index = 3 * construction_stage - 2
            current = pretension_a.copy()
            current[offset : offset + 2] = 0.0
            baseline = tracker.run(
                f"A stage {construction_stage:02d} baseline",
                stage_index,
                lambda current=current: self._evaluate_stage_a_ab(
                    strands, current, pretension_b, construction_stage
                ),
            )
            y0 = np.asarray(
                [baseline.backstay_tower_dx_m, baseline.main_stay_deck_uz_m],
                dtype=float,
            )
            full_range_columns = []
            for local_index, name in enumerate(("backstay", "main-stay")):
                group_index = offset + local_index
                probe_force = upper[group_index] * self.options.stage_a.probe_fraction
                probe = current.copy()
                probe[group_index] = probe_force
                response = tracker.run(
                    f"A stage {construction_stage:02d} {name} influence",
                    stage_index,
                    lambda probe=probe: self._evaluate_stage_a_ab(
                        strands, probe, pretension_b, construction_stage
                    ),
                )
                y_probe = np.asarray(
                    [
                        response.backstay_tower_dx_m,
                        response.main_stay_deck_uz_m,
                    ]
                )
                full_range_columns.append(
                    (y_probe - y0) / self.options.stage_a.probe_fraction
                )
            influence = np.column_stack(full_range_columns)
            solved = lsq_linear(
                influence,
                -y0,
                bounds=(np.zeros(2), np.ones(2)),
                lsmr_tol="auto",
            )
            normalized = np.clip(np.asarray(solved.x, dtype=float), 0.0, 1.0)
            pretension_a[offset : offset + 2] = normalized * upper[offset : offset + 2]
            actual = tracker.run(
                f"A stage {construction_stage:02d} validation",
                stage_index,
                lambda: self._evaluate_stage_a_ab(
                    strands, pretension_a, pretension_b, construction_stage
                ),
            )
            residual = np.asarray(
                [actual.backstay_tower_dx_m, actual.main_stay_deck_uz_m]
            )
            rank = int(np.linalg.matrix_rank(influence))
            condition = float(np.linalg.cond(influence))
            feasible = bool(
                np.max(np.abs(residual))
                <= self.options.stage_a.feasibility_tolerance_m
            )
            stable = bool(
                rank == 2
                and math.isfinite(condition)
                and condition <= self.options.stage_a.condition_limit
            )
            active = []
            names = ("backstay", "main_stay")
            for index, value in enumerate(normalized):
                if value <= 1.0e-8:
                    active.append(f"{names[index]}:lower")
                elif value >= 1.0 - 1.0e-8:
                    active.append(f"{names[index]}:upper")
            success = bool(solved.success and feasible and stable)
            message = (
                f"direct bounded influence solve; feasible={feasible}; stable={stable}; "
                f"condition={condition:.3e}; residual_max={np.max(np.abs(residual)) * 1000.0:.6f} mm"
            )
            controls.append(
                StageAControlResult3D(
                    response=actual,
                    backstay_a_N=float(pretension_a[offset]),
                    main_stay_a_N=float(pretension_a[offset + 1]),
                    backstay_ratio=1.0,
                    main_stay_ratio=1.0,
                    feasible=feasible,
                    stable=stable,
                    condition_number=condition,
                    matrix_rank=rank,
                    active_bounds=tuple(active),
                    success=success,
                    message=message,
                    nfev=4,
                    balance_basis=balance_basis,
                )
            )
            verdict = "FEASIBLE" if success else "CHECK"
            self._emit(
                f"A stage {construction_stage:02d} {verdict}: "
                f"TA(back/main)=({pretension_a[offset]:.2f}, "
                f"{pretension_a[offset + 1]:.2f}) N | "
                f"tower_dx={actual.backstay_tower_dx_m * 1000.0:.6f} mm | "
                f"deck_uz={actual.main_stay_deck_uz_m * 1000.0:.6f} mm | "
                f"rank={rank} cond={condition:.3e} bounds={tuple(active) or 'none'}"
            )
        if standalone:
            self._emit(
                f"A initial balance complete: feasible stages="
                f"{sum(item.success for item in controls)}/{len(controls)}"
            )
        return pretension_a, tuple(controls)

    @staticmethod
    def _response_arrays(
        evaluation: EvaluationResult3D,
        deck_ids: tuple[int, ...],
        anchor_ids: tuple[int, ...],
        cable_ids: tuple[int, ...],
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
        return (
            np.asarray([evaluation.deck_errors_m[node] for node in deck_ids]),
            float(evaluation.metrics.tower_top_dx_m),
            np.asarray([evaluation.tower_anchor_dx_m[node] for node in anchor_ids]),
            np.asarray([evaluation.cable_stress_mpa[cable] for cable in cable_ids]),
        )

    def build_secondary_affine_model(
        self,
        strands,
        pretension_a,
        *,
        tracker: _FEMProgress,
        curve_family: str | None = None,
        curve_basis: np.ndarray | None = None,
        baseline: EvaluationResult3D | None = None,
    ) -> tuple[SecondaryAffineModel3D, EvaluationResult3D]:
        """Build a low-dimensional completed-bridge B response surface."""

        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension_a = validate_tension_vector(pretension_a, self.layout)
        upper_b = np.maximum(0.0, self.tension_upper_bounds(strands) - pretension_a)
        zeros = np.zeros(self.layout.size, dtype=float)
        if curve_family is None:
            curve_family = (
                "bernstein"
                if self.options.secondary.curve_family == "auto"
                else self.options.secondary.curve_family
            )
        if curve_basis is None:
            curve_basis = build_stage_major_curve_basis(
                self.problem.n_seg,
                self.options.secondary.control_points_per_group,
                curve_family,
            )
        final_work = self.evaluator.build_plan(
            strands,
            *self._total_and_ratio(pretension_a, zeros),
        ).final_stage.index
        if baseline is None:
            baseline = tracker.run(
                "B response baseline",
                final_work,
                lambda: self._evaluate_ab(strands, pretension_a, zeros),
            )
        deck_ids = tuple(baseline.deck_errors_m)
        anchor_ids = tuple(baseline.tower_anchor_dx_m)
        cable_ids = tuple(self.layout.cable_ids)
        base_deck, base_tower, base_anchor, base_stress = self._response_arrays(
            baseline, deck_ids, anchor_ids, cable_ids
        )
        deck_columns = []
        tower_columns = []
        anchor_columns = []
        stress_columns = []
        for index in range(curve_basis.shape[1]):
            probe_b = upper_b * curve_basis[:, index]
            probe = tracker.run(
                f"B {curve_family} curve influence "
                f"{index + 1:02d}/{curve_basis.shape[1]}",
                final_work,
                lambda probe_b=probe_b: self._evaluate_ab(
                    strands, pretension_a, probe_b
                ),
            )
            deck, tower, anchors, stress = self._response_arrays(
                probe, deck_ids, anchor_ids, cable_ids
            )
            deck_columns.append(deck - base_deck)
            tower_columns.append(tower - base_tower)
            anchor_columns.append(anchors - base_anchor)
            stress_columns.append(stress - base_stress)
        return (
            SecondaryAffineModel3D(
                upper_b_N=upper_b,
                curve_family=curve_family,
                curve_basis=curve_basis,
                curve_control_coordinates=np.linspace(
                    0.0, 1.0, curve_basis.shape[1] // 2
                ),
                deck_node_ids=deck_ids,
                tower_anchor_ids=anchor_ids,
                cable_ids=cable_ids,
                baseline_deck_m=base_deck,
                deck_delta_m=np.column_stack(deck_columns),
                baseline_tower_dx_m=base_tower,
                tower_delta_m=np.asarray(tower_columns),
                baseline_anchor_dx_m=base_anchor,
                anchor_delta_m=np.column_stack(anchor_columns),
                baseline_stress_mpa=base_stress,
                stress_delta_mpa=np.column_stack(stress_columns),
            ),
            baseline,
        )

    def _final_residual_arrays(
        self,
        deck: np.ndarray,
        tower_dx: float,
        anchors: np.ndarray,
        stress: np.ndarray,
    ) -> np.ndarray:
        weights = self.problem.weights
        scale_m = weights.shape_scale_m
        stress_scale = weights.stress_scale_mpa
        violations = stress_violation_mpa(stress, self.problem.bounds)
        centered_stress = stress - float(np.mean(stress))
        pieces = []
        if deck.size:
            pieces.append(math.sqrt(weights.shape / deck.size) * deck / scale_m)
        pieces.append(
            np.asarray(
                [math.sqrt(weights.tower_displacement) * tower_dx / scale_m]
            )
        )
        if anchors.size:
            pieces.append(
                math.sqrt(weights.tower_anchor_displacement / anchors.size)
                * anchors
                / scale_m
            )
        if stress.size:
            pieces.append(
                math.sqrt(weights.stress_uniform / stress.size)
                * centered_stress
                / stress_scale
            )
            pieces.append(
                math.sqrt(weights.stress_violation / stress.size)
                * violations
                / stress_scale
            )
        return np.concatenate(pieces)

    def optimize_secondary_tension(
        self,
        strands,
        pretension_a,
        initial_b=None,
        *,
        tracker: _FEMProgress,
    ) -> SecondaryTensionResult3D:
        """Optimize one or more smooth B curves and retain the best validation."""

        try:
            from scipy.optimize import least_squares
        except ImportError as exc:
            raise RuntimeError("scipy is required for 3D cable optimization") from exc

        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension_a = validate_tension_vector(pretension_a, self.layout)
        initial = None
        if initial_b is not None:
            initial = validate_tension_vector(initial_b, self.layout)
        zeros = np.zeros(self.layout.size, dtype=float)
        final_work = self.evaluator.build_plan(
            strands,
            *self._total_and_ratio(pretension_a, zeros),
        ).final_stage.index
        baseline = tracker.run(
            "B response baseline shared by smooth curve candidates",
            final_work,
            lambda: self._evaluate_ab(strands, pretension_a, zeros),
        )
        candidates: list[_SecondaryCurveCandidate3D] = []
        trials: list[SmoothCurveTrial3D] = []
        for family, basis in self._curve_families_and_bases():
            model, _ = self.build_secondary_affine_model(
                strands,
                pretension_a,
                tracker=tracker,
                curve_family=family,
                curve_basis=basis,
                baseline=baseline,
            )
            initial_full = (
                0.5 * model.upper_b_N if initial is None else initial
            )
            normalized_full = np.clip(
                np.divide(
                    initial_full,
                    model.upper_b_N,
                    out=np.zeros_like(initial_full),
                    where=model.upper_b_N > 0.0,
                ),
                0.0,
                1.0,
            )
            control0, *_ = np.linalg.lstsq(
                model.curve_basis, normalized_full, rcond=None
            )
            control0 = np.clip(control0, 0.0, 1.0)
            solved = least_squares(
                lambda controls: self._final_residual_arrays(
                    *model.predict(controls)
                ),
                control0,
                bounds=(np.zeros(control0.size), np.ones(control0.size)),
                max_nfev=self.options.secondary.max_nfev,
                ftol=self.options.secondary.ftol,
                xtol=self.options.secondary.xtol,
                gtol=self.options.secondary.gtol,
            )
            controls = np.clip(np.asarray(solved.x, dtype=float), 0.0, 1.0)
            pretension_b = model.upper_b_N * model.expand_controls(controls)
            actual = tracker.run(
                f"B {family} interpolated full-design validation",
                final_work,
                lambda pretension_b=pretension_b: self._evaluate_ab(
                    strands,
                    pretension_a,
                    pretension_b,
                    keep_result=True,
                ),
            )
            predicted = model.predict(controls)
            actual_arrays = self._response_arrays(
                actual,
                model.deck_node_ids,
                model.tower_anchor_ids,
                model.cable_ids,
            )
            displacement_errors = [
                np.max(np.abs(predicted[0] - actual_arrays[0]))
                if predicted[0].size
                else 0.0,
                abs(predicted[1] - actual_arrays[1]),
                np.max(np.abs(predicted[2] - actual_arrays[2]))
                if predicted[2].size
                else 0.0,
            ]
            max_displacement_error = float(max(displacement_errors))
            max_stress_error = float(
                np.max(np.abs(predicted[3] - actual_arrays[3]))
            )
            validated = bool(
                max_displacement_error
                <= self.options.secondary.displacement_validation_tolerance_m
                and max_stress_error
                <= self.options.secondary.stress_validation_tolerance_mpa
            )
            message = (
                f"{family} smooth affine B response; "
                f"optimizer_success={bool(solved.success)}; validated={validated}; "
                f"displacement_error={max_displacement_error:.3e} m; "
                f"stress_error={max_stress_error:.3e} MPa"
            )
            candidate = _SecondaryCurveCandidate3D(
                model=model,
                evaluation=actual,
                pretension_b=pretension_b,
                controls=controls,
                optimizer_success=bool(solved.success),
                validated=validated,
                message=message,
                nfev=int(solved.nfev),
                max_displacement_error_m=max_displacement_error,
                max_stress_error_mpa=max_stress_error,
            )
            candidates.append(candidate)
            trials.append(
                SmoothCurveTrial3D(
                    family=family,
                    objective=float(actual.objective),
                    success=bool(solved.success and validated),
                    validated=validated,
                    nfev=int(solved.nfev),
                    fem_cases=basis.shape[1] + 1,
                )
            )
            self._emit(
                f"B curve trial {family}: controls={basis.shape[1]} | "
                f"objective={actual.objective:.6g} | "
                f"shape_rmse={actual.metrics.shape_rmse_m * 1000.0:.6f} mm | "
                f"surrogate={'PASS' if validated else 'FAIL'}"
            )

        selected = min(
            candidates,
            key=lambda item: (
                not (item.optimizer_success and item.validated),
                item.evaluation.objective,
            ),
        )
        success = bool(selected.optimizer_success and selected.validated)
        total_nfev = sum(item.nfev for item in candidates)
        total_cases = self.secondary_fem_cases_per_cycle()
        self._emit(
            "B smooth response solve complete: "
            f"selected={selected.model.curve_family}, "
            f"controls={selected.controls.size}, "
            f"matrix nfev={total_nfev} (no FEM in iterations) | "
            f"shape_rmse={selected.evaluation.metrics.shape_rmse_m * 1000.0:.6f} mm | "
            f"tower_dx={selected.evaluation.metrics.tower_top_dx_m * 1000.0:.6f} mm | "
            f"stress=[{selected.evaluation.metrics.stress_min_mpa:.3f}, "
            f"{selected.evaluation.metrics.stress_max_mpa:.3f}] MPa"
        )
        return SecondaryTensionResult3D(
            evaluation=selected.evaluation,
            pretension_b=selected.pretension_b,
            success=success,
            validated=selected.validated,
            message=selected.message,
            nfev=total_nfev,
            fem_cases=total_cases,
            max_displacement_prediction_error_m=(
                selected.max_displacement_error_m
            ),
            max_stress_prediction_error_mpa=selected.max_stress_error_mpa,
            curve_family=selected.model.curve_family,
            curve_control_coordinates=selected.model.curve_control_coordinates,
            curve_control_values=selected.controls,
            curve_trials=tuple(trials),
        )

    def optimize_continuous(
        self,
        strands,
        initial_pretension=None,
        initial_ratio=None,
        *,
        label: str = "continuous design",
    ) -> ContinuousDesignResult3D:
        requested_strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        strand_curve = self.project_strands(requested_strands)
        strands = strand_curve.interpolated_strands
        if not np.array_equal(requested_strands, strands):
            self._emit(
                f"{label}: projected strand counts to a monotone "
                f"{strand_curve.family} curve; changed groups="
                f"{int(np.count_nonzero(requested_strands != strands))}"
            )
        initial_total = (
            self.default_tension(strands)
            if initial_pretension is None
            else validate_tension_vector(initial_pretension, self.layout)
        )
        initial_ratios = (
            self.evaluator.default_pretension_a_ratio()
            if initial_ratio is None
            else validate_ratio_vector(initial_ratio, self.layout)
        )
        initial_b = initial_total * (1.0 - initial_ratios)
        zero = np.zeros(self.layout.size, dtype=float)
        final_index = self.evaluator.build_plan(
            strands, *self._total_and_ratio(zero, zero)
        ).final_stage.index
        a_work = 4.0 * sum(
            3 * stage - 2 for stage in range(1, self.problem.n_seg + 1)
        )
        b_cases = self.secondary_fem_cases_per_cycle()
        cycles = 1 + self.options.ab_correction_passes
        cases_per_cycle = 4 * self.problem.n_seg + b_cases
        total_cases = cycles * cases_per_cycle
        tracker = _FEMProgress(
            self,
            label=label,
            total_cases=total_cases,
            total_work=cycles * (a_work + b_cases * final_index),
        )
        start_cases = self._total_fem_cases
        start_seconds = self._total_fem_seconds
        self._emit(
            f"{label}: planned OpenSees cases={total_cases} "
            f"({cycles} A/B pass(es), each A={4 * self.problem.n_seg} partial + "
            f"B={b_cases} smooth-curve cases); "
            "ETA will calibrate after the first case"
        )
        pretension_b = initial_b
        controls = ()
        secondary = None
        for cycle in range(cycles):
            fixed_b = (
                np.zeros(self.layout.size, dtype=float)
                if cycle == 0
                else pretension_b
            )
            self._emit(
                f"{label}: A/B pass {cycle + 1}/{cycles} "
                f"({'preliminary B=0' if cycle == 0 else 'correction with previous B'})"
            )
            pretension_a, controls = self.calculate_initial_tension(
                strands,
                fixed_b=fixed_b,
                tracker=tracker,
            )
            secondary = self.optimize_secondary_tension(
                strands,
                pretension_a,
                pretension_b,
                tracker=tracker,
            )
            pretension_b = secondary.pretension_b
        evaluation = secondary.evaluation
        final_ratios = evaluation.design.pretension_a_ratio
        plan = self.evaluator.build_plan(
            evaluation.design.strands,
            evaluation.design.pretension,
            evaluation.design.pretension_a_ratio,
        )
        records_by_index = {
            record.stage_index: record
            for record in evaluation.staged_result.records
        }
        updated_controls = []
        for stage, control in enumerate(controls, start=1):
            final_response = self.evaluator.extract_stage_a_response(
                plan,
                records_by_index[3 * stage - 2],
                stage,
            )
            final_residual = max(
                abs(final_response.backstay_tower_dx_m),
                abs(final_response.main_stay_deck_uz_m),
            )
            updated_controls.append(
                replace(
                    control,
                    backstay_ratio=float(final_ratios[2 * (stage - 1)]),
                    main_stay_ratio=float(final_ratios[2 * (stage - 1) + 1]),
                    final_schedule_response=final_response,
                    final_schedule_feasible=(
                        final_residual
                        <= self.options.stage_a.feasibility_tolerance_m
                    ),
                )
            )
        controls = tuple(updated_controls)
        fem_cases = self._total_fem_cases - start_cases
        fem_seconds = self._total_fem_seconds - start_seconds
        success = bool(all(control.success for control in controls) and secondary.success)
        self._emit(
            f"{label} complete: FEM cases={fem_cases}, "
            f"FEM time={_format_duration(fem_seconds)}, "
            f"average={_format_duration(fem_seconds / max(fem_cases, 1))}/case"
        )
        return ContinuousDesignResult3D(
            evaluation=evaluation,
            controls=controls,
            secondary=secondary,
            success=success,
            message=(
                f"A feasible={sum(item.success for item in controls)}/{len(controls)}; "
                f"B: {secondary.message}"
            ),
            stage_a_fem_cases=sum(item.nfev for item in controls),
            secondary_matrix_nfev=secondary.nfev,
            fem_cases=fem_cases,
            fem_seconds=fem_seconds,
        )

    @staticmethod
    def _scale_tension_for_strands(evaluation, new_strands: np.ndarray) -> np.ndarray:
        old_strands = evaluation.design.strands.astype(float)
        return evaluation.design.pretension * np.divide(
            new_strands.astype(float),
            old_strands,
            out=np.ones_like(old_strands),
            where=old_strands > 0.0,
        )

    def optimize(
        self,
        initial_strands,
        initial_pretension=None,
        initial_ratio=None,
    ) -> Staged3DOptimizationResult:
        requested_strands = validate_strand_vector(
            initial_strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        initial_strand_curve = self.project_strands(requested_strands)
        strands = initial_strand_curve.interpolated_strands
        self._emit(
            f"start efficient 3D optimization: groups={self.layout.size}, "
            f"physical_strands={2 * int(np.sum(strands))}; "
            f"smooth_curve={self.options.secondary.curve_family}/"
            f"{initial_strand_curve.control_coordinates.size} controls per group; "
            f"strand_iterations={self.options.strands.iterations}; "
            f"ab_correction_passes={self.options.ab_correction_passes}"
        )
        if not np.array_equal(requested_strands, strands):
            self._emit(
                "initial strand counts were interpolated onto the outward "
                "non-decreasing smooth curve"
            )
        current = self.optimize_continuous(
            strands,
            initial_pretension,
            initial_ratio,
            label="base design",
        )
        best = current.evaluation
        controls = current.controls
        secondary = current.secondary
        history = [best]
        continuous_solves = 1
        completed = 0

        for iteration in range(self.options.strands.iterations):
            completed += 1
            stresses = np.asarray(
                [best.cable_stress_mpa[cable_id] for cable_id in self.layout.cable_ids]
            )
            upper = self.problem.bounds.stress_upper_mpa
            overstressed = np.flatnonzero(stresses > upper)
            if overstressed.size:
                index = int(overstressed[np.argmax(stresses[overstressed] - upper)])
                delta = self.options.strands.step
                reason = "repair upper stress violation"
            else:
                index = int(np.argmin(stresses))
                delta = -self.options.strands.step
                reason = "reduce lowest-utilized group"
            strand_curve = self.project_strands(best.design.strands)
            control_values = (
                strand_curve.backstay_control_strands.copy()
                if index % 2 == 0
                else strand_curve.main_stay_control_strands.copy()
            )
            stage = index // 2
            control_index = (
                0
                if control_values.size == 1 or self.problem.n_seg == 1
                else int(
                    round(
                        stage
                        * (control_values.size - 1)
                        / (self.problem.n_seg - 1)
                    )
                )
            )
            if delta > 0:
                control_values[control_index:] += self.options.strands.step
            else:
                control_values[: control_index + 1] -= self.options.strands.step
            control_values = np.clip(
                control_values,
                self.problem.bounds.strand_min,
                self.problem.bounds.strand_max,
            )
            basis = build_smooth_curve_basis(
                self.problem.n_seg,
                self.options.secondary.control_points_per_group,
                strand_curve.family,
            )
            changed_group = np.rint(basis @ control_values).astype(int)
            candidate_strands = best.design.strands.copy()
            candidate_strands[index % 2 :: 2] = changed_group
            candidate_strands = self.project_strands(candidate_strands).interpolated_strands
            if np.array_equal(candidate_strands, best.design.strands):
                self._emit(
                    "strand control-point move reaches a bound or vanishes after "
                    "smooth-curve interpolation; stopping"
                )
                break
            group_id = self.layout.cable_ids[index]
            self._emit(
                f"strand iteration {iteration + 1}: group={group_id}, "
                f"curve_control={control_index + 1}/{control_values.size}, "
                f"selected_count={best.design.strands[index]}->"
                f"{candidate_strands[index]} ({reason}); "
                f"this candidate requires another full "
                f"{(1 + self.options.ab_correction_passes) * (4 * self.problem.n_seg + self.secondary_fem_cases_per_cycle())}"
                "-case design"
            )
            candidate = self.optimize_continuous(
                candidate_strands,
                self._scale_tension_for_strands(best, candidate_strands),
                best.design.pretension_a_ratio,
                label=f"strand candidate {iteration + 1}",
            )
            continuous_solves += 1
            history.append(candidate.evaluation)
            if (
                candidate.evaluation.objective
                + self.options.strands.improvement_tol
                < best.objective
            ):
                best = candidate.evaluation
                controls = candidate.controls
                secondary = candidate.secondary
                self._emit(
                    f"accepted strand move: objective={best.objective:.6g}, "
                    f"physical_strands={best.metrics.total_strands}"
                )
            else:
                self._emit("strand candidate did not improve the full objective; stopping")
                break

        return Staged3DOptimizationResult(
            best=best,
            controls=controls,
            secondary=secondary,
            history=history,
            strand_iterations_completed=completed,
            continuous_solves=continuous_solves,
            fem_cases=self._total_fem_cases,
            fem_seconds=self._total_fem_seconds,
            strand_curve=self.project_strands(best.design.strands),
        )


__all__ = [
    "ContinuousDesignResult3D",
    "SecondaryAffineModel3D",
    "SecondaryTensionOptions3D",
    "SecondaryTensionResult3D",
    "SmoothCurveTrial3D",
    "StageAControlOptions",
    "StageAControlResult3D",
    "Staged3DOptimizationOptions",
    "Staged3DOptimizationResult",
    "StagedCableOptimizer3D",
    "StrandSearchOptions3D",
]
