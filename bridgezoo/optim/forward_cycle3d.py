"""Sequential forward cable tuning for the staged 3D single-tower bridge.

The algorithm deliberately follows construction order.  For cable group
``k`` it determines A at ``steel_and_A`` and then B at
``deck_weight_and_B``.  Every trial rebuilds the bridge and replays stages
from the beginning to the current substage.  Once a group's A/B values have
been accepted, later groups cannot change them.

After every group is locked, one complete replay supplies the final
``secondary_load`` cable stress.  Strand resizing is then a FEM-free force
conservation calculation against the requested target stress.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.optimize import lsq_linear

from bridgezoo.fem.single_staged import (
    SingleStagedDirectSolver3D,
    SingleStagedOpenSeesSolver3D,
)
from bridgezoo.optim.single_staged3d import (
    CableDesignEvaluator3D,
    EvaluationResult3D,
)
from bridgezoo.optim.variables import validate_strand_vector, validate_tension_vector


class ForwardTuningError(RuntimeError):
    """Raised when a construction substage cannot meet its local target."""


class ForwardSizingError(RuntimeError):
    """Raised when final cable force cannot define a tensile strand count."""


@dataclass(frozen=True)
class ForwardCycleOptions3D:
    """Numerical controls for one sequential forward/sizing cycle."""

    target_stress_mpa: float = 500.0
    displacement_tolerance_m: float = 1.0e-4
    linearity_tolerance_m: float = 1.0e-8
    probe_fraction: float = 0.10
    max_corrections: int = 2
    require_secondary_load: bool = True
    require_target: bool = True


@dataclass(frozen=True)
class ForwardLocalResponse3D:
    """The two displacement components controlled at one A or B substage."""

    tower_anchor_dx_m: float
    deck_anchor_relative_uz_m: float

    @property
    def vector(self) -> np.ndarray:
        return np.asarray(
            (self.tower_anchor_dx_m, self.deck_anchor_relative_uz_m),
            dtype=float,
        )

    @property
    def max_abs_m(self) -> float:
        return float(np.max(np.abs(self.vector)))

    @property
    def norm_m(self) -> float:
        return float(np.linalg.norm(self.vector))


@dataclass(frozen=True)
class ForwardSubstageResult3D:
    """Verified result for one group's A or B operation."""

    construction_stage: int
    phase: str
    stage_index: int
    stage_label: str
    displacement_basis: str
    target: ForwardLocalResponse3D
    response_before: ForwardLocalResponse3D
    predicted_response: ForwardLocalResponse3D
    response_after: ForwardLocalResponse3D
    backstay_tension_N: float
    main_stay_tension_N: float
    target_reached: bool
    best_feasible: bool
    influence_rank: int
    influence_condition: float
    active_bounds: tuple[str, ...]
    correction_passes: int
    fem_replays: int


@dataclass(frozen=True)
class ForwardCycleResult3D:
    """A completed forward construction replay followed by strand resizing."""

    cycle_index: int
    strands_before_sizing: np.ndarray
    strands_after_sizing: np.ndarray
    unclipped_target_strands: np.ndarray
    sizing_clipped: np.ndarray
    pretension_a: np.ndarray
    pretension_b: np.ndarray
    controls: tuple[ForwardSubstageResult3D, ...]
    final_evaluation: EvaluationResult3D
    final_stress_mpa: np.ndarray
    final_balanced_force_N: np.ndarray
    fem_replays: int
    fem_seconds: float


@dataclass(frozen=True)
class ForwardMilestone3D:
    """Restartable state after one construction group's A or B is accepted."""

    cycle_index: int
    construction_stage: int
    completed_phase: str
    strands: np.ndarray
    pretension_a: np.ndarray
    pretension_b: np.ndarray
    controls: tuple[ForwardSubstageResult3D, ...]
    birth_uz_m: dict[int, float]
    fem_replays: int
    fem_seconds: float


class ForwardCableCycleOptimizer3D:
    """Tune A/B once in construction order, then resize every cable group."""

    _SENSITIVITY_EPS = 1.0e-18
    _FORCE_EPS_N = 1.0e-6

    def __init__(
        self,
        evaluator: CableDesignEvaluator3D,
        options: ForwardCycleOptions3D | None = None,
        progress: Callable[[str], None] | None = None,
        milestone: Callable[[ForwardMilestone3D], None] | None = None,
    ) -> None:
        if evaluator.problem.backend not in {"direct", "opensees"}:
            raise ValueError("forward 3D tuning requires a direct or OpenSees backend")
        self.evaluator = evaluator
        self.problem = evaluator.problem
        self.layout = evaluator.layout
        self.options = options or ForwardCycleOptions3D()
        self.progress = progress
        self.milestone = milestone
        self.total_fem_replays = 0
        self.total_fem_seconds = 0.0
        self._validate_options()

    def _validate_options(self) -> None:
        options = self.options
        for name, value in (
            ("target stress", options.target_stress_mpa),
            ("displacement tolerance", options.displacement_tolerance_m),
            ("linearity tolerance", options.linearity_tolerance_m),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < options.probe_fraction <= 1.0:
            raise ValueError("probe fraction must be in (0, 1]")
        if isinstance(options.max_corrections, bool) or options.max_corrections < 1:
            raise ValueError("maximum corrections must be a positive integer")

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _solver(self):
        if self.problem.backend == "direct":
            return SingleStagedDirectSolver3D()
        return SingleStagedOpenSeesSolver3D()

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

    def tension_upper_bounds(self, strands: np.ndarray) -> np.ndarray:
        return (
            self.problem.bounds.tension_bound_stress_mpa
            * 1.0e6
            * self.problem.strand_area
            * strands.astype(float)
        )

    def _build_plan(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
    ):
        total, ratio = self._total_and_ratio(pretension_a, pretension_b)
        return self.evaluator.build_plan(strands, total, ratio)

    def _solve_record(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
        stage_index: int,
    ):
        plan = self._build_plan(strands, pretension_a, pretension_b)
        stage = next(
            (item for item in plan.stages if item.index == stage_index),
            None,
        )
        if stage is None:
            raise ForwardTuningError(f"unknown construction substage {stage_index}")
        started = time.perf_counter()
        record = self._solver().solve_stage(plan, stage.index, stage.label)
        duration = time.perf_counter() - started
        self.total_fem_replays += 1
        self.total_fem_seconds += duration
        if not record.converged:
            raise ForwardTuningError(
                f"{record.backend} did not converge at {record.stage_label!r}"
            )
        return plan, record

    @staticmethod
    def _group_nodes(plan, construction_stage: int) -> tuple[set[int], set[int]]:
        cables = [
            cable
            for cable in plan.model.cables.values()
            if cable.construction_stage == construction_stage
        ]
        backstays = [cable for cable in cables if cable.group == "backstay"]
        main_stays = [cable for cable in cables if cable.group == "main_stay"]
        if len(backstays) != 2 or len(main_stays) != 2:
            raise ForwardTuningError(
                f"3D group {construction_stage} must contain two backstays "
                "and two main stays"
            )
        return {cable.i for cable in backstays}, {cable.j for cable in main_stays}

    def _response(
        self,
        plan,
        record,
        construction_stage: int,
        birth_uz_m: dict[int, float] | None,
    ) -> tuple[ForwardLocalResponse3D, dict[int, float]]:
        tower_nodes, deck_nodes = self._group_nodes(plan, construction_stage)
        if birth_uz_m is None:
            missing = deck_nodes.difference(record.birth_displacement)
            if missing:
                raise ForwardTuningError(
                    f"stage {construction_stage} is missing tangent-birth "
                    f"displacement for deck nodes {sorted(missing)}"
                )
            birth_uz_m = {
                node_id: float(record.birth_displacement[node_id][2])
                for node_id in deck_nodes
            }
        elif set(birth_uz_m) != deck_nodes:
            raise ForwardTuningError(
                f"stage {construction_stage} tangent-birth node set changed"
            )
        response = ForwardLocalResponse3D(
            tower_anchor_dx_m=float(
                np.mean([record.displacement[node_id][0] for node_id in tower_nodes])
            ),
            deck_anchor_relative_uz_m=float(
                np.mean(
                    [
                        record.displacement[node_id][2] - birth_uz_m[node_id]
                        for node_id in deck_nodes
                    ]
                )
            ),
        )
        return response, birth_uz_m

    def _solve_response(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
        *,
        construction_stage: int,
        phase: str,
        birth_uz_m: dict[int, float] | None,
    ):
        stage_index = 3 * construction_stage - (2 if phase == "A" else 1)
        plan, record = self._solve_record(
            strands,
            pretension_a,
            pretension_b,
            stage_index,
        )
        response, resolved_birth = self._response(
            plan,
            record,
            construction_stage,
            birth_uz_m,
        )
        return response, resolved_birth, record

    @staticmethod
    def _as_response(values: np.ndarray) -> ForwardLocalResponse3D:
        return ForwardLocalResponse3D(
            tower_anchor_dx_m=float(values[0]),
            deck_anchor_relative_uz_m=float(values[1]),
        )

    def _target_response(
        self,
        construction_stage: int,
        phase: str,
    ) -> ForwardLocalResponse3D:
        target_uz_m = 0.0
        if phase == "B":
            x_m = self.evaluator.config.cable_station_x(construction_stage)
            target_uz_m = self.evaluator.config.pretension_b_target_uz_m(x_m)
        return ForwardLocalResponse3D(
            tower_anchor_dx_m=0.0,
            deck_anchor_relative_uz_m=target_uz_m,
        )

    def _probe_delta(self, value: float, upper: float) -> float:
        nominal = self.options.probe_fraction * upper
        if value + nominal <= upper:
            return nominal
        if value - nominal >= 0.0:
            return -nominal
        positive_room = upper - value
        negative_room = value
        return positive_room if positive_room >= negative_room else -negative_room

    def _bounded_candidate(
        self,
        response: np.ndarray,
        influence: np.ndarray,
        current: np.ndarray,
        upper: np.ndarray,
    ) -> tuple[np.ndarray, int, float]:
        scaled = influence * upper[None, :]
        rank = int(np.linalg.matrix_rank(scaled))
        condition = float(np.linalg.cond(scaled)) if rank else math.inf
        candidate = np.zeros(2, dtype=float)
        responsive = np.linalg.norm(influence, axis=0) > self._SENSITIVITY_EPS
        free = responsive & (upper > self._FORCE_EPS_N)
        if np.any(free):
            intercept = response - influence @ current
            solved = lsq_linear(
                influence[:, free],
                -intercept,
                bounds=(np.zeros(int(np.sum(free))), upper[free]),
                method="trf",
                tol=1.0e-12,
                lsmr_tol=1.0e-12,
                max_iter=200,
            )
            if not solved.success:
                raise ForwardTuningError(
                    f"bounded local balance solve failed: {solved.message}"
                )
            candidate[free] = solved.x
        return candidate, rank, condition

    def _optimize_substage(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
        *,
        construction_stage: int,
        phase: str,
        birth_uz_m: dict[int, float] | None,
    ) -> tuple[ForwardSubstageResult3D, dict[int, float]]:
        if phase not in {"A", "B"}:
            raise ValueError("forward substage phase must be A or B")
        target = self._target_response(construction_stage, phase)
        offset = 2 * (construction_stage - 1)
        target_values = pretension_a if phase == "A" else pretension_b
        other_values = pretension_b if phase == "A" else pretension_a
        upper = (
            self.tension_upper_bounds(strands)[offset : offset + 2]
            - other_values[offset : offset + 2]
        )
        if np.any(upper < -self._FORCE_EPS_N):
            raise ForwardTuningError(
                f"stage {construction_stage} {phase} starts above cable capacity"
            )
        upper = np.maximum(upper, 0.0)
        current = target_values[offset : offset + 2].copy()
        start_cases = self.total_fem_replays
        first_response = None
        first_residual = None
        last_prediction = None
        last_rank = 0
        last_condition = math.inf
        accepted_response = None
        accepted_record = None
        passes = 0

        for passes in range(1, self.options.max_corrections + 1):
            baseline_a = pretension_a.copy()
            baseline_b = pretension_b.copy()
            (baseline_a if phase == "A" else baseline_b)[
                offset : offset + 2
            ] = current
            baseline, resolved_birth, baseline_record = self._solve_response(
                strands,
                baseline_a,
                baseline_b,
                construction_stage=construction_stage,
                phase=phase,
                birth_uz_m=birth_uz_m,
            )
            birth_uz_m = resolved_birth
            if first_response is None:
                first_response = baseline
                first_residual = self._as_response(baseline.vector - target.vector)
            accepted_response = baseline
            accepted_record = baseline_record
            baseline_residual = self._as_response(baseline.vector - target.vector)
            if baseline_residual.max_abs_m <= self.options.displacement_tolerance_m:
                target_values[offset : offset + 2] = current
                last_prediction = baseline
                break

            influence = np.zeros((2, 2), dtype=float)
            for local_index in range(2):
                delta = self._probe_delta(current[local_index], upper[local_index])
                if abs(delta) <= self._FORCE_EPS_N:
                    continue
                probe = current.copy()
                probe[local_index] += delta
                probe_a = pretension_a.copy()
                probe_b = pretension_b.copy()
                (probe_a if phase == "A" else probe_b)[
                    offset : offset + 2
                ] = probe
                probe_response, _, _ = self._solve_response(
                    strands,
                    probe_a,
                    probe_b,
                    construction_stage=construction_stage,
                    phase=phase,
                    birth_uz_m=birth_uz_m,
                )
                influence[:, local_index] = (
                    probe_response.vector - baseline.vector
                ) / delta

            candidate, last_rank, last_condition = self._bounded_candidate(
                baseline_residual.vector,
                influence,
                current,
                upper,
            )
            residual_intercept = baseline_residual.vector - influence @ current
            last_prediction = self._as_response(
                target.vector + residual_intercept + influence @ candidate
            )
            if np.allclose(candidate, current, rtol=0.0, atol=self._FORCE_EPS_N):
                break

            candidate_a = pretension_a.copy()
            candidate_b = pretension_b.copy()
            (candidate_a if phase == "A" else candidate_b)[
                offset : offset + 2
            ] = candidate
            verified, _, verified_record = self._solve_response(
                strands,
                candidate_a,
                candidate_b,
                construction_stage=construction_stage,
                phase=phase,
                birth_uz_m=birth_uz_m,
            )
            prediction_error = float(
                np.max(np.abs(verified.vector - last_prediction.vector))
            )
            verified_residual = self._as_response(verified.vector - target.vector)
            if (
                verified_residual.norm_m
                > baseline_residual.norm_m + self.options.linearity_tolerance_m
            ):
                raise ForwardTuningError(
                    f"stage {construction_stage} {phase} correction worsened the "
                    "verified local displacement"
                )
            current = candidate
            accepted_response = verified
            accepted_record = verified_record
            if (
                verified_residual.max_abs_m
                <= self.options.displacement_tolerance_m
                or prediction_error <= self.options.linearity_tolerance_m
            ):
                target_values[offset : offset + 2] = current
                baseline = verified
                break
        else:
            accepted_response = verified
            accepted_record = verified_record

        target_values[offset : offset + 2] = current
        final_response = accepted_response
        if final_response is None or accepted_record is None:
            raise ForwardTuningError(
                f"stage {construction_stage} {phase} produced no verified response"
            )
        final_residual = self._as_response(final_response.vector - target.vector)
        target_reached = final_residual.max_abs_m <= self.options.displacement_tolerance_m
        best_feasible = final_residual.norm_m <= first_residual.norm_m + (
            self.options.linearity_tolerance_m
        )
        if self.options.require_target and not target_reached:
            raise ForwardTuningError(
                f"stage {construction_stage} {phase} did not reach the local "
                f"displacement target: actual tower="
                f"{final_response.tower_anchor_dx_m:.6g} m, deck="
                f"{final_response.deck_anchor_relative_uz_m:.6g} m; target tower="
                f"{target.tower_anchor_dx_m:.6g} m, deck="
                f"{target.deck_anchor_relative_uz_m:.6g} m"
            )

        active_bounds = []
        names = ("backstay", "main_stay")
        for index, name in enumerate(names):
            if current[index] <= self._FORCE_EPS_N:
                active_bounds.append(f"{name}:lower")
            elif upper[index] - current[index] <= self._FORCE_EPS_N:
                active_bounds.append(f"{name}:upper")
        if last_prediction is None:
            last_prediction = final_response
        basis = (
            "tower x is total stage displacement; deck z is relative to the "
            "steel_and_A tangent-birth position"
        )
        result = ForwardSubstageResult3D(
            construction_stage=construction_stage,
            phase=phase,
            stage_index=accepted_record.stage_index,
            stage_label=accepted_record.stage_label,
            displacement_basis=basis,
            target=target,
            response_before=first_response,
            predicted_response=last_prediction,
            response_after=final_response,
            backstay_tension_N=float(current[0]),
            main_stay_tension_N=float(current[1]),
            target_reached=target_reached,
            best_feasible=best_feasible,
            influence_rank=last_rank,
            influence_condition=last_condition,
            active_bounds=tuple(active_bounds),
            correction_passes=passes,
            fem_replays=self.total_fem_replays - start_cases,
        )
        return result, birth_uz_m

    def resize_strands(
        self,
        strands: np.ndarray,
        final_stress_mpa: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Resize strands by conserving each physical cable's final force.

        Returns ``(next, raw_target, clipped, force_N)``.  No FEM call is made.
        ``ceil`` is used so integer sizing does not exceed the target tensile
        stress before explicit strand bounds are applied.
        """

        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        stress = np.asarray(final_stress_mpa, dtype=float)
        if stress.shape != (self.layout.size,) or not np.all(np.isfinite(stress)):
            raise ValueError(
                f"final stress must have shape ({self.layout.size},) and be finite"
            )
        area = strands.astype(float) * self.problem.strand_area
        force_N = stress * 1.0e6 * area
        nonpositive = np.flatnonzero(force_N <= 0.0)
        if nonpositive.size:
            groups = [self.layout.cable_ids[index] for index in nonpositive]
            raise ForwardSizingError(
                "positive tensile final force is required for 500 MPa sizing; "
                f"non-tensile groups: {groups}"
            )
        exact = force_N / (
            self.options.target_stress_mpa * 1.0e6 * self.problem.strand_area
        )
        raw_target = np.ceil(exact - 1.0e-12).astype(int)
        next_strands = np.clip(
            raw_target,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        clipped = next_strands != raw_target
        return next_strands, raw_target, clipped, force_N

    def _evaluate_final(
        self,
        strands: np.ndarray,
        pretension_a: np.ndarray,
        pretension_b: np.ndarray,
    ) -> EvaluationResult3D:
        total, ratio = self._total_and_ratio(pretension_a, pretension_b)
        started = time.perf_counter()
        evaluation = self.evaluator.evaluate(
            strands,
            total,
            ratio,
            keep_result=True,
        )
        self.total_fem_replays += 1
        self.total_fem_seconds += time.perf_counter() - started
        if (
            self.options.require_secondary_load
            and evaluation.staged_result.final.stage_label != "secondary_load"
        ):
            raise ForwardTuningError(
                "forward cable sizing requires a final secondary_load stage"
            )
        return evaluation

    def run_cycle(
        self,
        strands,
        *,
        cycle_index: int = 1,
        start_construction_stage: int = 1,
        start_phase: str = "A",
        pretension_a=None,
        pretension_b=None,
        completed_controls: tuple[ForwardSubstageResult3D, ...] = (),
        pending_birth_uz_m: dict[int, float] | None = None,
    ) -> ForwardCycleResult3D:
        """Run or resume one forward pass, then perform FEM-free sizing.

        ``start_construction_stage`` and ``start_phase`` identify the first
        tensioning operation that still needs optimization.  Earlier A/B
        values and controls are accepted as locked milestone state and are
        only replayed by FEM; they are never optimized again.
        """

        if isinstance(cycle_index, bool) or cycle_index < 1:
            raise ValueError("cycle index must be a positive integer")
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        start_cases = self.total_fem_replays
        start_seconds = self.total_fem_seconds
        if (
            isinstance(start_construction_stage, bool)
            or not 1 <= start_construction_stage <= self.problem.n_seg + 1
        ):
            raise ValueError(
                f"start construction stage must be between 1 and "
                f"{self.problem.n_seg + 1}"
            )
        if start_phase not in {"A", "B"}:
            raise ValueError("start phase must be A or B")
        if start_construction_stage == self.problem.n_seg + 1 and start_phase != "A":
            raise ValueError("the post-construction resume point must use phase A")
        pretension_a = (
            np.zeros(self.layout.size, dtype=float)
            if pretension_a is None
            else validate_tension_vector(pretension_a, self.layout)
        )
        pretension_b = (
            np.zeros(self.layout.size, dtype=float)
            if pretension_b is None
            else validate_tension_vector(pretension_b, self.layout)
        )
        upper = self.tension_upper_bounds(strands)
        if np.any(pretension_a + pretension_b > upper * (1.0 + 1.0e-12)):
            raise ValueError("locked A/B milestone force exceeds cable capacity")
        expected_controls = 2 * (start_construction_stage - 1) + (
            1 if start_phase == "B" else 0
        )
        controls = list(completed_controls)
        if len(controls) != expected_controls:
            raise ValueError(
                f"resuming at group {start_construction_stage} requires "
                f"{expected_controls} locked A/B controls"
            )
        expected_order = [
            (stage, phase)
            for stage in range(1, start_construction_stage)
            for phase in ("A", "B")
        ]
        if start_phase == "B":
            expected_order.append((start_construction_stage, "A"))
        if [
            (item.construction_stage, item.phase) for item in controls
        ] != expected_order:
            raise ValueError("locked milestone controls are not in construction order")
        future_offset = 2 * (start_construction_stage - 1)
        future_a_offset = future_offset + (2 if start_phase == "B" else 0)
        if np.any(pretension_a[future_a_offset:] > self._FORCE_EPS_N) or np.any(
            pretension_b[future_offset:] > self._FORCE_EPS_N
        ):
            raise ValueError("unoptimized future cable groups must have zero A/B force")
        if start_phase == "B" and pending_birth_uz_m is None:
            raise ValueError("resuming at phase B requires the locked A tangent birth")

        for construction_stage in range(
            start_construction_stage, self.problem.n_seg + 1
        ):
            resume_at_b = (
                construction_stage == start_construction_stage
                and start_phase == "B"
            )
            if resume_at_b:
                birth_uz_m = dict(pending_birth_uz_m)
            else:
                self._emit(
                    f"cycle {cycle_index}: group {construction_stage}/"
                    f"{self.problem.n_seg} A"
                )
                control_a, birth_uz_m = self._optimize_substage(
                    strands,
                    pretension_a,
                    pretension_b,
                    construction_stage=construction_stage,
                    phase="A",
                    birth_uz_m=None,
                )
                controls.append(control_a)
                if self.milestone is not None:
                    self.milestone(
                        ForwardMilestone3D(
                            cycle_index=cycle_index,
                            construction_stage=construction_stage,
                            completed_phase="A",
                            strands=strands.copy(),
                            pretension_a=pretension_a.copy(),
                            pretension_b=pretension_b.copy(),
                            controls=tuple(controls),
                            birth_uz_m=dict(birth_uz_m),
                            fem_replays=self.total_fem_replays - start_cases,
                            fem_seconds=self.total_fem_seconds - start_seconds,
                        )
                    )
                    self._emit(
                        f"cycle {cycle_index}: group {construction_stage} A "
                        "milestone saved; next phase B"
                    )
            self._emit(
                f"cycle {cycle_index}: group {construction_stage}/{self.problem.n_seg} B"
            )
            control_b, _ = self._optimize_substage(
                strands,
                pretension_a,
                pretension_b,
                construction_stage=construction_stage,
                phase="B",
                birth_uz_m=birth_uz_m,
            )
            controls.append(control_b)
            if self.milestone is not None:
                self.milestone(
                    ForwardMilestone3D(
                        cycle_index=cycle_index,
                        construction_stage=construction_stage,
                        completed_phase="B",
                        strands=strands.copy(),
                        pretension_a=pretension_a.copy(),
                        pretension_b=pretension_b.copy(),
                        controls=tuple(controls),
                        birth_uz_m=dict(birth_uz_m),
                        fem_replays=self.total_fem_replays - start_cases,
                        fem_seconds=self.total_fem_seconds - start_seconds,
                    )
                )
                self._emit(
                    f"cycle {cycle_index}: group {construction_stage} milestone saved; "
                    f"next group {construction_stage + 1} phase A"
                )

        self._emit(f"cycle {cycle_index}: final secondary-load replay")
        evaluation = self._evaluate_final(
            strands,
            pretension_a,
            pretension_b,
        )
        final_stress = np.asarray(
            [
                evaluation.cable_stress_mpa[group_id]
                for group_id in self.layout.cable_ids
            ],
            dtype=float,
        )
        resized, raw_target, clipped, force_N = self.resize_strands(
            strands,
            final_stress,
        )
        self._emit(
            f"cycle {cycle_index}: sizing complete; next strand configuration saved"
        )
        return ForwardCycleResult3D(
            cycle_index=cycle_index,
            strands_before_sizing=strands.copy(),
            strands_after_sizing=resized,
            unclipped_target_strands=raw_target,
            sizing_clipped=clipped,
            pretension_a=pretension_a,
            pretension_b=pretension_b,
            controls=tuple(controls),
            final_evaluation=evaluation,
            final_stress_mpa=final_stress,
            final_balanced_force_N=force_N,
            fem_replays=self.total_fem_replays - start_cases,
            fem_seconds=self.total_fem_seconds - start_seconds,
        )


__all__ = [
    "ForwardCableCycleOptimizer3D",
    "ForwardCycleOptions3D",
    "ForwardCycleResult3D",
    "ForwardLocalResponse3D",
    "ForwardMilestone3D",
    "ForwardSizingError",
    "ForwardSubstageResult3D",
    "ForwardTuningError",
]
