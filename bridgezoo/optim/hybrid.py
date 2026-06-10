"""Hybrid integer-strand and continuous-pretension optimizer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from bridgezoo.optim.continuous import (
    ContinuousOptimizationResult,
    ContinuousOptions,
    FixedStrandTensionOptimizer,
)
from bridgezoo.optim.evaluator import CableDesignEvaluator, EvaluationResult
from bridgezoo.optim.linear import LinearTensionOptimizer
from bridgezoo.optim.problem import CableOptimizationProblem
from bridgezoo.optim.variables import validate_strand_vector


@dataclass(frozen=True)
class IntegerSearchOptions:
    outer_iterations: int = 4
    coordinate_step: int = 1
    random_trials: int = 0
    seed: int = 0
    improvement_tol: float = 1.0e-8
    stress_guided: bool = True
    # 每轮外层迭代先尝试按 σ/σ_target 比例整体改股(大步跳跃),再做 ±step 精修。
    resize: bool = True
    resize_target_stress_mpa: float | None = None  # None → 应力带中值


@dataclass(frozen=True)
class HybridOptions:
    continuous: ContinuousOptions = field(default_factory=ContinuousOptions)
    integer: IntegerSearchOptions = field(default_factory=IntegerSearchOptions)


@dataclass(frozen=True)
class HybridOptimizationResult:
    best: EvaluationResult
    history: list[EvaluationResult]
    # best 索股配置下 LP 最小可达最大违反量 s* [MPa](linear 连续层时填写)。
    feasibility_violation_mpa: float | None = None


class CableHybridOptimizer:
    """Coordinate-search over integer strands with SLSQP pretension solves."""

    def __init__(
        self,
        problem: CableOptimizationProblem,
        options: HybridOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.problem = problem
        self.options = options or HybridOptions()
        self.evaluator = CableDesignEvaluator(problem)
        self.layout = self.evaluator.layout
        self.progress = progress
        method = self.options.continuous.method
        if method == "linear":
            self.continuous = LinearTensionOptimizer(self.evaluator, self.options.continuous, progress=progress)
        elif method == "slsqp":
            self.continuous = FixedStrandTensionOptimizer(self.evaluator, self.options.continuous, progress=progress)
        else:
            raise ValueError(f"unknown continuous method: {method!r} (expected 'linear' or 'slsqp')")

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def default_strands(self) -> np.ndarray:
        value = min(max(20, self.problem.bounds.strand_min), self.problem.bounds.strand_max)
        return np.full(self.layout.size, value, dtype=int)

    @staticmethod
    def _scaled_pretension(old_pretension, old_strands, new_strands) -> np.ndarray:
        old = np.asarray(old_strands, dtype=float)
        new = np.asarray(new_strands, dtype=float)
        tension = np.asarray(old_pretension, dtype=float)
        return tension * np.divide(new, old, out=np.ones_like(new), where=old > 0)

    def _continuous_for(self, strands, initial_pretension=None) -> ContinuousOptimizationResult:
        self._emit(
            f"  optimize tensions: total_strands={int(np.sum(strands))} "
            f"min={int(np.min(strands))} max={int(np.max(strands))}"
        )
        return self.continuous.optimize(strands, initial_pretension=initial_pretension)

    def _resize_candidate(self, best: EvaluationResult) -> np.ndarray | None:
        """按 σ_i/σ_target 比例缩放索股(应力恒定近似:σ ∝ N/A,N 近似不变)。

        返回与当前不同的候选股数向量;无变化时返回 None。σ_i ≤ 0(受压/松弛)
        的索直接取 strand_min。
        """
        target = self.options.integer.resize_target_stress_mpa
        if target is None:
            target = 0.5 * (self.problem.bounds.stress_lower_mpa + self.problem.bounds.stress_upper_mpa)
        if target <= 0.0:
            raise ValueError(f"resize target stress must be positive, got {target!r}")
        current = np.asarray(best.design.strands, dtype=int)
        sigma = np.asarray([best.cable_stress_mpa[cid] for cid in self.layout.cable_ids], dtype=float)
        scaled = np.where(
            sigma > 0.0,
            np.rint(current * sigma / target),
            float(self.problem.bounds.strand_min),
        )
        candidate = np.clip(
            scaled.astype(int), self.problem.bounds.strand_min, self.problem.bounds.strand_max
        )
        return None if np.array_equal(candidate, current) else candidate

    def _strand_moves_for(self, best: EvaluationResult, idx: int) -> list[tuple[int, str]]:
        step = self.options.integer.coordinate_step
        cid = self.layout.cable_ids[idx]
        stress = best.cable_stress_mpa[cid]
        lower = self.problem.bounds.stress_lower_mpa
        upper = self.problem.bounds.stress_upper_mpa
        if self.options.integer.stress_guided:
            if stress < lower:
                return [(-step, f"stress low {stress:.1f} < {lower:.1f} MPa"), (+step, "fallback")]
            if stress > upper:
                return [(+step, f"stress high {stress:.1f} > {upper:.1f} MPa"), (-step, "fallback")]
            return [(-step, f"stress ok {stress:.1f} MPa; reduce strands if possible"), (+step, "fallback")]
        return [(-step, "coordinate search"), (+step, "coordinate search")]

    def optimize(self, initial_strands=None, initial_pretension=None) -> HybridOptimizationResult:
        if initial_strands is None:
            strands = self.default_strands()
        else:
            strands = validate_strand_vector(
                initial_strands,
                self.layout,
                self.problem.bounds.strand_min,
                self.problem.bounds.strand_max,
            )

        self._emit(
            f"start cable optimization: n_seg={self.problem.n_seg}, cables={self.layout.size}, "
            f"outer_iterations={self.options.integer.outer_iterations}, "
            f"random_trials={self.options.integer.random_trials}"
        )
        initial = self._continuous_for(strands, initial_pretension)
        best = initial.evaluation
        best_feasibility = initial.feasibility_violation_mpa
        history = [best]
        self._emit(f"initial best: objective={best.objective:.6g} total_strands={best.metrics.total_strands}")
        rng = np.random.default_rng(self.options.integer.seed)

        for trial_index in range(self.options.integer.random_trials):
            trial = rng.integers(
                self.problem.bounds.strand_min,
                self.problem.bounds.strand_max + 1,
                size=self.layout.size,
                dtype=int,
            )
            self._emit(f"random trial {trial_index + 1}/{self.options.integer.random_trials}")
            res = self._continuous_for(trial)
            ev = res.evaluation
            history.append(ev)
            if ev.objective + self.options.integer.improvement_tol < best.objective:
                self._emit(f"  accepted random trial: {best.objective:.6g} -> {ev.objective:.6g}")
                best = ev
                best_feasibility = res.feasibility_violation_mpa
            else:
                self._emit(f"  rejected random trial: objective={ev.objective:.6g}, best={best.objective:.6g}")

        for outer in range(self.options.integer.outer_iterations):
            self._emit(f"outer iteration {outer + 1}/{self.options.integer.outer_iterations}")
            improved = False
            if self.options.integer.resize:
                candidate = self._resize_candidate(best)
                if candidate is not None:
                    self._emit(
                        f"resize candidate: total_strands {int(np.sum(best.design.strands))} -> "
                        f"{int(np.sum(candidate))} (stress-ratio jump)"
                    )
                    warm = self._scaled_pretension(best.design.pretension, best.design.strands, candidate)
                    res = self._continuous_for(candidate, warm)
                    ev = res.evaluation
                    history.append(ev)
                    if ev.objective + self.options.integer.improvement_tol < best.objective:
                        self._emit(f"  accepted resize: {best.objective:.6g} -> {ev.objective:.6g}")
                        best = ev
                        best_feasibility = res.feasibility_violation_mpa
                        improved = True
                    else:
                        self._emit(f"  rejected resize: objective={ev.objective:.6g}, best={best.objective:.6g}")
            for idx in range(self.layout.size):
                current = best.design.strands
                for delta, reason in self._strand_moves_for(best, idx):
                    candidate = current.copy()
                    candidate[idx] += delta
                    if candidate[idx] < self.problem.bounds.strand_min:
                        self._emit(
                            f"candidate cable={self.layout.cable_ids[idx]} strand_delta={delta:+d} skipped: "
                            f"below strand_min ({reason})"
                        )
                        continue
                    if candidate[idx] > self.problem.bounds.strand_max:
                        self._emit(
                            f"candidate cable={self.layout.cable_ids[idx]} strand_delta={delta:+d} skipped: "
                            f"above strand_max ({reason})"
                        )
                        continue
                    cid = self.layout.cable_ids[idx]
                    self._emit(
                        f"candidate cable={cid} strand_delta={delta:+d} "
                        f"{int(current[idx])}->{int(candidate[idx])} ({reason})"
                    )
                    warm = self._scaled_pretension(best.design.pretension, current, candidate)
                    res = self._continuous_for(candidate, warm)
                    ev = res.evaluation
                    history.append(ev)
                    if ev.objective + self.options.integer.improvement_tol < best.objective:
                        self._emit(f"  accepted: {best.objective:.6g} -> {ev.objective:.6g}")
                        best = ev
                        best_feasibility = res.feasibility_violation_mpa
                        improved = True
                        break
                    self._emit(f"  rejected: objective={ev.objective:.6g}, best={best.objective:.6g}")
            if not improved:
                self._emit("no integer improvement in this outer iteration; stopping")
                break

        final = self.evaluator.evaluate(best.design.strands, best.design.pretension, keep_result=True)
        band_violation = max(
            0.0,
            self.problem.bounds.stress_lower_mpa - final.metrics.stress_min_mpa,
            final.metrics.stress_max_mpa - self.problem.bounds.stress_upper_mpa,
        )
        verdict = "SATISFIED" if band_violation <= 1e-6 else f"VIOLATED by {band_violation:.3f} MPa"
        feasibility_note = (
            f", LP bound s*={best_feasibility:.3f} MPa" if best_feasibility is not None else ""
        )
        self._emit(
            f"finished: objective={final.objective:.6g} total_strands={final.metrics.total_strands} "
            f"shape_rmse={final.metrics.shape_rmse_m * 1000.0:.3f} mm "
            f"stress=[{final.metrics.stress_min_mpa:.1f}, {final.metrics.stress_max_mpa:.1f}] MPa "
            f"band [{self.problem.bounds.stress_lower_mpa:g}, {self.problem.bounds.stress_upper_mpa:g}] MPa "
            f"{verdict}{feasibility_note}"
        )
        return HybridOptimizationResult(
            best=final, history=history, feasibility_violation_mpa=best_feasibility
        )
