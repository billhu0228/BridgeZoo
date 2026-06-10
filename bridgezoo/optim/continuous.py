"""Continuous pretension optimization for fixed cable strand counts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from bridgezoo.optim.evaluator import CableDesignEvaluator, EvaluationResult
from bridgezoo.optim.variables import validate_strand_vector, validate_tension_vector


@dataclass(frozen=True)
class ContinuousOptions:
    maxiter: int = 80
    ftol: float = 1.0e-7
    finite_diff_rel_step: float | None = None
    progress_every: int = 0
    # "linear": 仿射模型 LP+SLSQP(线性后端精确、无 FEM 内层,默认);
    # "slsqp": 直接在 FEM 上跑 SLSQP(保留给将来非线性后端)。
    method: str = "linear"


@dataclass(frozen=True)
class ContinuousOptimizationResult:
    evaluation: EvaluationResult
    success: bool
    message: str
    nfev: int | None = None
    # LP 可行性相得到的最小可达最大违反量 s* [MPa](仅 linear 方法填写)。
    feasibility_violation_mpa: float | None = None


class FixedStrandTensionOptimizer:
    """Optimize non-negative pretensions for a fixed integer strand vector."""

    def __init__(
        self,
        evaluator: CableDesignEvaluator,
        options: ContinuousOptions | None = None,
        progress: Callable[[str], None] | None = None,
    ):
        self.evaluator = evaluator
        self.options = options or ContinuousOptions()
        self.layout = evaluator.layout
        self.problem = evaluator.problem
        self.progress = progress

    def _emit(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def tension_bounds(self, strands: np.ndarray) -> list[tuple[float, float]]:
        area = self.problem.strand_area * strands.astype(float)
        upper = self.problem.bounds.tension_bound_stress_mpa * 1e6 * area
        return [(0.0, float(value)) for value in upper]

    def default_initial(self, strands: np.ndarray) -> np.ndarray:
        mid_stress = 0.5 * (self.problem.bounds.stress_lower_mpa + self.problem.bounds.stress_upper_mpa)
        return mid_stress * 1e6 * self.problem.strand_area * strands.astype(float)

    def _evaluate_cached(self, strands: np.ndarray, x, cache: dict) -> EvaluationResult:
        """同一 x 处 objective/约束/progress 共享一次 FEM(cache 生命周期 = 一次 optimize)。"""
        arr = np.asarray(x, dtype=float)
        key = arr.tobytes()
        if key not in cache:
            cache[key] = self.evaluator.evaluate(strands, arr)
        return cache[key]

    def _stress_margins(self, strands: np.ndarray, x, cache: dict, *, upper: bool) -> np.ndarray:
        """应力带 ineq 约束值(>=0 为满足)。

        评估失败(如优化器探测到无效张力)时返回有限的"严重违反"值,而不是让异常
        炸掉整个优化过程。
        """
        try:
            ev = self._evaluate_cached(strands, x, cache)
        except (ValueError, RuntimeError, FloatingPointError):
            return np.full(self.layout.size, -1.0e6)
        values = np.asarray([ev.cable_stress_mpa[cid] for cid in self.layout.cable_ids], dtype=float)
        if upper:
            return self.problem.bounds.stress_upper_mpa - values
        return values - self.problem.bounds.stress_lower_mpa

    @staticmethod
    def _progress_metrics(ev: EvaluationResult | None) -> str:
        if ev is None:
            return "metrics=unavailable"
        return (
            f"shape_rmse={ev.metrics.shape_rmse_m * 1000.0:.3f} mm "
            f"stress=[{ev.metrics.stress_min_mpa:.1f}, {ev.metrics.stress_max_mpa:.1f}] MPa "
            f"stress_violation_rms={ev.metrics.stress_violation_rms_mpa:.3f} MPa"
        )

    def optimize(self, strands, initial_pretension=None) -> ContinuousOptimizationResult:
        try:
            from scipy.optimize import minimize
        except ImportError as exc:
            raise RuntimeError("scipy is required for cable pretension optimization") from exc

        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        bounds = self.tension_bounds(strands)
        if initial_pretension is None:
            x0 = self.default_initial(strands)
        else:
            x0 = validate_tension_vector(initial_pretension, self.layout)
        lo = np.asarray([b[0] for b in bounds], dtype=float)
        hi = np.asarray([b[1] for b in bounds], dtype=float)
        x0 = np.clip(x0, lo, hi)
        eval_count = 0
        cache: dict[bytes, EvaluationResult] = {}

        def objective(x):
            nonlocal eval_count
            eval_count += 1
            try:
                ev = self._evaluate_cached(strands, x, cache)
                value = ev.objective
            except (ValueError, RuntimeError, FloatingPointError):
                ev, value = None, 1.0e30
            if self.options.progress_every > 0 and eval_count % self.options.progress_every == 0:
                self._emit(
                    f"    SLSQP eval {eval_count}: objective={value:.6g} "
                    f"{self._progress_metrics(ev)}"
                )
            return value

        def stress_lower_constraint(x):
            return self._stress_margins(strands, x, cache, upper=False)

        def stress_upper_constraint(x):
            return self._stress_margins(strands, x, cache, upper=True)

        options = {"maxiter": self.options.maxiter, "ftol": self.options.ftol, "disp": False}
        if self.options.finite_diff_rel_step is not None:
            options["finite_diff_rel_step"] = self.options.finite_diff_rel_step

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[
                {"type": "ineq", "fun": stress_lower_constraint},
                {"type": "ineq", "fun": stress_upper_constraint},
            ],
            options=options,
        )
        x = np.clip(np.asarray(res.x, dtype=float), lo, hi)
        evaluation = self.evaluator.evaluate(strands, x)
        self._emit(
            "    SLSQP done: "
            f"success={bool(res.success)} objective={evaluation.objective:.6g} "
            f"nfev={getattr(res, 'nfev', None)} "
            f"shape_rmse={evaluation.metrics.shape_rmse_m * 1000.0:.3f} mm "
            f"stress=[{evaluation.metrics.stress_min_mpa:.1f}, {evaluation.metrics.stress_max_mpa:.1f}] MPa "
            f"stress_violation_rms={evaluation.metrics.stress_violation_rms_mpa:.3f} MPa"
        )
        return ContinuousOptimizationResult(
            evaluation=evaluation,
            success=bool(res.success),
            message=str(res.message),
            nfev=getattr(res, "nfev", None),
        )
