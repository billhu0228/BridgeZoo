"""Forward evaluation for cable strand and pretension designs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bridgezoo.fem.staged import (
    StagedDirectBatchSolver,
    StagedDirectSolver,
    StagedOpenSeesSolver,
    build_staged_cantilever,
)
from bridgezoo.fem.staged.plan import StagedResult
from bridgezoo.optim.objectives import ObjectiveBreakdown, objective_breakdown, stress_violation_mpa
from bridgezoo.optim.problem import CableOptimizationProblem
from bridgezoo.optim.variables import CableLayout, validate_strand_vector, validate_tension_vector


@dataclass(frozen=True)
class CableDesign:
    strands: np.ndarray
    pretension: np.ndarray


@dataclass(frozen=True)
class DesignMetrics:
    shape_rmse_m: float
    shape_max_abs_m: float
    total_strands: int
    stress_mean_mpa: float
    stress_std_mpa: float
    stress_min_mpa: float
    stress_max_mpa: float
    stress_violation_rms_mpa: float
    stress_violation_max_mpa: float


@dataclass(frozen=True)
class EvaluationResult:
    design: CableDesign
    objective: float
    components: ObjectiveBreakdown
    metrics: DesignMetrics
    cable_ids: tuple[int, ...]
    deck_errors_m: dict[int, float] = field(default_factory=dict)
    cable_stress_mpa: dict[int, float] = field(default_factory=dict)
    staged_result: StagedResult | None = None


class CableDesignEvaluator:
    def __init__(self, problem: CableOptimizationProblem):
        self.problem = problem
        self.layout = CableLayout(problem.n_seg)

    def build_plan(self, strands, pretension):
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension = validate_tension_vector(pretension, self.layout)
        kwargs = self.problem.builder_kwargs()
        kwargs["strands"] = self.layout.as_int_mapping(strands)
        kwargs["pretension"] = self.layout.as_mapping(pretension)
        return build_staged_cantilever(**kwargs)

    def run_solver(self, plan) -> StagedResult:
        if self.problem.backend == "direct":
            return StagedDirectSolver().run(plan)
        if self.problem.backend == "opensees":
            return StagedOpenSeesSolver().run(plan)
        raise ValueError(f"unknown optimization backend: {self.problem.backend!r}")

    def _extract_case(self, result: StagedResult) -> tuple[dict[int, float], dict[int, float]]:
        """从一次 staged 结果末态提取 ``(deck_errors_m, cable_stress_mpa)``。

        单解与批量(:meth:`evaluate_batch`)共用,保证两条路径的后处理逐位一致。
        deck 节点按 x 升序;线形误差 = 末态 uy − 目标线形;索应力换算到 MPa。
        """
        if not result.records:
            raise RuntimeError("staged solver produced no records")
        final = result.records[-1]
        deck_errors: dict[int, float] = {}
        for nid in sorted(result.deck_ids, key=lambda node_id: result.coords[node_id][0]):
            if nid not in final.disp:
                continue
            x = result.coords[nid][0]
            deck_errors[nid] = float(final.disp[nid][1] - self.problem.target_line.uy(nid, x))
        cable_stress_mpa = {
            cid: float(final.cable_stress[cid] / 1e6) for cid in self.layout.cable_ids
        }
        return deck_errors, cable_stress_mpa

    def evaluate_batch(
        self, strands, tension_matrix
    ) -> list[tuple[dict[int, float], dict[int, float]]]:
        """同一 strands、多预张力工况的批量前向评估。

        ``tension_matrix`` 形状 ``(m, K)``,每列一个预张力工况;返回每列的
        ``(deck_errors_m, cable_stress_mpa)``。``direct`` 后端走
        :class:`StagedDirectBatchSolver`(每施工阶段刚度只分解一次,多右端共享),
        与逐列 :meth:`evaluate` 机器精度一致;其他后端回退为按列循环 :meth:`evaluate`。
        服务 :func:`bridgezoo.optim.linear.build_affine_model`。
        """
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        tension_matrix = np.asarray(tension_matrix, dtype=float)
        if tension_matrix.ndim != 2 or tension_matrix.shape[0] != self.layout.size:
            raise ValueError(
                f"tension_matrix must have shape (m={self.layout.size}, K); got {tension_matrix.shape}"
            )
        ncase = tension_matrix.shape[1]
        plans = [self.build_plan(strands, tension_matrix[:, k]) for k in range(ncase)]
        if self.problem.backend == "direct":
            results = StagedDirectBatchSolver().run_batch(plans)
            return [self._extract_case(r) for r in results]
        return [self._extract_case(self.run_solver(plan)) for plan in plans]

    def evaluate(self, strands, pretension, *, keep_result: bool = False) -> EvaluationResult:
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension = validate_tension_vector(pretension, self.layout)
        plan = self.build_plan(strands, pretension)
        result = self.run_solver(plan)
        deck_errors, cable_stress_mpa = self._extract_case(result)

        err = np.asarray(list(deck_errors.values()), dtype=float)
        shape_rmse = float(np.sqrt(np.mean(err * err))) if err.size else 0.0
        shape_max = float(np.max(np.abs(err))) if err.size else 0.0

        stress = np.asarray([cable_stress_mpa[cid] for cid in self.layout.cable_ids], dtype=float)
        violations = stress_violation_mpa(stress, self.problem.bounds)
        stress_std = float(np.std(stress))
        metrics = DesignMetrics(
            shape_rmse_m=shape_rmse,
            shape_max_abs_m=shape_max,
            total_strands=int(np.sum(strands)),
            stress_mean_mpa=float(np.mean(stress)),
            stress_std_mpa=stress_std,
            stress_min_mpa=float(np.min(stress)),
            stress_max_mpa=float(np.max(stress)),
            stress_violation_rms_mpa=float(np.sqrt(np.mean(violations * violations))),
            stress_violation_max_mpa=float(np.max(violations)),
        )
        components = objective_breakdown(
            shape_rmse_m=metrics.shape_rmse_m,
            total_strands=metrics.total_strands,
            stress_std_mpa=metrics.stress_std_mpa,
            stress_violation_rms_mpa=metrics.stress_violation_rms_mpa,
            weights=self.problem.weights,
        )
        return EvaluationResult(
            design=CableDesign(strands=strands.copy(), pretension=pretension.copy()),
            objective=components.total,
            components=components,
            metrics=metrics,
            cable_ids=self.layout.cable_ids,
            deck_errors_m=deck_errors,
            cable_stress_mpa=cable_stress_mpa,
            staged_result=result if keep_result else None,
        )

    def safe_objective(self, strands, pretension) -> float:
        try:
            return self.evaluate(strands, pretension).objective
        except (ValueError, RuntimeError, FloatingPointError):
            return 1.0e30
