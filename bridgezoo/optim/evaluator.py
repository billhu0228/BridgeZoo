"""Forward evaluation for cable strand and pretension designs."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from bridgezoo.fem.staged.plan import StagedResult
from bridgezoo.optim.objectives import ObjectiveBreakdown, objective_breakdown, stress_violation_mpa
from bridgezoo.optim.problem import CableOptimizationProblem
from bridgezoo.optim.variables import CableLayout, validate_strand_vector, validate_tension_vector

_AffineCase = tuple[dict[int, float], dict[int, float], float]
_AffineCaseWithAnchors = tuple[
    dict[int, float],
    dict[int, float],
    float,
    dict[int, float],
]


@dataclass(frozen=True)
class CableDesign:
    strands: np.ndarray
    pretension: np.ndarray
    # 3D staged designs may split each group's total pretension between
    # substage A and B.  The 2D optimizers deliberately leave this unset.
    pretension_a_ratio: np.ndarray | None = None


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
    tower_top_dx_m: float = 0.0
    tower_anchor_dx_rmse_m: float = 0.0


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
    tower_anchor_dx_m: dict[int, float] = field(default_factory=dict)


class CableDesignEvaluator:
    def __init__(self, problem: CableOptimizationProblem):
        self.problem = problem
        self.layout = CableLayout(problem.n_seg)
        self._builder, self._direct_solver, self._batch_solver, self._opensees_solver = _model_api(
            problem.model_family
        )

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
        return self._builder(**kwargs)

    def run_solver(self, plan) -> StagedResult:
        if self.problem.backend == "direct":
            return self._direct_solver().run(plan)
        if self.problem.backend == "opensees":
            return self._opensees_solver().run(plan)
        raise ValueError(f"unknown optimization backend: {self.problem.backend!r}")

    def _extract_case(
        self, result: StagedResult
    ) -> _AffineCaseWithAnchors:
        """从一次 staged 结果末态提取线形、索应力、塔顶及塔锚水平位移。

        单解与批量(:meth:`evaluate_batch`)共用,保证两条路径的后处理逐位一致。
        deck 节点按 x 升序;线形误差 = 末态 uy − 目标线形;索应力换算到 MPa;
        塔顶取 ``tower_ids`` 中设计高程最高的节点;塔锚取全部 ``anchor_ids``。
        两者的目标水平位移均为 0。
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
        tower_nodes = [
            nid for nid in result.tower_ids if nid in result.coords and nid in final.disp
        ]
        if not tower_nodes:
            raise RuntimeError("staged solver produced no displaced tower nodes")
        tower_top = max(tower_nodes, key=lambda nid: result.coords[nid][1])
        tower_top_dx_m = float(final.disp[tower_top][0])
        anchor_nodes = [
            nid for nid in result.anchor_ids if nid in result.coords and nid in final.disp
        ]
        if not anchor_nodes:
            raise RuntimeError("staged solver produced no displaced tower anchor nodes")
        tower_anchor_dx_m = {
            nid: float(final.disp[nid][0])
            for nid in sorted(anchor_nodes, key=lambda nid: result.coords[nid][1])
        }
        return deck_errors, cable_stress_mpa, tower_top_dx_m, tower_anchor_dx_m

    def evaluate_affine_batch(
        self, strands, tension_matrix, *, include_anchor_displacements: bool = False
    ) -> list[_AffineCase | _AffineCaseWithAnchors]:
        """批量提取仿射模型所需的线形、索应力及塔水平位移。

        默认维持原有三元组返回值;内部构造完整仿射模型时通过
        ``include_anchor_displacements=True`` 追加逐塔锚水平位移字典。
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
            results = self._batch_solver().run_batch(plans)
            cases = [self._extract_case(result) for result in results]
        else:
            cases = [self._extract_case(self.run_solver(plan)) for plan in plans]
        if include_anchor_displacements:
            return cases
        return [(deck, stress, tower) for deck, stress, tower, _ in cases]

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
        cases = self.evaluate_affine_batch(strands, tension_matrix)
        return [(deck_errors, cable_stress) for deck_errors, cable_stress, _ in cases]

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
        (
            deck_errors,
            cable_stress_mpa,
            tower_top_dx_m,
            tower_anchor_dx_m,
        ) = self._extract_case(result)

        err = np.asarray(list(deck_errors.values()), dtype=float)
        shape_rmse = float(np.sqrt(np.mean(err * err))) if err.size else 0.0
        shape_max = float(np.max(np.abs(err))) if err.size else 0.0

        stress = np.asarray([cable_stress_mpa[cid] for cid in self.layout.cable_ids], dtype=float)
        anchor_dx = np.asarray(list(tower_anchor_dx_m.values()), dtype=float)
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
            tower_top_dx_m=tower_top_dx_m,
            tower_anchor_dx_rmse_m=float(np.sqrt(np.mean(anchor_dx * anchor_dx))),
        )
        components = objective_breakdown(
            shape_rmse_m=metrics.shape_rmse_m,
            total_strands=metrics.total_strands,
            stress_std_mpa=metrics.stress_std_mpa,
            stress_violation_rms_mpa=metrics.stress_violation_rms_mpa,
            weights=self.problem.weights,
            tower_top_dx_m=metrics.tower_top_dx_m,
            tower_anchor_dx_rmse_m=metrics.tower_anchor_dx_rmse_m,
        )
        return EvaluationResult(
            design=CableDesign(strands=strands.copy(), pretension=pretension.copy()),
            objective=components.total,
            components=components,
            metrics=metrics,
            cable_ids=self.layout.cable_ids,
            deck_errors_m=deck_errors,
            cable_stress_mpa=cable_stress_mpa,
            tower_anchor_dx_m=tower_anchor_dx_m,
            staged_result=result if keep_result else None,
        )

    def safe_objective(self, strands, pretension) -> float:
        try:
            return self.evaluate(strands, pretension).objective
        except (ValueError, RuntimeError, FloatingPointError):
            return 1.0e30


def _model_api(model_family: str):
    """Resolve the staged model family without coupling their implementations."""

    if model_family == "staged":
        from bridgezoo.fem.staged import (
            StagedDirectBatchSolver,
            StagedDirectSolver,
            StagedOpenSeesSolver,
            build_staged_cantilever,
        )

        return build_staged_cantilever, StagedDirectSolver, StagedDirectBatchSolver, StagedOpenSeesSolver
    if model_family == "single_staged":
        from bridgezoo.fem.single_staged import (
            StagedDirectBatchSolver,
            StagedDirectSolver,
            StagedOpenSeesSolver,
            build_staged_cantilever,
        )

        return build_staged_cantilever, StagedDirectSolver, StagedDirectBatchSolver, StagedOpenSeesSolver
    raise ValueError(f"unknown optimization model family: {model_family!r}")
