"""Cable-design evaluation for the 3D single-tower staged bridge.

The optimizer retains the established stage-major ``2 * n_seg`` vector shape,
but each pair is interpreted as ``(backstay group, main-stay group)``.  A group
controls the two symmetric physical cables in the transverse cable planes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from bridgezoo.fem.single_staged import (
    SingleStaged3DConfig,
    SingleStagedDirectBatchSolver3D,
    SingleStagedDirectSolver3D,
    SingleStagedOpenSeesSolver3D,
    StagedResult3D,
    build_single_staged_3d,
)
from bridgezoo.optim.evaluator import (
    CableDesign,
    DesignMetrics,
    EvaluationResult,
)
from bridgezoo.optim.objectives import objective_breakdown, stress_violation_mpa
from bridgezoo.optim.problem import CableOptimizationProblem
from bridgezoo.optim.variables import (
    CableLayout,
    validate_ratio_vector,
    validate_strand_vector,
    validate_tension_vector,
)

_AffineCase = tuple[
    dict[int, float],
    dict[int, float],
    float,
    dict[int, float],
]


@dataclass(frozen=True)
class EvaluationResult3D(EvaluationResult):
    """Established optimization result plus 3D physical-cable traceability."""

    cable_group_members: dict[int, tuple[int, ...]] = field(default_factory=dict)
    physical_cable_stress_mpa: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class StageAControlResponse3D:
    """Local displacement targets used to set one stage's A/B split."""

    construction_stage: int
    stage_index: int
    backstay_tower_dx_m: float
    main_stay_deck_uz_m: float


class CableDesignEvaluator3D:
    """Evaluate symmetric 3D cable groups with either linear 3D backend.

    The objective is evaluated on the completed bridge, but ``solve_stage``
    replays every preceding construction substep so wet-deck loading and
    stress-free composite activation remain path-dependent.
    """

    def __init__(
        self,
        problem: CableOptimizationProblem,
        config: SingleStaged3DConfig,
    ):
        if problem.n_seg != config.n_seg:
            raise ValueError(
                f"optimization n_seg={problem.n_seg} does not match 3D config n_seg={config.n_seg}"
            )
        if not np.isclose(problem.strand_area, config.strand_area):
            raise ValueError(
                "optimization strand_area must match the physical 3D bridge config"
            )
        if problem.model_family != "single_staged_3d":
            raise ValueError("3D evaluator requires model_family='single_staged_3d'")
        self.problem = problem
        self.config = config
        self.layout = CableLayout(problem.n_seg)

    def default_pretension_a_ratio(self) -> np.ndarray:
        """Return the bridge-config A ratios in optimizer group order."""

        value = self.config.pretension_a_ratio
        if isinstance(value, (int, float)):
            flat = [value, value] * self.problem.n_seg
        else:
            raw = list(value)
            if len(raw) == self.problem.n_seg:
                flat = []
                for item in raw:
                    if isinstance(item, (tuple, list)):
                        if len(item) != 2:
                            raise ValueError(
                                "3D pretension coefficient stage pairs must contain two values"
                            )
                        flat.extend(item)
                    else:
                        flat.extend((item, item))
            elif len(raw) == self.layout.size:
                flat = raw
            else:
                raise ValueError("3D pretension coefficients do not match n_seg")
        return validate_ratio_vector(flat, self.layout)

    def build_plan(self, strands, pretension, pretension_a_ratio=None):
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension = validate_tension_vector(pretension, self.layout)
        ratios = (
            self.default_pretension_a_ratio()
            if pretension_a_ratio is None
            else validate_ratio_vector(pretension_a_ratio, self.layout)
        )
        strand_pairs = tuple(
            (int(backstay), int(main_stay))
            for backstay, main_stay in self.layout.stage_pairs(strands)
        )
        tension_pairs = tuple(self.layout.stage_pairs(pretension))
        ratio_pairs = tuple(self.layout.stage_pairs(ratios))
        config = replace(
            self.config,
            strands_per_cable=strand_pairs,
            pretension_per_cable=tension_pairs,
            pretension_a_ratio=ratio_pairs,
        )
        return build_single_staged_3d(config)

    def evaluate_stage_a(
        self,
        strands,
        pretension,
        pretension_a_ratio,
        construction_stage: int,
    ) -> StageAControlResponse3D:
        """Evaluate only the two local targets at one ``steel_and_A`` substage.

        The backstay coefficient controls the longitudinal displacement at its
        tower attachment.  The main-stay coefficient controls the mean vertical
        displacement of the two symmetric girder attachment nodes.  No B-stage
        load, completed-bridge response, or stress term enters this response.
        """

        if not 1 <= construction_stage <= self.problem.n_seg:
            raise ValueError(
                f"construction_stage must be between 1 and {self.problem.n_seg}"
            )
        plan = self.build_plan(strands, pretension, pretension_a_ratio)
        stage = next(
            item
            for item in plan.stages
            if item.construction_stage == construction_stage
            and item.phase == "steel_and_A"
        )
        if self.problem.backend == "direct":
            solver = SingleStagedDirectSolver3D()
        elif self.problem.backend == "opensees":
            solver = SingleStagedOpenSeesSolver3D()
        else:
            raise ValueError(f"unknown 3D optimization backend: {self.problem.backend!r}")
        record = solver.solve_stage(plan, stage.index, stage.label)
        if not record.converged:
            raise RuntimeError(
                f"{solver.name} did not converge for 3D substage {stage.label!r}"
            )
        return self.extract_stage_a_response(plan, record, construction_stage)

    def extract_stage_a_response(
        self,
        plan,
        record,
        construction_stage: int,
    ) -> StageAControlResponse3D:
        """Extract local A-stage targets from an already solved stage record."""

        backstays = [
            cable
            for cable in plan.model.cables.values()
            if cable.construction_stage == construction_stage
            and cable.group == "backstay"
        ]
        main_stays = [
            cable
            for cable in plan.model.cables.values()
            if cable.construction_stage == construction_stage
            and cable.group == "main_stay"
        ]
        if len(backstays) != 2 or len(main_stays) != 2:
            raise RuntimeError(
                f"3D stage {construction_stage} must contain two backstays and two main stays"
            )
        tower_nodes = {cable.i for cable in backstays}
        deck_nodes = {cable.j for cable in main_stays}
        return StageAControlResponse3D(
            construction_stage=construction_stage,
            stage_index=record.stage_index,
            backstay_tower_dx_m=float(
                np.mean([record.displacement[node_id][0] for node_id in tower_nodes])
            ),
            main_stay_deck_uz_m=float(
                np.mean([record.displacement[node_id][2] for node_id in deck_nodes])
            ),
        )

    def run_solver(self, plan, *, record_all: bool = False) -> StagedResult3D:
        if self.problem.backend == "direct":
            solver = SingleStagedDirectSolver3D()
        elif self.problem.backend == "opensees":
            solver = SingleStagedOpenSeesSolver3D()
        else:
            raise ValueError(f"unknown 3D optimization backend: {self.problem.backend!r}")
        if record_all:
            result = solver.run(plan)
        else:
            stage = plan.final_stage
            record = solver.solve_stage(plan, stage.index, stage.label)
            result = StagedResult3D(backend=solver.name, records=[record])
        if not result.final.converged:
            raise RuntimeError(
                f"{solver.name} did not converge for completed 3D stage "
                f"{plan.final_stage.label!r}"
            )
        return result

    def _cable_group_members(self, plan) -> dict[int, tuple[int, ...]]:
        groups: dict[int, tuple[int, ...]] = {}
        for stage in range(1, self.problem.n_seg + 1):
            for group_id, group_name in (
                (1000 + stage, "backstay"),
                (2000 + stage, "main_stay"),
            ):
                members = tuple(
                    sorted(
                        cable.id
                        for cable in plan.model.cables.values()
                        if cable.construction_stage == stage and cable.group == group_name
                    )
                )
                if len(members) != 2:
                    raise RuntimeError(
                        f"3D cable group {group_id} expected two transverse members; got {members}"
                    )
                groups[group_id] = members
        return groups

    def _extract_case(self, plan, result: StagedResult3D) -> _AffineCase:
        if not result.records:
            raise RuntimeError("3D staged solver produced no records")
        final = result.final
        model = plan.model

        deck_nodes = sorted(
            (
                node
                for node in model.nodes.values()
                if node.role.startswith("main_girder") and node.id in final.displacement
            ),
            key=lambda node: (node.x, node.y, node.id),
        )
        deck_errors = {
            node.id: float(
                final.displacement[node.id][2]
                - self.problem.target_line.uy(node.id, node.x)
            )
            for node in deck_nodes
        }

        group_members = self._cable_group_members(plan)
        cable_stress_mpa = {
            group_id: float(
                np.mean([final.cable_stress[member] for member in members]) / 1.0e6
            )
            for group_id, members in group_members.items()
        }

        tower_nodes = [
            node
            for node in model.nodes.values()
            if node.role in {"tower", "tower_anchor"} and node.id in final.displacement
        ]
        if not tower_nodes:
            raise RuntimeError("3D staged solver produced no displaced tower nodes")
        tower_top = max(tower_nodes, key=lambda node: node.z)
        tower_top_dx_m = float(final.displacement[tower_top.id][0])

        anchor_nodes = sorted(
            (
                node
                for node in model.nodes.values()
                if node.role == "tower_anchor" and node.id in final.displacement
            ),
            key=lambda node: (node.z, node.id),
        )
        if not anchor_nodes:
            raise RuntimeError("3D staged solver produced no displaced tower anchor nodes")
        tower_anchor_dx_m = {
            node.id: float(final.displacement[node.id][0]) for node in anchor_nodes
        }
        return (
            deck_errors,
            cable_stress_mpa,
            tower_top_dx_m,
            tower_anchor_dx_m,
        )

    def evaluate_affine_batch(
        self,
        strands,
        tension_matrix,
        *,
        include_anchor_displacements: bool = False,
    ):
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        tension_matrix = np.asarray(tension_matrix, dtype=float)
        if tension_matrix.ndim != 2 or tension_matrix.shape[0] != self.layout.size:
            raise ValueError(
                f"tension_matrix must have shape (m={self.layout.size}, K); "
                f"got {tension_matrix.shape}"
            )
        plans = [
            self.build_plan(strands, tension_matrix[:, column])
            for column in range(tension_matrix.shape[1])
        ]
        if self.problem.backend == "direct":
            stage = plans[0].final_stage
            records = SingleStagedDirectBatchSolver3D().solve_stage_batch(
                plans,
                stage.index,
                stage.label,
            )
            if not all(record.converged for record in records):
                raise RuntimeError(
                    f"direct3d batch did not converge for completed stage {stage.label!r}"
                )
            results = [
                StagedResult3D(backend="direct3d", records=[record])
                for record in records
            ]
            cases = [
                self._extract_case(plan, result)
                for plan, result in zip(plans, results)
            ]
        else:
            cases = [
                self._extract_case(plan, self.run_solver(plan))
                for plan in plans
            ]
        if include_anchor_displacements:
            return cases
        return [(deck, stress, tower) for deck, stress, tower, _ in cases]

    def evaluate_batch(self, strands, tension_matrix):
        cases = self.evaluate_affine_batch(strands, tension_matrix)
        return [(deck, stress) for deck, stress, _ in cases]

    def evaluate(
        self,
        strands,
        pretension,
        pretension_a_ratio=None,
        *,
        keep_result: bool = False,
    ) -> EvaluationResult3D:
        strands = validate_strand_vector(
            strands,
            self.layout,
            self.problem.bounds.strand_min,
            self.problem.bounds.strand_max,
        )
        pretension = validate_tension_vector(pretension, self.layout)
        ratios = (
            self.default_pretension_a_ratio()
            if pretension_a_ratio is None
            else validate_ratio_vector(pretension_a_ratio, self.layout)
        )
        plan = self.build_plan(strands, pretension, ratios)
        result = self.run_solver(plan, record_all=keep_result)
        (
            deck_errors,
            cable_stress_mpa,
            tower_top_dx_m,
            tower_anchor_dx_m,
        ) = self._extract_case(plan, result)

        errors = np.asarray(list(deck_errors.values()), dtype=float)
        stress = np.asarray(
            [cable_stress_mpa[group_id] for group_id in self.layout.cable_ids],
            dtype=float,
        )
        anchor_dx = np.asarray(list(tower_anchor_dx_m.values()), dtype=float)
        violations = stress_violation_mpa(stress, self.problem.bounds)
        metrics = DesignMetrics(
            shape_rmse_m=float(np.sqrt(np.mean(errors * errors))) if errors.size else 0.0,
            shape_max_abs_m=float(np.max(np.abs(errors))) if errors.size else 0.0,
            total_strands=2 * int(np.sum(strands)),
            stress_mean_mpa=float(np.mean(stress)),
            stress_std_mpa=float(np.std(stress)),
            stress_min_mpa=float(np.min(stress)),
            stress_max_mpa=float(np.max(stress)),
            stress_violation_rms_mpa=float(
                np.sqrt(np.mean(violations * violations))
            ),
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
        group_members = self._cable_group_members(plan)
        physical_stress = {
            cable_id: float(stress_pa / 1.0e6)
            for cable_id, stress_pa in result.final.cable_stress.items()
        }
        return EvaluationResult3D(
            design=CableDesign(
                strands=strands.copy(),
                pretension=pretension.copy(),
                pretension_a_ratio=ratios.copy(),
            ),
            objective=components.total,
            components=components,
            metrics=metrics,
            cable_ids=self.layout.cable_ids,
            deck_errors_m=deck_errors,
            cable_stress_mpa=cable_stress_mpa,
            staged_result=result if keep_result else None,
            tower_anchor_dx_m=tower_anchor_dx_m,
            cable_group_members=group_members,
            physical_cable_stress_mpa=physical_stress,
        )

    def safe_objective(self, strands, pretension, pretension_a_ratio=None) -> float:
        try:
            return self.evaluate(strands, pretension, pretension_a_ratio).objective
        except (ValueError, RuntimeError, FloatingPointError):
            return 1.0e30


__all__ = [
    "CableDesignEvaluator3D",
    "EvaluationResult3D",
    "StageAControlResponse3D",
]
