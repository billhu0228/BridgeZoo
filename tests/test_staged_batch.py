"""批量多右端求解器 :class:`StagedDirectBatchSolver` 与逐个标量 ``run`` 的逐位等价校核。

批量路径(同结构、异预张力/节点力)每个施工阶段只装配+分解一次刚度,K 个右端一次性
回代;由于直接刚度法对载荷线性、刚度与预张力无关,其结果必须与逐个
:meth:`StagedDirectSolver.run` 在机器精度内逐项一致。这是「批量加速不改结果」的护栏。
"""

import numpy as np
import pytest

from bridgezoo.fem.staged import (
    StagedDirectBatchSolver,
    StagedDirectSolver,
)
from bridgezoo.fem.staged.plan import (
    BalanceDof,
    BuildStep,
    NewFrame,
    NewNode,
    NodalLoad,
    StagedPlan,
)
from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator,
    CableLayout,
    CableOptimizationProblem,
    ObjectiveWeights,
)

# 容差说明:批量与标量做的是同一组线性方程,差异仅来自 LU 实现(scipy lu_factor/lu_solve
# vs numpy.linalg.solve)的浮点舍入,量级 ~1e-12 相对误差。取 rtol=1e-9、atol=1e-11
# 远紧于任何物理容差,却为浮点往返留足余量。
_RTOL = 1.0e-9
_ATOL = 1.0e-11


def _problem(n: int = 3) -> CableOptimizationProblem:
    return CableOptimizationProblem(
        n_seg=n,
        model_kwargs={
            "anchor_base_height": 20.0,
            "anchor_spacing": 3.0,
            "left_start": 6.0,
            "left_spacing": 8.0,
            "left_end": 4.0,
            "right_start": 6.0,
            "right_spacing": 8.0,
            "right_end": 4.0,
            "wg": 5.0e4,
        },
        bounds=CableBounds(strand_min=8, strand_max=40),
        weights=ObjectiveWeights(stress_violation=100.0),
    )


def _assert_results_match(scalar, batch) -> None:
    assert len(scalar.records) == len(batch.records)
    for rs, rb in zip(scalar.records, batch.records):
        assert rs.label == rb.label
        assert set(rs.disp) == set(rb.disp)
        for nid in rs.disp:
            assert np.allclose(rs.disp[nid], rb.disp[nid], rtol=_RTOL, atol=_ATOL), (
                f"disp mismatch at node {nid} step {rs.label!r}: {rs.disp[nid]} vs {rb.disp[nid]}"
            )
        assert set(rs.cable_force) == set(rb.cable_force)
        for cid in rs.cable_force:
            assert np.allclose(rs.cable_force[cid], rb.cable_force[cid], rtol=_RTOL, atol=_ATOL)
            assert np.allclose(rs.cable_stress[cid], rb.cable_stress[cid], rtol=_RTOL, atol=_ATOL)


def _affine_plans(evaluator: CableDesignEvaluator, strands: np.ndarray, step_n: float = 1.0e6):
    """复刻 :func:`build_affine_model` 的多右端:T=0 + 逐索单位扰动,共 m+1 个计划。"""
    m = evaluator.layout.size
    plans = [evaluator.build_plan(strands, np.zeros(m))]
    for j in range(m):
        tension = np.zeros(m)
        tension[j] = step_n
        plans.append(evaluator.build_plan(strands, tension))
    return plans


def test_batch_matches_scalar_on_affine_perturbations():
    problem = _problem(n=3)
    evaluator = CableDesignEvaluator(problem)
    strands = np.array([20, 18, 22, 16, 24, 14])  # m = 2*n = 6
    assert strands.size == CableLayout(problem.n_seg).size

    plans = _affine_plans(evaluator, strands)
    batch = StagedDirectBatchSolver().run_batch(plans)
    scalar = [StagedDirectSolver().run(p) for p in plans]

    assert len(batch) == len(plans)
    for rs, rb in zip(scalar, batch):
        _assert_results_match(rs, rb)


def test_batch_matches_scalar_with_nonuniform_tension():
    """更一般的多右端:各工况预张力互不相同(非单位扰动),仍应逐位等价。"""
    problem = _problem(n=4)
    evaluator = CableDesignEvaluator(problem)
    strands = np.full(CableLayout(problem.n_seg).size, 20)
    rng = np.random.default_rng(7)
    tensions = [rng.uniform(0.5e6, 3.0e6, size=strands.size) for _ in range(5)]

    plans = [evaluator.build_plan(strands, t) for t in tensions]
    batch = StagedDirectBatchSolver().run_batch(plans)
    scalar = [StagedDirectSolver().run(p) for p in plans]

    for rs, rb in zip(scalar, batch):
        _assert_results_match(rs, rb)


def _balance_plan(extra_fy: float) -> StagedPlan:
    """手工小算例:悬臂梁 + 末端 uy 平衡自由度(反力步),用于覆盖批量 balance 路径。

    两工况仅 ``NodalLoad`` 不同(结构、平衡自由度一致),据此校验批量反力计算。
    """
    e, a, mi = 2.1e11, 1.0, 1.0e-2
    return StagedPlan(
        name="balance-test",
        init_nodes=[NewNode(0, 0.0, 0.0)],
        supports=[(0, True, True, True)],
        steps=[
            BuildStep(
                label="build",
                new_nodes=[NewNode(1, 5.0, 0.0)],
                new_frames=[NewFrame(10, 0, 1, e, a, mi, udl_wy=-1.0e4)],
                nodal_loads=[NodalLoad(1, fy=extra_fy)],
                record=True,
            ),
            BuildStep(
                label="balance",
                balance_dofs=[BalanceDof(node=1, dof=1, target=0.0)],
                record=True,
            ),
        ],
    )


def test_batch_matches_scalar_with_balance_dofs():
    plans = [_balance_plan(0.0), _balance_plan(-2.0e4)]
    batch = StagedDirectBatchSolver().run_batch(plans)
    scalar = [StagedDirectSolver().run(p) for p in plans]

    for rs, rb in zip(scalar, batch):
        _assert_results_match(rs, rb)
        # 平衡步把末端 uy 拉回 0(反力步生效);两工况都应满足
        assert abs(rb.records[-1].disp[1][1]) < 1e-9
        # 反力(applied_loads)随工况不同而不同——确认确实测到了差异路径
    assert batch[0].records[-1].applied_loads != batch[1].records[-1].applied_loads


def test_run_batch_rejects_mismatched_structure():
    problem = _problem(n=3)
    evaluator = CableDesignEvaluator(problem)
    m = CableLayout(problem.n_seg).size
    p0 = evaluator.build_plan(np.full(m, 20), np.zeros(m))
    p1 = evaluator.build_plan(np.full(m, 18), np.zeros(m))  # 不同索股 → 索面积不同
    with pytest.raises(ValueError, match="new_cables structure"):
        StagedDirectBatchSolver().run_batch([p0, p1])


def test_run_batch_requires_at_least_one_plan():
    with pytest.raises(ValueError, match="at least one plan"):
        StagedDirectBatchSolver().run_batch([])
