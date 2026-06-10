import numpy as np
import pytest

from bridgezoo.optim import (
    CableBounds,
    CableDesign,
    CableDesignEvaluator,
    CableHybridOptimizer,
    CableLayout,
    CableOptimizationProblem,
    ContinuousOptions,
    DesignMetrics,
    EvaluationResult,
    FixedStrandTensionOptimizer,
    HybridOptions,
    IntegerSearchOptions,
    LinearTensionOptimizer,
    ObjectiveWeights,
    build_affine_model,
)
from bridgezoo.optim.objectives import ObjectiveBreakdown


def _problem(n=2, bounds=None):
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
        bounds=bounds or CableBounds(strand_min=8, strand_max=40),
        weights=ObjectiveWeights(stress_violation=100.0),
    )


def _fake_evaluation(problem, strands, stresses_mpa) -> EvaluationResult:
    """构造仅含 resize 所需字段的合成评价结果(其余字段填占位值)。"""
    layout = CableLayout(problem.n_seg)
    return EvaluationResult(
        design=CableDesign(strands=np.asarray(strands, dtype=int), pretension=np.zeros(layout.size)),
        objective=0.0,
        components=ObjectiveBreakdown(0.0, 0.0, 0.0, 0.0),
        metrics=DesignMetrics(0.0, 0.0, int(np.sum(strands)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        cable_ids=layout.cable_ids,
        cable_stress_mpa={cid: float(s) for cid, s in zip(layout.cable_ids, stresses_mpa)},
    )


def test_cable_design_evaluator_reports_metrics():
    problem = _problem()
    evaluator = CableDesignEvaluator(problem)
    strands = np.array([20, 18, 22, 16])
    pretension = np.array([2.0e6, 1.8e6, 2.2e6, 1.6e6])

    result = evaluator.evaluate(strands, pretension)

    assert np.isfinite(result.objective)
    assert result.metrics.total_strands == 76
    assert len(result.cable_stress_mpa) == 4
    assert len(result.deck_errors_m) > 0


def test_cable_design_evaluator_rejects_invalid_variables():
    problem = _problem()
    evaluator = CableDesignEvaluator(problem)

    with pytest.raises(ValueError, match="integers"):
        evaluator.evaluate([20.25, 20, 20, 20], [1.0e6] * 4)
    with pytest.raises(ValueError, match="non-negative"):
        evaluator.evaluate([20, 20, 20, 20], [1.0e6, -1.0, 1.0e6, 1.0e6])


def test_fixed_strand_tension_optimizer_runs():
    pytest.importorskip("scipy")
    problem = _problem()
    evaluator = CableDesignEvaluator(problem)
    optimizer = FixedStrandTensionOptimizer(evaluator, ContinuousOptions(maxiter=3, ftol=1.0e-6))

    result = optimizer.optimize([20, 20, 20, 20])

    assert np.isfinite(result.evaluation.objective)
    assert np.all(result.evaluation.design.pretension >= 0.0)
    assert np.all(result.evaluation.design.strands == np.array([20, 20, 20, 20]))


def test_hybrid_optimizer_keeps_integer_strands_and_nonnegative_tension():
    pytest.importorskip("scipy")
    problem = _problem()
    optimizer = CableHybridOptimizer(
        problem,
        HybridOptions(
            continuous=ContinuousOptions(maxiter=2, ftol=1.0e-6),
            integer=IntegerSearchOptions(outer_iterations=1, random_trials=0),
        ),
    )

    result = optimizer.optimize(initial_strands=[20, 20, 20, 20])

    assert len(result.history) >= 1
    assert np.isfinite(result.best.objective)
    assert np.all(result.best.design.pretension >= 0.0)
    assert np.all(result.best.design.strands == np.rint(result.best.design.strands))


def test_hybrid_strand_moves_follow_stress_direction():
    problem = _problem()
    optimizer = CableHybridOptimizer(problem)
    evaluator = CableDesignEvaluator(problem)
    # 高侧取 7e6 N:合龙改为"切线锁定支座"后终态应力重分布,5e6 N 已不再越上限
    # (1200 MPa);7e6 N 时 1 号索终态约 1400 MPa,稳定触发"应力过高"分支。
    low = evaluator.evaluate([20, 20, 20, 20], [1.0e5, 2.0e6, 2.0e6, 2.0e6])
    high = evaluator.evaluate([20, 20, 20, 20], [7.0e6, 2.0e6, 2.0e6, 2.0e6])

    assert optimizer._strand_moves_for(low, 0)[0][0] < 0
    assert optimizer._strand_moves_for(high, 0)[0][0] > 0


def test_default_bounds_and_continuous_method():
    # 用户决定:索股默认下限 1;连续层默认走线性模型路径。
    assert CableBounds().strand_min == 1
    assert ContinuousOptions().method == "linear"


def test_affine_model_matches_fem():
    """线性后端下仿射模型 σ=σ0+M·T、err=err0+D·T 应逐项精确(数值噪声内)。"""
    problem = _problem()
    evaluator = CableDesignEvaluator(problem)
    strands = np.array([20, 18, 22, 16])
    model = build_affine_model(evaluator, strands)

    rng = np.random.default_rng(42)
    tension = rng.uniform(0.5e6, 3.0e6, size=4)
    ev = evaluator.evaluate(strands, tension)
    sigma_fem = np.asarray([ev.cable_stress_mpa[cid] for cid in model.cable_ids])
    err_fem = np.asarray([ev.deck_errors_m[nid] for nid in model.deck_nodes])
    # 模型对线性求解器精确,容差只覆盖浮点累计噪声(应力 ~1e3 MPa、线形 ~0.1 m 量级)。
    assert float(np.max(np.abs(model.stress_mpa(tension) - sigma_fem))) < 1e-6
    assert float(np.max(np.abs(model.deck_err_m(tension) - err_fem))) < 1e-9


def test_linear_optimizer_finds_feasible_band_when_reachable():
    pytest.importorskip("scipy")
    # 该几何 + 每索 8 股时 [800,1200] MPa 可行(LP 验证 s*=0,σ∈[800,918])。
    problem = _problem()
    result = LinearTensionOptimizer(CableDesignEvaluator(problem)).optimize([8, 8, 8, 8])

    assert result.feasibility_violation_mpa == pytest.approx(0.0, abs=1e-6)
    lower, upper = problem.bounds.stress_lower_mpa, problem.bounds.stress_upper_mpa
    for sigma in result.evaluation.cable_stress_mpa.values():
        # 1e-3 MPa 余量覆盖 LP/SLSQP 收敛容差
        assert lower - 1e-3 <= sigma <= upper + 1e-3


def test_linear_optimizer_attains_lp_violation_bound():
    pytest.importorskip("scipy")
    # 窄带 [950,960] + 每索 12 股不可行(s*≈11.7 MPa)。回归:旧 SLSQP 路径在
    # 不可行问题上因硬约束不相容而停在远高于下界的违反量;线性路径的终解
    # 最大违反必须压到 LP 下界 s*。
    problem = _problem(
        bounds=CableBounds(strand_min=1, strand_max=40, stress_lower_mpa=950.0, stress_upper_mpa=960.0)
    )
    result = LinearTensionOptimizer(CableDesignEvaluator(problem)).optimize([12, 12, 12, 12])

    s_star = result.feasibility_violation_mpa
    assert s_star is not None and s_star > 1.0  # 确认该配置确实不可行,测试才有意义
    assert result.evaluation.metrics.stress_violation_max_mpa <= s_star + 1e-3


def test_hybrid_resize_candidate_jumps_by_stress_ratio():
    problem = _problem()  # band [800,1200] → σ_target=1000;股数界 [8,40]
    optimizer = CableHybridOptimizer(problem)

    best = _fake_evaluation(problem, [20, 20, 8, 40], [500.0, 2500.0, -10.0, 1000.0])
    candidate = optimizer._resize_candidate(best)
    # 20×500/1000=10;20×2500/1000=50→clip 40;σ≤0→strand_min 8;40×1000/1000 不变
    assert candidate is not None
    assert candidate.tolist() == [10, 40, 8, 40]

    unchanged = _fake_evaluation(problem, [10, 10, 10, 10], [1000.0] * 4)
    assert optimizer._resize_candidate(unchanged) is None


def test_slsqp_constraints_survive_evaluator_failure(monkeypatch):
    # 回归:约束函数遇评估失败(如越界张力)应返回有限"严重违反"值,
    # 而不是让 ValueError 炸掉整个优化过程。
    problem = _problem()
    evaluator = CableDesignEvaluator(problem)
    optimizer = FixedStrandTensionOptimizer(evaluator)

    def boom(*args, **kwargs):
        raise ValueError("synthetic evaluator failure")

    monkeypatch.setattr(evaluator, "evaluate", boom)
    margins = optimizer._stress_margins(np.array([20, 20, 20, 20]), np.full(4, 1.0e6), {}, upper=False)

    assert margins.shape == (4,)
    assert np.all(np.isfinite(margins))
    assert np.all(margins < 0.0)
