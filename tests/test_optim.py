import io
import json

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
from scripts import optimize_cables


class _TTYBuffer(io.StringIO):
    def isatty(self):
        return True


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds=1.0):
        self.now += seconds


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


def _single_problem(n=2):
    return CableOptimizationProblem(
        n_seg=n,
        model_kwargs={
            "anchor_base_height": 20.0,
            "anchor_spacing": 3.0,
            "anchor_top_free": 5.0,
            "left_start": 6.0,
            "left_spacing": 8.0,
            "left_end": 4.0,
            "right_start": 8.0,
            "right_spacing": 2.0,
            "right_end": 4.0,
            "right_fix": 3.0,
            "left_span": 5.0,
            "wg": 5.0e4,
            "dw": 2.0e4,
        },
        bounds=CableBounds(
            strand_min=8,
            strand_max=40,
            stress_lower_mpa=400.0,
            stress_upper_mpa=600.0,
        ),
        weights=ObjectiveWeights(stress_violation=100.0),
        model_family="single_staged",
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


@pytest.mark.parametrize("model_family", ["staged", "single_staged"])
def test_summary_lists_final_strand_counts_from_left_to_right(tmp_path, model_family):
    problem = _problem()
    best = _fake_evaluation(problem, [20, 18, 22, 16], [500.0] * 4)

    optimize_cables._write_outputs(
        tmp_path,
        best,
        [best],
        bridge_yaml="bridge.yaml",
        model_family=model_family,
    )

    summary = (tmp_path / "summary.txt").read_text(encoding="utf-8")
    assert "final strand counts (left to right): 16, 18, 20, 22\n" in summary


def test_resume_loader_accepts_legacy_design_without_search_metadata(tmp_path):
    problem = _problem()
    best = _fake_evaluation(problem, [20, 18, 22, 16], [500.0] * 4)
    optimize_cables._write_outputs(
        tmp_path,
        best,
        [best],
        bridge_yaml="bridge.yaml",
        model_family="staged",
    )

    state = optimize_cables._load_resume_state(
        tmp_path / "best_design.json",
        problem=problem,
        bridge_yaml="bridge.yaml",
        problem_metadata=optimize_cables._problem_metadata(problem),
    )

    assert np.array_equal(state.strands, [20, 18, 22, 16])
    assert np.array_equal(state.pretension, np.zeros(4))
    assert len(state.history_rows) == 1
    assert state.prior_budget_known is False


def test_progress_display_redraws_dashboard_with_progress_and_trend():
    stream = _TTYBuffer()
    clock = _Clock()
    display = optimize_cables._OptimizationProgressDisplay(
        total_outer=2,
        total_cables=4,
        refresh_interval=0.5,
        stream=stream,
        clock=clock,
    )
    messages = [
        "start cable optimization: n_seg=2, cables=4, outer_iterations=2, random_trials=0",
        "  optimize tensions: total_strands=800 min=200 max=200",
        "    linear LP: s*=12.500 MPa (band unreachable for these strands)",
        (
            "    linear QP done: success=False objective=10 "
            "shape_rmse=250.000 mm stress=[387.5, 612.5] MPa"
        ),
        "initial best: objective=10 total_strands=800",
        "outer iteration 1/2",
        "resize candidate: total_strands 800 -> 720 (stress-ratio jump)",
        "  optimize tensions: total_strands=720 min=170 max=190",
        (
            "    linear QP done: success=True objective=4 "
            "shape_rmse=120.000 mm stress=[400.0, 600.0] MPa"
        ),
        "accepted resize: 10 -> 4 s*=0.000 MPa",
        "candidate cable=2001 strand_delta=-1 180->179 (stress ok)",
        "finished: objective=3.5 total_strands=710 shape_rmse=110.000 mm "
        "stress=[410.0, 590.0] MPa LP bound s*=0.000 MPa",
    ]
    for message in messages:
        clock.advance()
        display.update(message)
    display.close()

    output = stream.getvalue()
    assert "\x1b[9F" in output
    assert "outer 1/2 | cable 2/4" in output
    assert "cable 2001: 180 → 179 strands (-1)" in output
    assert "shape RMSE 110 mm" in output
    assert "Trend    10 → 4 → 3.5" in output
    assert "candidate cable=" not in output


def test_progress_display_uses_sparse_checkpoints_when_not_a_terminal():
    stream = io.StringIO()
    clock = _Clock()
    display = optimize_cables._OptimizationProgressDisplay(
        total_outer=2,
        total_cables=4,
        stream=stream,
        clock=clock,
    )
    display.update("start cable optimization: n_seg=2, cables=4")
    display.update("candidate cable=1001 strand_delta=-1 200->199 (stress ok)")
    display.update("  optimize tensions: total_strands=799 min=199 max=200")
    display.update("outer iteration 1/2")
    display.update("candidate cable=2001 strand_delta=-1 200->199 (stress ok)")
    display.update("finished: objective=1 total_strands=798 shape_rmse=10 mm stress=[450, 550] MPa")

    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith("[   0.0s] initial design:")
    assert "outer=1/2" in lines[1]
    assert lines[2].startswith("[   0.0s] complete:")


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
    assert result.outer_iterations_completed == 1
    assert result.random_trials_completed == 0


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


def test_single_affine_model_matches_full_staged_fem():
    """Single 的锁定、辅助跨与 phase2 仍须保持同一精确仿射优化模型。"""

    problem = _single_problem()
    evaluator = CableDesignEvaluator(problem)
    strands = np.array([20, 18, 22, 16])
    model = build_affine_model(evaluator, strands)
    tension = np.array([1.1e6, 1.7e6, 1.3e6, 1.9e6])
    evaluation = evaluator.evaluate(strands, tension, keep_result=True)

    sigma_fem = np.asarray([evaluation.cable_stress_mpa[cid] for cid in model.cable_ids])
    err_fem = np.asarray([evaluation.deck_errors_m[nid] for nid in model.deck_nodes])
    assert float(np.max(np.abs(model.stress_mpa(tension) - sigma_fem))) < 1e-6
    assert float(np.max(np.abs(model.deck_err_m(tension) - err_fem))) < 1e-9
    assert 202 in model.deck_nodes
    assert evaluation.staged_result.records[-1].label == "phase2"


def test_single_cli_runs_shared_optimizer_and_records_model_family(tmp_path):
    pytest.importorskip("scipy")
    out = tmp_path / "single_opt"
    args = optimize_cables.parse_args([
        "--bridge", "omo",
        "--n", "2",
        "--outer-iterations", "0",
        "--random-trials", "0",
        "--quiet",
        "--out", str(out),
    ])

    result = optimize_cables.run(args)
    payload = json.loads((out / "best_design.json").read_text(encoding="utf-8"))

    assert np.isfinite(result.best.objective)
    assert result.best.components.total == pytest.approx(result.best.objective)
    assert result.best.staged_result.records[-1].label == "phase2"
    assert 202 in result.best.deck_errors_m
    assert payload["model_family"] == "single_staged"
    assert payload["bridge_yaml"] == "scripts/bridges/omo_bridge.yaml"
    assert "model family: single_staged" in (out / "summary.txt").read_text(encoding="utf-8")


def test_cli_resume_continues_design_budget_and_history(tmp_path):
    pytest.importorskip("scipy")
    out = tmp_path / "resume_opt"
    common = [
        "--bridge", "omo",
        "--n", "2",
        "--quiet",
        "--out", str(out),
    ]
    first = optimize_cables.run(optimize_cables.parse_args([
        *common,
        "--outer-iterations", "0",
        "--random-trials", "0",
    ]))
    first_payload = json.loads((out / "best_design.json").read_text(encoding="utf-8"))
    first_strands = np.asarray([cable["strands"] for cable in first_payload["cables"]])
    first_history_count = len((out / "history.csv").read_text(encoding="utf-8").splitlines()) - 1

    resumed = optimize_cables.run(optimize_cables.parse_args([
        *common,
        "--resume",
        "--outer-iterations", "1",
        "--random-trials", "1",
        "--seed", "17",
    ]))
    payload = json.loads((out / "best_design.json").read_text(encoding="utf-8"))
    history_count = len((out / "history.csv").read_text(encoding="utf-8").splitlines()) - 1

    assert np.array_equal(resumed.history[0].design.strands, first_strands)
    assert history_count == first_history_count + len(resumed.history)
    assert payload["search"]["run_index"] == 2
    assert payload["search"]["resumed"] is True
    assert payload["search"]["outer_iterations_completed"] == 1
    assert payload["search"]["outer_iterations_tracked_total"] == 1
    assert payload["search"]["random_trials_completed"] == 1
    assert payload["search"]["random_trials_tracked_total"] == 1
    assert payload["search"]["seed"] == 17
    assert payload["search"]["history_evaluations_previous"] == first_history_count
    assert payload["problem"] == first_payload["problem"]
    assert "search run: 2 (resumed)" in (out / "summary.txt").read_text(encoding="utf-8")
    assert first.outer_iterations_completed == 0

    branch = tmp_path / "resume_branch"
    branched = optimize_cables.run(optimize_cables.parse_args([
        "--bridge", "omo",
        "--n", "2",
        "--resume-from", str(out),
        "--out", str(branch),
        "--outer-iterations", "0",
        "--random-trials", "0",
        "--quiet",
    ]))
    branch_payload = json.loads((branch / "best_design.json").read_text(encoding="utf-8"))
    branch_history_count = len((branch / "history.csv").read_text(encoding="utf-8").splitlines()) - 1
    assert np.array_equal(branched.history[0].design.strands, resumed.best.design.strands)
    assert branch_payload["search"]["run_index"] == 3
    assert branch_payload["search"]["resume_source"] == str((out / "best_design.json").resolve())
    assert branch_history_count == history_count + len(branched.history)


def test_cli_resume_rejects_changed_problem_definition(tmp_path):
    pytest.importorskip("scipy")
    out = tmp_path / "resume_mismatch"
    common = [
        "--bridge", "omo",
        "--n", "2",
        "--outer-iterations", "0",
        "--random-trials", "0",
        "--quiet",
        "--out", str(out),
    ]
    optimize_cables.run(optimize_cables.parse_args(common))

    with pytest.raises(ValueError, match="bounds, or objective weights differ"):
        optimize_cables.run(optimize_cables.parse_args([*common, "--resume", "--strand-min", "101"]))


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


def test_linear_optimizer_matches_exact_lsq_on_shape_only():
    """回归(缩放失速):P4B 量级荷载/刚度下(T~1e7 N),未缩放变量的 SLSQP 曾
    停在离最优 >1e3 倍处。纯线形目标(带约束不活跃)时,优化器目标必须贴合
    同一仿射模型上 lsq_linear 的精确有界最小二乘最优。"""
    pytest.importorskip("scipy")
    from scipy.optimize import lsq_linear

    problem = CableOptimizationProblem(
        n_seg=2,
        model_kwargs={
            "anchor_base_height": 60.0,
            "anchor_spacing": 2.5,
            "anchor_top_free": 5.0,
            "left_start": 18.5,
            "left_spacing": 12.0,
            "left_end": 5.0,
            "right_start": 18.5,
            "right_spacing": 12.0,
            "right_end": 8.0,
            "wg": 3.2e5,
            "dw": 4.0e5,
            "beam_E": 200e9,
            "beam_A": 2.3,
            "beam_Iz": 2.0,
        },
        bounds=CableBounds(
            strand_min=100,
            strand_max=300,
            stress_lower_mpa=0.0,
            stress_upper_mpa=1.0e5,
            tension_bound_stress_mpa=1600.0,
        ),
        weights=ObjectiveWeights(
            shape=1.0, total_strands=0.0, stress_uniform=0.0, stress_violation=0.0,
            shape_scale_m=0.1,
        ),
    )
    evaluator = CableDesignEvaluator(problem)
    strands = np.full(4, 200, dtype=int)
    result = LinearTensionOptimizer(evaluator).optimize(strands)

    model = build_affine_model(evaluator, strands)
    hi = problem.bounds.tension_bound_stress_mpa * 1e6 * problem.strand_area * strands.astype(float)
    ls = lsq_linear(model.d_m_per_n, -model.err0_m, bounds=(np.zeros(4), hi), tol=1e-14)
    resid = model.err0_m + model.d_m_per_n @ ls.x
    exact = float(np.mean(resid * resid)) / problem.weights.shape_scale_m**2

    # lsq_linear 是同一凸模型上的精确下界:优化器不能更低(数值噪声内),也必须
    # 贴近(1e-3 相对 + 1e-8 绝对覆盖 SLSQP 收敛容差;失速回归时差距 >1e3 倍)。
    assert result.evaluation.objective >= exact - 1e-8
    assert result.evaluation.objective <= exact * (1.0 + 1e-3) + 1e-8


def test_hybrid_acceptance_prefers_band_feasibility():
    # 字典序接受:s*(LP 最小可达带违反)显著降低即接受(即使目标变差);
    # s* 持平(band_tol_mpa 内)时回退比较加权目标;s* 缺失(slsqp 路径)仅比目标。
    optimizer = CableHybridOptimizer(_problem())

    assert optimizer._accepts(0.0, 50.0, 5.0, 10.0)  # s* 5→0,目标变差 → 接受
    assert not optimizer._accepts(5.0, 1.0, 0.0, 10.0)  # s* 0→5 → 拒绝(目标更好也不行)
    assert optimizer._accepts(0.0, 9.0, 0.0, 10.0)  # s* 持平,目标改善 → 接受
    assert not optimizer._accepts(0.0, 11.0, 0.0, 10.0)  # s* 持平,目标变差 → 拒绝
    assert optimizer._accepts(None, 9.0, None, 10.0)  # s* 缺失 → 仅比目标


def test_hybrid_acceptance_band_priority_can_be_disabled():
    optimizer = CableHybridOptimizer(
        _problem(),
        HybridOptions(integer=IntegerSearchOptions(band_priority=False)),
    )
    # 关闭 band_priority 后退回旧行为:仅比较加权目标。
    assert optimizer._accepts(5.0, 1.0, 0.0, 10.0)
    assert not optimizer._accepts(0.0, 11.0, 0.0, 10.0)


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
