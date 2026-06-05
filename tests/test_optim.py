import numpy as np
import pytest

from bridgezoo.optim import (
    CableBounds,
    CableDesignEvaluator,
    CableHybridOptimizer,
    CableOptimizationProblem,
    ContinuousOptions,
    FixedStrandTensionOptimizer,
    HybridOptions,
    IntegerSearchOptions,
    ObjectiveWeights,
)


def _problem(n=2):
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
    low = evaluator.evaluate([20, 20, 20, 20], [1.0e5, 2.0e6, 2.0e6, 2.0e6])
    high = evaluator.evaluate([20, 20, 20, 20], [5.0e6, 2.0e6, 2.0e6, 2.0e6])

    assert optimizer._strand_moves_for(low, 0)[0][0] < 0
    assert optimizer._strand_moves_for(high, 0)[0][0] > 0
