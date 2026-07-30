"""Mathematical optimization tools for staged cable design."""

from bridgezoo.optim.continuous import ContinuousOptimizationResult, ContinuousOptions, FixedStrandTensionOptimizer
from bridgezoo.optim.evaluator import CableDesign, CableDesignEvaluator, DesignMetrics, EvaluationResult
from bridgezoo.optim.engineering_cycle3d import (
    EngineeringCableCycleOptimizer3D,
    EngineeringCycleOptions3D,
    EngineeringProgress3D,
    EngineeringCycleResult3D,
    EngineeringStageStatus3D,
    EngineeringSubstageControl3D,
)
from bridgezoo.optim.forward_cycle3d import (
    ForwardCableCycleOptimizer3D,
    ForwardCycleOptions3D,
    ForwardCycleResult3D,
    ForwardLocalResponse3D,
    ForwardMilestone3D,
    ForwardSizingError,
    ForwardSubstageResult3D,
    ForwardTuningError,
)
from bridgezoo.optim.hybrid import CableHybridOptimizer, HybridOptimizationResult, HybridOptions, IntegerSearchOptions
from bridgezoo.optim.linear import AffineCableModel, LinearTensionOptimizer, build_affine_model
from bridgezoo.optim.problem import CableBounds, CableOptimizationProblem, ObjectiveWeights, TargetLine
from bridgezoo.optim.single_staged3d import CableDesignEvaluator3D, EvaluationResult3D
from bridgezoo.optim.smooth_curves import (
    CURVE_FAMILIES,
    SmoothStrandCurve3D,
    build_smooth_curve_basis,
    build_stage_major_curve_basis,
    project_strands_to_smooth_curve,
)
from bridgezoo.optim.staged3d_optimizer import (
    SecondaryAffineModel3D,
    SecondaryTensionOptions3D,
    SecondaryTensionResult3D,
    SmoothCurveTrial3D,
    StageAControlOptions,
    StageAControlResult3D,
    Staged3DOptimizationOptions,
    Staged3DOptimizationResult,
    StagedCableOptimizer3D,
    StrandSearchOptions3D,
)
from bridgezoo.optim.variables import CableLayout

__all__ = [
    "AffineCableModel",
    "CableBounds",
    "CableDesign",
    "CableDesignEvaluator",
    "CableDesignEvaluator3D",
    "CableHybridOptimizer",
    "CableLayout",
    "CableOptimizationProblem",
    "CURVE_FAMILIES",
    "ContinuousOptimizationResult",
    "ContinuousOptions",
    "DesignMetrics",
    "EngineeringCableCycleOptimizer3D",
    "EngineeringCycleOptions3D",
    "EngineeringProgress3D",
    "EngineeringCycleResult3D",
    "EngineeringStageStatus3D",
    "EngineeringSubstageControl3D",
    "EvaluationResult",
    "EvaluationResult3D",
    "ForwardCableCycleOptimizer3D",
    "ForwardCycleOptions3D",
    "ForwardCycleResult3D",
    "ForwardLocalResponse3D",
    "ForwardMilestone3D",
    "ForwardSizingError",
    "ForwardSubstageResult3D",
    "ForwardTuningError",
    "SecondaryAffineModel3D",
    "SecondaryTensionOptions3D",
    "SecondaryTensionResult3D",
    "SmoothCurveTrial3D",
    "SmoothStrandCurve3D",
    "FixedStrandTensionOptimizer",
    "HybridOptimizationResult",
    "HybridOptions",
    "IntegerSearchOptions",
    "LinearTensionOptimizer",
    "ObjectiveWeights",
    "StageAControlOptions",
    "StageAControlResult3D",
    "Staged3DOptimizationOptions",
    "Staged3DOptimizationResult",
    "StagedCableOptimizer3D",
    "StrandSearchOptions3D",
    "TargetLine",
    "build_affine_model",
    "build_smooth_curve_basis",
    "build_stage_major_curve_basis",
    "project_strands_to_smooth_curve",
]
