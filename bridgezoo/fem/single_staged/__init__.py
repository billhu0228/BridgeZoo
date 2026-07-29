"""单塔逐阶段施工模型 —— "一套定义,两种后端,结果一致"。

``single_staged`` also exposes the detailed 3D architecture.  Its names
carry a ``3D`` suffix so the established 2D optimization API remains available
during migration.  The 3D path uses physical sections, a twin-main-girder
grillage, an eccentric deck-slab plane, three-substep incremental construction
and matching direct/OpenSees backends.

- :mod:`bridgezoo.fem.single_staged.plan` —— 与求解器无关的施工计划 IR + 结果容器 + 共享辅助。
- :mod:`bridgezoo.fem.single_staged.builder` —— 由参数构建 :class:`StagedPlan`
  (独塔左悬臂 + 固定地锚右索)。
- :mod:`bridgezoo.fem.single_staged.direct` —— 自研增量直接刚度法后端(RL 内核,线性小位移)。
- :mod:`bridgezoo.fem.single_staged.opensees` —— OpenSees 后端(切线激活),用于交叉校核。
- :mod:`bridgezoo.fem.single_staged.completed` —— 由施工计划派生的成桥(完成态)模型组装,
  是成桥工况的唯一建模入口(:func:`build_completed_model`)。
- :mod:`bridgezoo.fem.single_staged.sequence` —— RL 环境的施工阶段序列(骨架,M1/M2)。

公共 API 在此重导出,故 ``from bridgezoo.fem.single_staged import StagedDirectSolver`` 等写法保持稳定。
"""

from bridgezoo.fem.single_staged.builder import build_staged_cantilever
from bridgezoo.fem.single_staged.completed import build_completed_model
from bridgezoo.fem.single_staged.direct import StagedDirectBatchSolver, StagedDirectSolver
from bridgezoo.fem.single_staged.opensees import StagedOpenSeesSolver
from bridgezoo.fem.single_staged.plan import (
    BalanceDof,
    BuildStep,
    CompletedState,
    MemberLoad,
    NewCable,
    NewFrame,
    NewNode,
    NodalLoad,
    StagedPlan,
    StagedResult,
    StagedStepRecord,
)
from bridgezoo.fem.single_staged.builder3d import (
    SingleStaged3DConfig,
    build_single_staged_3d,
)
from bridgezoo.fem.single_staged.direct3d import (
    SingleStagedDirectBatchSolver3D,
    SingleStagedDirectSolver3D,
    solve_single_staged_3d,
)
from bridgezoo.fem.single_staged.model3d import (
    BridgeModel3D,
    CableElement3D,
    ConstructionStage3D,
    FrameElement3D,
    FrameLoad3D,
    Node3D,
    RigidLink3D,
    SingleStagedPlan3D,
    SolveResult3D,
    StagedResult3D,
    Support3D,
)
from bridgezoo.fem.single_staged.opensees3d import SingleStagedOpenSeesSolver3D
from bridgezoo.fem.single_staged.sections3d import (
    CABLE_STEEL,
    CONCRETE_C50,
    STEEL_Q345,
    ElasticMaterial3D,
    HSection3D,
    HollowBoxSection3D,
    RectangularSection3D,
)

__all__ = [
    "build_staged_cantilever",
    "build_completed_model",
    "StagedDirectSolver",
    "StagedDirectBatchSolver",
    "StagedOpenSeesSolver",
    "StagedPlan",
    "StagedResult",
    "StagedStepRecord",
    "CompletedState",
    "BuildStep",
    "NewNode",
    "NewFrame",
    "NewCable",
    "NodalLoad",
    "MemberLoad",
    "BalanceDof",
    "SingleStaged3DConfig",
    "build_single_staged_3d",
    "SingleStagedDirectBatchSolver3D",
    "SingleStagedDirectSolver3D",
    "SingleStagedOpenSeesSolver3D",
    "solve_single_staged_3d",
    "BridgeModel3D",
    "SingleStagedPlan3D",
    "ConstructionStage3D",
    "Node3D",
    "FrameElement3D",
    "CableElement3D",
    "RigidLink3D",
    "Support3D",
    "FrameLoad3D",
    "SolveResult3D",
    "StagedResult3D",
    "ElasticMaterial3D",
    "HSection3D",
    "HollowBoxSection3D",
    "RectangularSection3D",
    "STEEL_Q345",
    "CONCRETE_C50",
    "CABLE_STEEL",
    "plan",
    "builder",
    "direct",
    "completed",
    "opensees",
    "sequence",
]
