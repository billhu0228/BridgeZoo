# 项目架构

本文给出重构后 BridgeZoo 的目录结构、模块职责与数据流。总体研究设计见
[DESIGN_MAPPO.md](DESIGN_MAPPO.md)，开发进度见仓库根的 [TODO.md](../TODO.md)。

## 设计原则

1. **几何唯一真源**：所有几何/截面参数集中在 `bridgezoo/envs/geometry.py`，FEM、施工
   阶段、渲染均从此取数，杜绝历史代码中多处重复且不一致的几何公式。
2. **求解器分层**：RL 内核用自写**线性变刚度前进分析**求解器（快）；OpenSees 仅作
   **离线校核**（准）。二者解耦，互不进入对方主线。
3. **CTDE 边界清晰**：环境同时暴露"局部观测"（分散执行）与"全局状态"（集中训练），
   actor 只看局部、critic 只看全局。
4. **骨架先行**：先把接口、说明、测试位、里程碑钉死，再按 `TODO.md` 逐层实现，
   保证任意时刻仓库可 import、测试可运行（未实现项 skip）。

## 目录结构

```
BridgeZoo/
├── bridgezoo/                 # 主包
│   ├── __init__.py            # 版本 / 包说明
│   ├── fem/                   # 结构有限元（两种分析模式镜像对称）
│   │   ├── model.py           # 求解器无关 IR（StructuralModel/SolveResult）
│   │   ├── kernels.py         # 共享单元数值核（刚度/变换/等效荷载）
│   │   ├── completed/         # 一次成桥：direct.py（★自写）+ opensees.py
│   │   ├── staged/            # 分阶段施工：plan/builder/direct/opensees/completed/sequence（★RL 内核）
│   │   └── single_staged/     # 单塔 2D 兼容层 + 全新 3D 梁格 IR/builder/双后端
│   ├── envs/                  # 多智能体环境
│   │   ├── geometry.py        # ★桥梁几何/截面（已实现，唯一真源）
│   │   ├── cable_agent.py     # 索智能体状态/动作/观测（M2）
│   │   └── cable_construction.py  # ★施工+两次张拉 ParallelEnv（M2）
│   ├── mappo/                 # 自写 MAPPO（CTDE）
│   │   ├── config.py          # 超参 dataclass（已实现）
│   │   ├── actor_critic.py    # 共享 actor + 中心化 critic（M3）
│   │   ├── buffer.py          # rollout + GAE（M3）
│   │   └── trainer.py         # 训练主循环（M3/M4）
│   └── render/
│       └── pygame_render.py   # 施工/成桥可视化（M2/M4）
├── scripts/                   # 正式入口（python -m scripts.xxx）
│   ├── validate_fem.py        # 线性解 vs OpenSees 校核（M1，E1）
│   ├── train.py               # MAPPO 训练（M4，E2）
│   ├── evaluate.py            # 评估/导出指标图（M4）
│   └── baselines.py           # IPPO/启发式/一次成桥优化对比（M5，E3）
├── tools/                     # 开发辅助
│   ├── profile_fem.py         # 求解器性能基准
│   ├── export_dxf.py          # 模型 DXF 导出
│   └── reference/simple_beam.mct  # MIDAS 校核参考
├── tests/                     # pytest（testpaths=["tests"]）
│   ├── test_geometry.py       # ★已实现并通过
│   ├── test_completed_direct.py  # 直接刚度法解析解 + OpenSees 交叉校核
│   ├── test_env.py            # skip → M2
│   └── test_mappo.py          # skip → M3
├── docs/
│   ├── DESIGN_MAPPO.md        # 研究/算法总设计
│   └── ARCHITECTURE.md        # 本文
├── archive/                   # 历史实验代码（不参与构建/测试）
├── TODO.md                    # 里程碑与任务清单
├── README.md
├── pyproject.toml             # 打包；仅收录 bridgezoo*；含 pytest 配置
├── requirements.txt           # 运行依赖
└── requirements-dev.txt       # 开发依赖（pytest 等）
```

★ = 关键模块。

## 数据流（一个 episode）

```
reset()
  └─ staged_builder.build_stages(geometry)  → [Stage0, Stage1, ...]
  └─ StagedFrameModel 初始化（阶段0：塔+0#段自重）

step(actions)  // 每次推进一个施工阶段
  ┌─ 取当前阶段 active_cable
  ├─ CableAgent.apply_erection / apply_adjustment(动作)   // 改股数→改K / 改张拉→改荷载
  ├─ StagedFrameModel.activate(本阶段单元)               // 重装配 K_k（变刚度）
  ├─ apply_incremental_load + apply_cable_pretension      // 组装 ΔF_k
  ├─ solve_increment() → 累加位移 u                       // 线性增量
  ├─ accumulate() → 梁挠度 + 各索轴力 → CableAgent.update
  ├─ 组装 obs_i（局部）/ state（全局）/ 共享奖励（终局+势能塑形）
  └─ 推进阶段游标；成桥→terminated，超步→truncated

训练（MAPPO, CTDE）
  Actor(obs_i, mask) → 动作        // 分散执行
  CentralCritic(state) → V          // 集中训练
  RolloutBuffer + GAE → PPO 更新
```

## 编号约定（务必一致）

成桥/施工两套模型各自固定梁/塔节点与梁/索单元的 id 约定（`staged.builder` 为施工模型真源，
`completed/` 经 `staged.completed` 由同一施工计划派生），同模式内两后端必须共用同一套编号，否则无法交叉校核。

`single_staged/` 源码与内部导入完全独立，采用独塔左悬臂拓扑：右梁只有一个全固定端节点，
每根右索连接独立的全固定地锚；`tip_free` 安装并求解左侧自由端，随后最终
`left_tip_uy_lock` 阶段把节点 201 的竖向位移锁定在当前位置，之后 `left_span` 阶段按
YAML 同名长度向左切线激活辅助跨并把新端节点 202 的竖向位移锁定在诞生位置；当
`dw != 0` 时，最终 `phase2` 阶段对全部主梁单元施加二期均布荷载。
分析和优化统一使用 `scripts.staged_analysis` 与
`scripts.optimize_cables` 入口。桥梁 YAML 的 `bridge_type` 负责模型分派：`normal` 使用
`staged`，`single` 使用 `single_staged`；通用优化算法内部对应
`CableOptimizationProblem.model_family`。

## `single_staged` 3D 第一轮架构

3D 模型采用后缀明确的新 API（如 `build_single_staged_3d`、
`SingleStagedDirectSolver3D`），旧 2D 接口在迁移期间保持不变。坐标约定为
`x` 纵桥向、`y` 横桥向、`z` 竖向，每个节点有
`(ux, uy, uz, rx, ry, rz)` 六个自由度。

- `envs.geometry.SingleTowerGeometry3D` 是纵横向尺寸及 H/箱形截面尺寸的唯一真源，
  继承旧单塔模型的 `n_seg`、`anchor_*`、`left/right_*`、`right_fix`、`left_span`
  语义。
- `single_staged.sections3d` 从真实截面尺寸推导 `A/Iy/Iz/J`；默认主梁、横梁为钢制
  H 型截面，塔身为 C50 混凝土空心箱形截面，桥面板为 C50 混凝土等效矩形板条。
- `single_staged.model3d` 定义与后端无关的空间梁、索、支座、全局均布荷载、刚臂、
  阶段激活和统一结果 IR。
- 两根主梁沿横桥向对称布置，横梁使用独立的等间距网格并与主梁共节点；索锚点可额外
  细分主梁但不会破坏横梁等间距。桥面板纵横梁格位于主梁轴线上方的独立参考面，通过
  6 自由度刚臂表达偏心。当桥面宽度大于主梁间距时，板梁格在两根主梁外侧各增加一排
  边缘节点，横向板条显式形成两侧悬臂；纵向板条按实际影响宽度分配。纵横板条都参与
  刚度，板自重仅分配给纵向板条，避免重复计重且保持桥面板总质量不变。
- 单塔位于桥轴线，双索面分别连接两根主梁；每阶段同步激活一对主跨索和一对背索。
- 梁格、桥面板及可选辅助跨全部激活后，若配置了二期荷载则追加独立的
  `secondary_load` 最终阶段。`secondary_main_girder_line_load` 作为每根主梁的全局竖向
  线荷载施加到两条钢主梁，`secondary_deck_pressure` 按实际影响宽度转换为各条桥面纵向
  板条的线荷载；前者用于护栏等沿桥设施，后者用于沥青铺装等面荷载。
- `direct3d` 为自研线弹性 Euler-Bernoulli 空间梁 + 线性杆求解器，刚臂通过精确自由度
  凝聚实现；`SingleStagedDirectBatchSolver3D` 在固定索股时只组装、凝聚并分解一次刚度
  矩阵，将全部预张力扰动组成多右端矩阵并一次回代，完整恢复梁端力、索力和支座反力；
  `opensees3d` 使用同一 IR 建立 `elasticBeamColumn`、`Truss +
  InitStressMaterial` 和 `rigidLink beam`，用于离线对照。

第一轮的阶段语义是“累计激活后逐阶段线性重分析”，二期荷载阶段因此表示在完整结构上
重新进行累计线性分析，尚不包含路径相关的零应力诞生、
安装构形锁定、索垂度或几何非线性。`scripts/bridges/omo_bridge_3d.yaml` 是完整的 OMO
3D 物理输入，计算入口为 `python -m scripts.single_staged_3d --bridge omo3d`。

`render.staged3d` 消费同一份 `SingleStagedPlan3D/StagedResult3D`，因此自研和 OpenSees
后端共享渲染语义。它仿照 2D 逐阶段输出：未变形参考网格、位移放大后的空间梁格与箱塔
轴线、半透明桥面板、双索面、刚臂、支座和新增构件高亮，并可同时保存阶段 PNG 与 GIF。
独立的 `--dxf` 开关通过 `ezdxf` 输出最终状态的 3D DXF，可与任意 `--render` 模式组合；
节点、主梁、横梁、板条、塔、拉索、刚臂和桥面板分别置于命名图层。DXF 坐标采用模型
真实未变形坐标，单位为米，不使用绘图位移放大比例，便于 CAD 内直接检查几何尺寸。
未启用 `--dxf` 时图像渲染不会附带生成 DXF；所有渲染和导出都只做后处理，不进入求解器
或改变任何力学结果。

3D 索优化使用独立入口 `scripts.optimize_cables_3d` 和适配层
`optim.single_staged3d.CableDesignEvaluator3D`，但复用已验证的连续/整数混合搜索内核。
设计向量仍为逐阶段 `2*n_seg` 个变量，在 3D 中解释为 `(背索组, 主跨索组)`；每个组把
同一股数和单索预张力同步施加到横桥向两根实体索。线形目标取两根主梁所有节点的 `uz`，
塔目标取塔顶及各索塔锚点的 `ux`，组应力取两根实体索的均值，而材料量按两根实体索的
实际总股数计入目标。优化阶段只求解累计模型的最终阶段，因为当前 3D 各阶段是相互独立
的线性重分析；直接后端构造仿射模型时，`T=0` 与逐组单位扰动的 `2*n_seg+1` 个工况
共享一次自由度系统分解。输出保留组到实体索单元 ID 的映射，支持断点续跑和 OpenSees
校核。
