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
│   ├── fem/                   # 结构有限元
│   │   ├── linear_frame.py    # ★自写线性变刚度求解器（RL 内核，M1）
│   │   ├── staged_builder.py  # 几何 → 施工阶段序列（M1/M2）
│   │   └── opensees_ref.py    # OpenSees 一次成桥参考解（仅校核）
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
│   ├── test_linear_frame.py   # skip → M1
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

`geometry.py` 文档中固定了梁/塔节点与梁/索单元的 id 约定，`linear_frame`、
`staged_builder`、`opensees_ref` 三者必须共用同一套编号，否则无法交叉校核。
