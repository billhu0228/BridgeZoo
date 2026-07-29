# TODO / 路线图

里程碑与任务清单。详细设计见 [docs/DESIGN_MAPPO.md](docs/DESIGN_MAPPO.md)，
模块职责见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

图例：`[ ]` 待办 · `[~]` 进行中 · `[x]` 完成

---

## M0 —— 设计与脚手架 ✅

- [x] 研究/算法总设计文档（`docs/DESIGN_MAPPO.md`）
- [x] 架构文档（`docs/ARCHITECTURE.md`）
- [x] 清理并归档历史代码到 `archive/`
- [x] 建立目录骨架：`fem/ envs/ mappo/ render/ scripts/ tools/ tests/`
- [x] `geometry.py`（几何唯一真源）+ 测试通过
- [x] `MappoConfig` 超参 dataclass
- [x] `.gitignore` / `README.md` / `pyproject.toml` / 依赖文件
- [ ] 与导师/合作者评审本设计，确认 §12 待定接口（见下）

### 待确认接口（进入 M1 前）
- [ ] 理论线形 `y_target`：全零 or 给定预拱度？（默认全零）
- [ ] 主算例规模：先 `N=6`？
- [ ] 应力量纲与 `σ_allow` / `σ_min` / 单股面积 `A_s` 取值
- [ ] 是否仅对称双塔三跨（默认是）

---

## M1 —— 线性求解器（项目基石）

**后端无关架构（一套结构定义，两种后端，结果一致）✅**
- [x] `model.StructuralModel` / `SolveResult`：与求解器无关的结构 IR
- [x] `staged.build_completed_model`：由施工计划派生成桥 `StructuralModel`
- [x] `completed.direct.CompletedDirectSolver`：自研二维直接刚度法（梁+索+均布荷载+预张力）
- [x] `completed.opensees.CompletedOpenSeesSolver`：OpenSees 线性后端（Truss+InitStress，对照用）
- [x] `scripts/validate_fem.py`：成桥工况两后端逐项对比 → **相对误差 ~1e-14，通过**
- [x] `tests/test_completed_direct.py`：简支/悬臂解析解 + 单索预张力 + OpenSees 交叉校核

**施工阶段（变刚度 + 切线激活）✅**
- [x] `opensees_staged`：切线激活 + 顺序加载（已验证 staged≠one-shot，索力历程）
- [x] `staged.StagedPlan`：后端无关施工计划（节点切线附着 / 装段 / 张索）
- [x] `staged.StagedDirectSolver`：自研直接刚度法的**增量变刚度**求解器（切线激活 +
  位移锁定 + 索力历程）——RL 内核
- [x] `staged.StagedOpenSeesSolver`：OpenSees 后端，**可切换索单元** `cable_element`：
  `"linear"`（普通 Truss，与自研同为线性 → 研究初期逐项对照）/ `"corot"`（corotTruss
  几何精确 → 后续生产）
- [x] `scripts/validate_staged.py` + `tests/test_staged.py`：自研 vs OpenSees，stage-1
  误差 ~0.02%，大挠度算例 ~1.3%（线性 vs 几何精确，符合预期）

**剩余**
- [ ] `staged_builder.build_stages`：由全桥 `BridgeGeometry` 生成施工阶段（对称双悬臂/合龙）
- [ ] `tools/profile_fem.py`：单 episode 耗时基准 < 1~2 ms

**`single_staged` 3D 升级（详细施工阶段）✅**
- [x] `SingleTowerGeometry3D`：保留单塔纵向基本尺寸，新增双主梁、桥面板偏移及真实
  H/空心箱形截面尺寸
- [x] `single_staged.model3d`：6 自由度空间模型 IR、阶段激活、刚臂和统一结果
- [x] 双主梁 + 等间距横梁共节点梁格、偏心桥面板梁格、单塔箱形截面和双索面
- [x] `direct3d` / `opensees3d`：同模型线弹性空间梁/杆双后端及交叉测试
- [x] `scripts.single_staged_3d`：最小 CLI 输入、终端摘要及 JSON 输出
- [x] 完整 OMO 3D YAML 输入 + OpenSees 结果的阶段 PNG/GIF 空间渲染
- [x] 几何施工完成后追加独立二期荷载阶段：两根主梁线荷载 + 桥面板均布压强
- [x] `optim.single_staged3d` + `optim.staged3d_optimizer` + `scripts.optimize_cables_3d`：
  背索/主跨索对称分组的独立 3D 优化入口；A 张拉用逐阶段 2×2 位移影响矩阵直接求平衡
  并诊断可行性，B 张拉用低维 Bernstein/分段线性平滑曲线响应矩阵优化并恢复全部索组；
  股数投影为由内向外非递减的整数曲线，完整设计再由 OpenSees 实算校核。终端显示 FEM
  工况预算、耗时和 ETA。最外层股数搜索独立启用。新版优化后端暂仅 OpenSees，并输出可由
  `single_staged_3d --design` 复现的 v3 设计文件
- [x] 独立 `engineering_cycle3d` 工程调索：不建影响矩阵，每轮从一次完整回放读取全部
  施工组响应，A梁端使用切线相对位移，B梁端和A/B塔端使用当前阶段实际累计位移；采用
  阶段差分限幅反馈逐轮逼近A/B平衡与默认+300 mm全局软预拱度，并支持多种预拱度函数。
  默认等效调索步长约30 MPa，每个背/中索组的A/B分量独立应用、独立自适应并写入检查点；
  每循环第一轮固定根数只调索力，第二轮固定索力只按 B 阶段约500 MPa平衡应力调整平滑
  股数，默认每组最多变化1根；两轮独立实算和验收。首循环3次完整回放、同次运行后续每轮
  最多2次；劣化子轮单独回退，24组状态原地刷新，v4/v5检查点可迁移，原子检查点、历史与 `--resume` 续算均不覆盖
  低维全局优化器
- [x] 每个架设阶段细分为钢梁/拉索+A、湿桥面板等效线荷载+B、删除临时荷载定义并
  激活偏心桥面板组合体系；自研/OpenSees 双后端均保存零应力诞生和既有内力状态
- [ ] 第二轮：OpenSees `corotTruss`、大位移、索垂度及拉索仅受拉处理
- [ ] 第二轮：实体 H/箱形截面外轮廓、结果云图与交互式 3D 渲染

> 一次成桥（机器精度一致）与逐阶段施工（小位移 ~0.02% 一致）两条线均已闭环。
> 自研求解器可独立支撑 RL 训练，OpenSees 仅作离线校核。

---

## M2 —— 多智能体施工环境

- [ ] `cable_agent`：`apply_erection / apply_adjustment / build_observation`
- [ ] `cable_construction.reset / step`（驱动逐阶段求解，回填挠度/应力）
- [ ] `state()` 全局状态 + `action_masks()` 动作掩码
- [ ] 奖励：终局多目标 + 势能塑形；`final_metrics()`（与奖励解耦）
- [ ] `render/pygame_render`：逐阶段动画 + 目标线形 + 应力/股数面板
- [ ] `tests/test_env.py`：PettingZoo API、随机策略跑通、掩码、对称性

---

## M3 —— 自写精简 MAPPO

- [ ] `actor_critic.Actor`（共享、掩码）/ `CentralCritic`（全局）
- [ ] `buffer.RolloutBuffer` + GAE(λ)
- [ ] `trainer.MappoTrainer`：采样 / PPO 更新 / 日志 / checkpoint
- [ ] 归一化技巧：优势归一化、value normalization、obs 归一化
- [ ] `tests/test_mappo.py`：GAE 正确性、掩码概率为 0、玩具环境冒烟收敛
- [ ] （可选）移植历史 `resection` 玩具环境作最小冒烟测试

---

## M4 —— 训练与主结果（E2）

- [ ] `scripts/train.py` 串通；TensorBoard 记录回报与各 `J_*` 分量
- [ ] `N=6` 收敛，成桥线形/索力/股数达标
- [ ] `scripts/evaluate.py`：确定性回放、导出图与指标 CSV
- [ ] 课程学习（先小 N 后大 N，按需）

---

## M5 —— 对比与消融（E3/E4/E5）

- [ ] `baselines.py`：IPPO / 启发式（影响矩阵最小二乘）/ 一次成桥优化
- [ ] 消融：奖励塑形、中心化 critic、局部 vs 全局观测、离散粒度
- [ ] 规模/泛化：不同 `N`、塔高、跨径
- [ ] 每实验 ≥ 5 随机种子，报均值 ± std

---

## M6 —— 论文与复现

- [ ] 论文初稿（引言/相关工作/形式化/方法/实验/结论）
- [ ] 一键复现脚本与配置归档
- [ ] 图表脚本固化（线形图、收敛曲线、指标表）

---

## 工程债务 / 杂项

- [ ] 复查并更新 CI（`.github/workflows/publish.yml`）以适配新结构
- [ ] 决定是否将 `.idea/` 移出版本控制
- [ ] 大模型时直接刚度法（`completed.direct`/`staged.direct`）切换 `scipy.sparse` + 缓存因子分解
- [ ] 类型标注与 `ruff`/`mypy` 配置（可选）
