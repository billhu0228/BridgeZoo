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

## M1 —— 线性变刚度求解器（项目基石）

- [ ] `linear_frame.FrameElement.local_stiffness`（梁单元刚度 + 坐标变换）
- [ ] `linear_frame.CableElement.axial_stiffness`（索轴向刚度）
- [ ] `StagedFrameModel.activate / apply_incremental_load / apply_cable_pretension`
- [ ] `StagedFrameModel.solve_increment / accumulate`（增量求解 + 位移锁定）
- [ ] `staged_builder.build_stages`（几何 → 施工阶段序列，编号与 OpenSees 一致）
- [ ] `tests/test_linear_frame.py`：简支梁/单索解析解、逐阶段 vs OpenSees
- [ ] `scripts/validate_fem.py`：误差表，达标阈值（位移 2% / 索力 3%）→ 论文 E1
- [ ] `tools/profile_fem.py`：单 episode < 1~2 ms

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
- [ ] 大模型时 `linear_frame` 切换 `scipy.sparse` + 缓存因子分解
- [ ] 类型标注与 `ruff`/`mypy` 配置（可选）
