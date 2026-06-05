# 基于 MAPPO 的二维斜拉桥正向施工调索智能体 — 开发计划与架构文档

> 版本 v0.1 ｜ 2026-06-03
> 目标读者：开发者（程序）+ 论文作者（研究）
> 本文是"重新开发"的总纲。代码尚未编写，先确定架构、形式化定义、实验设计与里程碑。

---

## 0. 一句话定义

在二维斜拉桥**正向逐阶段拼装（前进分析）**与**分两次张拉**的过程中，用 **MAPPO** 训练一组"每根索一个智能体"的协作策略，使**成桥线形逼近理论线形**，同时让**拉索股数最小、最均匀、应力水平一致且处于安全范围**。研究重点是验证 **PPO/MAPPO 在调索问题上的可行性**，产出一套可复现的程序与一篇论文。

---

## 1. 研究目标与评价指标

### 1.1 多目标（成桥状态下评价）

| 目标 | 数学表达 | 说明 |
|------|----------|------|
| 线形逼近 | `J_shape = mean(|y_i − y_i^target|)` 或 RMSE | `y_i` 为成桥恒载下梁节点竖向位移，`y^target` 为理论线形 |
| 股数最小 | `J_total = Σ_i n_i` | `n_i` 为第 i 对索股数 |
| 股数均匀 | `J_even = std(n_i)` | 越小越均匀 |
| 应力一致 | `J_uni = std(σ_i)` | `σ_i` 为成桥索应力 |
| 安全 | `g_i: σ_min ≤ σ_i ≤ σ_allow`，且不松弛(σ_i>0) | 硬约束，违反重罚 |

成桥总评分（论文中用于对比，独立于 RL 奖励缩放）：
`Score = w1·J_shape + w2·J_total + w3·J_even + w4·J_uni + Penalty(g)`

### 1.2 研究问题（论文 RQ）

- **RQ1**：MAPPO 能否在"过程不可逆"的正向施工调索中收敛到接近最优的方案？
- **RQ2**：相比独立 PPO（IPPO，无中心化 critic）/ 启发式调索 / 一次成桥优化，MAPPO 的样本效率与最终性能如何？
- **RQ3**：去中心化执行（每索仅看局部信息）相对全局可观测损失多少性能？
- **RQ4**：奖励塑形（potential-based shaping）对收敛的贡献（消融）。

---

## 2. 物理模型与简化假设

延续现有 [cablebridge2.py](../bridgezoo/cablebridge/cablebridge2.py) 的几何参数体系（塔高、跨径、索间距由 `num_cables_per_side` 推导），但做如下明确化与简化：

1. **二维**、对称双塔（或单塔双跨，先做对称双塔三跨）；左右对称，**以"索对"为建模与决策单位**（沿用现有 `cab_per_side`，左右同股同力），智能体数 = `num_cables_per_side`。
2. 梁：Euler-Bernoulli 框架单元（`EA, EI`），恒载为线均布 `wg`（自重）。
3. 索：二节点桁架（只受拉），轴向刚度 `E_s·A_s·n_i`（`A_s` 为单股面积，`n_i` 股数）。张拉以**初应变 / 等效节点力**施加。
4. 索只能受拉：求解后若某索轴力 ≤ 0 视为松弛，触发安全罚（不在 FEM 内做接触非线性，简化为惩罚）。
5. **理论线形**：先取"成桥恒载下全梁节点竖向位移 = 0"（或给定预拱度向量）作为 `y^target`。
6. 材料、几何在每阶段内**线性**；阶段间刚度矩阵变化由"重装配 + 增量累加"体现（见 §3）。

---

## 3. 正向逐阶段施工建模（核心）

> 关键物理点（用户强调）：**计算可线性，但结构刚度矩阵 K 随施工阶段与股数不断变化**。因此**不能用单一全局影响矩阵**，必须每阶段重装配 K、线性增量求解、累加锁定位移/内力。

### 3.1 施工序列（对称悬臂拼装 + 二次张拉）

设单侧索对数 `N = num_cables_per_side`。施工阶段离散为：

```
阶段 0      : 索塔 + 0# 梁段（塔顶/根部）就位，仅自重
阶段 k (1..N): 安装第 k 对梁段 + 第 k 对索 → 施加【第一次张拉 T1_k】
阶段 N+1    : 合龙（中跨闭合，边界条件切换）
阶段 N+1+j (j=1..N): 对第 j 对索施加【第二次张拉/调索 T2_j】
成桥阶段    : 全部恒载就位，评价 §1 指标
```

- **第一次张拉 T1_k**：索安装时的初张力（决策：股数 `n_k` + 初应力 `σ1_k`）。
- **第二次张拉 T2_j**：调索阶段对已安装索的应力增量（决策：`Δσ2_j`），股数此时**冻结**。
- 这天然对应"分两次张拉"，且每个智能体**恰好动作两次**（安装期 + 调索期）。

### 3.2 变刚度 + 锁定位移的增量算法

维护"当前激活的节点/单元集合"`active_k`。每进入新阶段：

1. 把**本阶段新增**的梁段/索单元，按**上一阶段变形后的几何构型**接入（新单元"无应力长度"在安装时刻定义 → 之后增量才使其受力，实现**位移锁定/lock-in**）。
2. 用当前 `active_k`（含当前股数）**重新装配** `K_k`（线性、稀疏，DOF < ~150）。
3. 组装**本阶段增量荷载** `ΔF_k`：新增梁段自重 + 本阶段张拉等效力（T1 或 ΔT2）。
4. 解 `K_k · Δu_k = ΔF_k`（直接法，缓存符号结构/可重用因子分解）。
5. **累加**：`u ← u + Δu_k`；更新各索轴力 = 初张力 + 刚度引起的力变化。

> 说明：股数 `n_k` 改变会改变 `K`，所以影响矩阵不能脱离动作预计算；但**每阶段拓扑固定、解一个小线性方程组**，开销在微秒~毫秒级，足够 RL 用。张拉力 `T` 只进入荷载项（线性叠加），可在固定股数下对张拉做局部影响矩阵以进一步加速（可选优化）。

### 3.3 与现有一次成桥模型的关系

现有 [fem.py](../bridgezoo/cablebridge/fem.py) 是"建全桥 + 全索初应力 + 一次静力"——作为**最终校核器**和**线性求解器的正确性基准**保留，不作为 RL 内核。

---

## 4. 快速线性 FEM 求解器设计

为支撑数十万~数百万环境步，自写**轻量直接刚度法 2D 框架求解器**（numpy/scipy），替代每步重建 OpenSees 模型的高开销。

### 4.1 模块 `bridgezoo/fem/completed/direct.py`（成桥）与 `bridgezoo/fem/staged/direct.py`（施工，RL 内核）

- 单元：
  - `FrameElement2D`（梁，6 DOF，`EA, EI, L`，标准转换矩阵）。
  - `BarElement2D`（索，轴向，等效面积 `n·A_s`，初应变/初力）。
- 装配：稀疏 `scipy.sparse` + `splu`/直接解；DOF 映射缓存。
- API：
  ```python
  class StagedFrameModel:
      def add_node / add_frame / add_cable / set_support(...)
      def activate(stage_elements)         # 阶段激活
      def apply_incremental_load(dF)
      def apply_cable_pretension(elem, T)  # 等效节点力 + 锁定
      def solve_increment() -> du          # K_k Δu = ΔF
      def accumulate() -> u_total, cable_forces
  ```
- **正确性校核（论文必做）**：在"一次成桥"工况下与 OpenSees [fem.py](../bridgezoo/cablebridge/fem.py) 对比节点位移与索应力，给出最大相对误差表；再在 2~3 个简单分阶段工况手算/OpenSees 复核。

### 4.2 速度目标

单 episode（≈ 2N 阶段，每阶段一次小型 solve）目标 < 1~2 ms（CPU）。N=6 时单步求解 DOF≈40，远低于此。

---

## 5. MDP / Dec-POMDP 形式化

采用**合作型 Dec-POMDP**，CTDE（集中训练、分散执行）。

### 5.1 智能体

- `agent_i = cable_i`（索对），`i = 0..N−1`。参数共享单一策略网络。
- **时序**：episode 沿施工阶段推进。某阶段只有"当前动作的索"产生有效动作，其余被**动作掩码**屏蔽（mask 掉的 agent 输出 no-op、不计入该步 loss 的策略梯度，但仍参与 critic 的全局状态）。
  - 安装期阶段 k：`cable_k` 决策 `(n_k, σ1_k)`。
  - 调索期阶段 N+1+j：`cable_j` 决策 `Δσ2_j`。

> 备选形式化（论文可作对比）：**同步两轮**——第 1 轮所有索同时给 `(n, σ1)`，环境内部按施工序跑前进分析；第 2 轮所有索同时给 `Δσ2`。同步版更"标准 MAPPO"，分阶段版更贴合真实施工不可逆性。**默认实现分阶段版，附录给同步版对比。**

### 5.2 局部观测（分散执行，低维）

`obs_i`（约 8~12 维，全部归一化）：
- 自身：当前股数 `n_i`、当前应力 `σ_i`、应力利用率 `σ_i/σ_allow`、是否已安装。
- 局部线形：自身锚固点梁挠度 `y_i`，与目标差 `y_i − y_i^target`，相邻 1~2 节点挠度。
- 阶段信息：归一化阶段索引、当前相位（一次张拉 / 二次张拉 / 评价）、是否为当前活动 agent。

### 5.3 全局状态（中心化 critic）

`state`：
- 全梁节点挠度向量（线形）及其与 `y^target` 之差。
- 全部索的股数与应力向量。
- 阶段索引 + 相位 one-hot。

### 5.4 动作（离散）

- 安装期（factorized / `MultiDiscrete`，或展平为单 `Discrete`，沿用 [cablebridge_models.py](../bridgezoo/cablebridge/cablebridge_models.py) 的 9 动作思路）：
  - 股数档位 `Δn ∈ {−Δ, 0, +Δ}`（围绕默认值增减，下限 `n_min`）。
  - 初应力档位 `Δσ1 ∈ {−s, 0, +s}`（围绕上次平衡值）。
- 调索期：仅 `Δσ2 ∈ {−s, 0, +s}`（股数冻结）。
- 用 **action masking** 处理"非活动 agent / 已松弛 / 越界"等非法动作。

### 5.5 奖励（合作共享 + 势能塑形）

团队共享奖励（所有 agent 同一奖励，标准 MAPPO 合作设定）：

- **终局奖励**（成桥时）：
  `R_T = −(w1·J_shape + w2·Ĵ_total + w3·J_even + w4·J_uni) − Penalty(g)`
  各项做尺度归一化（除以参考量纲）；`Penalty` 对越界/松弛给大负值。
- **过程塑形**（每阶段，potential-based，保证最优策略不变）：
  `r_k = γ·Φ(s_{k+1}) − Φ(s_k)`，取 `Φ = −J_shape(当前)`，鼓励每次张拉改善线形。
- 消融：开/关塑形项（对应 RQ4）。

### 5.6 终止

- `terminated`：到达成桥阶段。
- `truncated`：线性求解失败 / 出现不可恢复的松弛或越界 / 超步数。

---

## 6. MAPPO 算法设计（自写精简版）

参考原始 MAPPO（Yu et al. 2022）的关键实现技巧，PyTorch 实现，约 400~600 行。

### 6.1 网络

- **Actor**（参数共享）：`obs_i → 离散动作分布`，MLP `[128,128]` + 动作掩码（对非法动作 logits 置 −inf）。可选 GRU 处理部分可观测（先用前馈，必要时加 RNN）。
- **Critic**（中心化）：`state → V(state)`，MLP `[128,128]`。共享一个 critic（合作团队值）。

### 6.2 训练要素（MAPPO tricks）

- GAE(λ)、PPO clip、value clip、entropy bonus、梯度裁剪。
- **优势归一化**、**PopArt / value normalization**（处理奖励尺度）。
- 并行环境（向量化，多个 episode 同时跑）收集 rollout。
- 共享参数 + agent-id 进 obs（或 one-hot），处理 agent 异质性。
- 离散动作熵正则、学习率退火。

### 6.3 训练循环（伪码）

```
for iter in range(num_iters):
    rollout = collect(parallel_envs, actor)        # (obs, state, act, mask, rew, done)
    adv, ret = gae(rollout, critic)
    adv = normalize(adv)
    for epoch in range(ppo_epochs):
        for mb in minibatches(rollout):
            L_pi = ppo_clip_loss(actor, mb, adv) - β·entropy
            L_v  = value_clip_loss(critic, mb, ret)
            step(L_pi + c·L_v)
    log(metrics); periodic eval + checkpoint
```

---

## 7. 重构后的软件架构

```
bridgezoo/
  fem/
    model.py               # 求解器无关 IR（StructuralModel/SolveResult）
    kernels.py             # 共享单元数值核（刚度/变换/等效荷载）
    completed/             # 一次成桥：direct.py（自写）+ opensees.py
    staged/                # 分阶段施工：plan/builder/direct（RL 内核）/opensees/completed/sequence
  envs/
    cable_construction.py   # 新环境：正向施工 + 两次张拉（PettingZoo ParallelEnv）
    cable_agent.py          # CableAgent：状态/动作/局部奖励
    geometry.py             # 桥梁几何参数（从 cablebridge2 抽出，去重）
  mappo/
    actor_critic.py         # 共享 actor + 中心化 critic
    buffer.py               # rollout buffer + GAE
    trainer.py              # 训练循环、日志、checkpoint
    config.py               # 超参 dataclass
  render/
    pygame_render.py        # 复用现有渲染，增加阶段/线形/应力可视化
scripts/
  validate_fem.py          # 线性求解器 vs OpenSees 校核
  train.py                 # 启动训练
  evaluate.py              # 评估 + 导出指标/图
  baselines.py             # IPPO / 启发式 / 一次成桥优化 对比
docs/
  DESIGN_MAPPO.md          # 本文
tests/
  test_completed_direct.py # 直接刚度法解析解 + 与 OpenSees 数值对比
  test_env.py              # 环境 API/掩码/对称性
```

> 现有 `backup/`、`resection*`、`pistonball*` 保留作参考，不进入新主线。`resection` 1D 玩具环境可保留为 MAPPO 流水线的最小冒烟测试。

---

## 8. 训练 / 评估 / 可视化

- **训练监控**：TensorBoard（沿用现有习惯）——回报、各 `J_*` 分量、约束违反率、熵、KL、value loss。
- **评估**：固定随机种子跑确定性策略，输出成桥线形图、索力/股数条形图、`Score` 表。
- **可视化**：扩展 [cablebridge2.py](../bridgezoo/cablebridge/cablebridge2.py) 的 pygame 渲染，逐阶段动画展示拼装 + 两次张拉、实时线形与应力。

---

## 9. 论文结构与实验设计

1. 引言：斜拉桥调索问题、施工不可逆性、RL 动机。
2. 相关工作：调索优化（影响矩阵/最小二乘/优化算法）、MARL、MAPPO。
3. 问题形式化：Dec-POMDP、正向施工 MDP、两次张拉。
4. 方法：线性变刚度前进分析求解器 + MAPPO（CTDE）。
5. 实验：
   - **E1** 求解器校核（vs OpenSees，误差表）。
   - **E2** 主结果：MAPPO 收敛曲线 + 成桥线形/索力/股数。
   - **E3** 对比：MAPPO vs IPPO vs 启发式 vs 一次成桥优化（LP/QP）。
   - **E4** 消融：奖励塑形、中心化 critic、局部 vs 全局观测、离散粒度。
   - **E5** 规模/泛化：不同 `N`、塔高、跨径。
6. 讨论与局限：线性简化、二维、对称假设。
7. 结论与展望（三维、非线性、随机扰动鲁棒性）。

**指标**：`J_shape, J_total, J_even, J_uni`、约束违反率、收敛步数、墙钟时间。每实验 ≥ 5 随机种子，报均值±std。

---

## 10. 里程碑与开发计划

| 阶段 | 交付物 | 验收 |
|------|--------|------|
| M0 设计 | 本文 + 接口确认 | 评审通过 |
| M1 求解器 | `completed/` + `staged/` 求解器 + 校核脚本 | vs OpenSees 误差 < 阈值（如位移 2%、索力 3%） |
| M2 环境 | `cable_construction.py` + 测试 | API/掩码/对称性通过；随机策略可跑通整个施工序列 |
| M3 MAPPO | `mappo/*` + `resection` 冒烟 | 玩具环境收敛，验证算法正确 |
| M4 训练 | 主结果 E2 | MAPPO 在 N=6 收敛，成桥指标达标 |
| M5 对比/消融 | E3/E4 baselines | 图表齐全 |
| M6 论文 | 初稿 + 复现脚本 | 可一键复现主结果 |

---

## 11. 风险与对策

| 风险 | 对策 |
|------|------|
| 每阶段重装配 K 仍偏慢 | 缓存因子分解；固定股数下对张拉做局部影响矩阵；向量化并行环境 |
| 离散动作粒度与最优解不匹配 | 多粒度消融；必要时混合（股数离散 + 张拉细档） |
| 稀疏终局奖励致收敛慢 | 势能塑形（已设计）；课程学习（先 N 小后 N 大） |
| 松弛/越界使 FEM 解病态 | 动作掩码 + 安全罚 + 截断；解失败回退处理 |
| 线性简化偏离真实 | M1 严格校核；论文明确局限；可选最终用 OpenSees 复核最优策略 |
| 参数共享掩盖索异质性 | obs 加 agent-id/位置编码；对比独立网络 |

---

## 12. 待确认接口问题（进入 M1 前）

1. 理论线形 `y^target`：取全零，还是给定预拱度向量？（默认全零）
2. 初始桥型规模：先 `N=6`（与现有一致）作为主算例？
3. 股数/应力的物理量纲与允许应力 `σ_allow`、`σ_min`、单股面积 `A_s` 的取值（默认沿用 `A_s=1.4e-4 m²`, `E_s=1.95e11`）。
4. 是否需要支持非对称/单塔，还是先锁定对称双塔三跨。

> 确认以上后即进入 M1（先写线性求解器并与 OpenSees 校核），这是整个流程的速度与正确性基石。
