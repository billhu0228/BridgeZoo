# BridgeZoo

> 基于 **MAPPO** 的二维斜拉桥**正向施工调索**强化学习研究框架

在斜拉桥**正向逐阶段拼装主梁与拉索、分两次张拉**的施工过程中，用多智能体强化学习
（MAPPO）训练一组"每根索一个智能体"的协作策略，使**成桥线形逼近理论线形**，同时让
**拉索股数最小、最均匀、应力水平一致且处于安全范围**。本项目面向研究与论文，重点验证
**PPO/MAPPO 在调索问题上的可行性**，并提供一套可复现的程序。

## 目标

| 目标 | 含义 |
|------|------|
| 线形逼近 | 成桥恒载下梁节点竖向位移逼近理论线形 |
| 股数最小 | 拉索总股数尽量少 |
| 股数均匀 | 各索股数标准差尽量小 |
| 应力一致 | 各索成桥应力标准差尽量小 |
| 安全 | 索应力处于容许范围、不松弛 |

## 方法概览

- **力学内核**：自写**线性变刚度前进分析**求解器——逐施工阶段重装配刚度矩阵、线性
  增量求解、累加锁定位移；OpenSeesPy 仅作离线校核。
- **环境**：PettingZoo 并行环境，建模正向逐阶段施工 + 两次张拉，合作型 Dec-POMDP。
- **算法**：自写精简 **MAPPO**（共享 actor + 中心化 critic，CTDE），离散动作 + 动作掩码，
  多目标合作奖励 + 势能塑形。

详见 [docs/DESIGN_MAPPO.md](docs/DESIGN_MAPPO.md)（研究/算法总设计）与
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（目录与模块职责）。

## 仓库结构

```
bridgezoo/   主包：fem（2D/3D 求解器）/ envs（环境）/ mappo（算法）/ render（可视化）
scripts/     正式入口：validate_fem / single_staged_3d / train / evaluate / baselines
tools/       开发辅助：profile_fem / export_dxf / 参考数据
tests/       pytest
docs/        设计与架构文档
archive/      历史实验代码（不参与构建）
```

## 安装

```bash
git clone https://github.com/billhu0228/BridgeZOO.git
cd BridgeZOO
pip install -e .            # 运行依赖
pip install -r requirements-dev.txt   # 开发/测试依赖
# 可选：FEM 校核需要 OpenSeesPy
pip install -e ".[ref]"
```

要求 Python ≥ 3.10。

## 快速开始

> ⚠️ 本仓库处于**重构起点（v0.1.x）**：几何模块与超参配置已实现，FEM 求解器、环境、
> MAPPO 为带完整说明的骨架，按 [TODO.md](TODO.md) 推进。当前可运行：

```bash
pytest                       # 运行测试（已实现的几何测试通过，其余 skip）
python -c "from bridgezoo.envs.geometry import BridgeGeometry; print(BridgeGeometry().summary())"
python -m scripts.single_staged_3d --bridge omo3d --n 3 --output results/single_staged_3d.json
python -m scripts.single_staged_3d --bridge omo3d --n 3 --backend opensees --render both --dxf
python -m scripts.optimize_cables_3d --bridge omo3d --backend opensees --out results/cable_opt_3d
python -m scripts.optimize_cables_3d_engineering --bridge omo3d --out results/cable_opt_3d_engineering
python -m scripts.single_staged_3d --bridge omo3d --design results/cable_opt_3d/best_design.json --backend opensees
```

`single_staged_3d` 是单塔双主梁 3D 梁格第一轮入口：使用真实材料与 H/空心箱形截面，
同时提供自研和 OpenSees 线弹性后端；`omo3d` 加载完整 OMO 3D YAML，`--render plot/both`
输出逐阶段 PNG、GIF。DXF 由独立的 `--dxf` 开关控制，可与任意 `--render` 模式组合；
它输出最终状态的真实未变形几何（单位 m、按构件分层，不采用位移放大比例），并可用
`--dxf-out` 指定路径。未指定 `--dxf` 时不会生成 DXF。
3D YAML 可分别设置每根主梁的
`secondary_main_girder_line_load`（N/m）与桥面 `secondary_deck_pressure`（N/m²）；
两者只在几何施工完成后的独立 `secondary_load` 最终阶段生效。

`optimize_cables_3d` 是独立的 3D 索优化入口。每阶段分别优化背索组和主跨索组，每组
同步控制横桥向两根对称实体索。高效算法不再把 A/B 系数放进反复 FEM 优化：初步张拉
`T_A` 在 B=0 条件下逐阶段构造 2×2 位移影响矩阵，以 back 索塔端 `ux=0` 和主跨索梁端
`uz=0` 一次有界求解，并报告秩、条件数、触界和残差来判断可行性。二次张拉 `T_B` 默认用
back 索和主跨索各4个控制点的三次 Bernstein 曲线表示；8个低维控制变量经平滑插值恢复
全部 `2*n_seg` 组张拉力。一个 B=0 基准、8个曲线方向探针和一个完整设计校核共10个
OpenSees 工况，有界最小二乘只在响应矩阵上迭代。也可选择 `--curve-family
piecewise-linear`，或用 `auto` 实际比较两种曲线。一次固定股数连续设计的默认 FEM 预算
因此约为 `4*n_seg+10` 次，其中 `4*n_seg` 次 A 工况只计算到各自施工阶段；`n=24` 从原先
146个工况降为106个，且最昂贵的完整施工分析由50次降为10次。终端逐工况显示进度、
单次耗时、累计耗时和动态 ETA。索股数也投影到同一类低维曲线，取整后强制由内向外
非递减；最终 JSON 仍保存每组完整整数股数。最外层股数搜索默认关闭，可在连续解稳定后
显式用 `--strand-iterations` 开启，候选同样经过单调平滑投影。优化后端当前仅支持
OpenSees。v3 `best_design.json` 同时保存股数、T、系数、A/B 分力、A 可行性诊断、B 响应
面校核误差、曲线控制点、完整插值值及 FEM 用时；B 最终验证工况还会记录实际 B 施工序列下各 A 子阶段的位移，
用于判断初步 B=0 平衡假设受到的后续扰动。必要时可用 `--ab-correction-passes 1` 做一次
“固定上轮 B 重算 A、再重建 B”的有界校正；每次校正会重复同一套低维曲线工况。
结果可通过 `single_staged_3d --design`
直接载入同一物理模型进行完整逐阶段求解、JSON 输出和 3D 渲染；载入时会校验几何、
材料、阶段、索组定义以及 T/系数/A/B 的一致性。旧 v1/v2 设计仍可读取，但不能作为新版
优化器的断点续算输入。

另有独立入口 `optimize_cables_3d_engineering`，不会替换上述低维全局优化器。一个工程循环
不建立影响矩阵。每轮先完整回放当前方案；只有初始索力 A 使用中间施工记录，控制新激活梁端
相对切线诞生位置的 `z=0`，同时控制该激活状态塔锚 `x=0`。B 索力、主梁位移、桥塔位移和
索应力全部统一读取最终 `secondary_load` 状态；最终主梁 `z` 和塔锚 `x` 的目标都固定为0，
不再使用 +300 mm 或任何预拱度曲线。
每个循环严格分成三个独立子轮。第一轮固定根数，只调整 A/B 索力；A 追踪激活切线零位移，
B 直接追踪最终 `secondary_load` 的主梁和塔锚零位移，再由一次完整回放独立验收。全量索力
候选被拒绝时，隔离当前归一化绝对残差最大的一个A/B索力分量
并额外回放一次验证；只更新该分量自己的缓存，其他索组不因整组耦合失败而误缩步长。
第二轮固定刚验收的索力，按每组最终 `secondary_load` 有符号应力，以
`n_new≈n_old·max(σ,0)/500 MPa` 调整股数；每组根数和步长缓存完全独立，不再拟合四控制点
Bernstein 曲线，也不施加外侧非递减约束。根数轮以500 MPa应力指标为最高优先级，只要
应力指标严格改善就接受，允许位移暂时扰动。若某组最终应力为负，则该组本轮强制减少至少
1/3钢束，不受普通松弛、步长缓存和单轮变化上限限制，但仍服从A+B张拉容量下界及全局根数
边界；这一强制候选经FEM回放后即接受。
第三轮固定A、全部钢束根数和全部背索B，只根据最终 `secondary_load` 主梁位移调整各组中跨索B。
完整候选只在 `max(abs(主梁z))` 严格降低时接受，不再被塔锚RMS或平均位移稀释；若完整候选
失败，则只保留当前绝对主梁位移最大位置的一个中跨索B，并额外回放至多一次。这样根数轮造成
的线形扰动会在同一循环内优先修复，同时保持各组B步长缓存独立。
因此通常全新首循环需要4个完整 FEM 回放（基准、索力、根数、线形修复），同一进程后续每循环3次；
触发局部索力或局部线形候选验证时各额外增加1次。连续 C 轮通常共 `3*C+1` 次，最坏再增加至多 `2*C` 次；
跨进程续算首轮需重新读取检查点方案。

终端使用原地刷新的固定仪表板，同时显示24个施工组的背索/中跨索根数、A+B索力（MN）、
A激活/最终/目标梁端 `z`、A激活/最终塔端 `x`、最终背索/中跨索应力，以及循环进度、耗时和 ETA，
不再逐工况追加打印。
复现入口 `single_staged_3d --render text` 会在控制台摘要中列出最后3个分析阶段的 n 个主梁
控制点位移（每个施工组横桥向两个主梁锚点取平均）、最终阶段全部物理拉索应力，以及最终
阶段塔顶三向位移；`--output` 仍独立保存完整 JSON。
每完成一循环就原子写入 `engineering_checkpoint.json` 和可由现有 `--design` 重放的
`best_design.json`；可用 `--resume` 随时从最后完整循环继续。检查点同时保存 A/B 张拉力；
历史分别记录索力轮、根数轮和最终线形修复轮的候选、验收与指标。索力采用约30 MPa的默认等效应力步长，
并为每个施工组的背索/中跨索分别保存 A、B 步长记忆（n=24共96项）：本分量残差显著改善
时步长放大15%，变差或跨过目标时单独减半，范围限制为基准的1/32～2倍。索力全量候选
劣化时可经额外FEM验证接受一个隔离的最差目标分量，未接受分量单独缩小步长；根数轮按
各组最终应力独立比例调整和更新缓存，位移不再否决应力改善候选。
最终位移会耦合所有索，因此不使用把全组同步变化误当成单索导数的割线加速。最终主梁/塔位移和最终
索应力既写入 JSON 供重放核对，也直接参与工程循环决策。当前 OpenSees 3D 构件仍是线性小位移；
工程采用前必须再做几何非线性、索垂度和仅受拉验证。

实现完成后（详见 TODO）：

```bash
python -m scripts.validate_fem --n 6      # M1：校核线性求解器
python -m scripts.train --n 6             # M4：训练 MAPPO
python -m scripts.evaluate --checkpoint runs/xxx.pt --render
```

## 路线图

见 [TODO.md](TODO.md)：M0 设计 → M1 求解器 → M2 环境 → M3 MAPPO → M4 训练 →
M5 对比/消融 → M6 论文。

## 许可证

见 [LICENCE](LICENCE)。
