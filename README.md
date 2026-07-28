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
同步控制横桥向两根对称实体索；算法保留 2D 版本的仿射 LP+SLSQP 连续优化、整数股数
缩放/坐标搜索、随机试探、断点续跑及 JSON/CSV/文本结果。`--backend direct` 可切换至
自研 3D 求解器，默认 OpenSees 后端适合完整 OMO 梁格。终端输出复用 2D 的节流仪表盘，
默认至多每秒刷新一次，可用 `--progress-refresh` 调整或用 `--quiet` 关闭。优化完成后的
`best_design.json` 可通过 `single_staged_3d --design` 直接载入同一物理模型进行完整逐阶段
求解、JSON 输出和 3D 渲染；载入时会校验几何、材料、阶段和索组定义。

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
