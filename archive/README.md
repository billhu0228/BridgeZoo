# archive/ —— 历史实验代码

本目录保存重构（v0.1）之前的实验性代码，**仅作参考与追溯**，不参与构建、不被
`pytest` 收集、不应被新代码 import。新开发请使用 `bridgezoo/` 主包。

| 子目录 | 内容 |
|--------|------|
| `bridgezoo_v0/` | 旧 `bridgezoo` 包：`cablebridge/`（含一次成桥 `fem.py`、AEC 环境 `cablebridge2.py` 等）、`pistonball/`、`resection/`、`rebar/`（钢筋布置 RL 子项目）及版本别名文件 |
| `tests_v0/` | 旧 `test/` 目录下的脚本与参考数据 |
| `backup/` | 更早的零散实验脚本（多版本 cablebridge、di 环境、力学/求解试验等） |

## 与新代码的关系

- 一次成桥 FEM 已迁移为 `bridgezoo/fem/opensees_ref.py`（校核用）。
- 几何公式已抽取去重为 `bridgezoo/envs/geometry.py`（唯一真源）。
- 离散动作设计、pygame 渲染思路在新模块中重构复用。
- `resection` 一维玩具环境可按需移植为 MAPPO 流水线的最小冒烟测试（见 TODO M3）。

> 如确认不再需要，可整体删除本目录而不影响主包。
