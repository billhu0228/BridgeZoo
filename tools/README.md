# tools/ —— 开发辅助工具

非训练主线的辅助脚本与参考数据，与 `scripts/`（正式入口）区分。

| 文件 | 用途 |
|------|------|
| `profile_fem.py` | 基准测试线性求解器单步耗时，确认满足 RL 采样速度要求（M1） |
| `export_dxf.py` | 导出桥梁几何/模型为 DXF，便于人工核对节点与单元布置 |
| `reference/simple_beam.mct` | MIDAS Civil 简支梁参考模型，用于交叉校核 FEM 结果 |

运行示例：

```bash
python -m tools.profile_fem --n 6 --iters 1000
python -m tools.export_dxf --n 6 --out model.dxf
```
