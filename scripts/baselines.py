"""对比基线（里程碑 M5）。

为论文 E3 提供对照：

- **IPPO**：独立 PPO（去掉中心化 critic，各智能体独立价值），验证中心化 critic 的价值。
- **启发式调索**：基于影响矩阵的最小二乘 / 规则法（传统调索思路）。
- **一次成桥优化**：直接对成桥索力做 LP/QP 优化（忽略施工过程），作为性能上界参考。

统一用 :meth:`CableConstructionEnv.final_metrics` 评价，输出可比表格。

用法::

    python -m scripts.baselines --method heuristic --n 6
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="调索基线对比")
    parser.add_argument("--method", choices=["ippo", "heuristic", "oneshot_opt"], required=True)
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    # TODO(M5): 按 method 分发到对应基线实现，统一评价并导出。
    raise NotImplementedError("TODO(M5): scripts.baselines.main")


if __name__ == "__main__":
    main()
