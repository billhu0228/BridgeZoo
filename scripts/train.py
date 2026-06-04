"""训练入口：用 MAPPO 训练调索策略（里程碑 M4）。

用法::

    python -m scripts.train --n 6 --total-steps 2000000 --device cpu
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPPO 训练斜拉桥调索智能体")
    parser.add_argument("--n", type=int, default=6, help="单侧索对数（偶数）")
    parser.add_argument("--total-steps", type=int, default=2_000_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-dir", type=str, default="runs")
    args = parser.parse_args()

    # TODO(M4): 构建 BridgeGeometry/env_fn 与 MappoConfig，实例化 MappoTrainer 并 learn()。
    raise NotImplementedError("TODO(M4): scripts.train.main")


if __name__ == "__main__":
    main()
