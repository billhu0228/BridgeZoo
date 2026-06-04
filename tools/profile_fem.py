"""基准测试线性求解器单步耗时（里程碑 M1）。

确认 :mod:`bridgezoo.fem.linear_frame` 的单个 episode（≈2N 阶段）耗时满足 RL 采样
要求（目标 < 1~2 ms）。用法::

    python -m tools.profile_fem --n 6 --iters 1000
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="线性求解器性能基准")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--iters", type=int, default=1000)
    args = parser.parse_args()

    # TODO(M1): 构建模型，循环跑 iters 次完整施工序列，统计单步/单 episode 耗时分布。
    raise NotImplementedError("TODO(M1): tools.profile_fem.main")


if __name__ == "__main__":
    main()
