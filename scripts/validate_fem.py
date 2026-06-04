"""校核自写线性求解器与 OpenSees 参考解（里程碑 M1）。

在"一次成桥"工况下对比 :mod:`bridgezoo.fem.linear_frame` 与
:mod:`bridgezoo.fem.opensees_ref` 的节点竖向位移与索应力，输出最大/相对误差表；
再在若干简单分阶段工况下复核。这是整个项目"线性简化是否可信"的关键证据，结果写入论文 E1。

用法::

    python -m scripts.validate_fem --n 6 --anchor-height 20
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="线性求解器 vs OpenSees 校核")
    parser.add_argument("--n", type=int, default=6, help="单侧索对数（偶数）")
    parser.add_argument("--anchor-height", type=float, default=20.0)
    parser.add_argument("--tol-disp", type=float, default=0.02, help="位移相对误差阈值")
    parser.add_argument("--tol-force", type=float, default=0.03, help="索力相对误差阈值")
    args = parser.parse_args()

    # TODO(M1): 用相同几何分别跑两套求解器，打印误差表并对阈值断言。
    raise NotImplementedError("TODO(M1): scripts.validate_fem.main")


if __name__ == "__main__":
    main()
