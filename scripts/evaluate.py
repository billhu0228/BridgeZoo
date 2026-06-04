"""评估入口：回放已训练策略并导出成桥指标/图（里程碑 M4）。

确定性策略跑若干 episode，输出成桥线形图、索力/股数条形图，以及
J_shape / J_total / J_even / J_uni / 约束违反率（写入 CSV 供论文表格）。

用法::

    python -m scripts.evaluate --checkpoint runs/xxx.pt --episodes 5 --render
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="评估并导出调索结果")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", type=str, default="results")
    args = parser.parse_args()

    # TODO(M4): 加载策略，跑确定性 episode，调用 env.final_metrics()，画图并存 CSV。
    raise NotImplementedError("TODO(M4): scripts.evaluate.main")


if __name__ == "__main__":
    main()
