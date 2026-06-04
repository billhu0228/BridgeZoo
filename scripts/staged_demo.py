"""正向逐阶段施工（悬臂拼装 + 张索）演示。

调用 :class:`bridgezoo.fem.opensees_staged.StagedCantileverCableBridge`，按"装节段→张索"
逐阶段推进，提取代表性结果：

- 每阶段的主梁线形（活动节点竖向位移）；
- 每根索的**应力变化历程**（某根索激活后，随后续节段安装与张索而变化的过程）。

默认各索安装张力按"承担自身节段重量"估算：``T_i = wg·L·L0_i / H``（L0_i 为索长）。

用法::

    python -m scripts.staged_demo --n 6
    python -m scripts.staged_demo --n 6 --plot results/cable_history.png

> 需要 openseespy（建议 Python 3.11–3.13）。
"""

from __future__ import annotations

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bridgezoo.fem.opensees_staged import StagedCantileverCableBridge


def default_pretension(n, seg_len, H, wg):
    """各索目标张力：使竖向分量约等于该节段自重。"""
    out = []
    for i in range(1, n + 1):
        L0 = math.hypot(i * seg_len, H)
        out.append(wg * seg_len * L0 / H)
    return out


def run(n: int, plot: str | None) -> None:
    model = StagedCantileverCableBridge(n_seg=n)
    strands = [20] * n
    pretension = default_pretension(n, model.seg_len, model.tower_height, model.wg)

    print("参数：")
    print(f"  节段数 N={n}  段长={model.seg_len} m  塔高={model.tower_height} m  wg={model.wg:.3e} N/m")
    print("  各索目标安装张力 (kN)：", [f"{t/1e3:.0f}" for t in pretension])

    history = model.run(strands, pretension)

    # ---- 逐阶段线形 + 索应力 ----
    print("\n=== 逐阶段结果 ===")
    for rec in history:
        tip = max(rec.deck_deflection)
        print(
            f"阶段 {rec.stage:2d} | 悬臂端挠度 dy={rec.deck_deflection[tip] * 1000:8.2f} mm | "
            f"活动索应力(MPa)="
            + " ".join(f"c{j}:{s:6.1f}" for j, s in rec.cable_stress.items())
        )

    # ---- 代表性：每根索的应力历程 ----
    print("\n=== 拉索应力变化历程（MPa）===")
    hist = model.cable_stress_history()
    for i, series in hist.items():
        s0 = series[0][1]
        s_end = series[-1][1]
        print(
            f"  索 c{i}: 安装(阶段{series[0][0]})={s0:7.1f} → 成形(阶段{series[-1][0]})={s_end:7.1f}"
            f"  Δ={s_end - s0:+7.1f}"
        )
    # 重点展示最早安装的 c1（经历后续所有施工扰动）
    if 1 in hist:
        print("\n  索 c1 完整历程：", " → ".join(f"S{st}:{s:.1f}" for st, s in hist[1]))

    if plot:
        _plot(hist, plot)


def _plot(hist, out_path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bridgezoo.render.mpl_cjk import use_cjk_font
    use_cjk_font()  # 配置中文字体（必须在建 figure 前）

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, series in hist.items():
        stages = [st for st, _ in series]
        vals = [s for _, s in series]
        ax.plot(stages, vals, "-o", ms=4, label=f"cable c{i}")
    ax.set_title("拉索应力变化历程 / Cable stress history")
    ax.set_xlabel("施工阶段 stage")
    ax.set_ylabel("应力 σ [MPa]")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n已保存图像：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="正向逐阶段施工 + 张索 演示")
    parser.add_argument("--n", type=int, default=6, help="悬臂节段数（= 索数）")
    parser.add_argument("--plot", type=str, default="Staged.png", help="输出索应力历程图路径")
    args = parser.parse_args()
    run(args.n, args.plot)


if __name__ == "__main__":
    main()
