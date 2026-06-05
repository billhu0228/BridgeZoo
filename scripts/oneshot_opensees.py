"""一次成桥（one-shot）OpenSees 计算示例。

构建一个基本的二维斜拉桥模型（由 :class:`BridgeGeometry` 给出几何），假定一组
均匀的拉索初应力与股数，调用 :mod:`bridgezoo.fem.oneshot.opensees_ref` 做**一次性成桥**
静力分析，并提取代表性结果：

- 主梁线形（各梁节点竖向位移）；
- 各拉索成桥应力（MPa）及其统计（最大/最小/均值/标准差）。

这是后续"正向逐阶段施工 + 调索"分析的对照基准（一次成桥 = 忽略施工过程，
所有索同时张拉到位）。

用法::

    python -m scripts.oneshot_opensees --n 6 --sigma 600 --strands 20
    python -m scripts.oneshot_opensees --n 6 --plot results/oneshot.png

> 注意：本脚本需要可用的 openseespy（其编译 DLL 对 Python 版本敏感）。若导入失败，
> 请在受支持的解释器（如 3.11/3.12）下安装 ``pip install openseespy`` 后运行。
"""

from __future__ import annotations

import argparse
import os
import sys

# 允许从仓库根目录直接 `python scripts/oneshot_opensees.py` 运行
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from bridgezoo.envs.geometry import BridgeGeometry
from bridgezoo.fem.oneshot.opensees_ref import build_oneshot_fem


def run(n: int, sigma: float, strands: int, plot: str | None) -> dict:
    geom = BridgeGeometry(num_cables_per_side=n)
    print("几何：", geom.summary())

    # 一组均匀的初始假定（每根索相同初应力与股数）作为基本算例
    cable_sigma = [float(sigma)] * n   # MPa
    cable_sizes = [int(strands)] * n   # 股

    fem = build_oneshot_fem(geom, cable_sigma, cable_sizes)
    beam_disp, cable_stress = fem.opensees()

    beam_disp = np.asarray(beam_disp, dtype=float)
    cable_stress = np.asarray(cable_stress, dtype=float)

    ok = bool(np.any(cable_stress)) and not np.all(beam_disp == 0)
    print("\n求解状态：", "成功" if ok else "失败（返回全零，检查模型/收敛）")

    # ---- 主梁线形 ----
    print("\n=== 主梁线形（节点 x[m] -> 竖向位移[mm]）===")
    for x, d in zip(geom.x_positions, beam_disp):
        print(f"  x={x:8.2f}   dy={d * 1000:8.2f} mm")
    print(
        f"  线形：max={beam_disp.max() * 1000:.2f} mm  "
        f"min={beam_disp.min() * 1000:.2f} mm  "
        f"RMS={np.sqrt(np.mean(beam_disp ** 2)) * 1000:.2f} mm"
    )

    # ---- 拉索成桥应力 ----
    print("\n=== 拉索成桥应力（MPa）===")
    for i, s in enumerate(cable_stress):
        print(f"  cable[{i:2d}]  sigma={s:8.2f} MPa")
    print(
        f"  应力：max={cable_stress.max():.2f}  min={cable_stress.min():.2f}  "
        f"mean={cable_stress.mean():.2f}  std={cable_stress.std():.2f} MPa"
    )

    if plot:
        _plot(geom, beam_disp, cable_stress, plot)

    return {"beam_disp": beam_disp, "cable_stress": cable_stress, "ok": ok}


def _plot(geom, beam_disp, cable_stress, out_path: str) -> None:
    """画主梁线形 + 拉索应力条形图并保存。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bridgezoo.render.mpl_cjk import use_cjk_font
    use_cjk_font()  # 配置中文字体（必须在建 figure 前）

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6))

    ax1.plot(geom.x_positions, beam_disp * 1000, "-o", ms=3)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_title("主梁线形 / Beam deflection")
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("dy [mm]")
    ax1.invert_yaxis()  # 下挠为正向下显示

    ax2.bar(range(len(cable_stress)), cable_stress)
    ax2.set_title("拉索成桥应力 / Cable stress")
    ax2.set_xlabel("cable index")
    ax2.set_ylabel("σ [MPa]")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n已保存图像：{out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="一次成桥 OpenSees 计算示例")
    parser.add_argument("--n", type=int, default=6, help="单侧索对数（偶数）")
    parser.add_argument("--sigma", type=float, default=600.0, help="各索初应力 (MPa)")
    parser.add_argument("--strands", type=int, default=20, help="各索股数")
    parser.add_argument("--plot", type=str, default=None, help="输出图片路径（如 results/oneshot.png）")
    args = parser.parse_args()
    run(args.n, args.sigma, args.strands, args.plot)


if __name__ == "__main__":
    main()
