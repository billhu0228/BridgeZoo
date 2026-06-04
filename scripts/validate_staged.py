"""逐阶段施工交叉校核：同一施工计划，自研直接刚度法 vs OpenSees，结果应一致。

用 :func:`bridgezoo.fem.staged.build_staged_cantilever` 构建**一个** StagedPlan，分别交给
:class:`StagedDirectSolver` 与 :class:`StagedOpenSeesSolver`，对比每个施工阶段的悬臂端
挠度与各索应力历程，输出最大误差并判定。

用法::

    python -m scripts.validate_staged --n 6

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

from bridgezoo.fem.staged import (
    StagedDirectSolver,
    StagedOpenSeesSolver,
    build_staged_cantilever,
)


def default_pretension(n, seg_len, H, wg):
    return [wg * seg_len * math.hypot(i * seg_len, H) / H for i in range(1, n + 1)]


def run(n: int, tol_rel: float) -> bool:
    wg = 4.7e5
    seg_len, H = 8.0, 20.0
    strands = [20] * n
    pre = default_pretension(n, seg_len, H, wg)
    plan = build_staged_cantilever(n_seg=n, seg_len=seg_len, tower_height=H, wg=wg,
                                   strands=strands, pretension=pre)

    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver().run(plan)

    # 逐阶段挠度对比：打印各阶段"当前悬臂端"，误差在所有共同节点上取最大
    print("=== 当前悬臂端竖向挠度 (mm)：direct vs opensees ===")
    max_disp_abs = 0.0
    disp_scale = 1e-12
    for a, b in zip(rd.records, ro.records):
        deck = [nid for nid in a.disp if 0 < nid < 900]   # 排除锚点 999/根部 0
        tip = max(deck) if deck else 0
        da, db = a.disp[tip][1], b.disp[tip][1]
        print(f"  {a.label:7s} (tip n{tip}): direct={da*1000:9.3f}  opensees={db*1000:9.3f}  Δ={(da-db)*1000:+.2e}")
        for nid in a.disp:
            max_disp_abs = max(max_disp_abs, abs(a.disp[nid][1] - b.disp[nid][1]))
            disp_scale = max(disp_scale, abs(a.disp[nid][1]), abs(b.disp[nid][1]))

    # 索应力历程对比
    hd = rd.cable_stress_history()
    ho = ro.cable_stress_history()
    max_stress_abs = 0.0
    stress_scale = 1e-12
    print("\n=== 索应力历程对比（成形态，MPa）===")
    for cid in sorted(hd):
        sd = hd[cid][-1][1]
        so = ho[cid][-1][1]
        max_stress_abs = max(max_stress_abs, abs(sd - so))
        stress_scale = max(stress_scale, abs(sd), abs(so))
        print(f"  cable {cid}: direct={sd/1e6:8.2f}  opensees={so/1e6:8.2f}  Δ={(sd-so)/1e6:+.3f}")
    # 全历程最大误差
    for cid in hd:
        for (_, sd), (_, so) in zip(hd[cid], ho[cid]):
            max_stress_abs = max(max_stress_abs, abs(sd - so))
            stress_scale = max(stress_scale, abs(sd), abs(so))

    drel = max_disp_abs / disp_scale
    srel = max_stress_abs / stress_scale
    print("\n=== 误差（按整体量级归一）===")
    print(f"  挠度: max|Δ|={max_disp_abs:.3e} m    rel={drel:.3e}")
    print(f"  索应力: max|Δ|={max_stress_abs/1e6:.3e} MPa rel={srel:.3e}")
    ok = max(drel, srel) < tol_rel
    print(f"\n判定：{'通过 ✅' if ok else '未通过 ❌'}  (相对阈值 {tol_rel:.1e})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="逐阶段施工：自研 vs OpenSees 交叉校核")
    parser.add_argument("--n", type=int, default=6)
    parser.add_argument("--tol", type=float, default=2e-2,
                        help="相对阈值；自研为线性、OpenSees 为几何精确，大挠度算例约 1~2%")
    args = parser.parse_args()
    sys.exit(0 if run(args.n, args.tol) else 1)


if __name__ == "__main__":
    main()
