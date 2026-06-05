"""交叉校核：同一结构定义，自研直接刚度法 vs OpenSees，结果应一致。

流程：由 :func:`bridgezoo.fem.staged.build_staged_cantilever` 的施工计划派生出成桥
(完成态) ``StructuralModel``（:func:`bridgezoo.fem.staged.build_completed_model`），
分别交给 :class:`CompletedDirectSolver` 与 :class:`CompletedOpenSeesSolver` 求解，逐项对比节点
位移、梁端力、索力，输出最大绝对/相对误差并按阈值判定通过。这是论文实验 E1 的证据。

> 校核对象为单塔双悬臂半桥（与施工阶段模型同源）。

用法::

    python -m scripts.validate_fem --n 6
    python -m scripts.validate_fem --n 6 --sigma 600 --strands 20

> 需要 openseespy（建议 Python 3.11–3.13）。
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bridgezoo.fem.completed import CompletedDirectSolver, CompletedOpenSeesSolver
from bridgezoo.fem.staged import build_completed_model, build_staged_cantilever

# 单股钢绞线截面积 (m²)，与 build_staged_cantilever 的默认一致，用于把目标初应力换算为预张力。
STRAND_AREA = 1.4e-4


def _max_abs_diff(a: dict, b: dict):
    """两个 {id: 标量 或 元组} 字典的最大绝对差 与 **按整体量级归一**的相对差。

    相对差用全体分量的最大幅值作分母（而非逐分量），避免近零分量（如轴力≈0）使
    逐分量相对误差虚高。
    """
    max_abs = 0.0
    scale = 1e-12
    for k in a:
        va, vb = a[k], b[k]
        comps = zip(va, vb) if isinstance(va, (list, tuple)) else [(va, vb)]
        for x, y in comps:
            max_abs = max(max_abs, abs(x - y))
            scale = max(scale, abs(x), abs(y))
    return max_abs, max_abs / scale


def run(n: int, sigma: float, strands: int, tol_rel: float) -> bool:
    # 由施工计划派生成桥状态：各索目标初应力 sigma(MPa) → 预张力 T = sigma·A。
    pretension = sigma * 1e6 * STRAND_AREA * strands
    plan = build_staged_cantilever(
        n_seg=n,
        strand_area=STRAND_AREA,
        strands=[int(strands)] * n,
        pretension=[pretension] * n,
    )
    model, _ = build_completed_model(plan)
    print("结构定义：", model.summary())

    r_direct = CompletedDirectSolver().solve(model)
    r_ops = CompletedOpenSeesSolver().solve(model)
    print(f"求解状态：direct converged={r_direct.converged}, opensees converged={r_ops.converged}")

    # 位移竖向分量
    uy_d = {k: v[1] for k, v in r_direct.disp.items()}
    uy_o = {k: v[1] for k, v in r_ops.disp.items()}
    da, dr = _max_abs_diff(uy_d, uy_o)

    # 梁端力（全部 6 分量）
    fa, fr = _max_abs_diff(r_direct.frame_force, r_ops.frame_force)

    # 索力
    ca, cr = _max_abs_diff(r_direct.cable_force, r_ops.cable_force)

    print("\n=== 直接刚度法 vs OpenSees 误差 ===")
    print(f"  节点竖向位移 Uy : max|Δ|={da:.3e} m   max rel={dr:.3e}")
    print(f"  梁端力          : max|Δ|={fa:.3e} N·(m) max rel={fr:.3e}")
    print(f"  索力 N          : max|Δ|={ca:.3e} N    max rel={cr:.3e}")

    # 抽样展示（右悬臂端节点 200 竖向位移；右塔根索 1001 轴力）
    tip = 200
    print("\n  抽样（右悬臂端节点竖向位移, mm）：")
    print(f"    direct  ={r_direct.uy(tip) * 1000:10.4f}  opensees={r_ops.uy(tip) * 1000:10.4f}")
    print("  抽样（索 1001 轴力, kN）：")
    print(f"    direct  ={r_direct.cable_force[1001] / 1e3:10.3f}  opensees={r_ops.cable_force[1001] / 1e3:10.3f}")

    ok = max(dr, fr, cr) < tol_rel
    print(f"\n判定：{'通过 [PASS]' if ok else '未通过 [FAIL]'}  (相对误差阈值 {tol_rel:.1e})")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="自研直接刚度法 vs OpenSees 交叉校核")
    parser.add_argument("--n", type=int, default=6, help="每侧索数（节段数）")
    parser.add_argument("--sigma", type=float, default=600.0, help="各索初应力 (MPa)")
    parser.add_argument("--strands", type=int, default=20, help="各索股数")
    parser.add_argument("--tol", type=float, default=1e-4, help="相对误差阈值")
    args = parser.parse_args()
    ok = run(args.n, args.sigma, args.strands, args.tol)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
