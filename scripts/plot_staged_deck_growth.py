"""Visualize staged girder growth solved by the staged solvers.

Draws the main girder after every recorded construction stage, so one can see:

1. the **double cantilever** grows segment by segment on both sides of the tower;
2. the active girder deforms as new self-weight and cables are added;
3. the **fan** of stay cables anchored at different tower heights.

Geometry is read from :class:`StagedResult` (``coords`` / ``cable_nodes`` /
``anchor_ids`` / ``deck_ids``) so it works for any (asymmetric, fan) layout —
no assumption that ``x = node_id * seg_len``.

Usage::

    py -3.12 -m scripts.plot_staged_deck_growth --n 6
    py -3.12 -m scripts.plot_staged_deck_growth --n 8 --scale 20 --backend opensees
    py -3.12 -m scripts.plot_staged_deck_growth --frames-dir results/staged_frames

Default output: ``results/staged_deck_growth.gif`` under the project root.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJECT_ROOT = Path(ROOT)

from bridgezoo.fem.staged import (
    StagedDirectSolver,
    StagedOpenSeesSolver,
    build_staged_cantilever,
)

MODEL_DEFAULTS = {
    "n": 8,
    "anchor_base": 32.0,
    "anchor_spacing": 2.0,
    "anchor_free": 10.0,
    "left_start": 12.0,
    "left_spacing": 8.0,
    "left_end": 4.0,
    "right_start": 12.0,
    "right_spacing": 12.0,
    "right_end": 4.0,
    "wg": 1.0e5,
}


def default_pretension(
    n,
    anchor_base,
    anchor_spacing,
    left_start,
    left_spacing,
    right_start,
    right_spacing,
    wg,
):
    """各索目标张力（粗估）：按左右各自几何估计每根索的竖向平衡分量。"""
    out = []
    for i in range(1, n + 1):
        h = anchor_base + (i - 1) * anchor_spacing
        right_dist = right_start + (i - 1) * right_spacing
        left_dist = left_start + (i - 1) * left_spacing
        right_len = math.hypot(right_dist, h)
        left_len = math.hypot(left_dist, h)
        out.append((wg * right_spacing * right_len / h, wg * left_spacing * left_len / h))
    return out


def run(args) -> None:
    n = args.n
    strands = [20] * n
    pretension = default_pretension(
        n,
        args.anchor_base,
        args.anchor_spacing,
        args.left_start,
        args.left_spacing,
        args.right_start,
        args.right_spacing,
        args.wg,
    )
    plan = build_staged_cantilever(
        n_seg=n,
        anchor_base_height=args.anchor_base,
        anchor_spacing=args.anchor_spacing,
        anchor_top_free=args.anchor_free,
        left_start=args.left_start, left_spacing=args.left_spacing, left_end=args.left_end,
        right_start=args.right_start, right_spacing=args.right_spacing, right_end=args.right_end,
        wg=args.wg,
        strands=strands,
        pretension=pretension,
    )
    solver = StagedOpenSeesSolver() if args.backend == "opensees" else StagedDirectSolver()
    result = solver.run(plan)
    if not result.records:
        raise RuntimeError("No staged records were produced.")

    _plot_animation(result, n=n, scale=args.scale, out=args.out,
                    frames_dir=args.frames_dir, fps=args.fps)


def _resolve_project_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _present_deck(result, record) -> list[int]:
    """本阶段已安装的主梁节点，按 x 升序（左端→右端）。"""
    ids = [nid for nid in result.deck_ids if nid in record.disp]
    return sorted(ids, key=lambda nid: result.coords[nid][0])


def _force_label(fx: float, fy: float, mz: float) -> str:
    parts = []
    if abs(fx) > 1e-9:
        parts.append(f"Fx={fx / 1e3:.1f} kN")
    if abs(fy) > 1e-9:
        parts.append(f"Fy={fy / 1e3:.1f} kN")
    if abs(mz) > 1e-9:
        parts.append(f"Mz={mz / 1e3:.1f} kN*m")
    return "\n".join(parts)


def _draw_applied_loads(ax, result, rec, scale: float, span: float) -> None:
    loads = getattr(rec, "applied_loads", {})
    if not loads:
        return

    force_norms = [math.hypot(fx, fy) for fx, fy, _ in loads.values()]
    max_force = max(force_norms + [1.0])
    arrow_len = 0.08 * span
    first_arrow = True

    for nid, (fx, fy, mz) in loads.items():
        if nid not in rec.disp or nid not in result.coords:
            continue
        x = result.coords[nid][0]
        y = result.coords[nid][1] + rec.disp[nid][1] * scale
        fmag = math.hypot(fx, fy)

        if fmag > 1e-9:
            dx = arrow_len * fx / max_force
            dy = arrow_len * fy / max_force
            ax.annotate(
                "",
                xy=(x + dx, y + dy),
                xytext=(x, y),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#7b3294"),
            )
            if first_arrow:
                ax.plot([], [], color="#7b3294", lw=2.2, label="闭合外荷载")
                first_arrow = False

        if abs(mz) > 1e-9:
            symbol = "Mz+" if mz > 0 else "Mz-"
            ax.text(x, y + 0.06 * span, symbol, color="#7b3294",
                    ha="center", va="center", fontsize=18, fontweight="bold")

        label = _force_label(fx, fy, mz)
        if label:
            ax.text(x + 0.015 * span, y + 0.025 * span, label,
                    color="#4a1486", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#7b3294", alpha=0.88))


def _plot_animation(result, n, scale, out, frames_dir, fps) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    from bridgezoo.render.mpl_cjk import use_cjk_font

    use_cjk_font()

    records = result.records
    coords = result.coords
    out_path = _resolve_project_path(out)
    frames_path = _resolve_project_path(frames_dir)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    if frames_path is not None:
        frames_path.mkdir(parents=True, exist_ok=True)
        for stale in frames_path.glob("stage_*.png"):
            stale.unlink()

    # 画幅范围（含塔、所有梁节点、放大后的变形）
    xs_all = [coords[nid][0] for nid in result.deck_ids]
    tower_top = max((coords[a][1] for a in result.anchor_ids), default=10.0)
    all_y = [coords[nid][1] + rec.disp[nid][1] * scale
             for rec in records for nid in result.deck_ids if nid in rec.disp]
    span = (max(xs_all) - min(xs_all)) or 1.0
    xmin, xmax = min(xs_all) - 0.12 * span, max(xs_all) + 0.12 * span
    ymin = min(all_y + [0.0]) - 0.10 * tower_top
    ymax = max(all_y + [tower_top]) + 0.10 * tower_top

    fig, ax = plt.subplots(figsize=(11, 5.6))

    def draw(k: int):
        rec = records[k]
        ax.clear()
        deck = _present_deck(result, rec)
        xs = [coords[nid][0] for nid in deck]
        ys = [rec.disp[nid][1] * scale for nid in deck]

        # 设计轴线 + 塔 + 锚点
        ax.plot([min(xs_all), max(xs_all)], [0, 0], color="0.82", lw=1.0, ls="--", label="设计主梁轴线")
        ax.plot([0, 0], [0, tower_top], color="0.35", lw=3.0, label="索塔")
        ax.plot([coords[a][0] for a in result.anchor_ids],
                [coords[a][1] for a in result.anchor_ids],
                ls="none", marker="_", ms=10, color="0.35")

        # 活动拉索（扇面）
        first_cable = True
        for cid, force in rec.cable_force.items():
            i, j = result.cable_nodes[cid]
            anc, deckn = (i, j) if i in result.anchor_ids else (j, i)
            ax.plot([coords[anc][0], coords[deckn][0]],
                    [coords[anc][1], rec.disp[deckn][1] * scale],
                    color="#4f8fba", lw=1.1, alpha=0.75,
                    label="拉索" if first_cable else None)
            first_cable = False

        # 已安装主梁（未变形 / 变形）
        ax.plot(xs, [0] * len(xs), color="0.6", lw=1.6, marker="o", ms=3, label="已安装主梁(未变形)")
        ax.plot(xs, ys, color="#d55e00", lw=2.6, marker="o", ms=5, label=f"变形主梁 x{scale:g}")

        # 高亮本阶段新增节段（相对上一记录新增的梁节点所在段）
        if k > 0:
            prev_nodes = set(records[k - 1].disp)
            new_nodes = [nid for nid in deck if nid not in prev_nodes]
            for nid in new_nodes:
                pos = deck.index(nid)
                for nb in (pos - 1, pos + 1):
                    if 0 <= nb < len(deck):
                        ax.plot([xs[pos], xs[nb]], [ys[pos], ys[nb]],
                                color="#b2182b", lw=4.0, alpha=0.9)

        tips = [nid for nid in (200, 201) if nid in rec.disp]
        tiptxt = "  ".join(f"{'右' if t == 200 else '左'}端 dy={rec.disp[t][1] * 1000:.1f}mm" for t in tips)
        if k == len(records) - 1:
            _draw_applied_loads(ax, result, rec, scale, span)
        ax.set_title(f"正向逐阶段半桥施工：{rec.label}   {tiptxt}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel(f"y [m]  (位移放大 {scale:g} 倍)")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        fig.tight_layout()
        return ax.lines

    if frames_path is not None:
        for i in range(len(records)):
            draw(i)
            fig.savefig(frames_path / f"stage_{i + 1:02d}_{records[i].label}.png", dpi=140)
        print(f"已保存阶段图片：{frames_path}")

    if out_path is not None:
        anim = FuncAnimation(fig, draw, frames=len(records), interval=1000 / max(fps, 1), blit=False)
        anim.save(out_path, writer=PillowWriter(fps=fps), dpi=130)
        print(f"已保存推进动画：{out_path}")

    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="绘制逐阶段双悬臂主梁增长与变形过程（扇面索）")
    p.add_argument("--n", type=int, default=MODEL_DEFAULTS["n"], help="每侧索数")
    p.add_argument("--backend", choices=["direct", "opensees"], default="opensees")
    # 扇面锚点
    p.add_argument("--anchor-base", type=float, default=MODEL_DEFAULTS["anchor_base"], help="参数a：最低锚点高度")
    p.add_argument("--anchor-spacing", type=float, default=MODEL_DEFAULTS["anchor_spacing"], help="参数b：锚点间距")
    p.add_argument("--anchor-free", type=float, default=MODEL_DEFAULTS["anchor_free"], help="参数c：顶部自由高度")
    # 双悬臂（左右）
    p.add_argument("--left-start", type=float, default=MODEL_DEFAULTS["left_start"])
    p.add_argument("--left-spacing", type=float, default=MODEL_DEFAULTS["left_spacing"])
    p.add_argument("--left-end", type=float, default=MODEL_DEFAULTS["left_end"])
    p.add_argument("--right-start", type=float, default=MODEL_DEFAULTS["right_start"])
    p.add_argument("--right-spacing", type=float, default=MODEL_DEFAULTS["right_spacing"])
    p.add_argument("--right-end", type=float, default=MODEL_DEFAULTS["right_end"])
    p.add_argument("--wg", type=float, default=MODEL_DEFAULTS["wg"], help="主梁自重线荷载 [N/m]")
    p.add_argument("--scale", type=float, default=10.0, help="竖向位移绘图放大倍数")
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--out", type=str, default="results/staged_deck_growth_ops.gif")
    p.add_argument("--frames-dir", type=str, default="results/frames")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
