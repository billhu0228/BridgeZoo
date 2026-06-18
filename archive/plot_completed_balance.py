"""Plot completed-bridge balance using the staged model parameters.

This script builds the same geometry, cable layout, material properties, and
loads as ``build_staged_cantilever`` with the defaults from
``plot_staged_deck_growth.py``.  Unlike staged construction, all girder
segments and all stay cables are active from the first and only solve.  The
completed bridge does not use the staged closure balancing loads; it activates
the final supports directly.

The backend can be switched between ``direct`` and ``opensees``.  The output is
meant to be compared with the final staged construction state: deck deflection
uses the same node ids, and cable forces/stresses use the same cable ids.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJECT_ROOT = Path(ROOT)

from bridgezoo.fem.completed import CompletedDirectSolver, CompletedOpenSeesSolver
from bridgezoo.fem.model import SolveResult, StructuralModel
from bridgezoo.fem.staged import (
    StagedDirectSolver,
    build_completed_model,
    build_staged_cantilever,
)
from scripts.plot_staged_deck_growth import MODEL_DEFAULTS, default_pretension

TOWER_DECK_NODE = 0
VERTICAL_SUPPORT_NODE = 201
MIDSPAN_NODE = 200


def build_completed_from_staged_params(args) -> tuple[StructuralModel, dict]:
    """Build a completed-bridge model from the staged plan geometry."""

    n = args.n
    strands = [args.strands] * n
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
        left_start=args.left_start,
        left_spacing=args.left_spacing,
        left_end=args.left_end,
        right_start=args.right_start,
        right_spacing=args.right_spacing,
        right_end=args.right_end,
        wg=args.wg,
        strands=strands,
        pretension=pretension,
    )

    model, meta = build_completed_model(plan, name=f"completed_from_staged_N{n}")
    meta["pretension"] = pretension
    meta["plan"] = plan
    return model, meta


def run(args) -> dict:
    model, meta = build_completed_from_staged_params(args)
    solver = CompletedOpenSeesSolver() if args.backend == "opensees" else CompletedDirectSolver()
    result = solver.solve(model)

    print("Completed bridge model:")
    print(f"  {model.summary()}")
    print(
        f"  n={args.n}, left spacing={args.left_spacing:g} m, "
        f"right spacing={args.right_spacing:g} m"
    )
    xs = [meta["coords"][nid][0] for nid in meta["deck_ids"]]
    print(f"  deck x range: {min(xs):.3f} m .. {max(xs):.3f} m")
    print(f"  converged: {result.converged} ({result.backend})")

    _print_deck_result(result, meta)
    _print_cable_result(result)

    if args.plot:
        _plot_balance_state(model, result, meta, args.plot, args.scale, args.beam_depth)
    if args.compare_staged:
        _compare_with_staged_final(result, meta)

    return {"model": model, "result": result, "meta": meta}


def _print_deck_result(result: SolveResult, meta: dict) -> None:
    print("\n=== Deck deflection, completed balance ===")
    uy_values = []
    for nid in meta["deck_ids"]:
        x = meta["coords"][nid][0]
        uy = result.disp[nid][1]
        uy_values.append(uy)
        print(f"  node={nid:>4d} x={x:>9.3f} m  uy={uy * 1000:>12.6f} mm")
    print(
        f"  uy max={max(uy_values) * 1000:.6f} mm, "
        f"min={min(uy_values) * 1000:.6f} mm"
    )


def _print_cable_result(result: SolveResult) -> None:
    print("\n=== Cable force/stress, completed balance ===")
    for cid in sorted(result.cable_force):
        force = result.cable_force[cid]
        stress = result.cable_stress[cid]
        print(f"  cable={cid:>4d}  N={force / 1e3:>12.6f} kN  sigma={stress / 1e6:>12.6f} MPa")


def _resolve_project_path(path: str | None) -> Path | None:
    if not path:
        return None
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p


def _plot_balance_state(model: StructuralModel, result: SolveResult, meta: dict, out: str,
                        scale: float, beam_depth: float = 1.0) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from bridgezoo.render.deformed_shape import deformed_chain_shape
    from bridgezoo.render.mpl_cjk import use_cjk_font

    use_cjk_font()

    out_path = _resolve_project_path(out)
    if out_path is None:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)

    coords = meta["coords"]
    deck = meta["deck_ids"]
    xs = [coords[nid][0] for nid in deck]
    ys0 = [coords[nid][1] for nid in deck]
    ys = [coords[nid][1] + result.disp[nid][1] * scale for nid in deck]
    # Hermite-interpolated deformed axis (nodal rotations shape the curve).
    xc, yc = deformed_chain_shape(coords, result.disp, deck, scale=scale)
    xn = [coords[nid][0] + result.disp[nid][0] * scale for nid in deck]

    tower_top = max(coords[nid][1] for nid in meta["anchor_ids"])
    span = (max(xs) - min(xs)) or 1.0
    all_y = ys + [0.0, tower_top]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(xs, ys0, color="0.65", lw=1.8, marker="o", ms=3, label="design deck")
    # Girder band at true depth (not displacement-scaled); vertical offset is
    # used instead of the section normal because the axes aspect is far from 1.
    if beam_depth > 0.0:
        ax.fill_between(xc, yc - 0.5 * beam_depth, yc + 0.5 * beam_depth,
                        color="#d55e00", alpha=0.35, lw=0)
    ax.plot(xc, yc, color="#d55e00", lw=1.6, label=f"completed deck x{scale:g}")
    ax.plot(xn, ys, ls="none", marker="o", ms=4, color="#d55e00")
    ax.plot([0.0, 0.0], [0.0, tower_top], color="0.35", lw=3.0, label="tower")
    ax.plot(
        [coords[nid][0] for nid in meta["anchor_ids"]],
        [coords[nid][1] for nid in meta["anchor_ids"]],
        ls="none",
        marker="_",
        ms=10,
        color="0.35",
    )

    stresses = [abs(result.cable_stress[cid]) for cid in result.cable_stress]
    max_stress = max(stresses + [1.0])
    first_cable = True
    for cid, (i, j) in sorted(meta["cable_nodes"].items()):
        anc, deckn = (i, j) if i in meta["anchor_ids"] else (j, i)
        stress_ratio = abs(result.cable_stress[cid]) / max_stress
        ax.plot(
            [coords[anc][0], coords[deckn][0] + result.disp[deckn][0] * scale],
            [coords[anc][1], coords[deckn][1] + result.disp[deckn][1] * scale],
            color=plt.cm.viridis(0.2 + 0.75 * stress_ratio),
            lw=0.8 + 1.8 * stress_ratio,
            alpha=0.85,
            label="cables" if first_cable else None,
        )
        first_cable = False

    ax.set_title("Completed bridge balance state")
    ax.set_xlabel("x [m]")
    ax.set_ylabel(f"y [m], vertical displacement x{scale:g}")
    ax.set_xlim(min(xs) - 0.08 * span, max(xs) + 0.08 * span)
    ax.set_ylim(min(all_y) - 0.12 * tower_top - 0.5 * beam_depth,
                max(all_y) + 0.12 * tower_top + 0.5 * beam_depth)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"\nSaved completed balance plot: {out_path}")


def _max_diff(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    max_abs = max((abs(a - b) for a, b in pairs), default=0.0)
    scale = max((max(abs(a), abs(b)) for a, b in pairs), default=1e-12)
    return max_abs, max_abs / max(scale, 1e-12)


def _compare_with_staged_final(completed: SolveResult, meta: dict) -> None:
    plan = meta["plan"]
    staged = StagedDirectSolver().run(plan)
    final = staged.records[-1]

    deck_pairs = [(completed.disp[nid][1], final.disp[nid][1]) for nid in meta["deck_ids"]]
    force_pairs = [
        (completed.cable_force[cid], final.cable_force[cid])
        for cid in sorted(meta["cable_nodes"])
        if cid in completed.cable_force and cid in final.cable_force
    ]
    stress_pairs = [
        (completed.cable_stress[cid], final.cable_stress[cid])
        for cid in sorted(meta["cable_nodes"])
        if cid in completed.cable_stress and cid in final.cable_stress
    ]

    uy_abs, uy_rel = _max_diff(deck_pairs)
    force_abs, force_rel = _max_diff(force_pairs)
    stress_abs, stress_rel = _max_diff(stress_pairs)
    print("\n=== Completed vs staged final direct comparison ===")
    print(f"  deck uy max diff     : {uy_abs * 1000:.6f} mm ({uy_rel * 100:.6f}%)")
    print(f"  cable force max diff : {force_abs / 1e3:.6f} kN ({force_rel * 100:.6f}%)")
    print(f"  cable stress max diff: {stress_abs / 1e6:.6f} MPa ({stress_rel * 100:.6f}%)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot completed-bridge balance with staged geometry.")
    parser.add_argument("--n", type=int, default=MODEL_DEFAULTS["n"], help="Number of cables per side.")
    parser.add_argument("--backend", choices=["direct", "opensees"], default="direct")
    parser.add_argument("--strands", type=int, default=20, help="Strands per cable.")
    parser.add_argument("--anchor-base", type=float, default=MODEL_DEFAULTS["anchor_base"])
    parser.add_argument("--anchor-spacing", type=float, default=MODEL_DEFAULTS["anchor_spacing"])
    parser.add_argument("--anchor-free", type=float, default=MODEL_DEFAULTS["anchor_free"])
    parser.add_argument("--left-start", type=float, default=MODEL_DEFAULTS["left_start"])
    parser.add_argument("--left-spacing", type=float, default=MODEL_DEFAULTS["left_spacing"])
    parser.add_argument("--left-end", type=float, default=MODEL_DEFAULTS["left_end"])
    parser.add_argument("--right-start", type=float, default=MODEL_DEFAULTS["right_start"])
    parser.add_argument("--right-spacing", type=float, default=MODEL_DEFAULTS["right_spacing"])
    parser.add_argument("--right-end", type=float, default=MODEL_DEFAULTS["right_end"])
    parser.add_argument("--wg", type=float, default=MODEL_DEFAULTS["wg"], help="Girder self-weight line load [N/m].")
    parser.add_argument("--plot", type=str, default="results/completed_balance.png")
    parser.add_argument("--scale", type=float, default=15.0, help="Vertical displacement plot scale.")
    parser.add_argument("--beam-depth", type=float, default=1.0,
                        help="Drawn girder depth [m] (true scale, not displacement-scaled; "
                             "0 = centerline only; default matches the builder section A=10/Iz=10/12).")
    parser.add_argument("--compare-staged", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
