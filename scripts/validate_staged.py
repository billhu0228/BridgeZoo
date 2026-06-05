"""Cross-check staged construction tip displacements with the OpenSees backend.

The validation case uses the same structural parameters as
``plot_staged_deck_growth.py``.  For every recorded construction stage, the
script compares the current farthest deck node displacement from the direct
solver and the OpenSees backend, reporting both absolute and percentage
differences.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bridgezoo.fem.staged import StagedDirectSolver, StagedOpenSeesSolver, build_staged_cantilever
from scripts.plot_staged_deck_growth import MODEL_DEFAULTS, default_pretension

RIGHT_TIP, LEFT_TIP = 200, 201


def _stage_tip_y_errors(rd, ro) -> list[dict[str, float | int | str]]:
    if len(rd.records) != len(ro.records):
        raise ValueError(f"record count mismatch: direct={len(rd.records)}, opensees={len(ro.records)}")

    rows = []
    for idx, (direct_rec, ops_rec) in enumerate(zip(rd.records, ro.records)):
        if direct_rec.label != ops_rec.label:
            raise ValueError(
                f"record label mismatch at index {idx}: "
                f"direct={direct_rec.label!r}, opensees={ops_rec.label!r}"
            )
        deck = [nid for nid in rd.deck_ids if nid in direct_rec.disp and nid in ops_rec.disp]
        if not deck:
            raise ValueError(f"no shared deck nodes at stage {direct_rec.label!r}")
        tip = max(deck, key=lambda nid: abs(rd.coords[nid][0]))
        direct_uy = direct_rec.disp[tip][1]
        ops_uy = ops_rec.disp[tip][1]
        diff = ops_uy - direct_uy
        scale = max(abs(direct_uy), abs(ops_uy), 1e-12)
        rows.append(
            {
                "label": direct_rec.label,
                "node": tip,
                "x": rd.coords[tip][0],
                "direct_uy": direct_uy,
                "opensees_uy": ops_uy,
                "diff": diff,
                "rel": abs(diff) / scale,
            }
        )
    return rows


def _closure_residual(ro) -> float:
    last = ro.records[-1]
    if last.label != "closure_balance":
        raise ValueError(f"last record is {last.label!r}, expected closure_balance")
    return max(abs(last.disp[LEFT_TIP][1]), abs(last.disp[RIGHT_TIP][0]), abs(last.disp[RIGHT_TIP][2]))


def run(args, tol_rel: float) -> bool:
    n = args.n
    strands = [20] * n
    pre = default_pretension(
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
        pretension=pre,
    )

    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element=args.cable_element).run(plan)

    print(f"OpenSees cable element: {args.cable_element}")
    print("Structural parameters:")
    print(
        f"  n={args.n}, anchor_base={args.anchor_base:g}, anchor_spacing={args.anchor_spacing:g}, "
        f"anchor_free={args.anchor_free:g}"
    )
    print(
        f"  left_start={args.left_start:g}, left_spacing={args.left_spacing:g}, left_end={args.left_end:g}"
    )
    print(
        f"  right_start={args.right_start:g}, right_spacing={args.right_spacing:g}, right_end={args.right_end:g}, "
        f"wg={args.wg:g}"
    )
    print("=== Recorded stages ===")
    print("  " + ", ".join(rec.label for rec in ro.records))

    rows = _stage_tip_y_errors(rd, ro)
    max_row = max(rows, key=lambda row: row["rel"])
    residual = _closure_residual(ro)

    print("\n=== Stage farthest-end uy comparison: direct vs OpenSees ===")
    header = (
        "  stage              node      x[m]   direct[mm]  OpenSees[mm]      diff[mm]      diff[%]"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for row in rows:
        print(
            f"  {row['label']:<16} {row['node']:>5d} {row['x']:>9.3f} "
            f"{row['direct_uy'] * 1000:>12.6f} {row['opensees_uy'] * 1000:>13.6f} "
            f"{row['diff'] * 1000:>13.6f} {row['rel'] * 100:>12.6f}"
        )
    print("\nMaximum staged farthest-end difference:")
    print(
        f"  stage={max_row['label']}, node={max_row['node']}, "
        f"abs diff={abs(max_row['diff']):.6e} m, percent={max_row['rel'] * 100:.6f}%"
    )

    last = ro.records[-1]
    print("\n=== closure_balance OpenSees residual ===")
    print(f"  left  uy target residual : {last.disp[LEFT_TIP][1]: .6e} m")
    print(f"  right ux target residual : {last.disp[RIGHT_TIP][0]: .6e} m")
    print(f"  right rz target residual : {last.disp[RIGHT_TIP][2]: .6e} rad")
    print(f"  max residual             : {residual:.6e}")

    ok = max_row["rel"] < tol_rel and residual < args.closure_tol
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    print(f"  relative tolerance: {tol_rel:.1e} ({tol_rel * 100:.3g}%)")
    print(f"  closure tolerance : {args.closure_tol:.1e}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate staged farthest-end displacement against OpenSees.")
    parser.add_argument("--n", type=int, default=MODEL_DEFAULTS["n"], help="Number of cables on each side.")
    parser.add_argument("--cable-element", choices=["linear", "corot"], default="linear")
    parser.add_argument("--tol", type=float, default=None, help="Relative tolerance for staged farthest-end uy comparison.")
    parser.add_argument("--closure-tol", type=float, default=1e-9)
    parser.add_argument("--anchor-base", type=float, default=MODEL_DEFAULTS["anchor_base"], help="Lowest cable anchor height.")
    parser.add_argument("--anchor-spacing", type=float, default=MODEL_DEFAULTS["anchor_spacing"], help="Vertical spacing between cable anchors.")
    parser.add_argument("--anchor-free", type=float, default=MODEL_DEFAULTS["anchor_free"], help="Tower free height above the highest anchor.")
    parser.add_argument("--left-start", type=float, default=MODEL_DEFAULTS["left_start"])
    parser.add_argument("--left-spacing", type=float, default=MODEL_DEFAULTS["left_spacing"])
    parser.add_argument("--left-end", type=float, default=MODEL_DEFAULTS["left_end"])
    parser.add_argument("--right-start", type=float, default=MODEL_DEFAULTS["right_start"])
    parser.add_argument("--right-spacing", type=float, default=MODEL_DEFAULTS["right_spacing"])
    parser.add_argument("--right-end", type=float, default=MODEL_DEFAULTS["right_end"])
    parser.add_argument("--wg", type=float, default=MODEL_DEFAULTS["wg"], help="Girder self-weight line load (N/m).")
    args = parser.parse_args()

    tol = args.tol if args.tol is not None else (2.5e-2 if args.cable_element == "linear" else 5e-2)
    sys.exit(0 if run(args, tol) else 1)


if __name__ == "__main__":
    main()
