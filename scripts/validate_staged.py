"""Cross-check staged construction results with the OpenSees backend.

The direct solver and OpenSees should agree closely before the closure
balancing step. During ``closure_balance`` the OpenSees backend now uses its
own tangent stiffness matrix from ``printA('-ret')`` to compute equivalent
closure loads, so the meaningful OpenSees check there is the residual of the
target closure DOFs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bridgezoo.fem.staged import StagedDirectSolver, StagedOpenSeesSolver, build_staged_cantilever

RIGHT_TIP, LEFT_TIP = 200, 201


def default_pretension(n, anchor_base, anchor_spacing, right_start, right_spacing, wg):
    """Rough vertical-equilibrium pretension estimate."""
    out = []
    for i in range(1, n + 1):
        h = anchor_base + (i - 1) * anchor_spacing
        dist = right_start + (i - 1) * right_spacing
        out.append(2.0 * wg * right_spacing * math.hypot(dist, h) / h)
    return out


def _record_index(result, label: str) -> int:
    for i, rec in enumerate(result.records):
        if rec.label == label:
            return i
    raise ValueError(f"record not found: {label}")


def _max_deck_y_error(rd, ro, rec_index: int) -> tuple[float, float]:
    a, b = rd.records[rec_index], ro.records[rec_index]
    deck = [nid for nid in rd.deck_ids if nid in a.disp and nid in b.disp]
    max_abs = max(abs(a.disp[nid][1] - b.disp[nid][1]) for nid in deck)
    scale = max(max(abs(a.disp[nid][1]), abs(b.disp[nid][1]), 1e-12) for nid in deck)
    return max_abs, max_abs / scale


def _stress_at_label(history: list[tuple[str, float]], label: str) -> float:
    for rec_label, value in history:
        if rec_label == label:
            return value
    raise ValueError(f"stress history label not found: {label}")


def _max_stress_error(rd, ro, label: str) -> tuple[float, float]:
    hd = rd.cable_stress_history()
    ho = ro.cable_stress_history()
    max_abs = 0.0
    scale = 1e-12
    for cid in hd:
        sd = _stress_at_label(hd[cid], label)
        so = _stress_at_label(ho[cid], label)
        max_abs = max(max_abs, abs(sd - so))
        scale = max(scale, abs(sd), abs(so))
    return max_abs, max_abs / scale


def _closure_residual(ro) -> float:
    last = ro.records[-1]
    if last.label != "closure_balance":
        raise ValueError(f"last record is {last.label!r}, expected closure_balance")
    return max(abs(last.disp[LEFT_TIP][1]), abs(last.disp[RIGHT_TIP][0]), abs(last.disp[RIGHT_TIP][2]))


def run(n: int, tol_rel: float, cable_element: str = "linear", wg: float = 1.0e5,
        closure_tol: float = 1e-9) -> bool:
    anchor_base, anchor_spacing, right_start, right_spacing = 25.0, 3.0, 6.0, 8.0
    strands = [20] * n
    pre = default_pretension(n, anchor_base, anchor_spacing, right_start, right_spacing, wg)
    plan = build_staged_cantilever(
        n_seg=n,
        anchor_base_height=anchor_base,
        anchor_spacing=anchor_spacing,
        right_start=right_start,
        right_spacing=right_spacing,
        wg=wg,
        strands=strands,
        pretension=pre,
    )

    rd = StagedDirectSolver().run(plan)
    ro = StagedOpenSeesSolver(cable_element=cable_element).run(plan)

    print(f"OpenSees cable element: {cable_element}")
    print("=== Recorded stages ===")
    print("  " + ", ".join(rec.label for rec in ro.records))

    tip_index = _record_index(ro, "tip_free")
    disp_abs, disp_rel = _max_deck_y_error(rd, ro, tip_index)
    stress_abs, stress_rel = _max_stress_error(rd, ro, "tip_free")
    residual = _closure_residual(ro)

    a, b = rd.records[tip_index], ro.records[tip_index]
    present = [nid for nid in rd.deck_ids if nid in a.disp and nid in b.disp]
    tip = max(present, key=lambda k: abs(rd.coords[k][0]))
    print("\n=== tip_free cross-check: direct vs OpenSees ===")
    print(f"  current tip node: {tip}")
    print(f"  direct tip uy   : {a.disp[tip][1] * 1000:12.6f} mm")
    print(f"  OpenSees tip uy : {b.disp[tip][1] * 1000:12.6f} mm")
    print(f"  deck uy max abs : {disp_abs:.6e} m")
    print(f"  deck uy rel     : {disp_rel:.6e}")
    print(f"  stress max abs  : {stress_abs / 1e6:.6e} MPa")
    print(f"  stress rel      : {stress_rel:.6e}")

    last = ro.records[-1]
    print("\n=== closure_balance OpenSees residual ===")
    print(f"  left  uy target residual : {last.disp[LEFT_TIP][1]: .6e} m")
    print(f"  right ux target residual : {last.disp[RIGHT_TIP][0]: .6e} m")
    print(f"  right rz target residual : {last.disp[RIGHT_TIP][2]: .6e} rad")
    print(f"  max residual             : {residual:.6e}")

    ok = max(disp_rel, stress_rel) < tol_rel and residual < closure_tol
    print(f"\nResult: {'PASS' if ok else 'FAIL'}")
    print(f"  relative tolerance: {tol_rel:.1e}")
    print(f"  closure tolerance : {closure_tol:.1e}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate staged direct solver against OpenSees.")
    parser.add_argument("--n", type=int, default=12)
    parser.add_argument("--wg", type=float, default=1.0e5, help="Girder self-weight line load (N/m).")
    parser.add_argument("--cable-element", choices=["linear", "corot"], default="linear")
    parser.add_argument("--tol", type=float, default=None, help="Relative tolerance for tip_free comparison.")
    parser.add_argument("--closure-tol", type=float, default=1e-9)
    args = parser.parse_args()

    tol = args.tol if args.tol is not None else (2.5e-2 if args.cable_element == "linear" else 5e-2)
    sys.exit(0 if run(args.n, tol, args.cable_element, args.wg, args.closure_tol) else 1)


if __name__ == "__main__":
    main()
