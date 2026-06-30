"""Optimize staged cable strands and pretensions.

Example::

    py -3.12 -m scripts.optimize_cables --n 4 --outer-iterations 3 --random-trials 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
PROJECT_ROOT = Path(ROOT)

from bridgezoo.optim import (  # noqa: E402
    CableBounds,
    CableDesignEvaluator,
    CableHybridOptimizer,
    CableOptimizationProblem,
    ContinuousOptions,
    HybridOptions,
    IntegerSearchOptions,
    ObjectiveWeights,
)
from scripts.staged_analysis import MODEL_DEFAULTS, default_pretension, P4B_DEFAULTS  # noqa: E402


def _model_kwargs(args) -> dict:
    return {
        "anchor_base_height": args.anchor_base,
        "anchor_spacing": args.anchor_spacing,
        "anchor_top_free": args.anchor_free,
        "left_start": args.left_start,
        "left_spacing": args.left_spacing,
        "left_end": args.left_end,
        "right_start": args.right_start,
        "right_spacing": args.right_spacing,
        "right_end": args.right_end,
        "wg": args.wg,
        "dw": args.dw,
        "beam_E": args.beam_E,
        "beam_A": args.beam_A,
        "beam_Iz": args.beam_Iz,
    }


def _flatten_stage_pairs(pairs) -> np.ndarray:
    out = []
    for right, left in pairs:
        out.extend((right, left))
    return np.asarray(out, dtype=float)


def _initial_pretension(args) -> np.ndarray:
    return _flatten_stage_pairs(
        default_pretension(
            args.n,
            args.anchor_base,
            args.anchor_spacing,
            args.left_start,
            args.left_spacing,
            args.right_start,
            args.right_spacing,
            args.wg,
        )
    )


def _evaluation_payload(ev) -> dict:
    return {
        "objective": ev.objective,
        "components": {
            "shape": ev.components.shape,
            "total_strands": ev.components.total_strands,
            "stress_uniform": ev.components.stress_uniform,
            "stress_violation": ev.components.stress_violation,
        },
        "metrics": {
            "shape_rmse_mm": ev.metrics.shape_rmse_m * 1000.0,
            "shape_max_abs_mm": ev.metrics.shape_max_abs_m * 1000.0,
            "total_strands": ev.metrics.total_strands,
            "stress_mean_mpa": ev.metrics.stress_mean_mpa,
            "stress_std_mpa": ev.metrics.stress_std_mpa,
            "stress_min_mpa": ev.metrics.stress_min_mpa,
            "stress_max_mpa": ev.metrics.stress_max_mpa,
            "stress_violation_rms_mpa": ev.metrics.stress_violation_rms_mpa,
            "stress_violation_max_mpa": ev.metrics.stress_violation_max_mpa,
        },
        "cables": [
            {
                "cable_id": cid,
                "stage": idx // 2 + 1,
                "side": "right" if idx % 2 == 0 else "left",
                "strands": int(ev.design.strands[idx]),
                "pretension_N": float(ev.design.pretension[idx]),
                "final_stress_MPa": ev.cable_stress_mpa[cid],
            }
            for idx, cid in enumerate(ev.cable_ids)
        ],
        "deck_errors_mm": {str(node): err * 1000.0 for node, err in ev.deck_errors_m.items()},
    }


def _band_verdict_line(best, result, stress_lower: float, stress_upper: float) -> str:
    violation = max(
        0.0,
        stress_lower - best.metrics.stress_min_mpa,
        best.metrics.stress_max_mpa - stress_upper,
    )
    verdict = "SATISFIED" if violation <= 1e-6 else "VIOLATED"
    s_star = result.feasibility_violation_mpa
    lp_note = f", LP bound s*={s_star:.3f} MPa" if s_star is not None else ""
    return (
        f"stress band [{stress_lower:g}, {stress_upper:g}] MPa: {verdict} "
        f"(max violation {violation:.3f} MPa{lp_note})"
    )


def _write_outputs(out_dir: Path, best, history, band_line: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "best_design.json").write_text(
        json.dumps(_evaluation_payload(best), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "objective",
            "shape_rmse_mm",
            "shape_max_abs_mm",
            "total_strands",
            "stress_mean_mpa",
            "stress_std_mpa",
            "stress_min_mpa",
            "stress_max_mpa",
            "stress_violation_rms_mpa",
        ])
        for i, ev in enumerate(history):
            writer.writerow([
                i,
                ev.objective,
                ev.metrics.shape_rmse_m * 1000.0,
                ev.metrics.shape_max_abs_m * 1000.0,
                ev.metrics.total_strands,
                ev.metrics.stress_mean_mpa,
                ev.metrics.stress_std_mpa,
                ev.metrics.stress_min_mpa,
                ev.metrics.stress_max_mpa,
                ev.metrics.stress_violation_rms_mpa,
            ])

    summary = [
        f"objective: {best.objective:.6g}",
        f"shape rmse: {best.metrics.shape_rmse_m * 1000.0:.6f} mm",
        f"shape max abs: {best.metrics.shape_max_abs_m * 1000.0:.6f} mm",
        f"total strands: {best.metrics.total_strands}",
        (
            "stress MPa: "
            f"mean={best.metrics.stress_mean_mpa:.3f}, "
            f"std={best.metrics.stress_std_mpa:.3f}, "
            f"min={best.metrics.stress_min_mpa:.3f}, "
            f"max={best.metrics.stress_max_mpa:.3f}"
        ),
        f"stress violation rms: {best.metrics.stress_violation_rms_mpa:.6f} MPa",
    ]
    if band_line is not None:
        summary.append(band_line)
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def run(args):
    problem = CableOptimizationProblem(
        n_seg=args.n,
        model_kwargs=_model_kwargs(args),
        bounds=CableBounds(
            strand_min=args.strand_min,
            strand_max=args.strand_max,
            stress_lower_mpa=args.stress_lower,
            stress_upper_mpa=args.stress_upper,
            tension_bound_stress_mpa=args.tension_bound_stress,
        ),
        weights=ObjectiveWeights(
            shape=args.weight_shape,
            total_strands=args.weight_strands,
            stress_uniform=args.weight_stress_uniform,
            stress_violation=args.weight_stress_violation,
            shape_scale_m=args.shape_scale_mm / 1000.0,
            stress_scale_mpa=args.stress_scale,
            strand_scale=args.strand_scale,
        ),
        backend="direct",
    )
    options = HybridOptions(
        continuous=ContinuousOptions(
            maxiter=args.continuous_maxiter,
            ftol=args.continuous_ftol,
            progress_every=0 if args.quiet else args.progress_every,
            method=args.continuous_method,
        ),
        integer=IntegerSearchOptions(
            outer_iterations=args.outer_iterations,
            coordinate_step=args.coordinate_step,
            random_trials=args.random_trials,
            seed=args.seed,
            stress_guided=not args.no_stress_guided_strands,
            resize=not args.no_strand_resize,
        ),
    )
    start_time = time.perf_counter()

    def progress(message: str) -> None:
        elapsed = time.perf_counter() - start_time
        print(f"[{elapsed:8.1f}s] {message}", flush=True)

    optimizer = CableHybridOptimizer(problem, options, progress=None if args.quiet else progress)
    initial_strands = np.full(2 * args.n, args.initial_strands, dtype=int)
    initial_pretension = _initial_pretension(args)
    result = optimizer.optimize(initial_strands=initial_strands, initial_pretension=initial_pretension)

    out_dir = PROJECT_ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    band_line = _band_verdict_line(result.best, result, args.stress_lower, args.stress_upper)
    _write_outputs(out_dir, result.best, result.history, band_line=band_line)

    print("Cable optimization complete")
    print(f"  objective: {result.best.objective:.6g}")
    print(f"  shape rmse: {result.best.metrics.shape_rmse_m * 1000.0:.6f} mm")
    print(f"  total strands: {result.best.metrics.total_strands}")
    print(
        "  stress MPa: "
        f"mean={result.best.metrics.stress_mean_mpa:.3f}, "
        f"std={result.best.metrics.stress_std_mpa:.3f}, "
        f"min={result.best.metrics.stress_min_mpa:.3f}, "
        f"max={result.best.metrics.stress_max_mpa:.3f}"
    )
    print(f"  {band_line}")
    print(f"  outputs: {out_dir}")

    if args.verify_opensees:
        verify_problem = replace(problem, backend="opensees")
        verify = CableDesignEvaluator(verify_problem).evaluate(
            result.best.design.strands,
            result.best.design.pretension,
        )
        print("OpenSees verification")
        print(f"  shape rmse: {verify.metrics.shape_rmse_m * 1000.0:.6f} mm")
        print(
            "  stress MPa: "
            f"mean={verify.metrics.stress_mean_mpa:.3f}, "
            f"std={verify.metrics.stress_std_mpa:.3f}, "
            f"min={verify.metrics.stress_min_mpa:.3f}, "
            f"max={verify.metrics.stress_max_mpa:.3f}"
        )

    return result


def build_parser() -> argparse.ArgumentParser:
    model_p = P4B_DEFAULTS
    p = argparse.ArgumentParser(description="Optimize staged cable strands and pretensions.")
    p.add_argument("--n", type=int, default=model_p["n"])
    p.add_argument("--out", default="results/cable_opt")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--anchor-base", type=float, default=model_p["anchor_base"])
    p.add_argument("--anchor-spacing", type=float, default=model_p["anchor_spacing"])
    p.add_argument("--anchor-free", type=float, default=model_p["anchor_free"])
    p.add_argument("--left-start", type=float, default=model_p["left_start"])
    p.add_argument("--left-spacing", type=float, default=model_p["left_spacing"])
    p.add_argument("--left-end", type=float, default=model_p["left_end"])
    p.add_argument("--right-start", type=float, default=model_p["right_start"])
    p.add_argument("--right-spacing", type=float, default=model_p["right_spacing"])
    p.add_argument("--right-end", type=float, default=model_p["right_end"])
    p.add_argument("--wg", type=float, default=model_p["wg"])
    p.add_argument("--dw", type=float, default=model_p["dw"])
    p.add_argument("--beam-E", type=float, default=model_p["beam_E"], help="主梁弹性模量 E [Pa]")
    p.add_argument("--beam-A", type=float, default=model_p["beam_A"], help="主梁截面积 A [m^2]")
    p.add_argument("--beam-Iz", type=float, default=model_p["beam_Iz"], help="主梁截面惯性矩 I [m^4]")

    p.add_argument("--strand-min", type=int, default=100)
    p.add_argument("--strand-max", type=int, default=300)
    p.add_argument("--initial-strands", type=int, default=200)
    p.add_argument("--stress-lower", type=float, default=400.0)
    p.add_argument("--stress-upper", type=float, default=600.0)
    p.add_argument("--tension-bound-stress", type=float, default=1600.0)

    p.add_argument("--weight-shape", type=float, default=1.0)
    p.add_argument("--weight-strands", type=float, default=0.02)
    p.add_argument("--weight-stress-uniform", type=float, default=0.2)
    p.add_argument("--weight-stress-violation", type=float, default=100.0)
    p.add_argument("--shape-scale-mm", type=float, default=1.0)
    p.add_argument("--stress-scale", type=float, default=100.0)
    p.add_argument("--strand-scale", type=float, default=20.0)

    p.add_argument(
        "--continuous-method",
        choices=["linear", "slsqp"],
        default="linear",
        help="Continuous tension solver: 'linear' = exact affine-model LP+SLSQP "
        "(linear backends), 'slsqp' = legacy SLSQP on the FEM.",
    )
    p.add_argument("--continuous-maxiter", type=int, default=80)
    p.add_argument("--continuous-ftol", type=float, default=1.0e-7)
    p.add_argument("--progress-every", type=int, default=10, help="Print every N SLSQP objective evaluations.")
    p.add_argument("--outer-iterations", type=int, default=4)
    p.add_argument("--coordinate-step", type=int, default=1)
    p.add_argument("--random-trials", type=int, default=0)
    p.add_argument(
        "--no-stress-guided-strands",
        action="store_true",
        help="Disable stress-guided strand add/remove ordering.",
    )
    p.add_argument(
        "--no-strand-resize",
        action="store_true",
        help="Disable the stress-ratio strand resize jump at the start of each outer iteration.",
    )
    p.add_argument("--quiet", action="store_true", help="Disable optimization progress output.")
    p.add_argument("--verify-opensees", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
