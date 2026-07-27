"""Optimize staged cable strands and pretensions.

Example::

    python -m scripts.optimize_cables --bridge p4b
    python -m scripts.optimize_cables --bridge p4b --outer-iterations 3 --random-trials 2
    从当前 results/cable_opt 接着增加 8 轮
    python -m scripts.optimize_cables --bridge omo --resume --outer-iterations 8
    只增加随机尝试：
    python -m scripts.optimize_cables --bridge omo --resume --outer-iterations 0 --random-trials 5 --seed 1
    从旧结果分叉，不覆盖原结果：
    python -m scripts.optimize_cables --bridge omo \
    --resume-from results/cable_opt \
    --out results/cable_opt_branch \
    --outer-iterations 4 --random-trials 3 --seed 2
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, replace
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
    CableLayout,
    CableOptimizationProblem,
    ContinuousOptions,
    HybridOptions,
    IntegerSearchOptions,
    ObjectiveWeights,
)
from scripts.bridge_config import (  # noqa: E402
    load_bridge_config,
    model_family_for_bridge_type,
    resolve_bridge_config,
)
from scripts.staged_analysis import default_pretension  # noqa: E402


_HISTORY_HEADER = [
    "index",
    "objective",
    "shape_rmse_mm",
    "shape_max_abs_mm",
    "tower_top_dx_mm",
    "tower_anchor_dx_rmse_mm",
    "total_strands",
    "stress_mean_mpa",
    "stress_std_mpa",
    "stress_min_mpa",
    "stress_max_mpa",
    "stress_violation_rms_mpa",
]


@dataclass(frozen=True)
class _ResumeState:
    design_path: Path
    strands: np.ndarray
    pretension: np.ndarray
    previous_objective: float
    history_rows: list[list[str]]
    run_index: int
    tracked_outer_iterations: int
    tracked_random_trials: int
    prior_budget_known: bool


class _OptimizationProgressDisplay:
    """Render optimizer messages as a throttled, fixed-height terminal dashboard."""

    _BAR_WIDTH = 28

    def __init__(
        self,
        *,
        total_outer: int,
        total_cables: int,
        refresh_interval: float = 0.5,
        stream=None,
        clock=None,
    ):
        if refresh_interval <= 0.0:
            raise ValueError("progress refresh interval must be positive")
        self.total_outer = total_outer
        self.total_cables = total_cables
        self.refresh_interval = refresh_interval
        self.stream = sys.stdout if stream is None else stream
        self.clock = time.perf_counter if clock is None else clock
        self.live = bool(getattr(self.stream, "isatty", lambda: False)())
        self.started_at = self.clock()
        self.last_rendered_at = float("-inf")
        self.rendered_lines = 0
        self.closed = False

        self.outer = 0
        self.cable_index = 0
        self.phase = "starting"
        self.solve_count = 0
        self.current_candidate = "—"
        self.last_decision = "—"
        self.total_strands = None
        self.strand_min = None
        self.strand_max = None
        self.current_objective = None
        self.initial_objective = None
        self.best_objective = None
        self.shape_rmse_mm = None
        self.stress_min_mpa = None
        self.stress_max_mpa = None
        self.s_star_mpa = None
        self.solver_success = None
        self.checkpoints: list[float] = []

    @staticmethod
    def _number(message: str, key: str) -> float | None:
        match = re.search(rf"\b{re.escape(key)}=(-?[0-9.eE+]+)", message)
        return None if match is None else float(match.group(1))

    @staticmethod
    def _integer(message: str, key: str) -> int | None:
        value = _OptimizationProgressDisplay._number(message, key)
        return None if value is None else int(value)

    @staticmethod
    def _objective_change(message: str) -> tuple[float, float] | None:
        match = re.search(r"(-?[0-9.eE+]+)\s*->\s*(-?[0-9.eE+]+)", message)
        if match is None:
            return None
        return float(match.group(1)), float(match.group(2))

    @staticmethod
    def _format_value(value: float | None, suffix: str = "") -> str:
        return "—" if value is None else f"{value:.6g}{suffix}"

    def _append_checkpoint(self, value: float | None) -> None:
        if value is None:
            return
        if not self.checkpoints or value != self.checkpoints[-1]:
            self.checkpoints.append(value)
            self.checkpoints = self.checkpoints[-7:]

    def _set_metrics(self, message: str) -> None:
        objective = self._number(message, "objective")
        if objective is not None:
            self.current_objective = objective
        shape = self._number(message, "shape_rmse")
        if shape is not None:
            self.shape_rmse_mm = shape
        stress = re.search(r"stress=\[(-?[0-9.eE+]+),\s*(-?[0-9.eE+]+)\]", message)
        if stress is not None:
            self.stress_min_mpa = float(stress.group(1))
            self.stress_max_mpa = float(stress.group(2))
        success = re.search(r"success=(True|False)", message)
        if success is not None:
            self.solver_success = success.group(1) == "True"

    def _cable_position(self, cable_id: int) -> int:
        stage = cable_id % 1000
        side_offset = 0 if cable_id < 2000 else 1
        return min(self.total_cables, 2 * (stage - 1) + side_offset + 1)

    def update(self, message: str) -> None:
        """Consume one existing optimizer progress message and update the display."""

        text = message.strip()
        event = "detail"
        force = False

        if text.startswith("start cable optimization"):
            self.phase = "initial design"
            event, force = "start", True
        elif text.startswith("optimize tensions"):
            self.phase = "tension optimization"
            self.solve_count += 1
            self.solver_success = None
            self.total_strands = self._integer(text, "total_strands")
            self.strand_min = self._integer(text, "min")
            self.strand_max = self._integer(text, "max")
        elif text.startswith("linear LP"):
            self.phase = "LP target-band warm-start"
            self.s_star_mpa = self._number(text, "s*")
        elif text.startswith(("linear QP done", "SLSQP done", "SLSQP eval")):
            self.phase = "continuous solve"
            self._set_metrics(text)
        elif text.startswith("initial best"):
            self.phase = "initial design complete"
            self.best_objective = self._number(text, "objective")
            self.initial_objective = self.best_objective
            self.total_strands = self._integer(text, "total_strands")
            self._append_checkpoint(self.best_objective)
            event, force = "initial", True
        elif text.startswith("random trial"):
            self.phase = text
            event, force = "trial", True
        elif text.startswith("outer iteration"):
            self._append_checkpoint(self.best_objective)
            match = re.search(r"outer iteration (\d+)/(\d+)", text)
            if match is not None:
                self.outer = int(match.group(1))
                self.total_outer = int(match.group(2))
            self.cable_index = 0
            self.phase = "integer search"
            event, force = "outer", True
        elif text.startswith("resize candidate"):
            change = re.search(r"total_strands (\d+)\s*->\s*(\d+)", text)
            if change is not None:
                self.current_candidate = f"global resize: total {change.group(1)} → {change.group(2)}"
            self.phase = "strand resize"
        elif text.startswith("candidate cable="):
            match = re.search(
                r"cable=(\d+) strand_delta=([+-]\d+) .*?(\d+)->(\d+)", text
            )
            if match is not None:
                cable_id, delta, old, new = match.groups()
                self.cable_index = self._cable_position(int(cable_id))
                self.current_candidate = f"cable {cable_id}: {old} → {new} strands ({delta})"
            self.phase = "coordinate search"
        elif text.startswith(("accepted resize", "accepted random trial", "accepted:")):
            change = self._objective_change(text)
            if change is not None:
                self.best_objective = change[1]
            s_star = self._number(text, "s*")
            if s_star is not None:
                self.s_star_mpa = s_star
            self.last_decision = f"ACCEPT  best objective → {self._format_value(self.best_objective)}"
            if text.startswith("accepted resize"):
                self._append_checkpoint(self.best_objective)
                event, force = "accepted_resize", True
        elif text.startswith(("rejected resize", "rejected random trial", "rejected:")):
            objective = self._number(text, "objective")
            self.last_decision = f"REJECT  candidate objective {self._format_value(objective)}"
        elif text.startswith("no integer improvement"):
            self.phase = "converged: no integer improvement"
            event, force = "converged", True
        elif text.startswith("finished:"):
            self.phase = "complete"
            self.cable_index = self.total_cables
            self._set_metrics(text)
            self.best_objective = self._number(text, "objective")
            self.total_strands = self._integer(text, "total_strands")
            s_star = self._number(text, "s*")
            if s_star is not None:
                self.s_star_mpa = s_star
            self._append_checkpoint(self.best_objective)
            event, force = "finished", True

        self._render(event=event, force=force)

    def _overall_fraction(self) -> float:
        if self.phase == "complete":
            return 1.0
        if self.total_outer <= 0 or self.outer <= 0:
            return 0.0
        within_outer = self.cable_index / max(1, self.total_cables)
        return min(1.0, ((self.outer - 1) + within_outer) / self.total_outer)

    def _dashboard_lines(self) -> list[str]:
        elapsed = self.clock() - self.started_at
        fraction = self._overall_fraction()
        filled = min(self._BAR_WIDTH, int(round(self._BAR_WIDTH * fraction)))
        bar = "█" * filled + "·" * (self._BAR_WIDTH - filled)
        solver = "—" if self.solver_success is None else ("OK" if self.solver_success else "NOT CONVERGED")
        strands = "—" if self.total_strands is None else str(self.total_strands)
        strand_range = (
            "—"
            if self.strand_min is None or self.strand_max is None
            else f"{self.strand_min}…{self.strand_max}"
        )
        stress = (
            "—"
            if self.stress_min_mpa is None or self.stress_max_mpa is None
            else f"{self.stress_min_mpa:.1f}…{self.stress_max_mpa:.1f} MPa"
        )
        improvement = "—"
        if self.initial_objective and self.best_objective is not None:
            improvement = f"{100.0 * (self.initial_objective - self.best_objective) / self.initial_objective:.2f}%"
        trend = " → ".join(f"{value:.6g}" for value in self.checkpoints) or "—"
        outer = f"{self.outer}/{self.total_outer}" if self.total_outer else "—"
        cable = f"{self.cable_index}/{self.total_cables}"
        return [
            f"Cable optimization | elapsed {elapsed:6.1f}s",
            f"Overall  [{bar}] {fraction * 100:5.1f}% | outer {outer} | cable {cable}",
            (
                f"Stage    {self.phase} | tension solve #{self.solve_count} | "
                f"solver {solver} | LP s*={self._format_value(self.s_star_mpa, ' MPa')}"
            ),
            f"Change   {self.current_candidate}",
            f"Decision {self.last_decision}",
            f"Design   total strands {strands} | per-cable range {strand_range}",
            (
                f"Current  objective {self._format_value(self.current_objective)} | "
                f"shape RMSE {self._format_value(self.shape_rmse_mm, ' mm')} | stress {stress}"
            ),
            f"Best     objective {self._format_value(self.best_objective)} | improvement {improvement}",
            f"Trend    {trend}",
        ]

    def _plain_line(self) -> str:
        return (
            f"[{self.clock() - self.started_at:6.1f}s] {self.phase}: "
            f"outer={self.outer}/{self.total_outer} cable={self.cable_index}/{self.total_cables} "
            f"best={self._format_value(self.best_objective)} strands={self.total_strands or '—'}"
        )

    def _render(self, *, event: str, force: bool) -> None:
        now = self.clock()
        if self.live:
            if not force and now - self.last_rendered_at < self.refresh_interval:
                return
            lines = self._dashboard_lines()
            if self.rendered_lines:
                self.stream.write(f"\x1b[{self.rendered_lines}F")
            for line in lines:
                self.stream.write(f"\x1b[2K{line}\n")
            self.stream.flush()
            self.rendered_lines = len(lines)
            self.last_rendered_at = now
        elif event in {"start", "initial", "trial", "outer", "accepted_resize", "converged", "finished"}:
            self.stream.write(self._plain_line() + "\n")
            self.stream.flush()

    def close(self) -> None:
        if self.closed:
            return
        if self.live:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _model_kwargs(args) -> dict:
    kwargs = {
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
        "tower_stiffness": args.bridge_defaults["tower_stiffness"],
        "tower_element_size": args.bridge_defaults["tower_element_size"],
        "tower_axial_rigidity": args.bridge_defaults["tower_axial_rigidity"],
    }
    if args.bridge_defaults["bridge_type"] == "single":
        kwargs["right_fix"] = args.bridge_defaults["right_fix"]
        kwargs["left_span"] = args.bridge_defaults["left_span"]
    return kwargs


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


def _bridge_yaml_reference(source: str | Path) -> str:
    """Return a stable YAML reference for persistence in optimization output."""

    path = resolve_bridge_config(source).resolve()
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _problem_metadata(problem: CableOptimizationProblem) -> dict:
    """Return the objective/model definition that must stay fixed across a resume."""

    metadata = {
        "n_seg": problem.n_seg,
        "model_family": problem.model_family,
        "backend": problem.backend,
        "strand_area": problem.strand_area,
        "model_kwargs": problem.model_kwargs,
        "bounds": asdict(problem.bounds),
        "weights": asdict(problem.weights),
    }
    # Normalize tuples/numpy-compatible scalars to the representation persisted by JSON.
    return json.loads(json.dumps(metadata))


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _resume_design_path(args, out_dir: Path) -> Path | None:
    if args.resume:
        source = out_dir
    elif args.resume_from is not None:
        source = _project_path(args.resume_from)
    else:
        return None
    if source.is_dir() or source.suffix.lower() != ".json":
        source = source / "best_design.json"
    return source.resolve()


def _read_history_rows(path: Path) -> list[list[str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows:
        return []
    if rows[0] != _HISTORY_HEADER:
        raise ValueError(f"cannot resume: incompatible history header in {path}")
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(_HISTORY_HEADER):
            raise ValueError(f"cannot resume: malformed history row {row_number} in {path}")
        try:
            int(row[0])
            metrics = [float(value) for value in row[1:]]
        except ValueError as exc:
            raise ValueError(f"cannot resume: invalid history row {row_number} in {path}") from exc
        if not all(np.isfinite(value) for value in metrics):
            raise ValueError(f"cannot resume: non-finite history row {row_number} in {path}")
    return rows[1:]


def _load_resume_state(
    design_path: Path,
    *,
    problem: CableOptimizationProblem,
    bridge_yaml: str,
    problem_metadata: dict,
) -> _ResumeState:
    """Load and validate a persisted best design as the next search starting point."""

    if not design_path.is_file():
        raise FileNotFoundError(f"resume design not found: {design_path}")
    try:
        payload = json.loads(design_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot resume: invalid design JSON {design_path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"cannot resume: design root must be an object in {design_path}")
    if payload.get("bridge_yaml") != bridge_yaml:
        raise ValueError(
            "cannot resume: bridge YAML mismatch "
            f"({payload.get('bridge_yaml')!r} != {bridge_yaml!r})"
        )
    if payload.get("model_family") != problem.model_family:
        raise ValueError(
            "cannot resume: model family mismatch "
            f"({payload.get('model_family')!r} != {problem.model_family!r})"
        )
    saved_problem = payload.get("problem")
    if saved_problem is not None and saved_problem != problem_metadata:
        raise ValueError(
            "cannot resume: model, bounds, or objective weights differ from the saved run; "
            "start a new output directory instead"
        )

    cables = payload.get("cables")
    if not isinstance(cables, list):
        raise ValueError(f"cannot resume: missing cables list in {design_path}")
    by_id: dict[int, dict] = {}
    for item in cables:
        if not isinstance(item, dict) or isinstance(item.get("cable_id"), bool):
            raise ValueError(f"cannot resume: malformed cable entry in {design_path}")
        try:
            raw_cable_id = float(item["cable_id"])
            cable_id = int(raw_cable_id)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot resume: invalid cable id in {design_path}") from exc
        if not np.isfinite(raw_cable_id) or raw_cable_id != cable_id:
            raise ValueError(f"cannot resume: invalid cable id in {design_path}")
        if cable_id in by_id:
            raise ValueError(f"cannot resume: duplicate cable id {cable_id} in {design_path}")
        by_id[cable_id] = item

    layout = CableLayout(problem.n_seg)
    if set(by_id) != set(layout.cable_ids):
        missing = sorted(set(layout.cable_ids) - set(by_id))
        extra = sorted(set(by_id) - set(layout.cable_ids))
        raise ValueError(f"cannot resume: cable ids differ (missing={missing}, extra={extra})")

    strands: list[int] = []
    pretension: list[float] = []
    for index, cable_id in enumerate(layout.cable_ids):
        item = by_id[cable_id]
        expected_stage = index // 2 + 1
        expected_side = "right" if index % 2 == 0 else "left"
        if item.get("stage") != expected_stage or item.get("side") != expected_side:
            raise ValueError(f"cannot resume: cable {cable_id} stage/side metadata is inconsistent")
        raw_strands = item.get("strands")
        if isinstance(raw_strands, bool):
            raise ValueError(f"cannot resume: cable {cable_id} strands must be an integer")
        try:
            strand_value = float(raw_strands)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"cannot resume: cable {cable_id} strands must be an integer") from exc
        rounded = int(round(strand_value))
        if not np.isfinite(strand_value) or strand_value != rounded:
            raise ValueError(f"cannot resume: cable {cable_id} strands must be an integer")
        if not problem.bounds.strand_min <= rounded <= problem.bounds.strand_max:
            raise ValueError(
                f"cannot resume: cable {cable_id} strands {rounded} outside current bounds "
                f"[{problem.bounds.strand_min}, {problem.bounds.strand_max}]"
            )
        if isinstance(item.get("pretension_N"), bool):
            raise ValueError(f"cannot resume: invalid pretension for cable {cable_id}")
        try:
            tension_value = float(item["pretension_N"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot resume: invalid pretension for cable {cable_id}") from exc
        tension_limit = (
            problem.bounds.tension_bound_stress_mpa
            * 1e6
            * problem.strand_area
            * rounded
        )
        if (
            not np.isfinite(tension_value)
            or tension_value < 0.0
            or tension_value > tension_limit * (1.0 + 1e-12)
        ):
            raise ValueError(
                f"cannot resume: cable {cable_id} pretension {tension_value:g} N "
                f"outside [0, {tension_limit:g}] N"
            )
        strands.append(rounded)
        pretension.append(tension_value)

    try:
        previous_objective = float(payload["objective"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"cannot resume: invalid objective in {design_path}") from exc
    if not np.isfinite(previous_objective):
        raise ValueError(f"cannot resume: objective must be finite in {design_path}")

    search = payload.get("search")
    has_search_metadata = isinstance(search, dict)
    prior_budget_known = has_search_metadata and bool(search.get("prior_budget_known", True))
    if has_search_metadata:
        try:
            run_index = int(search["run_index"])
            tracked_outer = int(search["outer_iterations_tracked_total"])
            tracked_random = int(search["random_trials_tracked_total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"cannot resume: invalid search metadata in {design_path}") from exc
        if run_index < 1 or tracked_outer < 0 or tracked_random < 0:
            raise ValueError(f"cannot resume: negative search metadata in {design_path}")
    else:
        # Legacy outputs contain a valid design but no reliable record of prior search budgets.
        run_index, tracked_outer, tracked_random = 1, 0, 0

    return _ResumeState(
        design_path=design_path,
        strands=np.asarray(strands, dtype=int),
        pretension=np.asarray(pretension, dtype=float),
        previous_objective=previous_objective,
        history_rows=_read_history_rows(design_path.with_name("history.csv")),
        run_index=run_index,
        tracked_outer_iterations=tracked_outer,
        tracked_random_trials=tracked_random,
        prior_budget_known=prior_budget_known,
    )


def _evaluation_payload(
    ev,
    bridge_yaml: str,
    model_family: str | None = None,
) -> dict:
    payload = {
        "bridge_yaml": bridge_yaml,
        "objective": ev.objective,
        "components": {
            "shape": ev.components.shape,
            "tower_displacement": ev.components.tower_displacement,
            "tower_anchor_displacement": ev.components.tower_anchor_displacement,
            "total_strands": ev.components.total_strands,
            "stress_uniform": ev.components.stress_uniform,
            "stress_violation": ev.components.stress_violation,
        },
        "metrics": {
            "shape_rmse_mm": ev.metrics.shape_rmse_m * 1000.0,
            "shape_max_abs_mm": ev.metrics.shape_max_abs_m * 1000.0,
            "tower_top_dx_mm": ev.metrics.tower_top_dx_m * 1000.0,
            "tower_anchor_dx_rmse_mm": ev.metrics.tower_anchor_dx_rmse_m * 1000.0,
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
        "tower_anchor_dx_mm": {
            str(node): dx * 1000.0 for node, dx in ev.tower_anchor_dx_m.items()
        },
    }
    if model_family is not None:
        payload["model_family"] = model_family
    return payload


def _band_verdict_line(best, result, stress_lower: float, stress_upper: float) -> str:
    violation = max(
        0.0,
        stress_lower - best.metrics.stress_min_mpa,
        best.metrics.stress_max_mpa - stress_upper,
    )
    verdict = "WITHIN TARGET" if violation <= 1e-6 else "OUTSIDE TARGET"
    s_star = result.feasibility_violation_mpa
    lp_note = f", target-band LP s*={s_star:.3f} MPa" if s_star is not None else ""
    return (
        f"target stress band [{stress_lower:g}, {stress_upper:g}] MPa: {verdict} "
        f"(max departure {violation:.3f} MPa{lp_note})"
    )


def _left_to_right_values(values) -> np.ndarray:
    """Return stage-major cable values in physical order: left tip to right tip."""

    values = np.asarray(values)
    return np.concatenate((values[1::2][::-1], values[0::2]))


def _left_to_right_strand_counts(best) -> list[int]:
    """Return strand counts in physical deck order: left tip to right tip."""

    strands = np.asarray(best.design.strands, dtype=int)
    return [int(value) for value in _left_to_right_values(strands)]


def _write_outputs(
    out_dir: Path,
    best,
    history,
    bridge_yaml: str,
    model_family: str,
    band_line: str | None = None,
    history_prefix: list[list[str]] | None = None,
    problem_metadata: dict | None = None,
    search_metadata: dict | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = _evaluation_payload(best, bridge_yaml, model_family=model_family)
    if problem_metadata is not None:
        payload["problem"] = problem_metadata
    if search_metadata is not None:
        payload["search"] = search_metadata
    (out_dir / "best_design.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    pretension_left_to_right = _left_to_right_values(best.design.pretension)
    stress_left_to_right = _left_to_right_values(
        [best.cable_stress_mpa[cid] for cid in best.cable_ids]
    )

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(_HISTORY_HEADER)
        prefix = history_prefix or []
        for i, row in enumerate(prefix):
            writer.writerow([i, *row[1:]])
        for i, ev in enumerate(history, start=len(prefix)):
            writer.writerow([
                i,
                ev.objective,
                ev.metrics.shape_rmse_m * 1000.0,
                ev.metrics.shape_max_abs_m * 1000.0,
                ev.metrics.tower_top_dx_m * 1000.0,
                ev.metrics.tower_anchor_dx_rmse_m * 1000.0,
                ev.metrics.total_strands,
                ev.metrics.stress_mean_mpa,
                ev.metrics.stress_std_mpa,
                ev.metrics.stress_min_mpa,
                ev.metrics.stress_max_mpa,
                ev.metrics.stress_violation_rms_mpa,
            ])

    summary = [
        f"model family: {model_family}",
        f"objective: {best.objective:.6g}",
        f"shape rmse: {best.metrics.shape_rmse_m * 1000.0:.6f} mm",
        f"shape max abs: {best.metrics.shape_max_abs_m * 1000.0:.6f} mm",
        f"tower top dx: {best.metrics.tower_top_dx_m * 1000.0:.6f} mm (target 0 mm)",
        (
            "tower anchor dx rmse: "
            f"{best.metrics.tower_anchor_dx_rmse_m * 1000.0:.6f} mm (target 0 mm)"
        ),
        f"total strands: {best.metrics.total_strands}",
        "final strand counts (left to right): "
        + ", ".join(str(value) for value in _left_to_right_strand_counts(best)),
        "pretension_N (left to right): "
        + ", ".join(f"{value:.2f}" for value in pretension_left_to_right),
        "final_stress_MPa (left to right): "
        + ", ".join(f"{value:.2f}" for value in stress_left_to_right),
        (
            "stress MPa: "
            f"mean={best.metrics.stress_mean_mpa:.3f}, "
            f"std={best.metrics.stress_std_mpa:.3f}, "
            f"min={best.metrics.stress_min_mpa:.3f}, "
            f"max={best.metrics.stress_max_mpa:.3f}"
        ),
        f"stress violation rms: {best.metrics.stress_violation_rms_mpa:.6f} MPa",
    ]
    if search_metadata is not None:
        summary.extend([
            (
                f"search run: {search_metadata['run_index']} "
                f"({'resumed' if search_metadata['resumed'] else 'fresh'})"
            ),
            (
                "search budget this run: "
                f"outer={search_metadata['outer_iterations_completed']}/"
                f"{search_metadata['outer_iterations_requested']}, "
                f"random={search_metadata['random_trials_completed']}"
            ),
            (
                "tracked cumulative search: "
                f"outer={search_metadata['outer_iterations_tracked_total']}, "
                f"random={search_metadata['random_trials_tracked_total']}"
            ),
        ])
        if search_metadata["resumed"]:
            summary.append(f"resume source: {search_metadata['resume_source']}")
        if not search_metadata["prior_budget_known"]:
            summary.append("resume note: cumulative search excludes untracked legacy runs")
    if band_line is not None:
        summary.append(band_line)
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


def run(args):
    out_dir = _project_path(args.out)
    bridge_yaml = _bridge_yaml_reference(args.bridge)
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
            tower_displacement=args.weight_tower_displacement,
            tower_anchor_displacement=args.weight_tower_anchor_displacement,
        ),
        backend="direct",
        model_family=model_family_for_bridge_type(args.bridge_defaults["bridge_type"]),
    )
    problem_metadata = _problem_metadata(problem)
    resume_path = _resume_design_path(args, out_dir)
    resume_state = None
    if resume_path is not None:
        resume_state = _load_resume_state(
            resume_path,
            problem=problem,
            bridge_yaml=bridge_yaml,
            problem_metadata=problem_metadata,
        )
        if not args.quiet:
            print(
                f"Resuming from {resume_state.design_path} | "
                f"objective={resume_state.previous_objective:.6g} | "
                f"history evaluations={len(resume_state.history_rows)}",
                flush=True,
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
            band_priority=args.band_priority,
        ),
    )
    progress_display = None
    if not args.quiet:
        progress_display = _OptimizationProgressDisplay(
            total_outer=args.outer_iterations,
            total_cables=2 * args.n,
            refresh_interval=args.progress_refresh,
        )

    optimizer = CableHybridOptimizer(
        problem,
        options,
        progress=None if progress_display is None else progress_display.update,
    )
    if resume_state is None:
        initial_strands = np.full(2 * args.n, args.initial_strands, dtype=int)
        initial_pretension = _initial_pretension(args)
    else:
        initial_strands = resume_state.strands
        initial_pretension = resume_state.pretension
    try:
        result = optimizer.optimize(initial_strands=initial_strands, initial_pretension=initial_pretension)
    finally:
        if progress_display is not None:
            progress_display.close()

    previous_outer = 0 if resume_state is None else resume_state.tracked_outer_iterations
    previous_random = 0 if resume_state is None else resume_state.tracked_random_trials
    search_metadata = {
        "run_index": 1 if resume_state is None else resume_state.run_index + 1,
        "resumed": resume_state is not None,
        "resume_source": None if resume_state is None else str(resume_state.design_path),
        "prior_budget_known": True if resume_state is None else resume_state.prior_budget_known,
        "outer_iterations_requested": args.outer_iterations,
        "outer_iterations_completed": result.outer_iterations_completed,
        "outer_iterations_tracked_total": previous_outer + result.outer_iterations_completed,
        "random_trials_completed": result.random_trials_completed,
        "random_trials_tracked_total": previous_random + result.random_trials_completed,
        "seed": args.seed,
        "history_evaluations_previous": 0 if resume_state is None else len(resume_state.history_rows),
        "history_evaluations_this_run": len(result.history),
        "history_evaluations_total": (
            len(result.history)
            if resume_state is None
            else len(resume_state.history_rows) + len(result.history)
        ),
    }
    band_line = _band_verdict_line(result.best, result, args.stress_lower, args.stress_upper)
    _write_outputs(
        out_dir,
        result.best,
        result.history,
        bridge_yaml=bridge_yaml,
        model_family=problem.model_family,
        band_line=band_line,
        history_prefix=None if resume_state is None else resume_state.history_rows,
        problem_metadata=problem_metadata,
        search_metadata=search_metadata,
    )

    print("Cable optimization complete")
    print(f"  model family: {problem.model_family}")
    print(f"  objective: {result.best.objective:.6g}")
    print(f"  shape rmse: {result.best.metrics.shape_rmse_m * 1000.0:.6f} mm")
    print(f"  tower top dx: {result.best.metrics.tower_top_dx_m * 1000.0:.6f} mm (target 0 mm)")
    print(
        "  tower anchor dx rmse: "
        f"{result.best.metrics.tower_anchor_dx_rmse_m * 1000.0:.6f} mm (target 0 mm)"
    )
    print(f"  total strands: {result.best.metrics.total_strands}")
    print(
        "  stress MPa: "
        f"mean={result.best.metrics.stress_mean_mpa:.3f}, "
        f"std={result.best.metrics.stress_std_mpa:.3f}, "
        f"min={result.best.metrics.stress_min_mpa:.3f}, "
        f"max={result.best.metrics.stress_max_mpa:.3f}"
    )
    print(f"  {band_line}")
    if resume_state is not None:
        print(
            "  resumed search: "
            f"run={search_metadata['run_index']} "
            f"outer+={result.outer_iterations_completed} "
            f"random+={result.random_trials_completed}"
        )
    print(f"  outputs: {out_dir}")

    if args.verify_opensees:
        verify_problem = replace(problem, backend="opensees")
        verify = CableDesignEvaluator(verify_problem).evaluate(
            result.best.design.strands,
            result.best.design.pretension,
        )
        print("OpenSees verification")
        print(f"  shape rmse: {verify.metrics.shape_rmse_m * 1000.0:.6f} mm")
        print(f"  tower top dx: {verify.metrics.tower_top_dx_m * 1000.0:.6f} mm (target 0 mm)")
        print(
            "  tower anchor dx rmse: "
            f"{verify.metrics.tower_anchor_dx_rmse_m * 1000.0:.6f} mm (target 0 mm)"
        )
        print(
            "  stress MPa: "
            f"mean={verify.metrics.stress_mean_mpa:.3f}, "
            f"std={verify.metrics.stress_std_mpa:.3f}, "
            f"min={verify.metrics.stress_min_mpa:.3f}, "
            f"max={verify.metrics.stress_max_mpa:.3f}"
        )

    return result


def build_parser(bridge_defaults: dict[str, object]) -> argparse.ArgumentParser:
    model_p = bridge_defaults
    p = argparse.ArgumentParser(description="Optimize staged cable strands and pretensions.")
    p.set_defaults(bridge_defaults=dict(model_p))
    p.add_argument(
        "--bridge",
        required=True,
        help="桥梁 YAML 配置：内置名称 model/p4b/omo，或 YAML 文件路径（必须显式指定）",
    )
    p.add_argument("--n", type=int, default=model_p["n"])
    p.add_argument("--out", default="results/cable_opt")
    resume = p.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume",
        action="store_true",
        help="Continue from best_design.json in the current --out directory.",
    )
    resume.add_argument(
        "--resume-from",
        metavar="PATH",
        help="Continue from another result directory or best_design.json file.",
    )
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

    p.add_argument("--strand-min", type=int, default=20)
    p.add_argument("--strand-max", type=int, default=500)
    p.add_argument("--initial-strands", type=int, default=50)
    p.add_argument("--stress-lower", type=float, default=600.0)
    p.add_argument("--stress-upper", type=float, default=700.0)
    p.add_argument("--tension-bound-stress", type=float, default=1600.0)

    p.add_argument("--weight-shape", type=float, default=1.0)
    p.add_argument("--weight-tower-displacement", type=float, default=1.0)
    p.add_argument("--weight-tower-anchor-displacement", type=float, default=1.0)
    p.add_argument("--weight-strands", type=float, default=0.02)
    p.add_argument("--weight-stress-uniform", type=float, default=0.2)
    p.add_argument("--weight-stress-violation", type=float, default=100.0)
    p.add_argument("--shape-scale-mm", type=float, default=100.0)
    p.add_argument("--stress-scale", type=float, default=100.0)
    p.add_argument("--strand-scale", type=float, default=8200.0)

    p.add_argument(
        "--continuous-method",
        choices=["linear", "slsqp"],
        default="linear",
        help="Continuous tension solver: 'linear' = exact affine-model LP+SLSQP "
        "(linear backends), 'slsqp' = legacy SLSQP on the FEM.",
    )
    p.add_argument("--continuous-maxiter", type=int, default=80)
    p.add_argument("--continuous-ftol", type=float, default=1.0e-7)
    p.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Update progress metrics every N direct-SLSQP objective evaluations.",
    )
    p.add_argument(
        "--progress-refresh",
        type=_positive_float,
        default=0.5,
        help="Minimum seconds between live dashboard refreshes (default: 0.5).",
    )
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
    band_priority = p.add_mutually_exclusive_group()
    band_priority.add_argument(
        "--band-priority",
        action="store_true",
        help="Prefer strand configurations whose target stress band is more reachable before "
        "comparing the weighted objective (legacy opt-in behavior).",
    )
    band_priority.add_argument(
        "--no-band-priority",
        dest="band_priority",
        action="store_false",
        help=argparse.SUPPRESS,
    )
    p.set_defaults(band_priority=False)
    p.add_argument("--quiet", action="store_true", help="Disable optimization progress output.")
    p.add_argument("--verify-opensees", action="store_true")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--bridge")
    known, _ = bootstrap.parse_known_args(argv)
    if known.bridge is None:
        bootstrap.error("--bridge is required; optimization must explicitly identify its YAML model")
    return build_parser(load_bridge_config(known.bridge)).parse_args(argv)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
