"""Write an optimized BridgeZoo cable design into a sibling MIDAS MCT file.

Run with the project defaults::

    python -m scripts.write_best_design

Or provide another design/template::

    python -m scripts.write_best_design results/cable_opt_3d_forward/best_design.json

The template is never overwritten.  The output is created beside it as
``<stem>_updated.mct`` (or ``_updated_2``, ``_updated_3``, ...).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MCT_FILE = Path("/Users/bill/Library/CloudStorage/GoogleDrive-billhu0228@gmail.com/我的云端硬盘/Project/OmoBridge/优化参数.mct")
DEFAULT_BEST_DESIGN = PROJECT_ROOT / "results/cable_opt_3d_forward/best_design.json"

_GROUP_RE = re.compile(r"^CF([AB])-(\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class _CableDesignValues:
    strands: int
    pretension_a_n: float
    pretension_b_n: float


def _load_design(path: Path) -> tuple[int, float, dict[tuple[int, str], _CableDesignValues]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    problem = payload.get("problem", {})
    n_seg = int(problem.get("n_seg", 0))
    strand_area = float(problem.get("strand_area", 0.0))
    if n_seg < 1:
        raise ValueError("best design must contain a positive problem.n_seg")
    if not math.isfinite(strand_area) or strand_area <= 0.0:
        raise ValueError("best design must contain a positive problem.strand_area")

    values: dict[tuple[int, str], _CableDesignValues] = {}
    for item in payload.get("cable_groups", []):
        stage = int(item.get("stage", 0))
        group = str(item.get("group", ""))
        key = (stage, group)
        if not 1 <= stage <= n_seg or group not in {"main_stay", "backstay"}:
            raise ValueError(f"invalid cable group in best design: {key}")
        if key in values:
            raise ValueError(f"duplicate cable group in best design: {key}")
        strands = int(item["strands_per_physical_cable"])
        pretension_a_n = float(item["pretension_A_per_physical_cable_N"])
        pretension_b_n = float(item["pretension_B_per_physical_cable_N"])
        if strands < 1:
            raise ValueError(f"cable group {key} must have a positive strand count")
        if any(
            not math.isfinite(value) or value < 0.0
            for value in (pretension_a_n, pretension_b_n)
        ):
            raise ValueError(f"cable group {key} has invalid A/B pretension")
        values[key] = _CableDesignValues(
            strands=strands,
            pretension_a_n=pretension_a_n,
            pretension_b_n=pretension_b_n,
        )

    expected = {
        (stage, group)
        for stage in range(1, n_seg + 1)
        for group in ("main_stay", "backstay")
    }
    missing = sorted(expected.difference(values))
    extra = sorted(set(values).difference(expected))
    if missing or extra:
        raise ValueError(
            f"best design cable groups do not match n_seg={n_seg}; "
            f"missing={missing}, extra={extra}"
        )
    return n_seg, strand_area, values


def _replace_token(token: str, value: str) -> str:
    """Replace a comma-separated value while preserving surrounding spaces."""

    leading = token[: len(token) - len(token.lstrip())]
    trailing = token[len(token.rstrip()) :]
    return f"{leading}{value}{trailing}"


def _section_key(section_id: int, n_seg: int) -> tuple[int, str] | None:
    if 101 <= section_id <= 100 + n_seg:
        return section_id - 100, "main_stay"
    if 201 <= section_id <= 200 + n_seg:
        return section_id - 200, "backstay"
    return None


def _pretension_key(element_id: int, n_seg: int) -> tuple[int, str] | None:
    # Two physical cable planes share one optimized per-cable value.
    if 101 <= element_id <= 100 + n_seg:
        return element_id - 100, "main_stay"
    if 201 <= element_id <= 200 + n_seg:
        return element_id - 200, "main_stay"
    if 301 <= element_id <= 300 + n_seg:
        return element_id - 300, "backstay"
    if 401 <= element_id <= 400 + n_seg:
        return element_id - 400, "backstay"
    return None


def _updated_path(template: Path) -> Path:
    candidate = template.with_name(f"{template.stem}_updated{template.suffix}")
    version = 2
    while candidate.exists():
        candidate = template.with_name(
            f"{template.stem}_updated_{version}{template.suffix}"
        )
        version += 1
    return candidate


def _rewrite_mct(
    source: str,
    *,
    n_seg: int,
    strand_area: float,
    values: dict[tuple[int, str], _CableDesignValues],
) -> tuple[str, int, int]:
    output: list[str] = []
    active_block = ""
    updated_sections: set[int] = set()
    updated_pretensions: set[tuple[int, str]] = set()

    for line in source.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        stripped = body.strip()
        if stripped.startswith("*"):
            active_block = stripped.split(";", 1)[0].strip().upper()

        fields = body.split(",")
        if active_block == "*SECTION" and len(fields) > 14:
            try:
                section_id = int(fields[0].strip())
            except ValueError:
                section_id = -1
            key = _section_key(section_id, n_seg)
            if key is not None:
                diameter_m = math.sqrt(
                    4.0 * values[key].strands * strand_area / math.pi
                )
                fields[14] = _replace_token(fields[14], f"{diameter_m:.10g}")
                body = ",".join(fields)
                if section_id in updated_sections:
                    raise ValueError(f"duplicate *SECTION id in MCT template: {section_id}")
                updated_sections.add(section_id)

        elif active_block == "*PRETENSION" and len(fields) >= 3:
            try:
                element_id = int(fields[0].strip())
            except ValueError:
                element_id = -1
            key = _pretension_key(element_id, n_seg)
            match = _GROUP_RE.fullmatch(fields[2].strip())
            if key is not None and match is not None:
                phase = match.group(1).upper()
                stage_from_group = int(match.group(2))
                if stage_from_group != key[0]:
                    raise ValueError(
                        f"MCT pretension stage mismatch for element {element_id}: "
                        f"{fields[2].strip()}"
                    )
                force_n = (
                    values[key].pretension_a_n
                    if phase == "A"
                    else values[key].pretension_b_n
                )
                fields[1] = _replace_token(fields[1], f"{force_n:.12g}")
                body = ",".join(fields)
                marker = (element_id, phase)
                if marker in updated_pretensions:
                    raise ValueError(
                        f"duplicate *PRETENSION entry in MCT template: {marker}"
                    )
                updated_pretensions.add(marker)

        output.append(body + ending)

    expected_sections = 2 * n_seg
    expected_pretensions = 8 * n_seg
    if len(updated_sections) != expected_sections:
        raise ValueError(
            f"MCT template updated {len(updated_sections)} cable sections; "
            f"expected {expected_sections}"
        )
    if len(updated_pretensions) != expected_pretensions:
        raise ValueError(
            f"MCT template updated {len(updated_pretensions)} pretensions; "
            f"expected {expected_pretensions}"
        )
    return "".join(output), len(updated_sections), len(updated_pretensions)


def run(best_design=DEFAULT_BEST_DESIGN, mct_file=DEFAULT_MCT_FILE) -> Path:
    """Create a non-overwriting updated MCT beside ``mct_file``."""

    design_path = Path(best_design).expanduser().resolve()
    template_path = Path(mct_file).expanduser().resolve()
    if not design_path.is_file():
        raise FileNotFoundError(f"best design not found: {design_path}")
    if not template_path.is_file():
        raise FileNotFoundError(f"MCT template not found: {template_path}")

    n_seg, strand_area, values = _load_design(design_path)
    source = template_path.read_text(encoding="utf-8")
    updated, section_count, pretension_count = _rewrite_mct(
        source,
        n_seg=n_seg,
        strand_area=strand_area,
        values=values,
    )
    output_path = _updated_path(template_path)
    with output_path.open("x", encoding="utf-8", newline="") as stream:
        stream.write(updated)
    print(
        f"created {output_path} "
        f"({section_count} sections, {pretension_count} pretensions)"
    )
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Write BridgeZoo optimized cable areas and A/B pretensions into a "
            "new sibling MIDAS MCT file without overwriting the template."
        )
    )
    parser.add_argument(
        "best_design",
        nargs="?",
        default=DEFAULT_BEST_DESIGN,
        help=f"optimization JSON (default: {DEFAULT_BEST_DESIGN})",
    )
    parser.add_argument(
        "mct_file",
        nargs="?",
        default=DEFAULT_MCT_FILE,
        help=f"MCT template (default: {DEFAULT_MCT_FILE})",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    run(args.best_design, args.mct_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
