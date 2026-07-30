import json
import math

import pytest

from scripts.write_best_design import run


def _design_payload():
    return {
        "problem": {"n_seg": 2, "strand_area": 0.00014},
        "cable_groups": [
            {
                "stage": 1,
                "group": "main_stay",
                "strands_per_physical_cable": 29,
                "pretension_A_per_physical_cable_N": 1_100_000.0,
                "pretension_B_per_physical_cable_N": 210_000.0,
            },
            {
                "stage": 1,
                "group": "backstay",
                "strands_per_physical_cable": 74,
                "pretension_A_per_physical_cable_N": 1_200_000.0,
                "pretension_B_per_physical_cable_N": 220_000.0,
            },
            {
                "stage": 2,
                "group": "main_stay",
                "strands_per_physical_cable": 84,
                "pretension_A_per_physical_cable_N": 1_300_000.0,
                "pretension_B_per_physical_cable_N": 230_000.0,
            },
            {
                "stage": 2,
                "group": "backstay",
                "strands_per_physical_cable": 77,
                "pretension_A_per_physical_cable_N": 1_400_000.0,
                "pretension_B_per_physical_cable_N": 240_000.0,
            },
        ],
    }


def _template_text():
    section_rows = "\n".join(
        f"  {section_id}, DBUSER, Cable{section_id}, CC, 0, 0, 0, 0, 0, 0, "
        "YES, NO, SR, 2, 0.1, 0"
        for section_id in (101, 102, 201, 202)
    )
    pretension_rows = []
    for element_base in (100, 200, 300, 400):
        for stage in (1, 2):
            for phase in ("B", "A"):
                pretension_rows.append(
                    f"   {element_base + stage}, 1e+06, CF{phase}-{stage}"
                )
    return (
        "*UNIT\n   N, M\n*SECTION\n"
        f"{section_rows}\n"
        "*USE-STLD, CableForce\n*PRETENSION\n"
        "; ELEM_LIST, TENS, GROUP\n"
        + "\n".join(pretension_rows)
        + "\n"
    )


def test_write_best_design_creates_versioned_mct_without_overwriting(tmp_path):
    design_path = tmp_path / "best_design.json"
    template_path = tmp_path / "template.mct"
    design_path.write_text(json.dumps(_design_payload()), encoding="utf-8")
    original = _template_text()
    template_path.write_text(original, encoding="utf-8")

    first = run(design_path, template_path)
    second = run(design_path, template_path)

    assert template_path.read_text(encoding="utf-8") == original
    assert first.name == "template_updated.mct"
    assert second.name == "template_updated_2.mct"
    updated = first.read_text(encoding="utf-8")
    main_diameter = math.sqrt(4.0 * 29 * 0.00014 / math.pi)
    back_diameter = math.sqrt(4.0 * 74 * 0.00014 / math.pi)
    assert f"{main_diameter:.10g}" in updated
    assert f"{back_diameter:.10g}" in updated
    assert "101, 1100000, CFA-1" in updated
    assert "201, 210000, CFB-1" in updated
    assert "301, 1200000, CFA-1" in updated
    assert "401, 220000, CFB-1" in updated


def test_write_best_design_rejects_incomplete_template(tmp_path):
    design_path = tmp_path / "best_design.json"
    template_path = tmp_path / "template.mct"
    design_path.write_text(json.dumps(_design_payload()), encoding="utf-8")
    template_path.write_text("*SECTION\n", encoding="utf-8")

    with pytest.raises(ValueError, match="updated 0 cable sections; expected 4"):
        run(design_path, template_path)

    assert not (tmp_path / "template_updated.mct").exists()
