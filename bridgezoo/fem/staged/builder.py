"""Build staged cantilever construction plans.

The generated :class:`StagedPlan` is backend independent and can be consumed by
both the direct staged solver and the OpenSees staged solver.

Node and element numbering convention, where ``n`` is the number of cables on
each side:

    root deck node            : 0
    tower anchor i            : 300 + i
    right cable deck node i   : i
    right free tip            : 200
    left cable deck node i    : 100 + i
    left free tip             : 201
    right girder element i    : 10 + i
    left girder element i     : 110 + i
    right free-tip element    : 90
    left free-tip element     : 190
    right cable i             : 1000 + i
    left cable i              : 2000 + i
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from bridgezoo.fem.staged.plan import (
    BuildStep,
    CompletedState,
    NewCable,
    NewFrame,
    NewNode,
    StagedPlan,
)

_ROOT = 0


def _anchor_id(i: int) -> int:
    return 300 + i


def _right_cable_id(i: int) -> int:
    return 1000 + i


def _left_cable_id(i: int) -> int:
    return 2000 + i


def _is_sequence_value(value) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _as_strand_count(value) -> int:
    if isinstance(value, bool) or isinstance(value, complex):
        raise ValueError("strands must be positive integer real values")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("strands must be positive integer real values") from exc
    if not math.isfinite(number) or not number.is_integer():
        raise ValueError("strands must be positive integer real values")
    count = int(number)
    if count <= 0:
        raise ValueError("strands must be positive")
    return count


def _normalize_strands(strands, n_seg: int) -> list[tuple[int, int]]:
    """Return stage-major ``(right, left)`` strand counts."""

    if strands is None:
        return [(20, 20)] * n_seg

    if isinstance(strands, Mapping):
        pairs = []
        missing = []
        for i in range(1, n_seg + 1):
            right_id = _right_cable_id(i)
            left_id = _left_cable_id(i)
            if right_id not in strands:
                missing.append(right_id)
            if left_id not in strands:
                missing.append(left_id)
            if right_id in strands and left_id in strands:
                pairs.append((_as_strand_count(strands[right_id]), _as_strand_count(strands[left_id])))
        if missing:
            raise ValueError(f"strands mapping is missing cable ids: {missing}")
        return pairs

    values = list(strands)
    if len(values) == n_seg:
        pairs = []
        for value in values:
            if _is_sequence_value(value):
                if len(value) != 2:
                    raise ValueError("each paired strands value must contain exactly 2 entries: (right, left)")
                pairs.append((_as_strand_count(value[0]), _as_strand_count(value[1])))
            else:
                count = _as_strand_count(value)
                pairs.append((count, count))
        return pairs

    if len(values) == 2 * n_seg:
        return [(_as_strand_count(values[2 * k]), _as_strand_count(values[2 * k + 1])) for k in range(n_seg)]

    raise ValueError(
        "strands length must be n_seg for paired input, 2*n_seg for independent "
        "stage-major input, or a mapping keyed by cable id"
    )


def _normalize_pretension(pretension, n_seg: int) -> list[tuple[float, float]]:
    """Return stage-major ``(right, left)`` pretensions.

    Backward compatible inputs with ``n_seg`` scalar values still mean the same
    target force is applied to the right and left cable of each construction
    stage.  New independent inputs may be supplied either as ``n_seg`` pairs or
    as a flat stage-major sequence: ``right1, left1, right2, left2, ...``.
    """

    if pretension is None:
        return [(0.0, 0.0)] * n_seg

    if isinstance(pretension, Mapping):
        pairs = []
        missing = []
        for i in range(1, n_seg + 1):
            right_id = _right_cable_id(i)
            left_id = _left_cable_id(i)
            if right_id not in pretension:
                missing.append(right_id)
            if left_id not in pretension:
                missing.append(left_id)
            if right_id in pretension and left_id in pretension:
                pairs.append((float(pretension[right_id]), float(pretension[left_id])))
        if missing:
            raise ValueError(f"pretension mapping is missing cable ids: {missing}")
        return pairs

    values = list(pretension)
    if len(values) == n_seg:
        pairs = []
        for value in values:
            if _is_sequence_value(value):
                if len(value) != 2:
                    raise ValueError("each paired pretension value must contain exactly 2 entries: (right, left)")
                pairs.append((float(value[0]), float(value[1])))
            else:
                force = float(value)
                pairs.append((force, force))
        return pairs

    if len(values) == 2 * n_seg:
        return [(float(values[2 * k]), float(values[2 * k + 1])) for k in range(n_seg)]

    raise ValueError(
        "pretension length must be n_seg for paired input, 2*n_seg for independent "
        "stage-major input, or a mapping keyed by cable id"
    )


def build_staged_cantilever(
    n_seg: int = 6,
    # Tower fan anchors.
    anchor_base_height: float = 20.0,
    anchor_spacing: float = 3.0,
    anchor_top_free: float = 5.0,
    # Deck geometry, measured from the tower at x=0.
    left_start: float = 6.0,
    left_spacing: float = 8.0,
    left_end: float = 4.0,
    right_start: float = 6.0,
    right_spacing: float = 8.0,
    right_end: float = 4.0,
    # Section and material properties.
    beam_E: float = 20e9,
    beam_A: float = 10.0,
    beam_Iz: float = 10.0 / 12.0,
    wg: float = 1.0e5,
    cable_Es: float = 1.95e11,
    strand_area: float = 1.4e-4,
    strands: Sequence[int | Sequence[int]] | Mapping[int, int] | None = None,
    pretension: Sequence[float | Sequence[float]] | Mapping[int, float] | None = None,
) -> StagedPlan:
    """Build a symmetric staged double-cantilever cable-stayed bridge plan.

    The first increment combines girder segment 1 and cable 1 in one solve,
    because the bare first girder increment is under-constrained while the
    tower-deck rotation is released.  Later increments keep the original
    two-step rhythm: install segment, then tension cable.

    The final ``tip_free`` step tangent-activates the closing tip segments
    stress-free and immediately adds closure supports (left tip uy, right tip
    ux+rz).  The constrained DOFs are locked at their tangent-birth values; no
    balancing reaction step drives them back to zero.
    """

    _ = anchor_top_free  # Geometry metadata for plotting callers; no plan DOF uses it.
    strand_pairs = _normalize_strands(strands, n_seg)
    pretension_pairs = _normalize_pretension(pretension, n_seg)
    assert n_seg < 90, "current numbering convention requires n_seg < 90"

    plan = StagedPlan(name=f"staged_half_bridge_N{n_seg}")

    plan.init_nodes = [NewNode(_ROOT, 0.0, 0.0)]
    # Tower-deck joint: keep translations coupled, release deck rotation.
    plan.supports = [(_ROOT, True, True, False)]
    for i in range(1, n_seg + 1):
        hy = anchor_base_height + (i - 1) * anchor_spacing
        plan.init_nodes.append(NewNode(_anchor_id(i), 0.0, hy))
        plan.supports.append((_anchor_id(i), True, True, True))

    def side_x(i: int, start: float, spacing: float, sign: int) -> float:
        return sign * (start + (i - 1) * spacing)

    sides = [
        dict(sign=+1, start=right_start, spacing=right_spacing, end=right_end,
             node=lambda i: i, tip=200, frame=lambda i: 10 + i, tip_frame=90, cable=_right_cable_id),
        dict(sign=-1, start=left_start, spacing=left_spacing, end=left_end,
             node=lambda i: 100 + i, tip=201, frame=lambda i: 110 + i, tip_frame=190, cable=_left_cable_id),
    ]

    prev = {+1: _ROOT, -1: _ROOT}
    for i in range(1, n_seg + 1):
        seg_nodes, seg_frames = [], []
        for sd in sides:
            nid = sd["node"](i)
            x = side_x(i, sd["start"], sd["spacing"], sd["sign"])
            seg_nodes.append(NewNode(nid, x, 0.0, attach=prev[sd["sign"]]))
            seg_frames.append(NewFrame(sd["frame"](i), prev[sd["sign"]], nid,
                                       beam_E, beam_A, beam_Iz, udl_wy=-wg))

        right_strands, left_strands = strand_pairs[i - 1]
        right_tension, left_tension = pretension_pairs[i - 1]
        seg_cables = [
            NewCable(
                sides[0]["cable"](i),
                _anchor_id(i),
                sides[0]["node"](i),
                cable_Es,
                strand_area * right_strands,
                tension=right_tension,
            ),
            NewCable(
                sides[1]["cable"](i),
                _anchor_id(i),
                sides[1]["node"](i),
                cable_Es,
                strand_area * left_strands,
                tension=left_tension,
            ),
        ]

        if i == 1:
            plan.steps.append(BuildStep(label="cable1", new_nodes=seg_nodes,
                                        new_frames=seg_frames, new_cables=seg_cables, record=True))
        else:
            plan.steps.append(BuildStep(label=f"seg{i}", new_nodes=seg_nodes,
                                        new_frames=seg_frames, record=False))
            plan.steps.append(BuildStep(label=f"cable{i}", new_cables=seg_cables, record=True))

        for sd in sides:
            prev[sd["sign"]] = sd["node"](i)

    tip_nodes, tip_frames = [], []
    for sd in sides:
        last_node = sd["node"](n_seg)
        last_x = side_x(n_seg, sd["start"], sd["spacing"], sd["sign"])
        tip_x = last_x + sd["sign"] * sd["end"]
        tip_nodes.append(NewNode(sd["tip"], tip_x, 0.0, attach=last_node))
        tip_frames.append(NewFrame(sd["tip_frame"], last_node, sd["tip"],
                                   beam_E, beam_A, beam_Iz, udl_wy=-wg))
    # 合龙:端段切线激活后就地添加支座(左端 uy / 右端 ux+rz),被锁自由度保持
    # 切线诞生位移——不再用 closure_balance 反力步把位移压回 0。
    plan.steps.append(BuildStep(
        label="tip_free", new_nodes=tip_nodes, new_frames=tip_frames,
        new_supports=[
            (sides[1]["tip"], False, True, False),
            (sides[0]["tip"], True, False, True),
        ],
        record=True,
    ))
    plan.completed = _build_completed_state(plan, left_tip=sides[1]["tip"], right_tip=sides[0]["tip"])

    return plan


def _build_completed_state(plan: StagedPlan, left_tip: int, right_tip: int) -> CompletedState:
    nodes = list(plan.init_nodes)
    frames = []
    cables = []
    nodal_loads = []
    for step in plan.steps:
        nodes.extend(step.new_nodes)
        frames.extend(step.new_frames)
        cables.extend(step.new_cables)
        nodal_loads.extend(step.nodal_loads)

    anchor_supports = [sp for sp in plan.supports if sp[0] != _ROOT]
    supports = [
        *anchor_supports,
        (_ROOT, True, True, False),
        (left_tip, False, True, False),
        (right_tip, True, False, True),
    ]
    return CompletedState(
        nodes=nodes,
        frames=frames,
        cables=cables,
        supports=supports,
        nodal_loads=nodal_loads,
    )
