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
    tower base                : 300
    tower mesh nodes          : 10000+
    tower frame elements      : 20000+
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from bridgezoo.fem.staged.plan import (
    BuildStep,
    CompletedState,
    MemberLoad,
    NewCable,
    NewFrame,
    NewNode,
    StagedPlan,
)

_ROOT = 0
_TOWER_BASE = 300
_TOWER_NODE_START = 10_000
_TOWER_FRAME_START = 20_000
_DEFAULT_TOWER_EI = 1.0e18
_DEFAULT_TOWER_EA = 1.0e15


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


def _normalize_tower_stiffness(
    stiffness: Sequence[Sequence[float]] | None,
) -> list[tuple[float, float]]:
    """Validate ``(z, EI)`` control points in metres and N·m²."""

    raw = [(0.0, _DEFAULT_TOWER_EI)] if stiffness is None else list(stiffness)
    if not raw:
        raise ValueError("tower_stiffness requires at least one (z, EI) pair")
    points: list[tuple[float, float]] = []
    for value in raw:
        if not _is_sequence_value(value) or len(value) != 2:
            raise ValueError("each tower_stiffness value must be a (z, EI) pair")
        z, ei = float(value[0]), float(value[1])
        if not math.isfinite(z) or z < 0.0:
            raise ValueError("tower stiffness elevations z must be finite and nonnegative")
        if not math.isfinite(ei) or ei <= 0.0:
            raise ValueError("tower flexural rigidity EI must be finite and positive")
        if points and z <= points[-1][0]:
            raise ValueError("tower stiffness elevations z must be strictly increasing")
        points.append((z, ei))
    return points


def _tower_ei_at(z: float, points: Sequence[tuple[float, float]]) -> float:
    """Piecewise-linear EI interpolation with constant endpoint extrapolation."""

    if z <= points[0][0]:
        return points[0][1]
    for (z0, ei0), (z1, ei1) in zip(points, points[1:]):
        if z <= z1:
            ratio = (z - z0) / (z1 - z0)
            return ei0 + ratio * (ei1 - ei0)
    return points[-1][1]


def _build_tower(
    anchor_heights: Sequence[float],
    tower_top: float,
    stiffness: Sequence[tuple[float, float]],
    max_element_length: float,
    axial_rigidity: float,
) -> tuple[list[NewNode], list[NewFrame]]:
    """Discretise the fixed-base tower, retaining every cable anchor as a node."""

    if not math.isfinite(max_element_length) or max_element_length <= 0.0:
        raise ValueError("tower_element_size must be finite and positive")
    if not math.isfinite(axial_rigidity) or axial_rigidity <= 0.0:
        raise ValueError("tower_axial_rigidity must be finite and positive")
    if tower_top <= 0.0:
        raise ValueError("tower top elevation must be positive")

    anchor_by_z = {float(z): _anchor_id(i) for i, z in enumerate(anchor_heights, start=1)}
    mandatory = {0.0, float(tower_top), *anchor_by_z.keys()}
    mandatory.update(z for z, _ in stiffness if 0.0 < z < tower_top)
    mandatory_z = sorted(mandatory)

    elevations = [mandatory_z[0]]
    for z0, z1 in zip(mandatory_z, mandatory_z[1:]):
        count = max(1, math.ceil((z1 - z0) / max_element_length))
        elevations.extend(z0 + (z1 - z0) * j / count for j in range(1, count))
        elevations.append(z1)

    nodes: list[NewNode] = []
    node_at_z: dict[float, int] = {}
    next_node = _TOWER_NODE_START
    previous: int | None = None
    for z in elevations:
        if math.isclose(z, 0.0, abs_tol=1e-12):
            node_id, role = _TOWER_BASE, "tower_base"
        elif z in anchor_by_z:
            node_id, role = anchor_by_z[z], "anchor"
        else:
            node_id, role = next_node, "tower"
            next_node += 1
        node_at_z[z] = node_id
        nodes.append(NewNode(node_id, 0.0, z, attach=previous, role=role))
        previous = node_id

    frames: list[NewFrame] = []
    for index, (z0, z1) in enumerate(zip(elevations, elevations[1:])):
        midpoint = 0.5 * (z0 + z1)
        # E=1 is a neutral decomposition: A stores EA and I stores EI.  Both
        # staged backends depend only on the products E*A and E*I.
        frames.append(
            NewFrame(
                _TOWER_FRAME_START + index,
                node_at_z[z0],
                node_at_z[z1],
                1.0,
                axial_rigidity,
                _tower_ei_at(midpoint, stiffness),
                group="tower",
            )
        )
    return nodes, frames


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
    wg: float = 1.0e5,        # 主梁自重线荷载 [N/m]（向下，施加为 udl_wy=-wg）
    dw: float = 0.0,          # 二期均布荷载 [N/m]（向下，成桥后对全主梁施加为 wy=-dw）
    cable_Es: float = 1.95e11,
    strand_area: float = 1.4e-4,
    strands: Sequence[int | Sequence[int]] | Mapping[int, int] | None = None,
    pretension: Sequence[float | Sequence[float]] | Mapping[int, float] | None = None,
    # Tower stiffness: (elevation z [m], flexural rigidity EI [N*m^2]).
    tower_stiffness: Sequence[Sequence[float]] | None = None,
    tower_element_size: float = 2.0,
    tower_axial_rigidity: float = _DEFAULT_TOWER_EA,
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

    When ``dw != 0`` a trailing ``phase2`` step applies the second-phase dead
    load (二期恒载, ``N/m`` downward) uniformly to **every** girder element of the
    completed deck and solves one more equilibrium increment.  The same load is
    folded into ``plan.completed`` so the staged-final state and the one-shot
    completed model stay consistent.  ``dw == 0`` (default) adds no extra step.

    The tower is a fixed-base Euler-Bernoulli frame. ``tower_stiffness`` gives
    ``(z, EI)`` control points (m, N·m²); EI is linearly interpolated at element
    midpoints and held constant outside the supplied z range.  Tower members
    have no self-weight in this model.  Their axial rigidity is the independent
    ``tower_axial_rigidity`` value (EA, N).
    """

    strand_pairs = _normalize_strands(strands, n_seg)
    pretension_pairs = _normalize_pretension(pretension, n_seg)
    tower_stiffness_points = _normalize_tower_stiffness(tower_stiffness)
    assert n_seg < 90, "current numbering convention requires n_seg < 90"
    if not math.isfinite(anchor_top_free) or anchor_top_free < 0.0:
        raise ValueError("anchor_top_free must be finite and nonnegative")

    plan = StagedPlan(name=f"staged_half_bridge_N{n_seg}")

    plan.init_nodes = [NewNode(_ROOT, 0.0, 0.0, role="deck")]
    # Tower-deck joint: keep translations coupled, release deck rotation.
    # The deformable tower uses a separate coincident fixed-base node so the
    # existing deck rotation release remains unchanged.
    plan.supports = [
        (_ROOT, True, True, False),
        (_TOWER_BASE, True, True, True),
    ]
    anchor_heights = []
    for i in range(1, n_seg + 1):
        hy = anchor_base_height + (i - 1) * anchor_spacing
        anchor_heights.append(hy)
    if any(not math.isfinite(z) or z <= 0.0 for z in anchor_heights):
        raise ValueError("tower anchor elevations must be finite and positive")
    if any(b <= a for a, b in zip(anchor_heights, anchor_heights[1:])):
        raise ValueError("tower anchor elevations must be strictly increasing")
    tower_top = anchor_heights[-1] + anchor_top_free
    tower_nodes, tower_frames = _build_tower(
        anchor_heights,
        tower_top,
        tower_stiffness_points,
        tower_element_size,
        tower_axial_rigidity,
    )

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
            seg_nodes.append(NewNode(nid, x, 0.0, attach=prev[sd["sign"]], role="deck"))
            seg_frames.append(NewFrame(sd["frame"](i), prev[sd["sign"]], nid,
                                       beam_E, beam_A, beam_Iz, udl_wy=-wg, group="deck"))

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
            plan.steps.append(BuildStep(
                label="cable1",
                new_nodes=[*tower_nodes, *seg_nodes],
                new_frames=[*tower_frames, *seg_frames],
                new_cables=seg_cables,
                record=True,
            ))
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
        tip_nodes.append(NewNode(sd["tip"], tip_x, 0.0, attach=last_node, role="deck"))
        tip_frames.append(NewFrame(sd["tip_frame"], last_node, sd["tip"],
                                   beam_E, beam_A, beam_Iz, udl_wy=-wg, group="deck"))
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

    # 二期恒载:成桥合龙后,对**全部已激活主梁单元**(各节段 + 合龙端段)一次性施加
    # 向下均布荷载 dw,再求解一次增量平衡。dw=0 时不追加该阶段(行为与既有完全一致)。
    if dw != 0.0:
        girder_members = [
            (fr.id, fr.i, fr.j)
            for step in plan.steps
            for fr in step.new_frames
            if fr.group == "deck"
        ]
        plan.steps.append(BuildStep(
            label="phase2",
            member_loads=[MemberLoad(mid, i, j, -dw) for (mid, i, j) in girder_members],
            record=True,
        ))

    plan.completed = _build_completed_state(
        plan, left_tip=sides[1]["tip"], right_tip=sides[0]["tip"], dw=dw
    )

    return plan


def _build_completed_state(
    plan: StagedPlan, left_tip: int, right_tip: int, dw: float = 0.0
) -> CompletedState:
    nodes = list(plan.init_nodes)
    frames = []
    cables = []
    nodal_loads = []
    for step in plan.steps:
        nodes.extend(step.new_nodes)
        for fr in step.new_frames:
            if dw != 0.0 and fr.group == "deck":
                # 成桥模型中,二期恒载与自重合并为单一分布荷载 udl_wy=-(wg+dw)。
                # 复制 NewFrame(不改 steps 共享的原对象),build_completed_model 会自动
                # 把 udl_wy*c 投影成 StructuralModel 的 member_udl。
                frames.append(
                    NewFrame(
                        fr.id,
                        fr.i,
                        fr.j,
                        fr.E,
                        fr.A,
                        fr.I,
                        udl_wy=fr.udl_wy - dw,
                        group=fr.group,
                    )
                )
            else:
                frames.append(fr)
        cables.extend(step.new_cables)
        nodal_loads.extend(step.nodal_loads)

    tower_supports = [sp for sp in plan.supports if sp[0] != _ROOT]
    supports = [
        *tower_supports,
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
