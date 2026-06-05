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

from bridgezoo.fem.staged.plan import (
    BalanceDof,
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
    strands: list[int] | None = None,
    pretension: list[float] | None = None,
) -> StagedPlan:
    """Build a symmetric staged double-cantilever cable-stayed bridge plan.

    The first increment combines girder segment 1 and cable 1 in one solve,
    because the bare first girder increment is under-constrained while the
    tower-deck rotation is released.  Later increments keep the original
    two-step rhythm: install segment, then tension cable.
    """

    _ = anchor_top_free  # Geometry metadata for plotting callers; no plan DOF uses it.
    strands = strands or [20] * n_seg
    pretension = pretension or [0.0] * n_seg
    assert len(strands) == n_seg and len(pretension) == n_seg, "strands/pretension length must equal n_seg"
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
             node=lambda i: i, tip=200, frame=lambda i: 10 + i, tip_frame=90, cable=lambda i: 1000 + i),
        dict(sign=-1, start=left_start, spacing=left_spacing, end=left_end,
             node=lambda i: 100 + i, tip=201, frame=lambda i: 110 + i, tip_frame=190, cable=lambda i: 2000 + i),
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

        area = strand_area * strands[i - 1]
        seg_cables = [
            NewCable(sd["cable"](i), _anchor_id(i), sd["node"](i), cable_Es, area, tension=pretension[i - 1])
            for sd in sides
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
    plan.steps.append(BuildStep(label="tip_free", new_nodes=tip_nodes, new_frames=tip_frames, record=True))
    plan.steps.append(BuildStep(label="closure_balance", balance_dofs=[
        BalanceDof(sides[1]["tip"], 1, 0.0),
        BalanceDof(sides[0]["tip"], 0, 0.0),
        BalanceDof(sides[0]["tip"], 2, 0.0),
    ], record=True))
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
