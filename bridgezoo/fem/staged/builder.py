"""施工计划构建器:由参数生成对称双悬臂 + 扇面索斜拉桥的 :class:`StagedPlan`。

与一次成桥侧的 :mod:`bridgezoo.fem.oneshot.builder` 对应,这是"施工过程"的建模落点:
调用一次得到 :class:`StagedPlan`,即可分别交给 :class:`bridgezoo.fem.staged.direct`
或 :class:`bridgezoo.fem.staged.opensees` 后端执行并对比。

节点 / 单元编号约定(n = 每侧索数;要求 n < 90)::

    根部(0#)节点         : 0
    塔上锚点 i (1..n)     : 300 + i            (y = anchor_base + (i-1)*anchor_spacing)
    右侧索点 i (1..n)     : i                  (x = +(right_start + (i-1)*right_spacing))
    右侧自由端(终止段端)  : 200
    左侧索点 i (1..n)     : 100 + i            (x = -(left_start + (i-1)*left_spacing))
    左侧自由端            : 201
    右侧梁单元 i          : 10 + i  ;右终止段梁: 90
    左侧梁单元 i          : 110 + i ;左终止段梁: 190
    右侧索 i              : 1000 + i ;左侧索 i : 2000 + i
"""

from __future__ import annotations

from bridgezoo.fem.staged.plan import (
    BalanceDof,
    BuildStep,
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
    # —— 塔上锚点(扇面)——
    anchor_base_height: float = 20.0,   # 参数 a:最低锚点高度(自梁面 y=0 起算)
    anchor_spacing: float = 3.0,        # 参数 b:相邻锚点竖向间距(向上)
    anchor_top_free: float = 5.0,       # 参数 c:最高锚点之上的自由高度(仅供绘塔参考)
    # —— 主梁双悬臂(左右可不同;以塔 x=0 为参考)——
    left_start: float = 6.0,            # 参数1(左):塔到第 1 索点距离
    left_spacing: float = 8.0,          # 参数2(左):相邻索点间距
    left_end: float = 4.0,              # 参数3(左):末索点到悬臂自由端距离(无索)
    right_start: float = 6.0,           # 参数1(右)
    right_spacing: float = 8.0,         # 参数2(右)
    right_end: float = 4.0,             # 参数3(右)
    # —— 截面 / 材料 ——
    beam_E: float = 20e9,
    beam_A: float = 10.0,
    beam_Iz: float = 10.0 / 12.0,
    wg: float = 1.0e5,
    cable_Es: float = 1.95e11,
    strand_area: float = 1.4e-4,
    strands: list[int] | None = None,       # 长度 n,左右同索号共用
    pretension: list[float] | None = None,  # 长度 n,左右同索号共用
) -> StagedPlan:
    """构建**对称双悬臂 + 扇面索**斜拉桥的施工计划。

    - 塔上锚点呈扇面:第 i 索锚在高度 ``a + (i-1)*b``(内侧低、外侧高),顶部留自由高 c。
    - 主梁自塔向两侧成对悬臂;左右各 n 个索点,间距等几何可不同(start/spacing/end)。
    - 索点:第 i 索点距塔 ``start + (i-1)*spacing``;``start`` 为塔到首索点的无索引段,
      ``end`` 为末索点到自由端的无索终止段。
    - 施工步序(平衡悬臂):每阶段同时装两侧第 i 节段(切线激活)+ 同时张两侧第 i 对索;
      最后装两侧终止段(无索)。塔为刚性(锚点固定),根部 0# 块固接(x=0 全固定)。

    strands / pretension 长度均为 n,左右同索号共用(如需左右各异,后续可扩展为按侧传入)。
    """
    strands = strands or [20] * n_seg
    pretension = pretension or [0.0] * n_seg
    assert len(strands) == n_seg and len(pretension) == n_seg, "strands/pretension 长度须 = n_seg"
    assert n_seg < 90, "当前编号方案要求 n_seg < 90"

    plan = StagedPlan(name=f"staged_half_bridge_N{n_seg}")

    # 初始:根部 0# 块 + 塔上 n 个锚点(均固定)
    plan.init_nodes = [NewNode(_ROOT, 0.0, 0.0)]
    # Tower-deck joint: keep translations coupled, release deck rotation.
    plan.supports = [(_ROOT, True, True, False)]
    for i in range(1, n_seg + 1):
        hy = anchor_base_height + (i - 1) * anchor_spacing
        plan.init_nodes.append(NewNode(_anchor_id(i), 0.0, hy))
        plan.supports.append((_anchor_id(i), True, True, True))

    # 各侧索点 x 坐标(左为负)与单元 id 偏移
    def side_x(i: int, start: float, spacing: float, sign: int) -> float:
        return sign * (start + (i - 1) * spacing)

    sides = [
        dict(sign=+1, start=right_start, spacing=right_spacing, end=right_end,
             node=lambda i: i, tip=200, frame=lambda i: 10 + i, tip_frame=90, cable=lambda i: 1000 + i),
        dict(sign=-1, start=left_start, spacing=left_spacing, end=left_end,
             node=lambda i: 100 + i, tip=201, frame=lambda i: 110 + i, tip_frame=190, cable=lambda i: 2000 + i),
    ]
    prev = {+1: _ROOT, -1: _ROOT}
    # 逐节段:每阶段同时装两侧第 i 节段,再同时张两侧第 i 对索
    for i in range(1, n_seg + 1):
        seg_nodes, seg_frames = [], []
        for sd in sides:
            nid = sd["node"](i)
            x = side_x(i, sd["start"], sd["spacing"], sd["sign"])
            seg_nodes.append(NewNode(nid, x, 0.0, attach=prev[sd["sign"]]))
            seg_frames.append(NewFrame(sd["frame"](i), prev[sd["sign"]], nid,
                                       beam_E, beam_A, beam_Iz, udl_wy=-wg))
        plan.steps.append(BuildStep(label=f"seg{i}", new_nodes=seg_nodes, new_frames=seg_frames, record=False))

        area = strand_area * strands[i - 1]
        seg_cables = [
            NewCable(sd["cable"](i), _anchor_id(i), sd["node"](i), cable_Es, area, tension=pretension[i - 1])
            for sd in sides
        ]
        plan.steps.append(BuildStep(label=f"cable{i}", new_cables=seg_cables, record=True))

        for sd in sides:
            prev[sd["sign"]] = sd["node"](i)

    # 终止段(两侧自由端,无索)
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
        BalanceDof(sides[1]["tip"], 1, 0.0),  # left vertical support target: uy = 0
        BalanceDof(sides[0]["tip"], 0, 0.0),  # right closure symmetry target: ux = 0
        BalanceDof(sides[0]["tip"], 2, 0.0),  # right closure symmetry target: rz = 0
    ], record=True))

    return plan
