"""由施工计划派生的成桥(完成态)模型组装。

:class:`bridgezoo.fem.staged.plan.StagedPlan` 在 :func:`build_staged_cantilever`
末尾会附带一个 :class:`~bridgezoo.fem.staged.plan.CompletedState`(成桥完成态:所有梁段/
拉索一次激活、用真实支座替代合龙平衡力)。本模块把该状态翻译成与求解器无关的
:class:`StructuralModel`,即可分别交给自研直接刚度法
:class:`~bridgezoo.fem.completed.direct.CompletedDirectSolver` 或
:class:`~bridgezoo.fem.completed.opensees.CompletedOpenSeesSolver` 求解并交叉校核。

这是"一套定义、两种后端、结果一致"在**成桥工况**上的唯一建模入口:施工过程
(逐阶段)与成桥(一次)都从同一个 ``StagedPlan`` 派生,几何/编号/索连接保持一致。
"""

from __future__ import annotations

import math

from bridgezoo.fem.model import StructuralModel
from bridgezoo.fem.staged.plan import StagedPlan


def build_completed_model(plan: StagedPlan, name: str | None = None) -> tuple[StructuralModel, dict]:
    """由施工计划的成桥状态构建完成态 :class:`StructuralModel`。

    Parameters
    ----------
    plan : StagedPlan
        必须带有 ``plan.completed``(由 :func:`build_staged_cantilever` 生成)。
    name : str, optional
        结构模型名;缺省由 plan 名派生。

    Returns
    -------
    (model, meta) : tuple[StructuralModel, dict]
        ``meta`` 含 ``coords`` / ``deck_ids`` / ``anchor_ids`` / ``cable_nodes``,
        供绘图/后处理按 id 取几何,免去外部反推。

    Raises
    ------
    ValueError
        当 ``plan.completed`` 为 ``None``(该计划未定义成桥状态)。
    """
    if plan.completed is None:
        raise ValueError("staged plan does not define a completed state")
    completed = plan.completed

    model = StructuralModel(name=name or f"{completed.label}_{plan.name}")
    coords: dict[int, tuple[float, float]] = {}
    deck_ids: list[int] = []
    anchor_ids: list[int] = []
    cable_nodes: dict[int, tuple[int, int]] = {}

    for nd in completed.nodes:
        model.add_node(nd.id, nd.x, nd.y)
        coords[nd.id] = (nd.x, nd.y)

    for nid, (_, y) in coords.items():
        if y > 1e-9:
            anchor_ids.append(nid)
        else:
            deck_ids.append(nid)
    deck_ids.sort(key=lambda nid: coords[nid][0])
    anchor_ids.sort(key=lambda nid: coords[nid][1])

    for node, ux, uy, rz in completed.supports:
        model.add_support(node, ux=ux, uy=uy, rz=rz)

    for fr in completed.frames:
        model.add_frame(fr.id, fr.i, fr.j, fr.E, fr.A, fr.I)
        xi, yi = coords[fr.i]
        xj, yj = coords[fr.j]
        length = math.hypot(xj - xi, yj - yi)
        c = (xj - xi) / length
        # StructuralModel stores OpenSees local transverse load Wy.  The staged
        # plan defines global gravity, so project it for each beam direction.
        # This matters for left-side beams with reversed x.
        model.add_member_udl(fr.id, fr.udl_wy * c)
    for cb in completed.cables:
        model.add_cable(cb.id, cb.i, cb.j, cb.E, cb.A, cb.tension)
        cable_nodes[cb.id] = (cb.i, cb.j)
    for load in completed.nodal_loads:
        model.add_nodal_load(load.node, load.fx, load.fy, load.mz)

    meta = {
        "coords": coords,
        "deck_ids": deck_ids,
        "anchor_ids": anchor_ids,
        "cable_nodes": cable_nodes,
    }
    return model, meta
