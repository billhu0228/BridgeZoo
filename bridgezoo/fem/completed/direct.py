"""一次成桥（完成态）自研二维直接刚度法求解器（求解后端之一）。

消费 :class:`bridgezoo.fem.model.StructuralModel`，返回 :class:`bridgezoo.fem.model.SolveResult`，
与 :mod:`bridgezoo.fem.completed.opensees` 接口一致、结果应一致（用于交叉校核）。

单元
----
- 梁：2D Euler-Bernoulli 框架单元，局部 6 自由度 ``[u_i, v_i, θ_i, u_j, v_j, θ_j]``，
  含轴向 + 弯曲刚度，按单元方向角做坐标变换。
- 索：仅受轴力的二节点杆，按方向余弦组装到两端平动自由度；初始轴力（预张力）以
  等效节点力施加，最终索力 = 预张力 + EA/L·轴向伸长。

荷载
----
- 节点荷载直接进入荷载向量。
- 梁单元局部横向均布荷载 :class:`MemberUDL`：采用**一致等效节点荷载**（fixed-end），
  保证节点位移与 OpenSees ``eleLoad -beamUniform`` 一致；单元端力回收时扣除等效项。

仅依赖 numpy；规模小（DOF<~数百）用稠密直接求解即可，后续大模型可换 scipy.sparse。
本求解器面向**线性、小位移**分析（成桥/单阶段）；逐阶段变刚度的扩展见
:mod:`bridgezoo.fem.staged`。底层单元核来自 :mod:`bridgezoo.fem.kernels`。

参见 ``docs/DESIGN_MAPPO.md`` 第 4 节。
"""

from __future__ import annotations

import math

import numpy as np

from bridgezoo.fem.kernels import (
    _frame_local_stiffness,
    _frame_transform,
    _udl_fixed_end_local,
)
from bridgezoo.fem.model import SolveResult, StructuralModel


class CompletedDirectSolver:
    """二维直接刚度法线性静力求解后端。"""

    name = "direct"

    def solve(self, model: StructuralModel) -> SolveResult:
        nodes = list(model.nodes.values())
        node_ids = [n.id for n in nodes]
        idx = {nid: k for k, nid in enumerate(node_ids)}  # node id -> 0..nnode-1
        ndof = 3 * len(nodes)

        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        def dofs_of(nid):
            b = 3 * idx[nid]
            return [b, b + 1, b + 2]

        # ---------- 组装梁单元 ----------
        for m in model.frames.values():
            xi, yi = model.node_xy(m.i)
            xj, yj = model.node_xy(m.j)
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy)
            c, s = dx / L, dy / L
            kl = _frame_local_stiffness(m.E, m.A, m.I, L)
            T = _frame_transform(c, s)
            kg = T.T @ kl @ T
            ed = dofs_of(m.i) + dofs_of(m.j)
            for a in range(6):
                for b in range(6):
                    K[ed[a], ed[b]] += kg[a, b]
            # 单元均布荷载等效节点荷载
            udl = model.member_udls.get(m.id)
            if udl is not None:
                feq_local = _udl_fixed_end_local(udl.wy, L)
                feq_global = T.T @ feq_local
                for a in range(6):
                    F[ed[a]] += feq_global[a]

        # ---------- 组装索单元（轴力杆 + 预张力等效节点力）----------
        for cab in model.cables.values():
            xi, yi = model.node_xy(cab.i)
            xj, yj = model.node_xy(cab.j)
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy)
            c, s = dx / L, dy / L
            ka = cab.E * cab.A / L
            b = np.array([-c, -s, c, s])
            kg4 = ka * np.outer(b, b)
            td = [dofs_of(cab.i)[0], dofs_of(cab.i)[1], dofs_of(cab.j)[0], dofs_of(cab.j)[1]]
            for a in range(4):
                for d in range(4):
                    K[td[a], td[d]] += kg4[a, d]
            # 预张力 N0：对 i 施加指向 j 的力，对 j 施加指向 i 的力
            N0 = cab.pretension
            if N0 != 0.0:
                F[td[0]] += N0 * c
                F[td[1]] += N0 * s
                F[td[2]] += -N0 * c
                F[td[3]] += -N0 * s

        # ---------- 节点荷载 ----------
        for nl in model.nodal_loads:
            d = dofs_of(nl.node)
            F[d[0]] += nl.fx
            F[d[1]] += nl.fy
            F[d[2]] += nl.mz

        # ---------- 约束 ----------
        fixed = np.zeros(ndof, dtype=bool)
        for sp in model.supports.values():
            d = dofs_of(sp.node)
            if sp.ux:
                fixed[d[0]] = True
            if sp.uy:
                fixed[d[1]] = True
            if sp.rz:
                fixed[d[2]] = True
        # 自动约束无刚度的自由转动自由度（如仅连索的自由节点），避免奇异
        for p in range(ndof):
            if not fixed[p] and abs(K[p, p]) < 1e-30 and np.allclose(K[p, :], 0.0):
                fixed[p] = True

        free = np.where(~fixed)[0]
        u = np.zeros(ndof)
        converged = True
        if free.size > 0:
            Kff = K[np.ix_(free, free)]
            Ff = F[free]
            try:
                u[free] = np.linalg.solve(Kff, Ff)
            except np.linalg.LinAlgError:
                converged = False

        # ---------- 结果回收 ----------
        result = SolveResult(backend=self.name, converged=converged)
        for n in nodes:
            d = dofs_of(n.id)
            result.disp[n.id] = (u[d[0]], u[d[1]], u[d[2]])

        for m in model.frames.values():
            xi, yi = model.node_xy(m.i)
            xj, yj = model.node_xy(m.j)
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy)
            c, s = dx / L, dy / L
            kl = _frame_local_stiffness(m.E, m.A, m.I, L)
            T = _frame_transform(c, s)
            ed = dofs_of(m.i) + dofs_of(m.j)
            d_global = u[ed]
            d_local = T @ d_global
            f_local = kl @ d_local
            udl = model.member_udls.get(m.id)
            if udl is not None:
                f_local = f_local - _udl_fixed_end_local(udl.wy, L)
            result.frame_force[m.id] = tuple(float(v) for v in f_local)

        for cab in model.cables.values():
            xi, yi = model.node_xy(cab.i)
            xj, yj = model.node_xy(cab.j)
            dx, dy = xj - xi, yj - yi
            L = math.hypot(dx, dy)
            c, s = dx / L, dy / L
            di, dj = dofs_of(cab.i), dofs_of(cab.j)
            elong = c * (u[dj[0]] - u[di[0]]) + s * (u[dj[1]] - u[di[1]])
            N = cab.pretension + cab.E * cab.A / L * elong
            result.cable_force[cab.id] = float(N)
            result.cable_stress[cab.id] = float(N / cab.A)

        return result


def solve(model: StructuralModel) -> SolveResult:
    """便捷函数：用直接刚度法求解。"""
    return CompletedDirectSolver().solve(model)
