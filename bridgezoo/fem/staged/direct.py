"""自研二维直接刚度法的逐阶段(增量、变刚度)求解器 —— RL 训练内核。

消费 :class:`bridgezoo.fem.staged.plan.StagedPlan`,返回 :class:`StagedResult`。纯 numpy,
**线性、小位移**假设(求快,RL 内核);与 :mod:`bridgezoo.fem.staged.opensees` 后端
执行同一计划、返回同一结果类型,逐项可比。

小位移下与 OpenSees ``linear`` 模式逐项吻合到机器精度;在刻意放大的大挠度算例下,
OpenSees ``corot`` 模式因几何非线性约有 1% 量级差异——这正好量化了线性近似的适用范围。
"""

from __future__ import annotations

import math

import numpy as np

from bridgezoo.fem.kernels import (
    _frame_local_stiffness,
    _frame_transform,
    _gravity_feq_global,
)
from bridgezoo.fem.staged.plan import (
    BalanceDof,
    BuildStep,
    NewNode,
    StagedPlan,
    StagedResult,
    StagedStepRecord,
    _attach_geometry,
)


class StagedDirectSolver:
    """自研二维直接刚度法的逐阶段(增量、变刚度)求解器。"""

    name = "direct"

    def run(self, plan: StagedPlan) -> StagedResult:
        self.coords: dict[int, tuple[float, float]] = {}
        self.u: dict[int, np.ndarray] = {}
        self.fixed: dict[int, tuple[bool, bool, bool]] = {}
        self.frames: list[dict] = []
        self.cables: list[dict] = []
        self._areas: dict[int, float] = {}

        for sp in plan.supports:
            self.fixed[sp[0]] = (sp[1], sp[2], sp[3])
        for nd in plan.init_nodes:
            self._add_node(nd)

        result = StagedResult(backend=self.name)
        for step in plan.steps:
            self._activate(step)
            dF = self._incremental_loads(step)
            self._last_applied_loads = self._explicit_nodal_loads(step)
            self._solve_increment(dF, step.balance_dofs)
            if step.record:
                result.records.append(self._record(step.label))
        _attach_geometry(result, plan)
        return result

    # --------------------------------------------------------------
    def _add_node(self, nd: NewNode) -> None:
        self.coords[nd.id] = (nd.x, nd.y)
        if nd.attach is not None:
            xI, yI = self.coords[nd.attach]
            uI = self.u[nd.attach]
            dx, dy = nd.x - xI, nd.y - yI
            rzI = uI[2]
            self.u[nd.id] = np.array([uI[0] - dy * rzI, uI[1] + dx * rzI, rzI])
        else:
            self.u[nd.id] = np.zeros(3)
        fx = self.fixed.get(nd.id)
        if fx is not None:
            for k, is_fixed in enumerate(fx):
                if is_fixed:
                    self.u[nd.id][k] = 0.0

    def _activate(self, step: BuildStep) -> None:
        for nd in step.new_nodes:
            self._add_node(nd)
        for fr in step.new_frames:
            birth = np.concatenate([self.u[fr.i].copy(), self.u[fr.j].copy()])
            self.frames.append(dict(o=fr, birth=birth))
        for cb in step.new_cables:
            birth_i = self.u[cb.i].copy()
            birth_j = self.u[cb.j].copy()
            self.cables.append(dict(o=cb, birth_i=birth_i, birth_j=birth_j, N0=cb.tension))
            self._areas[cb.id] = cb.A

    def _orig_geom(self, i: int, j: int) -> tuple[float, float, float]:
        """单元原始坐标下的长度与方向余弦(线性小位移参考)。"""
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        return L, dx / L, dy / L

    def _incremental_loads(self, step: BuildStep) -> dict[int, np.ndarray]:
        """本步新增的增量荷载(仅本步引入的自重 UDL + 索预张力等效节点力)。"""
        dF: dict[int, np.ndarray] = {}

        def add(nid, vec):
            dF.setdefault(nid, np.zeros(3))
            dF[nid] += vec

        for fr in step.new_frames:
            if fr.udl_wy != 0.0:
                L, c, s = self._orig_geom(fr.i, fr.j)
                feq_global = _gravity_feq_global(fr.udl_wy, c, s, L)
                add(fr.i, feq_global[0:3])
                add(fr.j, feq_global[3:6])
        for cb in step.new_cables:
            if cb.tension != 0.0:
                _, c, s = self._orig_geom(cb.i, cb.j)
                add(cb.i, np.array([cb.tension * c, cb.tension * s, 0.0]))
                add(cb.j, np.array([-cb.tension * c, -cb.tension * s, 0.0]))
        for nl in step.nodal_loads:
            add(nl.node, np.array([nl.fx, nl.fy, nl.mz], dtype=float))
        return dF

    @staticmethod
    def _explicit_nodal_loads(step: BuildStep) -> dict[int, np.ndarray]:
        loads: dict[int, np.ndarray] = {}
        for nl in step.nodal_loads:
            loads.setdefault(nl.node, np.zeros(3))
            loads[nl.node] += np.array([nl.fx, nl.fy, nl.mz], dtype=float)
        return loads

    def _solve_increment(self, dF: dict[int, np.ndarray], balance_dofs: list[BalanceDof] | None = None) -> None:
        active = list(self.coords.keys())
        idx = {nid: k for k, nid in enumerate(active)}
        ndof = 3 * len(active)
        K = np.zeros((ndof, ndof))
        F = np.zeros(ndof)

        def dofs(nid):
            b = 3 * idx[nid]
            return [b, b + 1, b + 2]

        for fr in self.frames:
            o = fr["o"]
            xi, yi = self.coords[o.i]
            xj, yj = self.coords[o.j]
            L = math.hypot(xj - xi, yj - yi)
            c, s = (xj - xi) / L, (yj - yi) / L
            kl = _frame_local_stiffness(o.E, o.A, o.I, L)
            T = _frame_transform(c, s)
            kg = T.T @ kl @ T
            ed = dofs(o.i) + dofs(o.j)
            for a in range(6):
                for b in range(6):
                    K[ed[a], ed[b]] += kg[a, b]
        for cb in self.cables:
            o = cb["o"]
            L, c, s = self._orig_geom(o.i, o.j)
            ka = o.E * o.A / L
            bvec = np.array([-c, -s, c, s])
            kg4 = ka * np.outer(bvec, bvec)
            td = [dofs(o.i)[0], dofs(o.i)[1], dofs(o.j)[0], dofs(o.j)[1]]
            for a in range(4):
                for d in range(4):
                    K[td[a], td[d]] += kg4[a, d]

        for nid, vec in dF.items():
            d = dofs(nid)
            F[d[0]] += vec[0]
            F[d[1]] += vec[1]
            F[d[2]] += vec[2]

        # 约束
        fixed = np.zeros(ndof, dtype=bool)
        for nid, fx in self.fixed.items():
            if nid in idx:
                d = dofs(nid)
                for k in range(3):
                    if fx[k]:
                        fixed[d[k]] = True
        for p in range(ndof):
            if not fixed[p] and abs(K[p, p]) < 1e-30 and np.allclose(K[p, :], 0.0):
                fixed[p] = True

        free = np.where(~fixed)[0]
        du = np.zeros(ndof)
        if free.size > 0:
            Kff = K[np.ix_(free, free)]
            Ff = F[free]
            if balance_dofs:
                current = np.zeros(ndof)
                for nid in active:
                    d = dofs(nid)
                    current[d] = self.u[nid]
                control = []
                target = []
                for bd in balance_dofs:
                    gdof = dofs(bd.node)[bd.dof]
                    if fixed[gdof]:
                        raise ValueError(f"balance dof is already fixed: node={bd.node}, dof={bd.dof}")
                    control.append(int(np.where(free == gdof)[0][0]))
                    target.append(float(bd.target))
                base_du = np.linalg.solve(Kff, Ff)
                unit = np.zeros((free.size, len(control)))
                for j, pos in enumerate(control):
                    unit[pos, j] = 1.0
                flex_cols = np.linalg.solve(Kff, unit)
                flex = flex_cols[control, :]
                rhs = np.array(target) - current[free[control]] - base_du[control]
                balance_load = np.linalg.solve(flex, rhs)
                Ff = Ff + unit @ balance_load
                for bd, value in zip(balance_dofs, balance_load):
                    self._last_applied_loads.setdefault(bd.node, np.zeros(3))
                    self._last_applied_loads[bd.node][bd.dof] += float(value)
            du[free] = np.linalg.solve(Kff, Ff)

        for nid in active:
            d = dofs(nid)
            self.u[nid] = self.u[nid] + np.array([du[d[0]], du[d[1]], du[d[2]]])

    def _record(self, label: str) -> StagedStepRecord:
        disp = {nid: tuple(float(v) for v in self.u[nid]) for nid in self.coords}
        cforce, cstress = {}, {}
        for cb in self.cables:
            o = cb["o"]
            L, c, s = self._orig_geom(o.i, o.j)
            dui = self.u[o.i] - cb["birth_i"]
            duj = self.u[o.j] - cb["birth_j"]
            elong = c * (duj[0] - dui[0]) + s * (duj[1] - dui[1])
            N = cb["N0"] + o.E * o.A / L * elong
            cforce[o.id] = float(N)
            cstress[o.id] = float(N / o.A)
        loads = {nid: tuple(float(x) for x in vec) for nid, vec in getattr(self, "_last_applied_loads", {}).items()}
        return StagedStepRecord(label=label, disp=disp, cable_force=cforce, cable_stress=cstress, applied_loads=loads)
