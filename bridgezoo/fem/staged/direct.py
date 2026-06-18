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


def _assemble_global_stiffness(coords, frames, cables, idx, ndof: int) -> np.ndarray:
    """按原始坐标(线性小位移)装配当前激活构件的整体刚度矩阵。

    与 :meth:`StagedDirectSolver._solve_increment` 内联装配逐位等价;刚度只依赖
    几何与 E/A/I,**与位移、预张力无关**,故可在多右端(同结构、异荷载)间复用。
    ``frames`` / ``cables`` 为求解器内部的 ``{"o": NewFrame|NewCable, ...}`` 字典列表。
    """
    K = np.zeros((ndof, ndof))

    def dofs(nid):
        b = 3 * idx[nid]
        return [b, b + 1, b + 2]

    for fr in frames:
        o = fr["o"]
        xi, yi = coords[o.i]
        xj, yj = coords[o.j]
        L = math.hypot(xj - xi, yj - yi)
        c, s = (xj - xi) / L, (yj - yi) / L
        kl = _frame_local_stiffness(o.E, o.A, o.I, L)
        T = _frame_transform(c, s)
        kg = T.T @ kl @ T
        ed = dofs(o.i) + dofs(o.j)
        K[np.ix_(ed, ed)] += kg
    for cb in cables:
        o = cb["o"]
        xi, yi = coords[o.i]
        xj, yj = coords[o.j]
        L = math.hypot(xj - xi, yj - yi)
        c, s = (xj - xi) / L, (yj - yi) / L
        ka = o.E * o.A / L
        bvec = np.array([-c, -s, c, s])
        kg4 = ka * np.outer(bvec, bvec)
        td = [dofs(o.i)[0], dofs(o.i)[1], dofs(o.j)[0], dofs(o.j)[1]]
        K[np.ix_(td, td)] += kg4
    return K


def _orig_geom(coords, i: int, j: int) -> tuple[float, float, float]:
    """单元原始坐标下的长度与方向余弦(线性小位移参考)。"""
    xi, yi = coords[i]
    xj, yj = coords[j]
    dx, dy = xj - xi, yj - yi
    L = math.hypot(dx, dy)
    return L, dx / L, dy / L


def _step_incremental_loads(step: BuildStep, coords) -> dict[int, np.ndarray]:
    """本步新增的增量荷载(仅本步引入的自重 UDL + 索预张力等效节点力 + 节点力)。

    单一来源,供标量与批量两条求解路径复用(只依赖 step 与原始坐标)。
    """
    dF: dict[int, np.ndarray] = {}

    def add(nid, vec):
        dF.setdefault(nid, np.zeros(3))
        dF[nid] += vec

    for fr in step.new_frames:
        if fr.udl_wy != 0.0:
            L, c, s = _orig_geom(coords, fr.i, fr.j)
            feq_global = _gravity_feq_global(fr.udl_wy, c, s, L)
            add(fr.i, feq_global[0:3])
            add(fr.j, feq_global[3:6])
    for cb in step.new_cables:
        if cb.tension != 0.0:
            _, c, s = _orig_geom(coords, cb.i, cb.j)
            add(cb.i, np.array([cb.tension * c, cb.tension * s, 0.0]))
            add(cb.j, np.array([-cb.tension * c, -cb.tension * s, 0.0]))
    for nl in step.nodal_loads:
        add(nl.node, np.array([nl.fx, nl.fy, nl.mz], dtype=float))
    return dF


def _assert_same_structure(p0: StagedPlan, pk: StagedPlan) -> None:
    """校验两个施工计划结构一致(仅 ``NewCable.tension`` 与 ``NodalLoad`` 可不同)。

    批量多右端复用同一刚度,要求几何/单元/支座/平衡自由度/记录标志逐项相同。
    """
    if p0.supports != pk.supports:
        raise ValueError("batched plans differ in supports")
    nodes0 = [(n.id, n.x, n.y, n.attach) for n in p0.init_nodes]
    nodesk = [(n.id, n.x, n.y, n.attach) for n in pk.init_nodes]
    if nodes0 != nodesk:
        raise ValueError("batched plans differ in init_nodes")
    if len(p0.steps) != len(pk.steps):
        raise ValueError("batched plans differ in step count")
    for s0, sk in zip(p0.steps, pk.steps):
        if s0.label != sk.label or s0.record != sk.record:
            raise ValueError(f"batched plans differ in step meta near {s0.label!r}")
        if [(n.id, n.x, n.y, n.attach) for n in s0.new_nodes] != [
            (n.id, n.x, n.y, n.attach) for n in sk.new_nodes
        ]:
            raise ValueError(f"batched plans differ in new_nodes at step {s0.label!r}")
        if [(f.id, f.i, f.j, f.E, f.A, f.I, f.udl_wy) for f in s0.new_frames] != [
            (f.id, f.i, f.j, f.E, f.A, f.I, f.udl_wy) for f in sk.new_frames
        ]:
            raise ValueError(f"batched plans differ in new_frames at step {s0.label!r}")
        if [(c.id, c.i, c.j, c.E, c.A) for c in s0.new_cables] != [
            (c.id, c.i, c.j, c.E, c.A) for c in sk.new_cables
        ]:
            raise ValueError(f"batched plans differ in new_cables structure at step {s0.label!r}")
        if [(b.node, b.dof, b.target) for b in s0.balance_dofs] != [
            (b.node, b.dof, b.target) for b in sk.balance_dofs
        ]:
            raise ValueError(f"batched plans differ in balance_dofs at step {s0.label!r}")
        if s0.new_supports != sk.new_supports:
            raise ValueError(f"batched plans differ in new_supports at step {s0.label!r}")


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
        for nid, ux, uy, rz in step.new_supports:
            # 切线激活后"就地固结":锁定当前位移(后续增量为 0),不清零。
            old = self.fixed.get(nid, (False, False, False))
            self.fixed[nid] = (old[0] or ux, old[1] or uy, old[2] or rz)

    def _orig_geom(self, i: int, j: int) -> tuple[float, float, float]:
        """单元原始坐标下的长度与方向余弦(线性小位移参考)。"""
        return _orig_geom(self.coords, i, j)

    def _incremental_loads(self, step: BuildStep) -> dict[int, np.ndarray]:
        """本步新增的增量荷载(仅本步引入的自重 UDL + 索预张力等效节点力)。"""
        return _step_incremental_loads(step, self.coords)

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
        K = _assemble_global_stiffness(self.coords, self.frames, self.cables, idx, ndof)
        F = np.zeros(ndof)

        def dofs(nid):
            b = 3 * idx[nid]
            return [b, b + 1, b + 2]

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


class StagedDirectBatchSolver:
    """同结构、多右端的批量逐阶段直接刚度求解器(去除冗余刚度分解)。

    与 :class:`StagedDirectSolver` 执行**同一**施工状态机,但同时承载 ``K`` 个仅在
    ``NewCable.tension`` / ``NodalLoad`` 上不同的载荷工况:每个施工阶段的整体刚度只
    装配、``lu_factor`` 分解**一次**,``K`` 个右端一次性 ``lu_solve`` 回代。

    服务 :func:`bridgezoo.optim.linear.build_affine_model` 的「T=0 + 逐索单位扰动」
    多右端构造;由于直接刚度法对载荷线性、刚度与预张力无关,本路径与逐个
    :meth:`StagedDirectSolver.run` 的结果在机器精度内逐位一致(见 ``tests/test_staged_batch.py``)。
    """

    name = "direct"

    def run_batch(self, plans: list[StagedPlan]) -> list[StagedResult]:
        if not plans:
            raise ValueError("run_batch requires at least one plan")
        ref = plans[0]
        for p in plans[1:]:
            _assert_same_structure(ref, p)
        ncase = len(plans)

        self.coords: dict[int, tuple[float, float]] = {}
        self.U: dict[int, np.ndarray] = {}            # {nid: (3, K)}
        self.fixed: dict[int, tuple[bool, bool, bool]] = {}
        self.frames: list[dict] = []                  # [{"o": NewFrame}]
        self.cables: list[dict] = []                  # [{"o", "birth_i":(3,K), "birth_j":(3,K), "N0":(K,)}]

        for sp in ref.supports:
            self.fixed[sp[0]] = (sp[1], sp[2], sp[3])
        for nd in ref.init_nodes:
            self._add_node(nd, ncase)

        results = [StagedResult(backend=self.name) for _ in range(ncase)]
        for si, step in enumerate(ref.steps):
            self._activate(step, plans, si, ncase)
            dF_cases = [_step_incremental_loads(p.steps[si], self.coords) for p in plans]
            applied = [StagedDirectSolver._explicit_nodal_loads(p.steps[si]) for p in plans]
            self._solve_increment_batch(dF_cases, step.balance_dofs, applied, ncase)
            if step.record:
                for k in range(ncase):
                    results[k].records.append(self._record_case(step.label, k, applied[k]))
        for k in range(ncase):
            _attach_geometry(results[k], plans[k])
        return results

    # --------------------------------------------------------------
    def _add_node(self, nd: NewNode, ncase: int) -> None:
        self.coords[nd.id] = (nd.x, nd.y)
        if nd.attach is not None:
            xI, yI = self.coords[nd.attach]
            uI = self.U[nd.attach]            # (3, K)
            dx, dy = nd.x - xI, nd.y - yI
            rzI = uI[2]                        # (K,)
            u_new = np.empty((3, ncase))
            u_new[0] = uI[0] - dy * rzI
            u_new[1] = uI[1] + dx * rzI
            u_new[2] = rzI
            self.U[nd.id] = u_new
        else:
            self.U[nd.id] = np.zeros((3, ncase))
        fx = self.fixed.get(nd.id)
        if fx is not None:
            for c, is_fixed in enumerate(fx):
                if is_fixed:
                    self.U[nd.id][c, :] = 0.0

    def _activate(self, step: BuildStep, plans: list[StagedPlan], si: int, ncase: int) -> None:
        for nd in step.new_nodes:
            self._add_node(nd, ncase)
        for fr in step.new_frames:
            # 框架诞生位移仅在标量路径记录而不参与任何输出,此处无需追踪。
            self.frames.append(dict(o=fr))
        for ci, cb in enumerate(step.new_cables):
            birth_i = self.U[cb.i].copy()     # (3, K)
            birth_j = self.U[cb.j].copy()
            n0 = np.array(
                [plans[k].steps[si].new_cables[ci].tension for k in range(ncase)], dtype=float
            )
            self.cables.append(dict(o=cb, birth_i=birth_i, birth_j=birth_j, N0=n0))
        for nid, ux, uy, rz in step.new_supports:
            old = self.fixed.get(nid, (False, False, False))
            self.fixed[nid] = (old[0] or ux, old[1] or uy, old[2] or rz)

    def _solve_increment_batch(
        self,
        dF_cases: list[dict[int, np.ndarray]],
        balance_dofs: list[BalanceDof] | None,
        applied: list[dict[int, np.ndarray]],
        ncase: int,
    ) -> None:
        from scipy.linalg import lu_factor, lu_solve

        active = list(self.coords.keys())
        idx = {nid: k for k, nid in enumerate(active)}
        ndof = 3 * len(active)
        Kmat = _assemble_global_stiffness(self.coords, self.frames, self.cables, idx, ndof)

        def dofs(nid):
            b = 3 * idx[nid]
            return [b, b + 1, b + 2]

        F = np.zeros((ndof, ncase))
        for k in range(ncase):
            for nid, vec in dF_cases[k].items():
                d = dofs(nid)
                F[d[0], k] += vec[0]
                F[d[1], k] += vec[1]
                F[d[2], k] += vec[2]

        fixed = np.zeros(ndof, dtype=bool)
        for nid, fx in self.fixed.items():
            if nid in idx:
                d = dofs(nid)
                for c in range(3):
                    if fx[c]:
                        fixed[d[c]] = True
        for p in range(ndof):
            if not fixed[p] and abs(Kmat[p, p]) < 1e-30 and np.allclose(Kmat[p, :], 0.0):
                fixed[p] = True

        free = np.where(~fixed)[0]
        du = np.zeros((ndof, ncase))
        if free.size > 0:
            Kff = Kmat[np.ix_(free, free)]
            Ff = F[free, :]                              # (nf, K)
            lu = lu_factor(Kff)                          # 每步只分解一次,K 个右端共享
            if balance_dofs:
                current = np.zeros((ndof, ncase))
                for nid in active:
                    current[dofs(nid), :] = self.U[nid]
                control = []
                target = []
                for bd in balance_dofs:
                    gdof = dofs(bd.node)[bd.dof]
                    if fixed[gdof]:
                        raise ValueError(f"balance dof is already fixed: node={bd.node}, dof={bd.dof}")
                    control.append(int(np.where(free == gdof)[0][0]))
                    target.append(float(bd.target))
                base_du = lu_solve(lu, Ff)               # (nf, K)
                unit = np.zeros((free.size, len(control)))
                for j, pos in enumerate(control):
                    unit[pos, j] = 1.0
                flex_cols = lu_solve(lu, unit)           # (nf, ncontrol),跨工况共享
                flex = flex_cols[control, :]             # (ncontrol, ncontrol)
                rhs = np.array(target)[:, None] - current[free[control], :] - base_du[control, :]
                balance_load = np.linalg.solve(flex, rhs)  # (ncontrol, K)
                Ff = Ff + unit @ balance_load              # (nf, K)
                for j, bd in enumerate(balance_dofs):
                    for k in range(ncase):
                        applied[k].setdefault(bd.node, np.zeros(3))
                        applied[k][bd.node][bd.dof] += float(balance_load[j, k])
            du[free, :] = lu_solve(lu, Ff)

        for nid in active:
            self.U[nid] = self.U[nid] + du[dofs(nid), :]

    def _record_case(self, label: str, k: int, applied_k: dict[int, np.ndarray]) -> StagedStepRecord:
        disp = {
            nid: (float(self.U[nid][0, k]), float(self.U[nid][1, k]), float(self.U[nid][2, k]))
            for nid in self.coords
        }
        cforce, cstress = {}, {}
        for cb in self.cables:
            o = cb["o"]
            L, c, s = _orig_geom(self.coords, o.i, o.j)
            dui = self.U[o.i][:, k] - cb["birth_i"][:, k]
            duj = self.U[o.j][:, k] - cb["birth_j"][:, k]
            elong = c * (duj[0] - dui[0]) + s * (duj[1] - dui[1])
            N = cb["N0"][k] + o.E * o.A / L * elong
            cforce[o.id] = float(N)
            cstress[o.id] = float(N / o.A)
        loads = {nid: tuple(float(x) for x in vec) for nid, vec in applied_k.items()}
        return StagedStepRecord(label=label, disp=disp, cable_force=cforce, cable_stress=cstress, applied_loads=loads)
