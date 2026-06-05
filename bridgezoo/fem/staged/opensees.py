"""OpenSees 逐阶段后端(setNodeDisp 切线激活)—— 校核自研增量求解器。

执行与 :mod:`bridgezoo.fem.staged.direct` **同一个** :class:`StagedPlan`,返回**同一种**
:class:`StagedResult`,逐项可比。openseespy 惰性导入(仅 :meth:`run` 时需要),其编译
DLL 对 Python 版本敏感(建议 3.11–3.13)。
"""

from __future__ import annotations

import math

import numpy as np

from bridgezoo.fem.kernels import _gravity_feq_global
from bridgezoo.fem.staged.plan import (
    StagedPlan,
    StagedResult,
    StagedStepRecord,
    _attach_geometry,
)


class StagedOpenSeesSolver:
    """OpenSees 逐阶段后端(setNodeDisp 切线激活),执行同一 StagedPlan。

    拉索单元可切换(``cable_element``):

    - ``"linear"``(默认,研究初期):普通线性 ``Truss``。OpenSees 中新建的 Truss 以
      **创建时刻的变形构型**为零应变参考,故预张力直接 ``σ0 = T/A``(无需扣几何应变)。
      此时与自研线性直接刚度法**逐项吻合到机器精度**,便于校核自研求解器。
    - ``"corot"``(后续生产):几何精确 ``corotTruss`` + ``InitStrain``。它以**原始节点
      坐标**为参考,故预张力用 ``initStrain = T/(E·A) − ε_geo``。与线性自研解在小位移
      下吻合,大挠度下体现几何非线性差异。

    两种模式都用 setNodeDisp 切线激活实现梁节段零应力诞生。
    """

    name = "opensees"
    _TRANSF = 1

    def __init__(self, cable_element: str = "linear"):
        if cable_element not in ("linear", "corot"):
            raise ValueError("cable_element 必须是 'linear' 或 'corot'")
        self.cable_element = cable_element

    def run(self, plan: StagedPlan) -> StagedResult:
        import openseespy.opensees as ops

        self.ops = ops
        self.coords: dict[int, tuple[float, float]] = {}
        self._areas: dict[int, float] = {}
        self._cable_ids: list[int] = []
        self.frames: list = []
        self.cables: list = []
        self._pat = 0

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.geomTransf("Linear", self._TRANSF)

        fixed = {sp[0]: (sp[1], sp[2], sp[3]) for sp in plan.supports}
        self.fixed = fixed
        for nd in plan.init_nodes:
            ops.node(nd.id, nd.x, nd.y)
            self.coords[nd.id] = (nd.x, nd.y)
            if nd.id in fixed:
                fx = fixed[nd.id]
                ops.fix(nd.id, int(fx[0]), int(fx[1]), int(fx[2]))

        result = StagedResult(backend=self.name)
        for step in plan.steps:
            self._activate(step, fixed)
            self._apply_and_solve(step)
            if step.record:
                result.records.append(self._record(step.label))
        _attach_geometry(result, plan)
        return result

    def _activate(self, step, fixed):
        ops = self.ops
        for nd in step.new_nodes:
            ops.node(nd.id, nd.x, nd.y)
            self.coords[nd.id] = (nd.x, nd.y)
            if nd.attach is not None:
                xI, yI = self.coords[nd.attach]
                uxI, uyI, rzI = (ops.nodeDisp(nd.attach, 1), ops.nodeDisp(nd.attach, 2), ops.nodeDisp(nd.attach, 3))
                dx, dy = nd.x - xI, nd.y - yI
                ops.setNodeDisp(nd.id, 1, uxI - dy * rzI, "-commit")
                ops.setNodeDisp(nd.id, 2, uyI + dx * rzI, "-commit")
                ops.setNodeDisp(nd.id, 3, rzI, "-commit")
            if nd.id in fixed:
                fx = fixed[nd.id]
                for k, is_fixed in enumerate(fx, start=1):
                    if is_fixed:
                        ops.setNodeDisp(nd.id, k, 0.0, "-commit")
                ops.fix(nd.id, int(fx[0]), int(fx[1]), int(fx[2]))
        for fr in step.new_frames:
            ops.element("elasticBeamColumn", fr.id, fr.i, fr.j, fr.A, fr.E, fr.I, self._TRANSF)
            self.frames.append(fr)
        for cb in step.new_cables:
            emat, imat = 600000 + cb.id, 700000 + cb.id
            ops.uniaxialMaterial("Elastic", emat, cb.E)
            if self.cable_element == "linear":
                # 线性 Truss:以创建时刻变形构型为零应变参考 → σ0 = T/A(无需扣几何应变)。
                # 与自研线性直接刚度法逐项一致(机器精度)。
                sigma0 = cb.tension / cb.A
                ops.uniaxialMaterial("InitStressMaterial", imat, emat, float(sigma0))
                ops.element("Truss", cb.id, cb.i, cb.j, cb.A, imat)
            else:
                # corotTruss:以原始坐标为参考 → initStrain = T/(EA) − ε_geo。
                xi, yi = self.coords[cb.i]
                xj, yj = self.coords[cb.j]
                L0 = math.hypot(xj - xi, yj - yi)
                uxi, uyi = ops.nodeDisp(cb.i, 1), ops.nodeDisp(cb.i, 2)
                uxj, uyj = ops.nodeDisp(cb.j, 1), ops.nodeDisp(cb.j, 2)
                Lcur = math.hypot((xj + uxj) - (xi + uxi), (yj + uyj) - (yi + uyi))
                eps_geo = (Lcur - L0) / L0
                init_strain = cb.tension / (cb.E * cb.A) - eps_geo
                ops.uniaxialMaterial("InitStrainMaterial", imat, emat, float(init_strain))
                ops.element("corotTruss", cb.id, cb.i, cb.j, cb.A, imat)
            self._areas[cb.id] = cb.A
            self._cable_ids.append(cb.id)
            self.cables.append(cb)

    @staticmethod
    def _explicit_nodal_loads(step) -> dict[int, np.ndarray]:
        loads: dict[int, np.ndarray] = {}
        for nl in step.nodal_loads:
            loads.setdefault(nl.node, np.zeros(3))
            loads[nl.node] += np.array([nl.fx, nl.fy, nl.mz], dtype=float)
        return loads

    def _step_loads(self, step) -> dict[int, np.ndarray]:
        dF: dict[int, np.ndarray] = {}

        def add(nid, vec):
            dF.setdefault(nid, np.zeros(3))
            dF[nid] += vec

        for fr in step.new_frames:
            if fr.udl_wy != 0.0:
                xi, yi = self.coords[fr.i]
                xj, yj = self.coords[fr.j]
                L = math.hypot(xj - xi, yj - yi)
                c, s = (xj - xi) / L, (yj - yi) / L
                feq_global = _gravity_feq_global(fr.udl_wy, c, s, L)
                add(fr.i, feq_global[0:3])
                add(fr.j, feq_global[3:6])
        for nl in step.nodal_loads:
            add(nl.node, np.array([nl.fx, nl.fy, nl.mz], dtype=float))
        return dF

    def _balance_loads(self, step, dF: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        if not step.balance_dofs:
            return {}

        ops = self.ops
        ops.wipeAnalysis()
        ops.system("FullGeneral")
        ops.numberer("Plain")
        ops.constraints("Transformation")
        ops.integrator("LoadControl", 0.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        ok = ops.analyze(1)
        if ok != 0:
            raise RuntimeError(f"failed to form OpenSees tangent for balance step: {step.label}")

        neq = int(ops.systemSize())
        raw = ops.printA("-ret")
        Kff = np.asarray(raw, dtype=float)
        if Kff.size != neq * neq:
            raise RuntimeError(f"OpenSees printA returned {Kff.size} entries for system size {neq}")
        Kff = Kff.reshape((neq, neq))
        Ff = np.zeros(neq)

        def eq_dof(nid: int, dof: int) -> int:
            eqs = ops.nodeDOFs(nid)
            eq = int(eqs[dof])
            if eq < 0:
                raise ValueError(f"balance dof is constrained or inactive: node={nid}, dof={dof}")
            return eq

        for nid, vec in dF.items():
            eqs = ops.nodeDOFs(nid)
            for dof, value in enumerate(vec):
                eq = int(eqs[dof])
                if eq >= 0:
                    Ff[eq] += float(value)

        control = []
        target = []
        for bd in step.balance_dofs:
            control.append(eq_dof(bd.node, bd.dof))
            target.append(float(bd.target))

        base_du = np.linalg.solve(Kff, Ff)
        unit = np.zeros((neq, len(control)))
        for j, pos in enumerate(control):
            unit[pos, j] = 1.0
        flex_cols = np.linalg.solve(Kff, unit)
        flex = flex_cols[control, :]
        current = np.array([ops.nodeDisp(bd.node, bd.dof + 1) for bd in step.balance_dofs], dtype=float)
        rhs = np.array(target) - current - base_du[control]
        loads = np.linalg.solve(flex, rhs)

        out: dict[int, np.ndarray] = {}
        for bd, value in zip(step.balance_dofs, loads):
            out.setdefault(bd.node, np.zeros(3))
            out[bd.node][bd.dof] += value
        return out

    def _apply_and_solve(self, step):
        ops = self.ops
        try:
            ops.domainChange()
        except Exception:
            pass
        ops.wipeAnalysis()
        self._pat += 1
        ts = pat = 10000 + self._pat
        dF = self._step_loads(step)
        balance_loads = self._balance_loads(step, dF)
        self._last_applied_loads = self._explicit_nodal_loads(step)
        for nid, vec in balance_loads.items():
            self._last_applied_loads.setdefault(nid, np.zeros(3))
            self._last_applied_loads[nid] += vec
        ops.wipeAnalysis()
        has_load = any(fr.udl_wy != 0.0 for fr in step.new_frames) or bool(step.nodal_loads) or bool(balance_loads)
        if has_load:
            ops.timeSeries("Linear", ts)
            ops.pattern("Plain", pat, ts)
            for fr in step.new_frames:
                if fr.udl_wy != 0.0:
                    # 全局重力 → 局部 (Wy=gy*c, Wx=gy*s),使反向梁自重方向也正确
                    xi, yi = self.coords[fr.i]
                    xj, yj = self.coords[fr.j]
                    Lf = math.hypot(xj - xi, yj - yi)
                    c, s = (xj - xi) / Lf, (yj - yi) / Lf
                    ops.eleLoad("-ele", fr.id, "-type", "-beamUniform", fr.udl_wy * c, fr.udl_wy * s)
            for nl in step.nodal_loads:
                ops.load(nl.node, nl.fx, nl.fy, nl.mz)
            for nid, vec in balance_loads.items():
                ops.load(nid, float(vec[0]), float(vec[1]), float(vec[2]))
        ops.system("BandGeneral")
        ops.numberer("Plain")
        ops.constraints("Transformation")
        ops.integrator("LoadControl", 1.0)
        ops.algorithm("Linear")
        ops.analysis("Static")
        ops.analyze(1)
        ops.loadConst("-time", 0.0)

    def _record(self, label):
        ops = self.ops
        disp = {nid: (ops.nodeDisp(nid, 1), ops.nodeDisp(nid, 2), ops.nodeDisp(nid, 3)) for nid in self.coords}
        cforce, cstress = {}, {}
        for cid in self._cable_ids:
            N = 0.0
            for resp in ("axialForce", "basicForce", "force"):
                try:
                    r = ops.eleResponse(cid, resp)
                except Exception:
                    r = None
                if r:
                    N = float(r[0]) if isinstance(r, (list, tuple)) else float(r)
                    break
            cforce[cid] = N
            cstress[cid] = N / self._areas[cid]
        loads = {nid: tuple(float(x) for x in vec) for nid, vec in getattr(self, "_last_applied_loads", {}).items()}
        return StagedStepRecord(label=label, disp=disp, cable_force=cforce, cable_stress=cstress, applied_loads=loads)
