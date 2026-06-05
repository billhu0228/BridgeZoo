"""逐阶段施工（变刚度 + 切线激活）—— 后端无关的施工计划 + 两种求解后端。

延续"一套定义，两种后端，结果一致"的思路，但对象是**施工过程**：

- :class:`StagedPlan` —— 与求解器无关的施工计划：一串 :class:`BuildStep`，每个 step 是
  一次"激活构件 + 施加增量荷载 + 求解增量"的施工动作（装节段 / 张索）。
- :class:`StagedDirectSolver` —— 自研直接刚度法的**增量变刚度**求解器（RL 内核）。
- :class:`StagedOpenSeesSolver` —— OpenSees 线性后端（Truss + 切线激活），用于校核。

两个后端执行**同一个** StagedPlan，返回**同一种** :class:`StagedResult`，逐项可比。

切线激活（零应力诞生）
----------------------
新节点按附着节点的**变形后切线**安放（小转角线性）::

    ux_J = ux_I - dy * rz_I,   uy_J = uy_I + dx * rz_I,   rz_J = rz_I

其中 (dx, dy) 为新单元的设计矢量、I 为附着节点。这样新梁单元安装即零应力；之后各
增量步测得的相对变形才使其受力（位移锁定）。索按"安装时刻达到目标张力 T"处理，随
后续施工重分布而变化——即索力历程。

自研直接刚度法为**线性、小位移**（RL 内核，求快）；OpenSees 后端用 **corotTruss 几何精确**
（与 :mod:`bridgezoo.fem.opensees_staged` 同一稳健做法）。二者在小位移下逐项吻合
（成桥调索的目标即把位移压小，stage-1 误差 ~0.02%）；在刻意放大的大挠度算例下因几何
非线性约有 1% 量级差异——这正好量化了线性近似的适用范围。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from bridgezoo.fem.linear_frame import _frame_local_stiffness, _frame_transform, _udl_fixed_end_local


# ============================================================ 施工计划（IR）
@dataclass
class NewNode:
    id: int
    x: float
    y: float
    attach: int | None = None  # 切线附着节点；None 表示按零位移就位（如根部/锚点）


@dataclass
class NewFrame:
    id: int
    i: int
    j: int
    E: float
    A: float
    I: float
    udl_wy: float = 0.0  # 本步施加的**全局竖向**均布荷载（自重，向下为负；按单元方向投影到局部）


@dataclass
class NewCable:
    id: int
    i: int
    j: int
    E: float
    A: float
    tension: float = 0.0  # 安装时刻目标张力 (N)


@dataclass
class NodalLoad:
    node: int
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0


@dataclass
class BalanceDof:
    node: int
    dof: int  # 0=ux, 1=uy, 2=rz
    target: float = 0.0


@dataclass
class BuildStep:
    """一次施工动作（= 一次增量求解）。"""

    label: str
    new_nodes: list[NewNode] = field(default_factory=list)
    new_frames: list[NewFrame] = field(default_factory=list)
    new_cables: list[NewCable] = field(default_factory=list)
    nodal_loads: list[NodalLoad] = field(default_factory=list)
    balance_dofs: list[BalanceDof] = field(default_factory=list)
    record: bool = True


@dataclass
class StagedPlan:
    name: str
    init_nodes: list[NewNode] = field(default_factory=list)        # 阶段0已存在的节点
    supports: list[tuple[int, bool, bool, bool]] = field(default_factory=list)
    steps: list[BuildStep] = field(default_factory=list)


# ============================================================ 结果
@dataclass
class StagedStepRecord:
    label: str
    disp: dict[int, tuple[float, float, float]]
    cable_force: dict[int, float]
    cable_stress: dict[int, float]
    applied_loads: dict[int, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class StagedResult:
    backend: str
    records: list[StagedStepRecord] = field(default_factory=list)
    # 静态几何信息（供可视化/后处理使用，免去外部按 id 反推坐标）
    coords: dict[int, tuple[float, float]] = field(default_factory=dict)   # {node_id: (x, y)}
    cable_nodes: dict[int, tuple[int, int]] = field(default_factory=dict)  # {cable_id: (i, j)}
    anchor_ids: list[int] = field(default_factory=list)                    # 塔上锚点 node id
    deck_ids: list[int] = field(default_factory=list)                      # 主梁 node id（含根部）

    def cable_stress_history(self) -> dict[int, list[tuple[str, float]]]:
        out: dict[int, list[tuple[str, float]]] = {}
        for rec in self.records:
            for cid, s in rec.cable_stress.items():
                out.setdefault(cid, []).append((rec.label, s))
        return out

    def final_disp(self) -> dict[int, tuple[float, float, float]]:
        return self.records[-1].disp if self.records else {}


def _gravity_feq_global(gy: float, c: float, s: float, L: float) -> np.ndarray:
    """**全局竖向**重力线荷载 gy（向下为负）的一致等效节点荷载（全局 6 向量）。

    重力 (0, gy) 投影到单元局部：横向 Wy=gy*c、轴向 Wx=gy*s（局部 x 沿单元方向 (c,s)，
    局部 y 为 (-s, c)）；再按一致固端力转回全局。水平梁 (s=0) 时退化为 Wy=gy*c。
    解决了"局部横向荷载"在反向（-x）梁上方向翻转、导致左右自重不对称的问题。
    """
    q_t = gy * c   # 局部横向
    q_a = gy * s   # 局部轴向
    feq_local = np.array([
        q_a * L / 2.0, q_t * L / 2.0, q_t * L * L / 12.0,
        q_a * L / 2.0, q_t * L / 2.0, -q_t * L * L / 12.0,
    ])
    return _frame_transform(c, s).T @ feq_local


def _attach_geometry(result: StagedResult, plan: StagedPlan) -> None:
    """把施工计划的静态几何（节点坐标、索连接、锚点/梁节点分类）写入结果。"""
    nodes: list[NewNode] = list(plan.init_nodes)
    for step in plan.steps:
        nodes.extend(step.new_nodes)
    for nd in nodes:
        result.coords[nd.id] = (nd.x, nd.y)
        if nd.y > 1e-9:
            result.anchor_ids.append(nd.id)   # 塔上锚点（y>0）
        else:
            result.deck_ids.append(nd.id)     # 主梁节点（y≈0，含根部）
    for step in plan.steps:
        for cb in step.new_cables:
            result.cable_nodes[cb.id] = (cb.i, cb.j)


# ============================================================ 自研直接刚度法（增量）
class StagedDirectSolver:
    """自研二维直接刚度法的逐阶段（增量、变刚度）求解器。"""

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
        """单元原始坐标下的长度与方向余弦（线性小位移参考）。"""
        xi, yi = self.coords[i]
        xj, yj = self.coords[j]
        dx, dy = xj - xi, yj - yi
        L = math.hypot(dx, dy)
        return L, dx / L, dy / L

    def _incremental_loads(self, step: BuildStep) -> dict[int, np.ndarray]:
        """本步新增的增量荷载（仅本步引入的自重 UDL + 索预张力等效节点力）。"""
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


# ============================================================ OpenSees 后端（线性，校核用）
class StagedOpenSeesSolver:
    """OpenSees 逐阶段后端（setNodeDisp 切线激活），执行同一 StagedPlan。

    拉索单元可切换（``cable_element``）：

    - ``"linear"``（默认，研究初期）：普通线性 ``Truss``。OpenSees 中新建的 Truss 以
      **创建时刻的变形构型**为零应变参考，故预张力直接 ``σ0 = T/A``（无需扣几何应变）。
      此时与自研线性直接刚度法**逐项吻合到机器精度**，便于校核自研求解器。
    - ``"corot"``（后续生产）：几何精确 ``corotTruss`` + ``InitStrain``。它以**原始节点
      坐标**为参考，故预张力用 ``initStrain = T/(E·A) − ε_geo``。与线性自研解在小位移
      下吻合，大挠度下体现几何非线性差异。

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
        self.frames: list[NewFrame] = []
        self.cables: list[NewCable] = []
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
                # 线性 Truss：以创建时刻变形构型为零应变参考 → σ0 = T/A（无需扣几何应变）。
                # 与自研线性直接刚度法逐项一致（机器精度）。
                sigma0 = cb.tension / cb.A
                ops.uniaxialMaterial("InitStressMaterial", imat, emat, float(sigma0))
                ops.element("Truss", cb.id, cb.i, cb.j, cb.A, imat)
            else:
                # corotTruss：以原始坐标为参考 → initStrain = T/(EA) − ε_geo。
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
                    # 全局重力 → 局部 (Wy=gy*c, Wx=gy*s)，使反向梁自重方向也正确
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


# ============================================================ 施工计划构建器
# ---- 节点 / 单元编号约定（n = 每侧索数；要求 n < 90）----
#   根部(0#)节点         : 0
#   塔上锚点 i (1..n)     : 300 + i            （y = anchor_base + (i-1)*anchor_spacing）
#   右侧索点 i (1..n)     : i                  （x = +(right_start + (i-1)*right_spacing)）
#   右侧自由端(终止段端)  : 200
#   左侧索点 i (1..n)     : 100 + i            （x = −(left_start + (i-1)*left_spacing)）
#   左侧自由端            : 201
#   右侧梁单元 i          : 10 + i  ；右终止段梁: 90
#   左侧梁单元 i          : 110 + i ；左终止段梁: 190
#   右侧索 i              : 1000 + i ；左侧索 i : 2000 + i

_ROOT = 0


def _anchor_id(i: int) -> int:
    return 300 + i


def build_staged_cantilever(
    n_seg: int = 6,
    # —— 塔上锚点（扇面）——
    anchor_base_height: float = 20.0,   # 参数 a：最低锚点高度（自梁面 y=0 起算）
    anchor_spacing: float = 3.0,        # 参数 b：相邻锚点竖向间距（向上）
    anchor_top_free: float = 5.0,       # 参数 c：最高锚点之上的自由高度（仅供绘塔参考）
    # —— 主梁双悬臂（左右可不同；以塔 x=0 为参考）——
    left_start: float = 6.0,            # 参数1(左)：塔到第 1 索点距离
    left_spacing: float = 8.0,          # 参数2(左)：相邻索点间距
    left_end: float = 4.0,              # 参数3(左)：末索点到悬臂自由端距离（无索）
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
    strands: list[int] | None = None,       # 长度 n，左右同索号共用
    pretension: list[float] | None = None,  # 长度 n，左右同索号共用
) -> StagedPlan:
    """构建**对称双悬臂 + 扇面索**斜拉桥的施工计划。

    - 塔上锚点呈扇面：第 i 索锚在高度 ``a + (i-1)*b``（内侧低、外侧高），顶部留自由高 c。
    - 主梁自塔向两侧成对悬臂；左右各 n 个索点，间距等几何可不同（start/spacing/end）。
    - 索点：第 i 索点距塔 ``start + (i-1)*spacing``；``start`` 为塔到首索点的无索引段，
      ``end`` 为末索点到自由端的无索终止段。
    - 施工步序（平衡悬臂）：每阶段同时装两侧第 i 节段（切线激活）+ 同时张两侧第 i 对索；
      最后装两侧终止段（无索）。塔为刚性（锚点固定），根部 0# 块固接（x=0 全固定）。

    strands / pretension 长度均为 n，左右同索号共用（如需左右各异，后续可扩展为按侧传入）。
    """
    strands = strands or [20] * n_seg
    pretension = pretension or [0.0] * n_seg
    assert len(strands) == n_seg and len(pretension) == n_seg, "strands/pretension 长度须 = n_seg"
    assert n_seg < 90, "当前编号方案要求 n_seg < 90"

    plan = StagedPlan(name=f"staged_half_bridge_N{n_seg}")

    # 初始：根部 0# 块 + 塔上 n 个锚点（均固定）
    plan.init_nodes = [NewNode(_ROOT, 0.0, 0.0)]
    # Tower-deck joint: keep translations coupled, release deck rotation.
    plan.supports = [(_ROOT, True, True, False)]
    for i in range(1, n_seg + 1):
        hy = anchor_base_height + (i - 1) * anchor_spacing
        plan.init_nodes.append(NewNode(_anchor_id(i), 0.0, hy))
        plan.supports.append((_anchor_id(i), True, True, True))

    # 各侧索点 x 坐标（左为负）与单元 id 偏移
    def side_x(i: int, start: float, spacing: float, sign: int) -> float:
        return sign * (start + (i - 1) * spacing)

    sides = [
        dict(sign=+1, start=right_start, spacing=right_spacing, end=right_end,
             node=lambda i: i, tip=200, frame=lambda i: 10 + i, tip_frame=90, cable=lambda i: 1000 + i),
        dict(sign=-1, start=left_start, spacing=left_spacing, end=left_end,
             node=lambda i: 100 + i, tip=201, frame=lambda i: 110 + i, tip_frame=190, cable=lambda i: 2000 + i),
    ]
    prev = {+1: _ROOT, -1: _ROOT}
    # 逐节段：每阶段同时装两侧第 i 节段，再同时张两侧第 i 对索
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

    # 终止段（两侧自由端，无索）
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
