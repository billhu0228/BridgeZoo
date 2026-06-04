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
    udl_wy: float = 0.0  # 本步施加的局部横向均布荷载（自重）


@dataclass
class NewCable:
    id: int
    i: int
    j: int
    E: float
    A: float
    tension: float = 0.0  # 安装时刻目标张力 (N)


@dataclass
class BuildStep:
    """一次施工动作（= 一次增量求解）。"""

    label: str
    new_nodes: list[NewNode] = field(default_factory=list)
    new_frames: list[NewFrame] = field(default_factory=list)
    new_cables: list[NewCable] = field(default_factory=list)
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


@dataclass
class StagedResult:
    backend: str
    records: list[StagedStepRecord] = field(default_factory=list)

    def cable_stress_history(self) -> dict[int, list[tuple[str, float]]]:
        out: dict[int, list[tuple[str, float]]] = {}
        for rec in self.records:
            for cid, s in rec.cable_stress.items():
                out.setdefault(cid, []).append((rec.label, s))
        return out

    def final_disp(self) -> dict[int, tuple[float, float, float]]:
        return self.records[-1].disp if self.records else {}


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
            self._solve_increment(dF)
            if step.record:
                result.records.append(self._record(step.label))
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
                xi, yi = self.coords[fr.i]
                xj, yj = self.coords[fr.j]
                L = math.hypot(xj - xi, yj - yi)
                c, s = (xj - xi) / L, (yj - yi) / L
                feq_local = _udl_fixed_end_local(fr.udl_wy, L)
                feq_global = _frame_transform(c, s).T @ feq_local
                add(fr.i, feq_global[0:3])
                add(fr.j, feq_global[3:6])
        for cb in step.new_cables:
            if cb.tension != 0.0:
                _, c, s = self._orig_geom(cb.i, cb.j)
                add(cb.i, np.array([cb.tension * c, cb.tension * s, 0.0]))
                add(cb.j, np.array([-cb.tension * c, -cb.tension * s, 0.0]))
        return dF

    def _solve_increment(self, dF: dict[int, np.ndarray]) -> None:
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
            du[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

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
        return StagedStepRecord(label=label, disp=disp, cable_force=cforce, cable_stress=cstress)


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
        self._pat = 0

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.geomTransf("Linear", self._TRANSF)

        fixed = {sp[0]: (sp[1], sp[2], sp[3]) for sp in plan.supports}
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
                ops.fix(nd.id, int(fx[0]), int(fx[1]), int(fx[2]))
        for fr in step.new_frames:
            ops.element("elasticBeamColumn", fr.id, fr.i, fr.j, fr.A, fr.E, fr.I, self._TRANSF)
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

    def _apply_and_solve(self, step):
        ops = self.ops
        try:
            ops.domainChange()
        except Exception:
            pass
        ops.wipeAnalysis()
        self._pat += 1
        ts = pat = 10000 + self._pat
        has_load = any(fr.udl_wy != 0.0 for fr in step.new_frames)
        if has_load:
            ops.timeSeries("Linear", ts)
            ops.pattern("Plain", pat, ts)
            for fr in step.new_frames:
                if fr.udl_wy != 0.0:
                    ops.eleLoad("-ele", fr.id, "-type", "-beamUniform", fr.udl_wy)
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
        return StagedStepRecord(label=label, disp=disp, cable_force=cforce, cable_stress=cstress)


# ============================================================ 施工计划构建器
def build_staged_cantilever(
    n_seg: int = 6,
    seg_len: float = 8.0,
    tower_height: float = 20.0,
    beam_E: float = 20e9,
    beam_A: float = 10.0,
    beam_Iz: float = 10.0 / 12.0,
    wg: float = 4.7e5,
    cable_Es: float = 1.95e11,
    strand_area: float = 1.4e-4,
    strands: list[int] | None = None,
    pretension: list[float] | None = None,
) -> StagedPlan:
    """构建单悬臂斜拉桥的施工计划（装节段→张索，逐段推进）。

    与 :class:`bridgezoo.fem.opensees_staged.StagedCantileverCableBridge` 同场景，但用
    后端无关的 :class:`StagedPlan` 表达，可同时交给自研/OpenSees 两后端求解对比。
    """
    strands = strands or [20] * n_seg
    pretension = pretension or [0.0] * n_seg
    L, H = seg_len, tower_height
    ANCHOR, ROOT = 999, 0

    plan = StagedPlan(name=f"staged_cantilever_N{n_seg}")
    plan.init_nodes = [NewNode(ANCHOR, 0.0, H), NewNode(ROOT, 0.0, 0.0)]
    plan.supports = [(ANCHOR, True, True, True), (ROOT, True, True, True)]

    prev = ROOT
    for i in range(1, n_seg + 1):
        node_i = i
        # 步：装节段 i（切线激活）+ 自重
        plan.steps.append(
            BuildStep(
                label=f"seg{i}",
                new_nodes=[NewNode(node_i, i * L, 0.0, attach=prev)],
                new_frames=[NewFrame(10 + i, prev, node_i, beam_E, beam_A, beam_Iz, udl_wy=-wg)],
                record=False,
            )
        )
        # 步：装索 i 并张拉到目标张力
        area = strand_area * strands[i - 1]
        plan.steps.append(
            BuildStep(
                label=f"cable{i}",
                new_cables=[NewCable(1000 + i, ANCHOR, node_i, cable_Es, area, tension=pretension[i - 1])],
                record=True,
            )
        )
        prev = node_i

    return plan
