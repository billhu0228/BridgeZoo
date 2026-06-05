"""一次成桥（完成态）OpenSees 求解后端：消费与求解器无关的 :class:`StructuralModel`。

与 :mod:`bridgezoo.fem.completed.direct`（自研直接刚度法）接口一致、结果应一致，用于
**交叉校核自研求解器**。为保证可比性，本后端刻意采用与自研求解器相同的**线性、
小位移**力学假设：

- 梁：``elasticBeamColumn`` + ``geomTransf Linear``（线性、Euler-Bernoulli）。
- 索：线性 ``Truss`` + ``InitStressMaterial``（初应力 σ0 = pretension/A），而非
  ``corotTruss``——避免几何非线性带来的差异，确保与线性直接刚度法逐项对齐。
- 均布荷载：``eleLoad -beamUniform Wy``（局部横向），与直接刚度法一致等效节点荷载对应。

> 注意：施工阶段分析（几何非线性、切线激活）请用 :mod:`bridgezoo.fem.staged`，
> 那里用 corotTruss。本后端专注于"成桥/单阶段线性"工况的对照。

需要 openseespy（惰性导入；其 DLL 对 Python 版本敏感，建议 3.11–3.13）。
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager

from bridgezoo.fem.model import SolveResult, StructuralModel

_TRANSF = 1
_CABLE_EMAT_OFFSET = 600000   # 索弹性材料 tag 偏移
_CABLE_IMAT_OFFSET = 700000   # 索初应力材料 tag 偏移


@contextmanager
def _suppress(suppress=True):
    if not suppress:
        yield
        return
    with open(os.devnull, "w") as devnull:
        o, e = sys.stdout, sys.stderr
        try:
            sys.stdout = devnull
            sys.stderr = devnull
            yield
        finally:
            sys.stdout, sys.stderr = o, e


class CompletedOpenSeesSolver:
    """OpenSees 线性静力求解后端。"""

    name = "opensees"

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def solve(self, model: StructuralModel) -> SolveResult:
        import openseespy.opensees as ops  # 惰性导入

        with _suppress(not self.verbose):
            ops.wipe()
            ops.model("basic", "-ndm", 2, "-ndf", 3)

            for n in model.nodes.values():
                ops.node(n.id, n.x, n.y)

            for sp in model.supports.values():
                ops.fix(sp.node, int(sp.ux), int(sp.uy), int(sp.rz))

            ops.geomTransf("Linear", _TRANSF)

            for m in model.frames.values():
                ops.element("elasticBeamColumn", m.id, m.i, m.j, m.A, m.E, m.I, _TRANSF)

            for cab in model.cables.values():
                sigma0 = cab.pretension / cab.A  # 初应力 (Pa)
                emat = _CABLE_EMAT_OFFSET + cab.id
                imat = _CABLE_IMAT_OFFSET + cab.id
                ops.uniaxialMaterial("Elastic", emat, cab.E)
                ops.uniaxialMaterial("InitStressMaterial", imat, emat, float(sigma0))
                ops.element("Truss", cab.id, cab.i, cab.j, cab.A, imat)

            # 荷载
            ops.timeSeries("Constant", 1)
            ops.pattern("Plain", 1, 1)
            for nl in model.nodal_loads:
                ops.load(nl.node, nl.fx, nl.fy, nl.mz)
            for udl in model.member_udls.values():
                ops.eleLoad("-ele", udl.member, "-type", "-beamUniform", udl.wy)

            # 线性静力分析
            ops.system("BandGeneral")
            ops.numberer("Plain")
            ops.constraints("Transformation")
            ops.integrator("LoadControl", 1.0)
            ops.algorithm("Linear")
            ops.analysis("Static")
            ok = ops.analyze(1)

            result = SolveResult(backend=self.name, converged=(ok == 0))
            for n in model.nodes.values():
                d = ops.nodeDisp(n.id)
                result.disp[n.id] = (d[0], d[1], d[2])
            for m in model.frames.values():
                f = ops.eleResponse(m.id, "localForce")
                result.frame_force[m.id] = tuple(float(v) for v in f)
            for cab in model.cables.values():
                N = self._truss_axial(ops, cab.id)
                result.cable_force[cab.id] = N
                result.cable_stress[cab.id] = N / cab.A

            ops.wipe()
            return result

    @staticmethod
    def _truss_axial(ops, tag: int) -> float:
        """稳健读取 Truss 轴力 (N, +拉)。"""
        for resp in ("axialForce", "basicForce", "force"):
            try:
                r = ops.eleResponse(tag, resp)
            except Exception:
                r = None
            if r:
                return float(r[0]) if isinstance(r, (list, tuple)) else float(r)
        return 0.0


def solve(model: StructuralModel, verbose: bool = False) -> SolveResult:
    """便捷函数：用 OpenSees 求解。"""
    return CompletedOpenSeesSolver(verbose=verbose).solve(model)
