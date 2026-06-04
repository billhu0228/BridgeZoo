"""正向逐阶段施工（悬臂拼装 + 张索）OpenSees 模型——切线激活方案。

参考 ``docs/施工阶段验证/verify_cons_ops.py`` 的"切线激活（tangent activation）"做法，
在**持久化模型**（两次 analyze 之间不 wipe）中实现单元的零应力诞生：

- **梁节段切线激活**：新节段的远端节点先建在设计坐标，再用 ``setNodeDisp(..., '-commit')``
  把它的**初始位移**设到上一悬臂端的"切线延长位置"::

      uy_install = uy_tip + rz_tip * L_seg      # 沿切线竖向延伸
      rz_install = rz_tip                        # 转角随端截面

  这样新梁单元两端处于同一刚体构型 → 安装即零应力（小位移近似）。
- **拉索安装/张拉**：索用 ``corotTruss`` + ``InitStrainMaterial``。初应变 ``initStrain``
  既能让索"零应力安装"（``initStrain = -ε_几何``），也能直接给定**目标张力**::

      initStrain = T_target / (E_s * A) - ε_几何   # 安装即带张力 T_target

  （本 OpenSees 约定为 σ = E·(ε + initStrain)，与参考脚本一致。）
- **顺序加载**：每阶段新建 pattern 施加新增荷载，``analyze`` 后用 ``loadConst('-time',0)``
  冻结，再进入下一阶段；``wipeAnalysis`` + 重设分析对象。

简化假设（v1，可行性研究用，后续可放宽）：
1. 单塔、**单悬臂臂**（固定根部的悬臂），代表"装节段→张索"的基本循环；对称双臂/双塔
   是直接的扩展（镜像 + 平衡）。
2. **索塔刚性**：拉索锚固在固定的塔顶点（忽略塔变形）。
3. **小位移**：切线激活为一阶近似，残余 ghost force 为二阶小量。
4. 自重为梁单元均布荷载；恒载一次性随节段就位。

> 本模块仅在调用 :meth:`StagedCantileverCableBridge.run` 时需要 openseespy；
> 其编译 DLL 对 Python 版本敏感（实测 3.14 无法加载，请用 3.11–3.13）。

参见 ``docs/DESIGN_MAPPO.md`` 第 3 节、``docs/施工阶段验证/verify_cons_ops.py``。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

try:  # 惰性容错：无 openseespy 时仍可 import 本模块
    import openseespy.opensees as ops
except Exception:  # pragma: no cover
    ops = None


_TRANSF = 1  # geomTransf tag


def _require_ops():
    if ops is None:
        raise ImportError(
            "需要 openseespy 才能运行分阶段分析。请在受支持的解释器（3.11–3.13）下"
            "`pip install openseespy`。当前环境未能加载 openseespy。"
        )


def _setup_analysis():
    """每阶段统一的（线性静力）分析设置。"""
    ops.system("BandGeneral")
    ops.numberer("Plain")
    ops.constraints("Plain")
    ops.integrator("LoadControl", 1.0)
    ops.algorithm("Linear")
    ops.analysis("Static")


# ---- 单元/材料/节点编号约定 ----
def _beam_tag(i: int) -> int:
    return 10 + i           # 梁单元：11, 12, ...


def _cable_tag(i: int) -> int:
    return 1000 + i         # 索单元：1001, 1002, ...


def _cable_emat(i: int) -> int:
    return 2000 + i         # 索弹性材料


def _cable_imat(i: int) -> int:
    return 3000 + i         # 索 InitStrain 材料


_ANCHOR = 999              # 固定塔顶锚点节点
_ROOT = 0                  # 悬臂固定根部节点（塔处 0# 段）


@dataclass
class StageRecord:
    """单个施工阶段记录。"""

    stage: int                                   # 阶段序号（= 刚安装的节段/索）
    deck_deflection: dict[int, float]            # {梁节点id: 竖向位移 m}（活动节点）
    cable_force: dict[int, float]                # {索序号 i: 轴力 N}
    cable_stress: dict[int, float]               # {索序号 i: 应力 MPa}


@dataclass
class StagedCantileverCableBridge:
    """单悬臂斜拉桥的正向逐阶段施工（切线激活）模型。

    Parameters
    ----------
    n_seg : int
        悬臂节段数（= 拉索数）。
    seg_len : float
        每个节段长度 (m)。
    tower_height : float
        塔顶锚点相对梁面高度 (m)。
    beam_E, beam_A, beam_Iz : float
        主梁弹性模量 (Pa)、截面积 (m²)、惯矩 (m⁴)。
    wg : float
        主梁自重线荷载 (N/m)。
    cable_Es, strand_area : float
        拉索弹性模量 (Pa)、单股面积 (m²)。
    """

    n_seg: int = 6
    seg_len: float = 8.0
    tower_height: float = 20.0
    beam_E: float = 20e9
    beam_A: float = 10.0
    beam_Iz: float = 10.0 * 1.0 ** 3 / 12.0
    wg: float = 4.7e5
    cable_Es: float = 1.95e11
    strand_area: float = 1.4e-4

    history: list[StageRecord] = field(default_factory=list, init=False)

    # ------------------------------------------------------------------ API
    def run(self, cable_strands: list[int], cable_pretension: list[float]) -> list[StageRecord]:
        """执行完整的"装节段→张索"逐阶段分析。

        Parameters
        ----------
        cable_strands : list[int]
            各索股数，长度 = n_seg。
        cable_pretension : list[float]
            各索**安装时的目标张力** (N)，长度 = n_seg。

        Returns
        -------
        list[StageRecord]
            逐阶段记录（也存于 ``self.history``）。
        """
        _require_ops()
        assert len(cable_strands) == self.n_seg, "cable_strands 长度须 = n_seg"
        assert len(cable_pretension) == self.n_seg, "cable_pretension 长度须 = n_seg"

        self.history = []
        L = self.seg_len
        H = self.tower_height

        ops.wipe()
        ops.model("basic", "-ndm", 2, "-ndf", 3)
        ops.geomTransf("Linear", _TRANSF)

        # 固定塔顶锚点（刚性塔简化）与悬臂固定根部
        ops.node(_ANCHOR, 0.0, H)
        ops.fix(_ANCHOR, 1, 1, 1)
        ops.node(_ROOT, 0.0, 0.0)
        ops.fix(_ROOT, 1, 1, 1)

        prev_tip = _ROOT
        uy_tip, rz_tip = 0.0, 0.0
        installed_deck_nodes: list[int] = []

        for i in range(1, self.n_seg + 1):
            # ====== 1) 安装第 i 节段（切线激活）======
            node_i = i
            xi = i * L
            ops.node(node_i, xi, 0.0)
            # 切线延长：新端点的初始位移设到上一端切线位置 → 新梁单元零应力诞生
            uy_install = uy_tip + rz_tip * L
            ops.setNodeDisp(node_i, 1, 0.0, "-commit")
            ops.setNodeDisp(node_i, 2, uy_install, "-commit")
            ops.setNodeDisp(node_i, 3, rz_tip, "-commit")
            ops.element(
                "elasticBeamColumn", _beam_tag(i), prev_tip, node_i,
                self.beam_A, self.beam_E, self.beam_Iz, _TRANSF,
            )
            self._safe_domain_change()

            # 施加该节段自重，求平衡
            ops.wipeAnalysis()
            ts = pat = _beam_tag(i)  # 复用唯一 tag
            ops.timeSeries("Linear", ts)
            ops.pattern("Plain", pat, ts)
            ops.eleLoad("-ele", _beam_tag(i), "-type", "-beamUniform", -self.wg)
            _setup_analysis()
            if ops.analyze(1) != 0:
                raise RuntimeError(f"阶段 {i}：装节段后求解失败")
            ops.loadConst("-time", 0.0)

            prev_tip = node_i
            uy_tip = ops.nodeDisp(node_i, 2)
            rz_tip = ops.nodeDisp(node_i, 3)
            installed_deck_nodes.append(node_i)

            # ====== 2) 安装并张拉第 i 对索（目标张力 cable_pretension[i-1]）======
            area = self.strand_area * cable_strands[i - 1]
            T = float(cable_pretension[i - 1])
            self._install_cable(i, node_i, xi, H, area, T)
            self._safe_domain_change()

            ops.wipeAnalysis()
            _setup_analysis()
            if ops.analyze(1) != 0:
                raise RuntimeError(f"阶段 {i}：张索后求解失败")
            ops.loadConst("-time", 0.0)

            # 张索后端部状态变化，更新供下一节段切线激活
            uy_tip = ops.nodeDisp(prev_tip, 2)
            rz_tip = ops.nodeDisp(prev_tip, 3)

            # ====== 记录本阶段所有活动索的内力/应力 + 线形 ======
            self.history.append(self._record_stage(i, installed_deck_nodes, cable_strands))

        return self.history

    # -------------------------------------------------------------- 内部
    def _install_cable(self, i: int, deck_node: int, xi: float, H: float, area: float, T: float) -> None:
        """安装第 i 索（corotTruss + InitStrain），使其安装即带目标张力 T。"""
        # 参考（无应力）长度由节点坐标决定
        L0 = math.hypot(xi - 0.0, 0.0 - H)
        # 当前几何长度（含锚点固定、梁端已提交位移）
        ux = ops.nodeDisp(deck_node, 1)
        uy = ops.nodeDisp(deck_node, 2)
        Lcur = math.hypot((xi + ux) - 0.0, (0.0 + uy) - H)
        eps_geo = (Lcur - L0) / L0
        # 约定 σ = E·(ε + initStrain) → 安装力 = E·A·(ε_geo + initStrain) = T
        init_strain = T / (area * self.cable_Es) - eps_geo

        ops.uniaxialMaterial("Elastic", _cable_emat(i), self.cable_Es)
        ops.uniaxialMaterial("InitStrainMaterial", _cable_imat(i), _cable_emat(i), init_strain)
        ops.element("corotTruss", _cable_tag(i), _ANCHOR, deck_node, area, _cable_imat(i))

    def _record_stage(self, stage: int, deck_nodes: list[int], strands: list[int]) -> StageRecord:
        defl = {nd: ops.nodeDisp(nd, 2) for nd in deck_nodes}
        force, stress = {}, {}
        for j in range(1, stage + 1):
            area = self.strand_area * strands[j - 1]
            N = self._cable_axial_force(_cable_tag(j))  # N
            force[j] = N
            stress[j] = (N / area) * 1e-6  # MPa
        return StageRecord(stage=stage, deck_deflection=defl, cable_force=force, cable_stress=stress)

    @staticmethod
    def _cable_axial_force(cable_tag: int) -> float:
        """稳健读取索（truss）轴力 (N)。

        不同 OpenSees 版本/单元对响应名支持不一，按优先级依次尝试：
        ``axialForce`` → ``basicForce`` → ``forces``（取首项）。供运行时调试参考。
        """
        for resp_type in ("axialForce", "basicForce", "forces"):
            try:
                resp = ops.eleResponse(cable_tag, resp_type)
            except Exception:
                resp = None
            if resp:
                return float(resp[0]) if isinstance(resp, (list, tuple)) else float(resp)
        return 0.0

    @staticmethod
    def _safe_domain_change() -> None:
        try:
            ops.domainChange()
        except Exception:
            pass

    # -------------------------------------------------------------- 后处理
    def cable_stress_history(self) -> dict[int, list[tuple[int, float]]]:
        """整理为每根索的应力历程。

        Returns
        -------
        dict[int, list[(stage, stress_MPa)]]
            键为索序号 i；值为该索自安装阶段起、各阶段的 (阶段号, 应力MPa)。
        """
        out: dict[int, list[tuple[int, float]]] = {}
        for rec in self.history:
            for i, s in rec.cable_stress.items():
                out.setdefault(i, []).append((rec.stage, s))
        return out
