"""基于 OpenSeesPy 的"一次成桥"参考求解器。

本模块从历史代码 ``bridgezoo/cablebridge/fem.py`` 迁移而来，作为新自写线性求解器
(:mod:`bridgezoo.fem.linear_frame`) 的**正确性基准**：

- 它一次性建立全桥模型（全部梁单元 + 全部拉索初应力），做一次静力分析；
- **不**进入 RL 训练回路（每步重建模型开销过大）；
- 仅用于 ``scripts/validate_fem.py`` 中与线性求解器对比节点位移与索应力。

边界条件、单元/材料编号约定与历史几何模型保持一致，请勿随意改动，否则会破坏
与历史结果的可比性。

参见 ``docs/DESIGN_MAPPO.md`` 第 3.3、4.1 节。
"""

import os
import sys
from contextlib import contextmanager

import ezdxf
from openseespy.opensees import *  # noqa: F401,F403  (OpenSees 全局函数式 API)


@contextmanager
def suppress_openseespy_output(suppress=True):
    """临时屏蔽 OpenSeesPy 写到 stdout/stderr 的求解日志。"""
    if not suppress:
        yield
    else:
        with open(os.devnull, "w") as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                sys.stdout = original_stdout
                sys.stderr = original_stderr


class FEM:
    """一次成桥的 2D 斜拉桥有限元参考模型。

    Parameters
    ----------
    cables_per_side : int
        单侧索对数。
    Wg : float
        梁单元线均布荷载 (N/m)。
    tensions : Sequence[float]
        各索初应力 (MPa)，用于 ``InitStressMaterial``。
    """

    def __init__(self, cables_per_side: int, Wg: float, tensions):
        self.nodes = []  # 节点坐标信息
        self.elements = []  # 单元连接节点信息
        self.element_types = {}  # 单元类型信息
        self.materials = {}  # 单元的材料信息
        self.cab_per_side = cables_per_side  # 单侧索对
        self.Wg = Wg  # 梁单元线荷载 N/m
        self.tensions = tensions

    def add_node(self, node_id, x, y):
        self.nodes.append({"id": node_id, "x": x, "y": y})

    def add_element(self, element_id, node1_id, node2_id, element_type, material_id):
        self.elements.append(
            {
                "id": element_id,
                "node1": node1_id,
                "node2": node2_id,
                "type": element_type,
                "material": material_id,
            }
        )

    def add_element_type(self, type_id, description):
        self.element_types[type_id] = description

    def add_material(self, material_id, properties):
        self.materials[material_id] = properties

    def generate_dxf(self, filename):
        """导出模型几何为 DXF，便于人工核对节点/单元布置。"""
        doc = ezdxf.new(dxfversion="R2010")
        msp = doc.modelspace()
        color_map = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6}
        for i in range(100):
            color_map[1001 + i] = 100 + i
        for n in self.nodes:
            msp.add_circle(center=(n["x"], n["y"]), radius=0.1, dxfattribs={"color": 7})
        for e in self.elements:
            node1 = next(n for n in self.nodes if n["id"] == e["node1"])
            node2 = next(n for n in self.nodes if n["id"] == e["node2"])
            color = color_map.get(e["material"], 1)
            msp.add_line(
                start=(node1["x"], node1["y"]),
                end=(node2["x"], node2["y"]),
                dxfattribs={"color": color},
            )
        doc.saveas(filename)

    def opensees(self):
        """运行一次静力分析，返回 (梁节点竖向位移列表, 索应力列表[MPa])。

        求解失败时返回全零列表（与历史行为一致）。
        """
        with suppress_openseespy_output(True):
            wipe()
            model("basic", "-ndm", 2, "-ndf", 3)
            for nd in self.nodes:
                node(nd["id"], nd["x"], nd["y"])
                if nd["id"] > 1000:
                    fix(nd["id"], 1, 1, 1)
            fix(1, 0, 1, 0)  # 桥台
            fix(self.cab_per_side // 2 + 2, 0, 1, 0)  # 索塔
            fix(self.cab_per_side + 3, 1, 0, 1)  # 跨中
            fix(self.cab_per_side // 2 * 3 + 4, 0, 1, 0)  # 索塔
            fix(self.cab_per_side * 2 + 5, 0, 1, 0)  # 桥台
            geomTransf("Linear", 1)
            A = self.materials[1]["A"]
            E = self.materials[1]["E"]
            Iz = self.materials[1]["I"]
            # 定义拉索属性
            Es = 1.95e11  # 弹性模量，单位：Pa (N/m²)
            As = 0.00014
            uniaxialMaterial("Elastic", 2, Es)
            for ed in self.elements:
                if ed["type"] == 1:
                    element("elasticBeamColumn", ed["id"], ed["node1"], ed["node2"], A, E, Iz, 1)
                else:
                    n1 = next(n for n in self.nodes if n["id"] == ed["node1"])
                    mat = self.materials[n1["id"] % 1000 + 1000]
                    uniaxialMaterial("InitStressMaterial", ed["id"], 2, float(mat["sigma"] * 1e6))
                    area = As * mat["Ns"]  # 截面面积，单位：m²
                    element("corotTruss", ed["id"], ed["node1"], ed["node2"], area, ed["id"])
            # 设置均布荷载
            timeSeries("Linear", 1)
            pattern("Plain", 1, 1)
            for i in range(1, self.cab_per_side * 2 + 4):
                eleLoad("-ele", i, "-type", "-beamUniform", -self.Wg)
            system("ProfileSPD")
            constraints("Plain")
            numberer("RCM")
            test("NormUnbalance", 1.0e-6, 100, 2)
            steps = 100
            integrator("LoadControl", 1.0 / steps)
            algorithm("Newton")
            analysis("Static")
            ret = analyze(steps)
            res = []
            e_res = []
            for nd in self.nodes:
                if nd["id"] < 1000:
                    res.append(nodeDisp(nd["id"])[1])
            for i in range(self.cab_per_side // 2):
                eid = 1001 + i
                fx, fy = eleForce(eid)[0:2]
                mat = self.materials[eid]
                sig = (fx ** 2 + fy ** 2) ** 0.5 / (mat["Ns"] * As)
                e_res.append(sig * 1e-6)
            for i in range(self.cab_per_side // 2):
                eid = 2000 + self.cab_per_side // 2 - i
                fx, fy = eleForce(eid)[0:2]
                mat = self.materials[eid]
                sig = (fx ** 2 + fy ** 2) ** 0.5 / (mat["Ns"] * As)
                e_res.append(sig * 1e-6)

            wipe()
            if ret != 0:
                return [0 for _ in res], [0 for _ in res]
            return res, e_res
