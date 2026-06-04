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

# 说明：openseespy 为可选重依赖（且其编译 DLL 对 Python 版本敏感）。这里**不在模块顶层
# 导入**，而是在真正求解时（FEM.opensees 内）惰性导入 ``import openseespy.opensees as ops``，
# 这样在没有/无法加载 openseespy 的环境里仍可 import 本模块、组装模型（build_oneshot_fem）。


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
        import openseespy.opensees as ops  # 惰性导入：仅求解时需要 openseespy

        with suppress_openseespy_output(True):
            ops.wipe()
            ops.model("basic", "-ndm", 2, "-ndf", 3)
            for nd in self.nodes:
                ops.node(nd["id"], nd["x"], nd["y"])
                if nd["id"] > 1000:
                    ops.fix(nd["id"], 1, 1, 1)
            ops.fix(1, 0, 1, 0)  # 桥台
            ops.fix(self.cab_per_side // 2 + 2, 0, 1, 0)  # 索塔
            ops.fix(self.cab_per_side + 3, 1, 0, 1)  # 跨中
            ops.fix(self.cab_per_side // 2 * 3 + 4, 0, 1, 0)  # 索塔
            ops.fix(self.cab_per_side * 2 + 5, 0, 1, 0)  # 桥台
            ops.geomTransf("Linear", 1)
            A = self.materials[1]["A"]
            E = self.materials[1]["E"]
            Iz = self.materials[1]["I"]
            # 定义拉索属性
            Es = 1.95e11  # 弹性模量，单位：Pa (N/m²)
            As = 0.00014
            ops.uniaxialMaterial("Elastic", 2, Es)
            for ed in self.elements:
                if ed["type"] == 1:
                    ops.element("elasticBeamColumn", ed["id"], ed["node1"], ed["node2"], A, E, Iz, 1)
                else:
                    n1 = next(n for n in self.nodes if n["id"] == ed["node1"])
                    mat = self.materials[n1["id"] % 1000 + 1000]
                    ops.uniaxialMaterial("InitStressMaterial", ed["id"], 2, float(mat["sigma"] * 1e6))
                    area = As * mat["Ns"]  # 截面面积，单位：m²
                    ops.element("corotTruss", ed["id"], ed["node1"], ed["node2"], area, ed["id"])
            # 设置均布荷载
            ops.timeSeries("Linear", 1)
            ops.pattern("Plain", 1, 1)
            for i in range(1, self.cab_per_side * 2 + 4):
                ops.eleLoad("-ele", i, "-type", "-beamUniform", -self.Wg)
            ops.system("ProfileSPD")
            ops.constraints("Plain")
            ops.numberer("RCM")
            ops.test("NormUnbalance", 1.0e-6, 100, 2)
            steps = 100
            ops.integrator("LoadControl", 1.0 / steps)
            ops.algorithm("Newton")
            ops.analysis("Static")
            ret = ops.analyze(steps)
            res = []
            e_res = []
            for nd in self.nodes:
                if nd["id"] < 1000:
                    res.append(ops.nodeDisp(nd["id"])[1])
            for i in range(self.cab_per_side // 2):
                eid = 1001 + i
                fx, fy = ops.eleForce(eid)[0:2]
                mat = self.materials[eid]
                sig = (fx ** 2 + fy ** 2) ** 0.5 / (mat["Ns"] * As)
                e_res.append(sig * 1e-6)
            for i in range(self.cab_per_side // 2):
                eid = 2000 + self.cab_per_side // 2 - i
                fx, fy = ops.eleForce(eid)[0:2]
                mat = self.materials[eid]
                sig = (fx ** 2 + fy ** 2) ** 0.5 / (mat["Ns"] * As)
                e_res.append(sig * 1e-6)

            ops.wipe()
            if ret != 0:
                return [0 for _ in res], [0 for _ in res]
            return res, e_res


def build_oneshot_fem(geometry, cable_sigma, cable_sizes):
    """由 :class:`bridgezoo.envs.geometry.BridgeGeometry` 组装一次成桥 :class:`FEM`。

    节点/单元/材料编号约定与历史 ``extract_fem_model`` 完全一致（见 geometry.py 文档），
    以保证与历史结果及自写求解器可交叉校核。

    Parameters
    ----------
    geometry : BridgeGeometry
        桥梁几何（提供 x_positions、塔顶锚点、截面/材料、自重 wg 等）。
    cable_sigma : Sequence[float]
        长度 N（=num_cables_per_side）的各索初应力 (MPa)。
    cable_sizes : Sequence[float|int]
        长度 N 的各索股数。

    Returns
    -------
    FEM
    """
    N = geometry.num_cables_per_side
    nbp = geometry.num_beam_points  # = N + 3

    fem = FEM(N, geometry.wg, cable_sigma)

    # 梁节点（id 1..2N+5，按 x 升序）与梁单元（id 1..2N+4）
    for i, x in enumerate(geometry.x_positions):
        fem.add_node(i + 1, float(x), 0.0)
    for i in range(len(geometry.x_positions) - 1):
        fem.add_element(i + 1, i + 1, i + 2, 1, 1)  # 梁单元，类型1，材料1

    # 索塔锚点 + 拉索单元（左塔 1001+i / 2001+i，右塔 3001+i / 4001+i）
    for i in range(N // 2):
        lx, ly = float(geometry.left_tower_pts[i, 0]), float(geometry.left_tower_pts[i, 1])
        rx, ry = float(geometry.right_tower_pts[i, 0]), float(geometry.right_tower_pts[i, 1])
        fem.add_node(1001 + i, lx, ly)
        fem.add_node(3001 + i, rx, ry)

        beam_index_left = i + 2
        beam_index_right = nbp + 1
        fem.add_element(1001 + i, 1001 + i, beam_index_left, 2, 1001 + i)
        fem.add_element(2001 + i, 1001 + i, nbp - 1 - i, 2, 1001 + i)
        fem.add_element(3001 + i, 3001 + i, beam_index_right + i, 2, 1001 + i)
        fem.add_element(4001 + i, 3001 + i, nbp * 2 - 2 - i, 2, 1001 + i)

        fem.add_material(1001 + i, {"Ns": cable_sizes[i], "sigma": cable_sigma[i]})
        fem.add_material(2001 + i, {"Ns": cable_sizes[N - 1 - i], "sigma": cable_sigma[N - 1 - i]})
        fem.add_material(3001 + i, {"Ns": cable_sizes[i], "sigma": cable_sigma[i]})
        fem.add_material(4001 + i, {"Ns": cable_sizes[N - 1 - i], "sigma": cable_sigma[N - 1 - i]})

    fem.add_element_type(1, "Beam")
    fem.add_element_type(2, "Cable")
    fem.add_material(
        1,
        {
            "E": geometry.beam_E,
            "A": geometry.beam_area,
            "I": geometry.beam_Iz,
            "W": geometry.beam_w,
            "H": geometry.beam_h,
        },
    )
    return fem
