import ezdxf
from openseespy.opensees import *
import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_openseespy_output():
    with open(os.devnull, 'w') as devnull:
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
    def __init__(self, cables_per_side: int, Wg: float, tensions):
        self.nodes = []  # 节点坐标信息
        self.elements = []  # 单元连接节点信息
        self.element_types = {}  # 单元类型信息
        self.materials = {}  # 单元的材料信息
        self.cab_per_side = cables_per_side  # 单侧索对
        self.Wg = Wg  # 梁单元线荷载 N/m
        self.tensions = tensions

    def add_node(self, node_id, x, y):
        self.nodes.append({'id': node_id, 'x': x, 'y': y})

    def add_element(self, element_id, node1_id, node2_id, element_type, material_id):
        self.elements.append({'id': element_id, 'node1': node1_id, 'node2': node2_id, 'type': element_type, 'material': material_id})

    def add_element_type(self, type_id, description):
        self.element_types[type_id] = description

    def add_material(self, material_id, properties):
        self.materials[material_id] = properties

    def generate_dxf(self, filename):
        """
        生成 DXF 文件
        :param filename: 要保存的 DXF 文件名
        """
        # 创建一个新的 DXF 文档
        doc = ezdxf.new(dxfversion='R2010')
        # 添加一个新的模型空间
        msp = doc.modelspace()
        # 定义颜色映射
        color_map = {
            1: 1,  # 红色
            2: 2,  # 黄色
            3: 3,  # 绿色
            4: 4,  # 青色
            5: 5,  # 蓝色
            6: 6,  # 紫色
        }
        for i in range(100):
            color_map[1001 + i] = 100 + i
        # 添加节点
        for n in self.nodes:
            msp.add_circle(center=(n['x'], n['y']), radius=0.1, dxfattribs={'color': 7})
        # 添加单元
        for e in self.elements:
            node1 = next(n for n in self.nodes if n['id'] == e['node1'])
            node2 = next(n for n in self.nodes if n['id'] == e['node2'])
            color = color_map.get(e['material'], 1)  # 默认颜色为红色
            msp.add_line(start=(node1['x'], node1['y']), end=(node2['x'], node2['y']), dxfattribs={'color': color})
        # 保存 DXF 文件
        doc.saveas(filename)

    def opensees(self):
        """
        计算
        """
        with suppress_openseespy_output():
            wipe()
            model('basic', '-ndm', 2, '-ndf', 3)
            for nd in self.nodes:
                node(nd['id'], nd['x'], nd['y'])
                if nd['id'] > 1000:
                    fix(nd['id'], 1, 1, 1)
            fix(1, 0, 1, 0)  # 桥台
            fix(self.cab_per_side // 2 + 2, 0, 1, 0)  # 索塔
            fix(self.cab_per_side + 3, 1, 0, 1)  # 跨中
            fix(self.cab_per_side // 2 * 3 + 4, 0, 1, 0)  # 索塔
            fix(self.cab_per_side * 2 + 5, 0, 1, 0)  # 桥台
            geomTransf('Linear', 1)
            A = self.materials[1]['A']
            E = self.materials[1]['E']
            Iz = self.materials[1]['I']
            # 定义拉索属性
            Es = 1.95e11  # 弹性模量，单位：Pa (N/m²)
            As = 0.00014
            uniaxialMaterial('Elastic', 2, Es)
            for ed in self.elements:
                if ed['type'] == 1:
                    element('elasticBeamColumn', ed['id'], ed['node1'], ed['node2'], A, E, Iz, 1)
                else:
                    # continue
                    n1 = next(n for n in self.nodes if n['id'] == ed['node1'])
                    mat = self.materials[n1['id'] % 1000 + 1000]
                    uniaxialMaterial('InitStressMaterial', ed['id'], 2, float(mat['sigma'] * 1e6))
                    area = As * mat['Ns']  # 截面面积，单位：m²
                    element('corotTruss', ed['id'], ed['node1'], ed['node2'], area, ed['id'])
            # 设置均布荷载
            timeSeries('Linear', 1)
            pattern('Plain', 1, 1)
            for i in range(1, self.cab_per_side * 2 + 4):
                eleLoad('-ele', i, '-type', '-beamUniform', -self.Wg)
            system('ProfileSPD')
            constraints('Plain')
            numberer('RCM')
            # test('NormDispIncr', 1.0e-3, 100, 0)
            test('NormUnbalance', 1.0e-6, 100)
            steps = 100
            integrator('LoadControl', 1.0 / steps)
            algorithm("Newton")
            # algorithm("Newton")
            analysis("Static")
            # printModel()
            analyze(steps)
            res = []
            e_res = []
            for nd in self.nodes:
                if nd['id'] < 1000:
                    res.append(nodeDisp(nd['id'])[1])
            for i in range(self.cab_per_side // 2):
                eid = 1001 + i
                fx, fy = eleForce(eid)[0:2]
                mat = self.materials[eid]
                sig = (fx ** 2 + fy ** 2) ** 0.5 / (mat['Ns'] * As)
                e_res.append(sig * 1e-6)
            for i in range(self.cab_per_side // 2):
                eid = 2000 + self.cab_per_side // 2 - i
                fx, fy = eleForce(eid)[0:2]
                mat = self.materials[eid]
                sig = (fx ** 2 + fy ** 2) ** 0.5 / (mat['Ns'] * As)
                e_res.append(sig * 1e-6)
            wipe()
            # print(len(res))
            return res, e_res
