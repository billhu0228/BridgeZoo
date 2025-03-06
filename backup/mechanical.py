import subprocess
from typing import Union

import ezdxf
import os

from ansys.mapdl.core.mapdl_console import MapdlConsole
from ansys.mapdl.core.mapdl_grpc import MapdlGrpc


class FEM2:
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

    def ansys(self, apdl: Union[MapdlGrpc, "MapdlConsole"]):
        with apdl.non_interactive:
            W = self.materials[1]['W']
            E = self.materials[1]['E']
            H = self.materials[1]['H']
            apdl.clear()
            apdl.finish()
            apdl.prep7()
            apdl.mp('ex', 1, E)
            apdl.mp('ex', 2, 1.95e11)
            apdl.et(1, '188')
            apdl.et(2, '180')
            apdl.sectype(1, 'beam', 'rect')
            apdl.secdata(W, H)
            for nd in self.nodes:
                apdl.n(nd['id'], nd['x'], nd['y'], 0)
                if nd['id'] > 1000:
                    apdl.d(nd['id'], 'all')
            apdl.d(1, 'uy')
            apdl.d(self.cab_per_side * 2 + 5, 'uy')  # 桥台
            apdl.d(self.cab_per_side // 2 + 2, 'uy')  # 索塔
            apdl.d(self.cab_per_side // 2 * 3 + 4, 'uy')  # 索塔
            apdl.d(self.cab_per_side + 3, 'ux')  # 跨中
            apdl.d(self.cab_per_side + 3, 'rotz')
            eid = 1
            for ed in self.elements:
                if ed['type'] == 1:
                    apdl.secnum(1)
                    apdl.mat(1)
                    apdl.type(1)
                    apdl.e(ed['node1'], ed['node2'])
                    eid += 1
                    apdl.sfbeam(ed['id'], '2', 'PRES', self.Wg)
                else:
                    n1 = next(n for n in self.nodes if n['id'] == ed['node1'])
                    mat = self.materials[n1['id'] % 1000 + 1000]
                    apdl.sectype(ed['id'], 'link')
                    apdl.secdata(0.00014 * mat['Ns'])
                    apdl.seccontrol(0, 1)
                    apdl.secnum(ed['id'])
                    apdl.mat(2)
                    apdl.type(2)
                    apdl.e(ed['node1'], ed['node2'])
                    apdl.inistate('DEFINE', eid, 'all', 'all', 'all', float(mat['sigma'] * 1e6))
                    eid += 1
            apdl.allsel()
            apdl.d("all", "uz")
            apdl.d("all", "rotx")
            apdl.d("all", "roty")
            apdl.solution()
            apdl.allsel()
            apdl.antype(0)
            apdl.nsubst(1)
            apdl.time(1)
            apdl.solve()
            apdl.finish()
            apdl.post1()
            apdl.set(1, 1)
        ff = apdl.prnsol('u', 'y')
        gg = apdl.presol('smisc', '1')
        n_res = [x[1] for x in ff.to_list() if x[0] < 1000]
        e_res = []
        e_res_dict = {}
        for e, f in gg.to_list():
            e_res_dict[int(e)] = f
        # print(e_res_dict)
        beam_max_n = self.cab_per_side * 2 + 5
        cable_left_last = beam_max_n + (self.cab_per_side // 2 - 1) * 4 + 1
        for i in range(self.cab_per_side // 2):
            e_res.append(e_res_dict[beam_max_n + i * 4])
        for i in range(self.cab_per_side // 2):
            e_res.append(e_res_dict[cable_left_last - i * 4])
        return n_res, e_res

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
