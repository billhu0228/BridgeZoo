import numpy as np
from gymnasium import spaces
import ezdxf
from openseespy.opensees import *
import os
import sys
from contextlib import contextmanager


class Cable:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        # 索力限制
        self.min_stress = 300
        self.max_stress = 1000
        self.min_num_strands = 5
        self.max_num_strands = 100
        self.stress_init = 0
        self.num_strands = 0
        self.stress_after = np.nan
        self.deform = None
        self.action = (None, None)
        self._reward = None

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.float32(-10),
            high=np.float32(10),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        stress_act_space = spaces.Box(
            low=np.float32(0),
            high=np.float32(1),
            shape=(1,),
            dtype=np.float32,
        )
        strands_act_space = spaces.Discrete(int(self.max_num_strands - self.min_num_strands + 1))
        return spaces.Tuple([stress_act_space, strands_act_space])

    def reset(self):
        action = self.action_space.sample()
        self.stress_init = np.float32(self.min_stress + action[0][0] * (self.max_stress - self.min_stress))
        self.num_strands = self.min_num_strands + action[1]
        self.stress_after = np.nan
        self.action = action
        self.deform = None

    def step(self, norm_action):
        self.stress_init = np.float32(self.min_stress + norm_action[0] * (self.max_stress - self.min_stress))
        self.num_strands = self.min_num_strands + norm_action[1]
        return

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        self.deform = deform

    def done(self):
        return self.num_strands <= 0 or self.stress_init <= 0 or self.stress_init >= 2000

    def reward2(self):
        EPS = 0.01
        if abs(self.deform) <= EPS:
            s_score = {0: -1, 1: 2, 2: -1}
        else:
            if self.deform > 0:
                s_score = {0: 2, 1: -1, 2: -1}
            else:
                s_score = {0: -1, 1: -1, 2: 2}
        if self.stress_after <= 300:
            n_score = {0: 1, 1: 0, 2: -1}
        elif self.stress_after <= 800:
            n_score = {0: 0, 1: 1, 2: 0}
        else:
            n_score = {0: -2, 1: -1, 2: +2}
        self._reward = s_score[self.action[0]] + n_score[self.action[1]]
        return self._reward

    def reward3(self):
        self._reward = -abs(self.deform) * 100
        return self._reward


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
        with suppress_openseespy_output(False):
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
                    # uniaxialMaterial('InitStressMaterial', ed['id'], 2, float(mat['sigma'] * 1e6))
                    uniaxialMaterial('InitStrainMaterial', ed['id'], 2, float(mat['sigma']) / Es)
                    area = As * mat['Ns']  # 截面面积，单位：m²
                    element('corotTruss', ed['id'], ed['node1'], ed['node2'], area, ed['id'])
            # 设置均布荷载
            timeSeries('Linear', 1)
            pattern('Plain', 1, 1)
            for i in range(1, self.cab_per_side * 2 + 4):
                eleLoad('-ele', i, '-type', '-beamUniform', -self.Wg)
            system('BandSPD')
            constraints('Plain')
            numberer('Plain')
            # test('NormDispIncr', 1.0e-3, 100, 0)
            test('NormUnbalance', 1.0e-6, 100)
            integrator('LoadControl', 1)
            algorithm("Linear")
            # algorithm("Newton")

            analysis("Static")

            analyze(1)
            # printModel()
            res = []
            e_res = []
            for nd in self.nodes:
                if nd['id'] < 1000:
                    res.append(nodeDisp(nd['id'])[1] + nodeCoord(nd['id'])[1])
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


class FEM2:
    """
    第2版有限元：增加了对初始Y坐标调整的支持（变相支持无应力长度）
    """

    def __init__(self, cables_per_side: int, Wg: float, tensions, positions):
        self.nodes = []  # 节点坐标信息
        self.elements = []  # 单元连接节点信息
        self.element_types = {}  # 单元类型信息
        self.materials = {}  # 单元的材料信息
        self.cab_per_side = cables_per_side  # 单侧索对
        self.Wg = Wg  # 梁单元线荷载 N/m
        self.tensions = tensions
        self.positions = positions  # 拉索初始位置

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
        with suppress_openseespy_output(False):
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
                    # uniaxialMaterial('InitStressMaterial', ed['id'], 2, float(mat['sigma'] * 1e6))
                    uniaxialMaterial('InitStrainMaterial', ed['id'], 2, float(mat['sigma']) / Es)
                    area = As * mat['Ns']  # 截面面积，单位：m²
                    element('corotTruss', ed['id'], ed['node1'], ed['node2'], area, ed['id'])
            # 设置均布荷载
            timeSeries('Linear', 1)
            pattern('Plain', 1, 1)
            for i in range(1, self.cab_per_side * 2 + 4):
                eleLoad('-ele', i, '-type', '-beamUniform', -self.Wg)
            system('BandSPD')
            constraints('Plain')
            numberer('Plain')
            # test('NormDispIncr', 1.0e-3, 100, 0)
            test('NormUnbalance', 1.0e-6, 100)
            integrator('LoadControl', 1)
            algorithm("Linear")
            # algorithm("Newton")

            analysis("Static")

            analyze(1)
            # printModel()
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


class CableFixStrands:
    def __init__(self, num_strands, stress_init, stress_delta):
        # self.obs_dim = obs_dim
        self.stress_init = stress_init
        self.stress_exert = stress_init
        self.stress_delta = stress_delta
        self.stress_after = np.nan
        self.action_history = []
        self.num_strands = num_strands
        self.deform = None
        self.action = None
        self._reward = None

    def __str__(self):
        return "Cable(e:%i,a:%i)" % (self.stress_exert, self.stress_after)

    #    @property
    #    def observation_space(self):
    #        return spaces.Box(
    #            low=np.float32(-10),
    #            high=np.float32(10),
    #            shape=(self.obs_dim,),
    #            dtype=np.float32,
    #        )

    @property
    def action_space(self):
        stress_act_space = spaces.Discrete(3)
        return stress_act_space

    def reset(self):
        action = self.action_space.sample()
        self.stress_exert = self.stress_init
        self.stress_after = np.nan
        self.action = action
        self.deform = None

    def step(self, action):
        a = action - 1
        self.action_history.append(action)
        self.stress_exert = self.stress_after + a * self.stress_delta
        return

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        self.deform = deform

    def done(self):
        return False

    def reward2(self):
        EPS = 0.01
        if abs(self.deform) <= EPS:
            s_score = {0: -1, 1: 2, 2: -1}
        else:
            if self.deform > 0:
                s_score = {0: 2, 1: -1, 2: -1}
            else:
                s_score = {0: -1, 1: -1, 2: 2}
        if self.stress_after <= 300:
            n_score = {0: 1, 1: 0, 2: -1}
        elif self.stress_after <= 800:
            n_score = {0: 0, 1: 1, 2: 0}
        else:
            n_score = {0: -2, 1: -1, 2: +2}
        self._reward = s_score[self.action[0]] + n_score[self.action[1]]
        return self._reward

    def reward3(self):
        self._reward = -abs(self.deform) * 100
        return self._reward


class CableMoveY:
    def __init__(self, num_strands, stress_init, dy):
        # self.obs_dim = obs_dim
        self.stress_init = stress_init
        self.stress_exert = stress_init

        self.stress_after = np.nan
        self.action_history = []
        self.position_history = []
        self.num_strands = num_strands

        self.position_input = 0
        self.delta_y = dy
        self.position_after = None
        self.action = None
        self._reward = None

    def __str__(self):
        return "Cable(e:%i,a:%i)" % (self.stress_exert, self.stress_after)

    @property
    def action_space(self):
        stress_act_space = spaces.Discrete(3)
        return stress_act_space

    def reset(self):
        action = self.action_space.sample()
        self.stress_exert = self.stress_init
        self.stress_after = np.nan
        self.action = action
        self.position_input = 0
        self.position_after = None

    def step(self, action):
        a = action - 1
        self.action_history.append(action)
        self.position_input = self.position_input + a * self.delta_y
        return

    def update(self, balance_stress, balance_position):
        self.stress_after = balance_stress
        self.position_after = balance_position
        self.position_history.append(self.position_after)

    def done(self):
        return False

    def reward(self):
        return self.reward1

    @property
    def reward1(self):
        if (self.action_history[-1] - 1) * self.position_history[-2] < 0:
            self._reward = 1
        else:
            self._reward = 0
        if self.action_history[-1] == 1 and abs(self.position_history[-2]) * 1e3 < 30:
            self._reward += 0.5
        return self._reward

    @property
    def reward2(self):
        EPS = 0.01
        if abs(self.position_after) <= EPS:
            s_score = {0: -1, 1: 2, 2: -1}
        else:
            if self.position_after > 0:
                s_score = {0: 2, 1: -1, 2: -1}
            else:
                s_score = {0: -1, 1: -1, 2: 2}
        if self.stress_after <= 300:
            n_score = {0: 1, 1: 0, 2: -1}
        elif self.stress_after <= 800:
            n_score = {0: 0, 1: 1, 2: 0}
        else:
            n_score = {0: -2, 1: -1, 2: +2}
        self._reward = s_score[self.action[0]] + n_score[self.action[1]]
        return self._reward

    @property
    def reward3(self):
        self._reward = -abs(self.position_after) * 100
        return self._reward


@contextmanager
def suppress_openseespy_output(enable=True):
    """
    上下文管理器，用于屏蔽 OpenSeesPy 的输出（stdout 和 stderr）。
    参数:
        enable (bool): 如果为 True，则屏蔽输出；如果为 False，则不屏蔽输出。
    """
    if enable:
        with open(os.devnull, 'w') as devnull:
            original_stdout = sys.stdout
            original_stderr = sys.stderr
            try:
                # 将标准输出和错误重定向到 /dev/null
                sys.stdout = devnull
                sys.stderr = devnull
                yield
            finally:
                # 恢复原来的标准输出和错误
                sys.stdout = original_stdout
                sys.stderr = original_stderr
    else:
        # 如果不屏蔽，直接执行代码块
        yield


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
        计算一次平衡状态。
        2024-11-15：一次平衡状态并不准确
        """
        with suppress_openseespy_output(False):
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
            system('BandSPD')
            constraints('Plain')
            numberer('Plain')
            test('NormDispIncr', 1.0e-12, 10000, 0)
            # test('NormUnbalance', 1.0e-12, 100)
            # test('EnergyIncr', 1e-12, 1000)
            integrator('LoadControl', 1)
            algorithm("Linear")
            # algorithm("Newton")
            analysis("Static")
            printModel()
            analyze(1)
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
            return res, e_res
