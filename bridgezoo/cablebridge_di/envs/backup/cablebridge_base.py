from collections import OrderedDict

import gymnasium as gym
import numpy as np
import pygame

from ezdxf.math import Matrix44, Vec2
from gymnasium import spaces
from gymnasium.utils import seeding

from .cablebridge_models import FEM, Cable


class CableBridgeBase(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
            self,
            beam_e=30e9,
            beam_w=10.0,
            beam_h=1.0,
            num_cables_per_side=30,
            anchor_height=80,
            max_cycles=10,
            render_mode=None,
            DEF_SCALE=10,
            FPS=10,
    ):

        self.clock = pygame.time.Clock()
        self.DEF_SCALE = DEF_SCALE
        self.max_cycles = max_cycles
        # 物理参数 Start --------------------------------------------------
        w = beam_w
        h = beam_h
        self.beam_area = h * w
        self.beam_Iz = w * h ** 3 / 12.0
        self.beam_E = beam_e
        self.wg = w * h * 1 * 2400 * 9.806
        self.num_cables_per_side = num_cables_per_side
        self.middle_spacing = 10
        self.outside_spacing = 8
        self.end_to_first_spacing = 4
        self.center_to_adjacent_spacing = 2
        self.vertical_spacing = 2
        self.anchor_height = anchor_height  # 上塔柱高度
        self.num_beam_points = self.num_cables_per_side + 3  # 只考虑一侧的梁节点
        self.span = 2 * (self.num_cables_per_side * 0.5 * self.middle_spacing + self.center_to_adjacent_spacing)
        self.side_span = self.end_to_first_spacing + self.num_cables_per_side * 0.5 * self.outside_spacing
        self.beam_length = self.side_span * 2 + self.span
        self.x_positions = np.zeros(self.num_cables_per_side * 2 + 5)
        self.left_tower_top = self.right_tower_top = Vec2(0, 0)
        self.left_tower_base = self.right_tower_base = Vec2(0, 0)
        self.left_tower_pts = []
        self.right_tower_pts = []
        # 主梁参数
        x1 = np.linspace(-0.5 * self.beam_length + self.end_to_first_spacing, -self.outside_spacing - self.span * 0.5, self.num_cables_per_side // 2)
        x2 = np.linspace(-self.span * 0.5, -self.center_to_adjacent_spacing, self.num_cables_per_side // 2 + 1)
        self.x_positions = np.hstack((-self.beam_length * 0.5, x1, x2, 0, x2 * -1, x1 * -1, 0.5 * self.beam_length))
        self.x_positions.sort()
        # 索塔参数
        self.left_tower_top = Vec2([-0.5 * self.span, self.anchor_height])
        self.right_tower_top = Vec2([0.5 * self.span, self.anchor_height])
        self.left_tower_base = Vec2(-0.5 * self.span, 0)
        self.right_tower_base = Vec2(0.5 * self.span, 0)
        self.left_tower_pts = []
        self.right_tower_pts = []
        for i in range(self.num_cables_per_side // 2):
            left_anchor = self.left_tower_top + Vec2(0, -self.vertical_spacing) * i
            right_anchor = self.right_tower_top + Vec2(0, -self.vertical_spacing) * i
            self.left_tower_pts.append(left_anchor)
            self.right_tower_pts.append(right_anchor)

        # 物理参数 End  --------------------------------------------------
        obv_dim = self.num_cables_per_side + 1
        self.cables = OrderedDict({"Cable_%i" % i: Cable(obv_dim) for i in range(self.num_cables_per_side)})
        self.render_mode = render_mode
        self.screen = None
        self.frames = 0
        self.num_agents = self.num_cables_per_side
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_cables_per_side * 2,), dtype=np.float32)
        act_space = [c.action_space for i, c in self.cables.items()]
        act_n = self.cables['Cable_0'].action_space[1].n
        self.action_space = spaces.Dict({
            "discrete_actions": spaces.MultiDiscrete([act_n, ] * self.num_cables_per_side),  # 4组离散动作，每组有4个值
            "continuous_actions": spaces.Box(low=0, high=1.0, shape=(self.num_cables_per_side,), dtype=np.float32)  # 2个连续动作，取值范围[-1, 1]
        })
        # self.action_space = spaces.Tuple(act_space)
        self.screen_width = 1080
        self.screen_height = 600
        self.FPS = FPS
        self.screen_scale = (self.screen_width - 100) / self.beam_length  # 视频比例尺
        self.seed()
        self.state = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def reset(self, **kwargs):
        self.frames = 0
        self.cables = OrderedDict({"Cable_" + str(r): Cable(self.num_cables_per_side + 1) for r in range(self.num_cables_per_side)})
        for i, c in self.cables.items():
            c.reset()
        cable_stress = [c.stress_init for i, c in self.cables.items()]
        cable_no = [c.num_strands for i, c in self.cables.items()]
        beam_pos, cable_stress_after = self.update_fem(cable_stress, cable_no)
        beam_pos = beam_pos[:self.num_cables_per_side + 3]
        beam_pos = beam_pos[1:self.num_cables_per_side // 2 + 1] + beam_pos[self.num_cables_per_side // 2 + 1 + 1:-1]
        for i, (key, c) in enumerate(self.cables.items()):
            c.update(cable_stress_after[i], beam_pos[i])
        self.state = np.hstack((beam_pos, cable_stress_after), dtype=np.float32)
        return self.state, {}

    def step(self, action_dict):
        dis_act = action_dict['discrete_actions']
        con_act = action_dict['continuous_actions']
        for i, (k, c) in enumerate(self.cables.items()):
            self.cables[k].step((con_act[i], dis_act[i]))

        cable_stress = [c.stress_init for i, c in self.cables.items()]
        cable_no = [c.num_strands for i, c in self.cables.items()]
        beam_pos, cable_stress_after = self.update_fem(cable_stress, cable_no)
        beam_pos = beam_pos[:self.num_cables_per_side + 3]
        beam_pos = beam_pos[1:self.num_cables_per_side // 2 + 1] + beam_pos[self.num_cables_per_side // 2 + 1 + 1:-1]
        for i, (key, c) in enumerate(self.cables.items()):
            c.update(cable_stress_after[i], beam_pos[i])
        self.state = np.hstack((beam_pos, cable_stress_after), dtype=np.float32)
        rewards = [c.reward3() for i, c in self.cables.items()]
        done = False
        truncated = self.frames >= self.max_cycles
        self.frames += 1
        if self.render_mode == "human":
            self.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()
        return self.state, np.sum(rewards), done, truncated, {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if self.screen is None:
            if self.render_mode == "human":
                pygame.init()
                self.screen_width = 1080
                self.screen_height = 600
                self.screen_scale = (self.screen_width - 100) / self.beam_length  # 视频比例尺
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption('Cable-Stayed Bridge Environment')

        beam_positions = np.hstack(
            (
                [0, ],
                self.state[:self.num_cables_per_side // 2],
                [0, ],
                self.state[self.num_cables_per_side // 2:self.num_cables_per_side],
                self.state[self.num_cables_per_side - 1],
            )
        )
        cable_stress_after = [cable.stress_after for i, cable in self.cables.items()]
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen_width = 1080
            self.screen_height = 600
            self.screen_scale = (self.screen_width - 100) / self.beam_length  # 视频比例尺
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption('Cable-Stayed Bridge Environment')
        self.screen.fill((255, 255, 255))  # 清空屏幕并设置背景为白色
        width, height = self.screen.get_size()
        screen_center = (width // 2, int(height * 3 / 4))
        mat = self.create_transformation_matrix(self.screen_scale, self.screen_scale, 0, screen_center[0], screen_center[1])

        # 绘制淡淡的网格线，水平间距为50米
        center_x, center_y = width // 2, int(height * 3 / 4)
        for i in range(center_x % 50, width, 50):
            pygame.draw.line(self.screen, (200, 200, 200), (i, 0), (i, height))
        for j in range(center_y % 50, height, 50):
            pygame.draw.line(self.screen, (200, 200, 200), (0, j), (width, j))

        # 绘制梁的位置
        y_positions = np.hstack((beam_positions, beam_positions[::-1][1:]))
        transformed_points = self.trans(mat, list(zip(self.x_positions, y_positions * self.DEF_SCALE)))
        pygame.draw.lines(self.screen, (0, 0, 0), False, transformed_points, 5)
        for pt in transformed_points:
            pygame.draw.line(self.screen, (255, 0, 0), Vec2(pt) + Vec2(0, 10), Vec2(pt) + Vec2(0, -10), 1)

        # 绘制固定点
        pygame.draw.circle(self.screen, (0, 0, 0), transformed_points[0], 5)
        pygame.draw.circle(self.screen, (0, 0, 0), transformed_points[-1], 5)

        # 计算左侧和右侧拉索锚点基准位置
        for i in range(self.num_cables_per_side // 2):
            left_anchor = self.left_tower_pts[i]
            right_anchor = self.right_tower_pts[i]
            tr2 = self.trans(mat, (left_anchor, right_anchor))
            beam_index_left = i + 1
            beam_index_right = self.num_beam_points
            pygame.draw.line(self.screen, (0, 0, 255), tr2[0], transformed_points[beam_index_left], 2)
            pygame.draw.line(self.screen, (0, 0, 255), tr2[0], transformed_points[self.num_beam_points - 2 - i], 2)
            pygame.draw.line(self.screen, (0, 0, 255), tr2[1], transformed_points[beam_index_right + i], 2)
            pygame.draw.line(self.screen, (0, 0, 255), tr2[1], transformed_points[self.num_beam_points * 2 - 3 - i], 2)

        # 绘制参考线
        left_anchor_base = self.trans(mat, self.left_tower_top)
        right_anchor_base = self.trans(mat, self.right_tower_top)
        left_tower_base = self.trans(mat, self.left_tower_base)
        right_tower_base = self.trans(mat, self.right_tower_base)
        pygame.draw.line(self.screen, (0, 0, 0), left_tower_base, left_anchor_base, 4)
        pygame.draw.line(self.screen, (0, 0, 0), right_tower_base, right_anchor_base, 4)

        # 显示梁的总长度、固定点高度和网格间距
        font = pygame.font.SysFont('fs', 20)
        text = font.render('Tower Height: %.1f m' % self.anchor_height, True, (0, 0, 0))
        self.screen.blit(text, (10, 10))
        text = font.render('Span Length:%.1f m' % self.span, True, (0, 0, 0))
        self.screen.blit(text, (10, 25))
        text = font.render(f'EI: {self.beam_area}', True, (0, 0, 0))
        self.screen.blit(text, (10, 40))
        text = font.render('Beam0Y: %.0f mm' % (float(beam_positions[-1]) * 1000), True, (0, 0, 0))
        self.screen.blit(text, (10, 55))
        text = font.render('Stress: %.1f MPa(Max. ) | %.1f MPa(Min.)' % (max(cable_stress_after), min(cable_stress_after)), True, (0, 0, 0))
        self.screen.blit(text, (10, 70))
        pygame.display.flip()
        self.clock.tick(self.FPS)

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.update()
        return

    @staticmethod
    def create_transformation_matrix(sx, sy, angle, tx, ty):
        """
        创建一个二维的旋转、平移和缩放变换矩阵。

        :param sx: x方向的缩放因子
        :param sy: y方向的缩放因子
        :param angle: 逆时针旋转角度（以弧度表示）
        :param tx: x方向的平移量
        :param ty: y方向的平移量
        :return: 一个表示旋转、平移和缩放的组合变换矩阵
        """
        # 创建旋转矩阵（绕z轴逆时针旋转）
        rotation_matrix = Matrix44.z_rotate(angle)

        # 创建平移矩阵
        translation_matrix = Matrix44.translate(tx, ty, 0)

        # 创建缩放矩阵
        scaling_matrix = Matrix44.scale(sx, sy, 1)

        # 组合变换矩阵（先旋转，再平移，最后缩放）
        transformation_matrix = scaling_matrix @ translation_matrix @ rotation_matrix
        return transformation_matrix

    @staticmethod
    def trans(matrix, points):
        """
        应用变换矩阵到点集合上，返回变换后的点。

        :param matrix: 变换矩阵
        :param points: 点的集合，每个点为一个元组(x, y)
        :return: 变换后的点的集合
        """
        # 将点转换为 Vec2 对象并应用变换矩阵,上下反转
        if not isinstance(points, Vec2):
            transformed_points = [matrix.transform(Vec2(p[0], -p[1])) for p in points]
            return [(p.x, p.y) for p in transformed_points]
        else:
            transformed_points = matrix.transform(Vec2(points.x, -points.y))
            return transformed_points.vec2

    def extract_fem_model(self, cable_sigma, cable_sizes):
        """
        从CableSystemEnv提取信息，生成FEM实例
        """
        # cable_tensions = self.state[self.num_beam_points:self.num_beam_points + self.num_cables_per_side]
        fem_model = FEM(self.num_cables_per_side, self.wg, cable_sigma)
        # 提取梁的节点信息
        for i, x in enumerate(self.x_positions):
            fem_model.add_node(i + 1, x, 0)
        # 提取梁的单元信息
        for i in range(len(self.x_positions) - 1):
            fem_model.add_element(i + 1, i + 1, i + 2, 1, 1)  # 梁单元类型和材料假设为1
        # 提取索塔坐标
        for i in range(self.num_cables_per_side // 2):
            fem_model.add_node(1001 + i, self.left_tower_pts[i].x, self.left_tower_pts[i].y)
            fem_model.add_node(3001 + i, self.right_tower_pts[i].x, self.right_tower_pts[i].y)
            beam_index_left = i + 2
            beam_index_right = self.num_beam_points + 1
            fem_model.add_element(1001 + i, 1001 + i, beam_index_left, 2, 1001 + i)  #
            fem_model.add_element(2001 + i, 1001 + i, self.num_beam_points - 1 - i, 2, 1001 + i)
            fem_model.add_element(3001 + i, 3001 + i, beam_index_right + i, 2, 1001 + i)
            fem_model.add_element(4001 + i, 3001 + i, self.num_beam_points * 2 - 2 - i, 2, 1001 + i)
            Ns1001 = cable_sizes[i]
            Nf1001 = cable_sigma[i]
            Ns2001 = cable_sizes[self.num_cables_per_side - 1 - i]
            Nf2001 = cable_sigma[self.num_cables_per_side - 1 - i]
            fem_model.add_material(1001 + i, {'Ns': Ns1001, 'sigma': Nf1001})
            fem_model.add_material(2001 + i, {'Ns': Ns2001, 'sigma': Nf2001})
            fem_model.add_material(3001 + i, {'Ns': Ns1001, 'sigma': Nf1001})
            fem_model.add_material(4001 + i, {'Ns': Ns2001, 'sigma': Nf2001})

        # 添加单元类型和材料信息
        fem_model.add_element_type(1, 'Beam')
        fem_model.add_element_type(2, 'Cable')
        fem_model.add_material(1, {'E': self.beam_E, 'A': self.beam_area, 'I': self.beam_Iz})

        return fem_model

    def update_fem(self, cable_force, cable_soq):
        fem = self.extract_fem_model(cable_force, cable_soq)
        return fem.opensees()
