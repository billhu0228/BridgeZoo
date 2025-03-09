from functools import cached_property

import gymnasium
import numpy as np
import pygame
from ezdxf.math import Vec2, Matrix44
from gymnasium.utils import EzPickle, seeding

from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn

from bridgezoo.cablebridge.fem import FEM

_image_library = {}

__all__ = ["env", "parallel_env", "raw_env"]


class CableAgent:
    def __init__(self):
        self.obs_dim = 1
        self.stress_init = 1000
        self.stress_before = self.stress_init
        self.stress_after = 1000
        self.num_strands = 40
        self.stress_step = 500
        self.observation = 0
        self.deform = 0
        self.action = None
        self.the_reward = None
        self.real_action = None
        self.reset()

    def reset(self):
        self.stress_init = 1000
        self.stress_before = self.stress_init
        self.stress_after = 1000
        self.num_strands = 20
        self.stress_step = 500
        self.deform = 0.0
        self.observation = np.array([0.], dtype=np.float32)
        self.action = None
        self.the_reward = None
        self.real_action = None

    @cached_property
    def observation_space(self):
        return gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
        )

    @cached_property
    def action_space(self):
        # return gymnasium.spaces.Discrete(3)
        return gymnasium.spaces.Box(low=-1, high=1, shape=(1,))

    def step(self, act):
        self.action = act
        self.stress_before = self.stress_init + act * self.stress_step
        self.stress_before = max(0, self.stress_before)

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        # self.stress_init = balance_stress
        self.deform = deform
        self.observation = np.array([deform, ], dtype=np.float32)


    def done(self):
        ret = bool(self.stress_after <= self.stress_step * 2)
        return ret

    def reward(self):
        return 1


def env(**kwargs):
    the_env = raw_env(**kwargs)
    the_env = wrappers.ClipOutOfBoundsWrapper(the_env)
    the_env = wrappers.OrderEnforcingWrapper(the_env)
    return the_env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "text"],
        "name": "cablebridge_v2",
        "is_parallelizable": True,
    }

    def __init__(
            self,
            time_penalty=-0.1,
            beam_w=10.0,
            beam_h=1.0,
            num_cables_per_side=6,
            anchor_height=20,
            max_cycles=125,
            render_mode=None,
            fps=10,
            DEF_SCALE=10,
    ):
        EzPickle.__init__(
            self,
            beam_w=beam_w,
            beam_h=beam_h,
            num_cables_per_side=num_cables_per_side,
            anchor_height=anchor_height,
            max_cycles=max_cycles,
            render_mode=render_mode,
            fps=fps,
            DEF_SCALE=DEF_SCALE,
        )
        self.clock = pygame.time.Clock()
        self.fps = fps  # Frames Per Second
        self.DEF_SCALE = DEF_SCALE
        self.max_cycles = max_cycles
        # 系统随机参数
        self.beam_E = 20e9
        # 物理参数 Start --------------------------------------------------
        w = beam_w
        h = beam_h
        self.beam_area = h * w
        self.beam_h = h
        self.beam_w = w
        self.beam_Iz = w * h ** 3 / 12.0
        self.wg = w * h * 1 * 2000 * 9.806 * 2
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
        x1 = np.linspace(-0.5 * self.beam_length + self.end_to_first_spacing, -self.outside_spacing - self.span * 0.5,
                         self.num_cables_per_side // 2)
        x2 = np.linspace(-self.span * 0.5, -self.center_to_adjacent_spacing, self.num_cables_per_side // 2 + 1)
        self.x_positions = np.hstack((-self.beam_length * 0.5, x1, x2, 0, x2 * -1, x1 * -1, 0.5 * self.beam_length))
        self.x_positions.sort()
        # 索塔参数
        self.system_obv = 0
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

        assert self.num_cables_per_side > 1, "cables must be greater than 1"
        self.agents = ["cable_" + str(r) for r in range(self.num_cables_per_side)]
        self.cables = {"cable_" + str(r): CableAgent() for r in range(self.num_cables_per_side)}
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_cables_per_side))))
        self._agent_selector = agent_selector(self.agents)

        self.observation_spaces = dict(
            zip(
                self.agents,
                [
                    gymnasium.spaces.Box(
                        low=-1,
                        high=1,
                        shape=(1,),
                    )
                ]
                * self.num_cables_per_side,
            )
        )

        self.action_spaces = dict(
            zip(
                self.agents,
                # [gymnasium.spaces.Discrete(3)] * self.num_cables_per_side,
                [gymnasium.spaces.Box(low=-1, high=1, shape=(1,))] * self.num_cables_per_side,
            )
        )

        self.state_space = gymnasium.spaces.Box(
            low=-1,
            high=1,
            shape=(self.num_cables_per_side + 1,),
        )

        pygame.init()
        self.render_mode = render_mode
        self.renderOn = False
        self.screen = None
        self.max_cycles = max_cycles
        self.screen_width = 1080
        self.screen_height = 600
        self.screen_scale = (self.screen_width - 100) / self.beam_length  # 视频比例尺

        self.cableList = []
        self.cableRewards = []  # Keeps track of individual rewards
        self.time_penalty = time_penalty
        self.terminate = False
        self.truncate = False
        self.frames = 0
        self._seed()

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent):
        return self.cables[agent].observation

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._seed(seed)
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.terminate = False
        self.truncate = False
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self.terminations = dict(zip(self.agents, [False for _ in self.agents]))
        self.truncations = dict(zip(self.agents, [False for _ in self.agents]))
        self.infos = dict(zip(self.agents, [{} for _ in self.agents]))
        self.frames = 0
        self.system_obv = 0

    def render(self, agent_id=None):
        if self.render_mode is None:
            gymnasium.logger.warn(
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
        beam_positions = [c.deform for _, c in self.cables.items()]
        beam_positions = [0] + beam_positions[0:3] + [0, ] + beam_positions[3:] + [self.system_obv]
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
            if i == agent_id:
                color = (255, 0, 0)
            else:
                color = (0, 0, 255)
            pygame.draw.line(self.screen, color, tr2[0], transformed_points[beam_index_left], 2)
            pygame.draw.line(self.screen, color, tr2[0], transformed_points[self.num_beam_points - 2 - i], 2)
            pygame.draw.line(self.screen, color, tr2[1], transformed_points[beam_index_right + i], 2)
            pygame.draw.line(self.screen, color, tr2[1], transformed_points[self.num_beam_points * 2 - 3 - i], 2)

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
        self.clock.tick(self.fps)

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
        fem_model.add_material(1, {'E': self.beam_E, 'A': self.beam_area, 'I': self.beam_Iz, 'W': self.beam_w, 'H': self.beam_h})
        return fem_model

    def update_fem(self, cable_force, cable_soq):
        fem = self.extract_fem_model(cable_force, cable_soq)
        return fem.opensees()

    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(None)
            return

        action = np.asarray(action)
        agent = self.agent_selection
        self.cables[agent].step(action)

        if self._agent_selector.is_last():
            cable_stress = [c.stress_before for i, c in self.cables.items()]
            cable_no = [c.num_strands for i, c in self.cables.items()]
            beam_pos, cable_stress_after = self.update_fem(cable_stress, cable_no)
            self.system_obv = beam_pos[self.num_cables_per_side + 3]
            beam_pos = beam_pos[:self.num_cables_per_side + 3]
            beam_pos = beam_pos[1:self.num_cables_per_side // 2 + 1] + beam_pos[self.num_cables_per_side // 2 + 1 + 1:]
            for i, (key, the_cable) in enumerate(self.cables.items()):
                the_cable.update(cable_stress_after[i], beam_pos[i])
            if self.is_even(beam_pos):
                self.terminate = True
            self.rewards = dict(zip(self.agents, [0, ] * self.num_cables_per_side))
            self.frames += 1

        else:
            self._clear_rewards()

        self.truncate = self.frames >= self.max_cycles

        if self._agent_selector.is_last():
            self.terminations = dict(
                zip(self.agents, [self.terminate for _ in self.agents])
            )
            self.truncations = dict(
                zip(self.agents, [self.truncate for _ in self.agents])
            )

        self._cumulative_rewards[agent] = 0
        self._accumulate_rewards()
        if self._agent_selector.is_last():
            if self.render_mode == "human":
                self.render()
            elif self.render_mode == 'text':
                self.report()
        self.agent_selection = self._agent_selector.next()

    def is_even(self, beam_pos):
        return False

    def report(self):
        cable_stress = [c.stress_before for c in self.cables.values()]
        beam_pos = [c.deform for c in self.cables.values()]
        cable_stress_after = [c.stress_after for c in self.cables.values()]
        cable_nos = [c.num_strands for c in self.cables.values()]
        actions = [c.action for c in self.cables.values()]
        # print(env.unwrapped.env.cables, obs_str)
        # print("BeamE:%.2e | Wg= %.2e | %s | %s" % (self.env.beam_E, self.env.wg, self.env.cables, obs_str))
        text_render = "(%3i)  " % self.frames
        for s in cable_stress:
            text_render += "%5i" % s
        text_render += " | "
        for s in beam_pos:
            text_render += "%6.3f  " % s
        text_render += " | "
        for s in cable_stress_after:
            text_render += "%5i" % s
        text_render += " | "
        for s in actions:
            if s is None:
                text_render += "  \033[91m×\033[0m  "
            else:
                text_render += " %5.2f " % s
        text_render += " | "
        for i, s in self.rewards.items():
            text_render += "%3i  " % s

        print(text_render)
