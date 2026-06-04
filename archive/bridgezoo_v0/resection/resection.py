import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Box, Discrete
import functools
import pygame


def env(render_mode=None):
    """
    创建一维空间两点移动环境的实例
    """
    return raw_env(render_mode=render_mode)


class raw_env(AECEnv):
    """
    一维空间两点移动环境

    环境描述：
    - 两个智能体（点）在一维空间上移动
    - 每个智能体可以选择向左移动、不动或向右移动
    - 目标是两点保持一定距离，不要太近也不要太远
    """
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "name": "point_movement_v0",
        "is_parallelizable": False,
        "render_fps": 10,
    }

    def __init__(self, render_mode=None):
        super().__init__()

        # 环境参数设置
        self.min_position = -10.0  # 一维空间最小值
        self.max_position = 10.0  # 一维空间最大值
        self.max_steps = 100  # 最大步数
        self.move_size = 0.5  # 每步移动距离
        self.min_distance = 1.0  # 最小距离要求
        self.max_absolute_value = 9.0  # 坐标绝对值上限

        # 设置智能体
        self.possible_agents = ["agent_0", "agent_1"]
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )

        # 动作空间: 0-向左移动，1-不动，2-向右移动
        self.action_spaces = {
            agent: Discrete(3) for agent in self.possible_agents
        }

        # 观察空间: [自身位置, 另一个智能体位置]
        self.observation_spaces = {
            agent: Box(low=self.min_position, high=self.max_position, shape=(2,), dtype=np.float32)
            for agent in self.possible_agents
        }

        # 渲染相关设置
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        if self.render_mode == "human":
            pygame.init()
            self.screen_width = 800
            self.screen_height = 200
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Resection Environment")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        """重置环境"""
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 重置环境状态
        self.step_count = 0
        self.current_positions = np.array([-5.0, 5.0], dtype=np.float32)  # 初始位置

        # 选择第一个行动的智能体
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        # 初始观察
        self._last_obs = self._get_obs()

        return self._last_obs

    def step(self, action):
        """环境步进"""
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            # 如果当前智能体已经结束，则选择下一个智能体
            return self._was_dead_step(action)

        agent = self.agent_selection
        agent_idx = self.agent_name_mapping[agent]

        # 执行动作
        if action == 0:  # 向左移动
            self.current_positions[agent_idx] = max(self.min_position, self.current_positions[agent_idx] - self.move_size)
        elif action == 2:  # 向右移动
            self.current_positions[agent_idx] = min(self.max_position, self.current_positions[agent_idx] + self.move_size)
        # action == 1 时不移动

        # 计算奖励
        current_distance = abs(self.current_positions[0] - self.current_positions[1])
        pos0_squared = self.current_positions[0] ** 2
        pos1_squared = self.current_positions[1] ** 2

        # 检查约束条件
        distance_constraint_met = current_distance >= self.min_distance
        value_constraint_0_met = abs(self.current_positions[0]) <= self.max_absolute_value
        value_constraint_1_met = abs(self.current_positions[1]) <= self.max_absolute_value

        # 奖励函数:
        # 1. 两点坐标的平方和尽可能大
        # 2. 但是如果约束条件不满足，则给予惩罚
        if distance_constraint_met and value_constraint_0_met and value_constraint_1_met:
            # 所有约束满足，奖励为坐标平方之和的标准化值
            max_possible_value = 2 * (self.max_absolute_value ** 2)  # 两个点最大可能的平方和
            reward = (pos0_squared + pos1_squared) / max_possible_value
        else:
            # 约束不满足，给予惩罚
            penalty = 0.0

            if not distance_constraint_met:
                # 距离约束不满足，惩罚与违反程度成比例
                penalty -= 1.0 + (self.min_distance - current_distance)

            if not value_constraint_0_met:
                # 坐标约束不满足，惩罚与违反程度成比例
                penalty -= 0.5 * (abs(self.current_positions[0]) - self.max_absolute_value)

            if not value_constraint_1_met:
                # 坐标约束不满足，惩罚与违反程度成比例
                penalty -= 0.5 * (abs(self.current_positions[1]) - self.max_absolute_value)

            reward = penalty

        self.rewards[agent] = reward

        # 检查是否达到最大步数
        self.step_count += 1
        if self.step_count >= self.max_steps:
            self.truncations = {agent: True for agent in self.agents}

        # 更新观察空间并选择下一个智能体
        self._last_obs = self._get_obs()

        if self._agent_selector.is_last():
            self._cumulative_rewards = {
                agent: self._cumulative_rewards[agent] + self.rewards[agent] for agent in self.agents
            }
            self.rewards = {agent: 0 for agent in self.agents}

        self.agent_selection = self._agent_selector.next()

        # 渲染环境（如果需要）
        if self.render_mode == "human":
            self.render()

        return self._last_obs

    def observe(self, agent):
        """获取指定智能体的观察"""
        return self._last_obs[agent]

    def _get_obs(self):
        """构建所有智能体的观察"""
        return {
            self.agents[i]: np.array(
                [self.current_positions[i], self.current_positions[1 - i]],
                dtype=np.float32
            )
            for i in range(len(self.agents))
        }

    def render(self):
        """渲染环境"""
        if self.render_mode is None:
            return

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen_width = 800
            self.screen_height = 200
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Resection Environment")
            self.clock = pygame.time.Clock()

        # 清空屏幕
        self.screen.fill((255, 255, 255))  # 白色背景

        # 计算坐标转换
        scale = self.screen_width / (self.max_position - self.min_position)
        offset = -self.min_position * scale

        # 绘制网格线和刻度
        for x in range(int(self.min_position), int(self.max_position) + 1):
            screen_x = int(x * scale + offset)
            # 绘制垂直网格线
            pygame.draw.line(self.screen, (200, 200, 200), (screen_x, 0), (screen_x, self.screen_height))
            # 绘制刻度值
            if x % 2 == 0:  # 每隔2个单位显示一个刻度
                font = pygame.font.Font(None, 24)
                text = font.render(str(x), True, (0, 0, 0))
                self.screen.blit(text, (screen_x - 10, self.screen_height - 20))

        # 绘制中心线
        pygame.draw.line(self.screen, (100, 100, 100), 
                        (self.screen_width//2, 0), 
                        (self.screen_width//2, self.screen_height))

        # 绘制最大绝对值约束区域
        max_abs_x = int(self.max_absolute_value * scale + offset)
        min_abs_x = int(-self.max_absolute_value * scale + offset)
        constraint_surface = pygame.Surface((max_abs_x - min_abs_x, self.screen_height))
        constraint_surface.fill((200, 255, 200))  # 浅绿色
        constraint_surface.set_alpha(100)
        self.screen.blit(constraint_surface, (min_abs_x, 0))

        # 绘制两个智能体
        for i, pos in enumerate(self.current_positions):
            screen_x = int(pos * scale + offset)
            color = (0, 0, 255) if i == 0 else (255, 0, 0)  # 蓝色和红色
            pygame.draw.circle(self.screen, color, (screen_x, self.screen_height//2), 10)

            # 绘制最小距离约束
            if i == 0:  # 只对第一个智能体绘制
                min_dist_left = int((pos - self.min_distance) * scale + offset)
                min_dist_right = int((pos + self.min_distance) * scale + offset)
                constraint_surface = pygame.Surface((min_dist_right - min_dist_left, self.screen_height))
                constraint_surface.fill((255, 200, 200))  # 浅红色
                constraint_surface.set_alpha(100)
                self.screen.blit(constraint_surface, (min_dist_left, 0))

        # 显示信息
        font = pygame.font.Font(None, 24)
        distance = abs(self.current_positions[0] - self.current_positions[1])
        info_text = f"Step: {self.step_count}  Distance: {distance:.2f}"
        text = font.render(info_text, True, (0, 0, 0))
        self.screen.blit(text, (10, 10))

        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            pygame.quit()
            self.screen = None