from __future__ import annotations
import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector
from gymnasium.spaces import Box, Discrete
import functools
from dataclasses import dataclass
from enum import Enum
import pygame


class BoundaryShape(Enum):
    """边界形状枚举类"""
    RECTANGLE = "rectangle"  # 矩形
    T_SHAPE = "t_shape"     # T型
    H_SHAPE = "h_shape"     # H型
    CIRCLE = "circle"       # 圆形

@dataclass
class EnvConfig:
    """环境配置类"""
    num_agents: int = 2                # 智能体数量
    render_mode: str = None            # 渲染模式
    render_fps: int = 30              # 渲染帧率
    min_x: float = -10.0              # x轴最小值
    max_x: float = 10.0               # x轴最大值
    min_y: float = -5.0               # y轴最小值
    max_y: float = 5.0                # y轴最大值
    max_steps: int = 500              # 最大步数
    move_size: float = 0.1            # 每步移动距离
    min_distance: float = 1.0         # 最小距离要求
    window_width: int = 800           # 窗口宽度
    window_height: int = 400          # 窗口高度
    random_seed: int = None           # 随机种子，用于控制初始位置
    
    # 通用边界参数
    boundary_shape: BoundaryShape = BoundaryShape.RECTANGLE  # 边界形状
    max_absolute_x: float = 9.0       # x坐标绝对值上限（所有形状通用）
    max_absolute_y: float = 4.0       # y坐标绝对值上限（所有形状正常区域的y范围）
    cover: float = 0.3               # 边界覆盖范围

    # T型和H型边界共用参数
    normal_x: float = 3.0            # 正常区域的x范围（T型为负，H型为正）
    narrow_y: float = 2.0            # 窄区域的y范围

def env(config: EnvConfig = None, **kwargs):
    """
    创建一维空间多点移动环境的实例
    
    参数:
        config: EnvConfig 实例，包含环境配置
        **kwargs: 其他配置参数，用于覆盖 config 中的默认值
    """
    if config is None:
        config = EnvConfig(**kwargs)
    elif kwargs:
        # 使用 kwargs 更新 config
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    return raw_env(config=config)


class raw_env(AECEnv):
    """
    一维空间多点移动环境

    环境描述：
    - 多个智能体（点）在一维空间上移动
    - 每个智能体可以选择向左移动、不动或向右移动
    - 目标是所有点之间保持一定距离，不要太近也不要太远
    """
    metadata = {
        "render_modes": ["human"],
        "name": "point_movement_v0",
        "is_parallelizable": True,
        "render_fps": 30,
    }

    def __init__(self, config: EnvConfig):
        """初始化环境"""
        # 必须首先设置 possible_agents
        self.possible_agents = [f"agent_{i}" for i in range(config.num_agents)]
        
        # 调用父类初始化
        super().__init__()
        
        # 保存配置
        self.config = config
        
        # 对H型边界的normal_x进行自动修正（确保为正值）
        if self.config.boundary_shape == BoundaryShape.H_SHAPE:
            self.config.normal_x = abs(self.config.normal_x)
        
        # 环境参数设置
        self.min_x = config.min_x
        self.max_x = config.max_x
        self.min_y = config.min_y
        self.max_y = config.max_y
        self.max_steps = config.max_steps
        self.move_size = config.move_size
        self.min_distance = config.min_distance
        self.num_rebars = config.num_agents
        self.step_count = 0
        
        # pygame相关设置
        self.window_width = config.window_width
        self.window_height = config.window_height
        
        # 计算统一的缩放比例
        # 使用最大的x和y范围来计算缩放比例，确保图形不会超出窗口
        if self.config.boundary_shape == BoundaryShape.CIRCLE:
            # 对于圆形边界，考虑半径和覆盖层
            max_radius = self.config.max_absolute_x + self.config.cover
            max_x_range = max_radius
            max_y_range = max_radius
        else:
            max_x_range = max(abs(self.config.max_absolute_x), abs(self.config.normal_x)) + self.config.cover
            max_y_range = max(abs(self.config.max_absolute_y), abs(self.config.narrow_y)) + self.config.cover
        
        # 计算x和y方向的缩放比例，考虑窗口边距
        margin_ratio = 0.1  # 留出10%的边距
        available_width = self.window_width * (1 - 2 * margin_ratio)
        available_height = self.window_height * (1 - 2 * margin_ratio)
        
        # 计算x和y方向的缩放比例，取较小值以确保完整显示
        scale_x = available_width / (2 * max_x_range)
        scale_y = available_height / (2 * max_y_range)
        
        # 使用较小的缩放比例，确保图形完整显示且保持比例
        self.scale = min(scale_x, scale_y)
        self.scale_x = self.scale
        self.scale_y = self.scale
        
        # 计算实际绘制区域的大小
        self.render_width = 2 * max_x_range * self.scale
        self.render_height = 2 * max_y_range * self.scale
        
        # 计算绘制区域的偏移量，使图形居中
        self.offset_x = (self.window_width - self.render_width) / 2
        self.offset_y = (self.window_height - self.render_height) / 2
        
        self.screen = None
        self.clock = None
        
        # 设置智能体映射
        self.agent_name_mapping = dict(
            zip(self.possible_agents, list(range(len(self.possible_agents))))
        )
        
        # 动作空间: 0-左, 1-右, 2-上, 3-下, 4-不动
        self.action_spaces = {
            agent: Discrete(5) for agent in self.possible_agents
        }
        
        # 观察空间: [x位置, y位置] * num_agents
        self.observation_spaces = {
            agent: Box(
                low=np.array([self.min_x, self.min_y] * config.num_agents),
                high=np.array([self.max_x, self.max_y] * config.num_agents),
                shape=(2 * config.num_agents,),
                dtype=np.float32
            )
            for agent in self.possible_agents
        }
        
        # 渲染模式
        self.render_mode = config.render_mode
        
        # 用于并行执行的动作缓存
        self.cached_actions = {}
        
        # 初始化智能体列表
        self.agents = self.possible_agents.copy()
        
        # 初始化奖励和状态
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # 初始化智能体选择器
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # 清空动作缓存
        self.cached_actions.clear()
        
        # 初始化智能体位置
        self.current_positions = self._initialize_positions()
        
        # 初始观察
        self._last_obs = self._get_obs()

    def _initialize_positions(self):
        """初始化智能体位置，使用随机种子控制随机性"""
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
        
        if self.config.boundary_shape == BoundaryShape.CIRCLE:
            # 对于圆形边界，在圆内随机分布
            radius = self.config.max_absolute_x * 0.7  # 使用70%的半径来确保在边界内
            angles = np.random.uniform(0, 2*np.pi, self.num_rebars)
            x_positions = radius * np.cos(angles)
            y_positions = radius * np.sin(angles)
        else:
            # 其他边界形状，只在x轴上随机分布，y保持为0
            x_positions = np.random.uniform(-self.config.max_absolute_x, 
                                          self.config.max_absolute_x, 
                                          self.num_rebars)
            y_positions = np.zeros(self.num_rebars)  # y坐标保持为0
        
        return np.column_stack([x_positions, y_positions]).astype(np.float32)

    def reset(self, seed=None, options=None):
        """重置环境"""
        # 重置步数
        self.step_count = 0
        
        # 重置智能体状态
        self.agents = self.possible_agents[:]
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        
        # 清空动作缓存
        self.cached_actions.clear()
        
        # 重置随机种子（如果提供了新的种子）
        if seed is not None:
            self.config.random_seed = seed
        
        # 重新初始化智能体位置
        self.current_positions = self._initialize_positions()
        
        # 选择第一个行动的智能体
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()
        
        # 初始观察
        self._last_obs = self._get_obs()
        
        return self._last_obs

    def _get_obs(self):
        """构建所有智能体的观察"""
        return {
            agent: self.current_positions.flatten()
            for agent in self.agents
        }

    def _calculate_reward(self) -> float:
        """计算奖励值
        
        基于绝对值计算奖励，考虑：
        1. 钢筋间距约束（平均间距越大越好）
        2. 绝对承载力（基于二次矩/惯性矩）
        
        返回:
            float: 计算得到的奖励值
        """
        # 计算所有钢筋对之间的平均间距
        total_spacing = 0
        pair_count = 0
        for i in range(self.num_rebars):
            for j in range(i + 1, self.num_rebars):
                dist = np.linalg.norm(self.current_positions[i] - self.current_positions[j])
                total_spacing += dist
                pair_count += 1
        
        # 计算平均间距
        avg_spacing = total_spacing / pair_count if pair_count > 0 else 0
        
        # 计算间距得分（平均间距越大越好，但需要满足最小间距要求）
        spacing_score = min(1.0, avg_spacing / (self.min_distance * 2))  # 使用2倍最小间距作为参考值
        
        # 计算二次矩（惯性矩）
        # 假设每个钢筋的面积为1，中性轴在x轴
        distances_to_neutral = np.abs(self.current_positions[:, 0])  # 到中性轴的距离
        second_moment = np.sum(distances_to_neutral ** 2)  # 二次矩
        
        # 计算最大可能的二次矩（所有钢筋都在最大x距离处）
        max_possible_moment = self.num_rebars * (self.config.max_absolute_x ** 2)
        
        # 计算承载力得分
        capacity_score = second_moment / max_possible_moment
        
        # 综合评分（间距约束权重0.4，承载力权重0.6）
        reward = 0.4 * spacing_score + 0.6 * capacity_score
        
        return reward

    def _is_point_in_boundary(self, x: float, y: float) -> bool:
        """检查点是否在边界内"""
        if self.config.boundary_shape == BoundaryShape.RECTANGLE:
            return (abs(x) <= self.config.max_absolute_x and 
                    abs(y) <= self.config.max_absolute_y)
        
        elif self.config.boundary_shape == BoundaryShape.T_SHAPE:
            # T型边界检查
            # 在左侧宽区域内（从-max_absolute_x到normal_x）
            if x <= self.config.normal_x:
                return abs(y) <= self.config.max_absolute_y and x >= -self.config.max_absolute_x
            # 在右侧窄区域内（从normal_x到max_absolute_x）
            else:
                # 确保x在正确范围内，且y在窄区域内
                return (x <= self.config.max_absolute_x and 
                       abs(y) <= self.config.narrow_y)
        
        elif self.config.boundary_shape == BoundaryShape.H_SHAPE:
            # H型边界检查（哑铃形状）
            abs_x = abs(x)
            abs_y = abs(y)
            
            # 在收缩区域内
            if abs_x <= self.config.normal_x:
                # 确保y在窄区域内
                return abs_y <= self.config.narrow_y
            # 在正常区域内
            else:
                # 确保x和y都在正常区域内
                return (abs_x <= self.config.max_absolute_x and 
                       abs_y <= self.config.max_absolute_y)
        
        elif self.config.boundary_shape == BoundaryShape.CIRCLE:
            # 圆形边界检查：使用max_absolute_x作为半径
            distance_squared = x * x + y * y
            radius_squared = self.config.max_absolute_x * self.config.max_absolute_x
            # 添加一个小的容差以处理浮点数精度问题
            epsilon = 1e-10
            return distance_squared <= radius_squared + epsilon
        
        return False

    def _handle_boundary(self, new_pos: np.ndarray, old_pos: np.ndarray, action: int, bounce: bool = False) -> np.ndarray:
        """处理边界问题，确保智能体在边界内
        
        参数:
            new_pos: 移动后的新位置
            old_pos: 移动前的原始位置
            action: 执行的动作（0-左, 1-右, 2-上, 3-下, 4-不动）
            bounce: 是否在碰到边界时反弹，True表示反弹，False表示沿边界滑行
            
        返回:
            np.ndarray: 处理后的位置
        """
        # 检查是否在边界内
        if not self._is_point_in_boundary(new_pos[0], new_pos[1]):
            if bounce:
                # 反弹模式：计算反弹后的位置
                if self.config.boundary_shape == BoundaryShape.CIRCLE:
                    # 对于圆形边界，沿着半径方向反弹
                    radius = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
                    if radius > 0:  # 避免除以零
                        scale = self.config.max_absolute_x / radius
                        new_pos[0] *= scale
                        new_pos[1] *= scale
                else:
                    # 对于其他边界形状，分别处理x和y方向的反弹
                    # 如果超出x方向边界，反转x方向的移动
                    if action in [0, 1]:
                        new_pos[0] = old_pos[0] - (new_pos[0] - old_pos[0])
                    # 如果超出y方向边界，反转y方向的移动
                    if action in [2, 3]:
                        new_pos[1] = old_pos[1] - (new_pos[1] - old_pos[1])
            else:
                # 滑行模式：将位置限制在边界内
                if self.config.boundary_shape == BoundaryShape.CIRCLE:
                    # 对于圆形边界，将位置限制在圆上
                    radius = np.sqrt(new_pos[0]**2 + new_pos[1]**2)
                    if radius > 0:  # 避免除以零
                        scale = self.config.max_absolute_x / radius
                        new_pos[0] *= scale
                        new_pos[1] *= scale
                else:
                    # 对于其他边界形状，将位置限制在边界内
                    if self.config.boundary_shape == BoundaryShape.T_SHAPE:
                        # T型边界特殊处理
                        if old_pos[0] > self.config.normal_x:  # 在右侧窄区域
                            # 如果超出y方向边界，根据动作方向进行反弹或滑行
                            if action in [2, 3]:  # 上下移动
                                if new_pos[1] > self.config.narrow_y:
                                    new_pos[1] = self.config.narrow_y
                                elif new_pos[1] < -self.config.narrow_y:
                                    new_pos[1] = -self.config.narrow_y
                            else: # 左右移动
                                if new_pos[0] > self.config.max_absolute_x:
                                    new_pos[0] = self.config.max_absolute_x
                                elif new_pos[0] < -self.config.normal_x:
                                    new_pos[0] = -self.config.normal_x
                        else: # 在左侧宽区域
                            # 如果超出x方向边界，根据动作方向进行反弹或滑行
                            if action in [0, 1]:  # 左右移动
                                if new_pos[0] > self.config.normal_x:
                                    new_pos[0] = self.config.normal_x
                                elif new_pos[0] < -self.config.max_absolute_x:
                                    new_pos[0] = -self.config.max_absolute_x
                            # 如果超出y方向边界，根据动作方向进行反弹或滑行
                            if action in [2, 3]:  # 上下移动
                                if new_pos[1] > self.config.max_absolute_y:
                                    new_pos[1] = self.config.max_absolute_y
                                elif new_pos[1] < -self.config.max_absolute_y:
                                    new_pos[1] = -self.config.max_absolute_y
                    
                    elif self.config.boundary_shape == BoundaryShape.H_SHAPE:
                        assert self.config.normal_x > 0, "normal_x must be positive for H-shape"
                        if abs(old_pos[0]) < self.config.normal_x:  # 在中间窄区域
                            # 如果超出y方向边界，根据动作方向进行反弹或滑行
                            if action in [2, 3]:  # 上下移动
                                if new_pos[1] > self.config.narrow_y:
                                    new_pos[1] = self.config.narrow_y
                                elif new_pos[1] < -self.config.narrow_y:
                                    new_pos[1] = -self.config.narrow_y
                        elif old_pos[0] >= self.config.normal_x: # 在右侧宽区域
                            if new_pos[0] > self.config.max_absolute_x:
                                new_pos[0] = self.config.max_absolute_x
                            elif new_pos[0] < self.config.normal_x:
                                new_pos[0] = self.config.normal_x
                        else: # 在左侧宽区域
                            if action in [0, 1]:  # 左右移动
                                if new_pos[0] < -self.config.max_absolute_x:
                                    new_pos[0] = -self.config.max_absolute_x
                                elif new_pos[0] > -self.config.normal_x:
                                    new_pos[0] = -self.config.normal_x                    
                    else:  # 矩形边界
                        # 如果超出x方向边界，根据动作方向进行反弹或滑行
                        if action in [0, 1]:  # 左右移动
                            if new_pos[0] > self.config.max_absolute_x:
                                new_pos[0] = self.config.max_absolute_x
                            elif new_pos[0] < -self.config.max_absolute_x:
                                new_pos[0] = -self.config.max_absolute_x
                        # 如果超出y方向边界，根据动作方向进行反弹或滑行
                        if action in [2, 3]:  # 上下移动
                            if new_pos[1] > self.config.max_absolute_y:
                                new_pos[1] = self.config.max_absolute_y
                            elif new_pos[1] < -self.config.max_absolute_y:
                                new_pos[1] = -self.config.max_absolute_y
        
        return new_pos

    def _world_to_screen(self, x: float, y: float) -> tuple[float, float]:
        """将世界坐标转换为屏幕坐标"""
        screen_x = self.window_width/2 + x * self.scale
        screen_y = self.window_height/2 - y * self.scale  # 注意y轴方向是相反的
        return screen_x, screen_y

    def _draw_boundary(self):
        """绘制边界"""
        if self.config.boundary_shape == BoundaryShape.RECTANGLE:
            # 绘制矩形边界
            center_x, center_y = self._world_to_screen(0, 0)
            
            # 绘制外轮廓（深色）- 向外偏移
            outer_x_left = center_x + (self.config.max_absolute_x + self.config.cover) * self.scale_x
            outer_x_right = center_x - (self.config.max_absolute_x + self.config.cover) * self.scale_x
            outer_y_top = center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y
            outer_y_bottom = center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y
            pygame.draw.rect(self.screen, (150, 200, 150), 
                           (outer_x_right, outer_y_top, 
                            outer_x_left - outer_x_right, 
                            outer_y_bottom - outer_y_top))
            
            # 绘制内部（浅色）
            inner_x_left = center_x + self.config.max_absolute_x * self.scale_x
            inner_x_right = center_x - self.config.max_absolute_x * self.scale_x
            inner_y_top = center_y - self.config.max_absolute_y * self.scale_y
            inner_y_bottom = center_y + self.config.max_absolute_y * self.scale_y
            pygame.draw.rect(self.screen, (200, 255, 200), 
                           (inner_x_right, inner_y_top, 
                            inner_x_left - inner_x_right, 
                            inner_y_bottom - inner_y_top))
            
            # 绘制边框
            pygame.draw.rect(self.screen, (0, 150, 0), 
                           (inner_x_right, inner_y_top, 
                            inner_x_left - inner_x_right, 
                            inner_y_bottom - inner_y_top), 2)
        
        elif self.config.boundary_shape == BoundaryShape.T_SHAPE:
            # 转换T型边界坐标
            center_x = self.window_width // 2
            center_y = self.window_height // 2
            
            # 创建外轮廓点列表（深色）- 向外偏移
            outer_points = []
            outer_points.append((center_x - (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x - (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.normal_x + self.config.cover) * self.scale_x, 
                               center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.normal_x + self.config.cover) * self.scale_x, 
                               center_y + (self.config.narrow_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y + (self.config.narrow_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y - (self.config.narrow_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.normal_x + self.config.cover) * self.scale_x, 
                               center_y - (self.config.narrow_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.normal_x + self.config.cover) * self.scale_x, 
                               center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            
            # 绘制外轮廓（深色）
            pygame.draw.polygon(self.screen, (150, 200, 150), outer_points)
            
            # 创建内部点列表（浅色）
            inner_points = []
            inner_points.append((center_x - self.config.max_absolute_x * self.scale_x, 
                               center_y - self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x - self.config.max_absolute_x * self.scale_x, 
                               center_y + self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y + self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y + self.config.narrow_y * self.scale_y))
            inner_points.append((center_x + self.config.max_absolute_x * self.scale_x, 
                               center_y + self.config.narrow_y * self.scale_y))
            inner_points.append((center_x + self.config.max_absolute_x * self.scale_x, 
                               center_y - self.config.narrow_y * self.scale_y))
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y - self.config.narrow_y * self.scale_y))
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y - self.config.max_absolute_y * self.scale_y))
            
            # 绘制内部（浅色）
            pygame.draw.polygon(self.screen, (200, 255, 200), inner_points)
            
            # 绘制边框
            pygame.draw.polygon(self.screen, (0, 150, 0), inner_points, 2)
        
        elif self.config.boundary_shape == BoundaryShape.H_SHAPE:
            # 转换H型边界坐标（哑铃形状）
            center_x = self.window_width // 2
            center_y = self.window_height // 2
            
            # 创建外轮廓点列表（深色）- 向外偏移
            outer_points = []
            # 左侧矩形区域
            outer_points.append((center_x - (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x - (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x - (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x - (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y + (self.config.narrow_y + self.config.cover) * self.scale_y))
            # 中间连接区域
            outer_points.append((center_x + (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y + (self.config.narrow_y + self.config.cover) * self.scale_y))
            # 右侧矩形区域
            outer_points.append((center_x + (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y + (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.max_absolute_x + self.config.cover) * self.scale_x, 
                               center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x + (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y - (self.config.narrow_y + self.config.cover) * self.scale_y))
            # 中间连接区域返回
            outer_points.append((center_x - (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y - (self.config.narrow_y + self.config.cover) * self.scale_y))
            outer_points.append((center_x - (self.config.normal_x - self.config.cover) * self.scale_x, 
                               center_y - (self.config.max_absolute_y + self.config.cover) * self.scale_y))
            
            # 绘制外轮廓（深色）
            pygame.draw.polygon(self.screen, (150, 200, 150), outer_points)
            
            # 创建内部点列表（浅色）
            inner_points = []
            # 左侧矩形区域
            inner_points.append((center_x - self.config.max_absolute_x * self.scale_x, 
                               center_y - self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x - self.config.max_absolute_x * self.scale_x, 
                               center_y + self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x - self.config.normal_x * self.scale_x, 
                               center_y + self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x - self.config.normal_x * self.scale_x, 
                               center_y + self.config.narrow_y * self.scale_y))
            # 中间连接区域
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y + self.config.narrow_y * self.scale_y))
            # 右侧矩形区域
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y + self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x + self.config.max_absolute_x * self.scale_x, 
                               center_y + self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x + self.config.max_absolute_x * self.scale_x, 
                               center_y - self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y - self.config.max_absolute_y * self.scale_y))
            inner_points.append((center_x + self.config.normal_x * self.scale_x, 
                               center_y - self.config.narrow_y * self.scale_y))
            # 中间连接区域返回
            inner_points.append((center_x - self.config.normal_x * self.scale_x, 
                               center_y - self.config.narrow_y * self.scale_y))
            inner_points.append((center_x - self.config.normal_x * self.scale_x, 
                               center_y - self.config.max_absolute_y * self.scale_y))
            
            # 绘制内部（浅色）
            pygame.draw.polygon(self.screen, (200, 255, 200), inner_points)
            
            # 绘制边框
            pygame.draw.polygon(self.screen, (0, 150, 0), inner_points, 2)
        
        elif self.config.boundary_shape == BoundaryShape.CIRCLE:
            # 转换圆形边界坐标
            center_x = self.window_width // 2
            center_y = self.window_height // 2
            
            # 绘制外轮廓（深色）- 向外偏移
            outer_radius = (self.config.max_absolute_x + self.config.cover) * self.scale_x
            pygame.draw.circle(self.screen, (150, 200, 150), 
                             (center_x, center_y), int(outer_radius))
            
            # 绘制内部（浅色）
            inner_radius = self.config.max_absolute_x * self.scale_x
            pygame.draw.circle(self.screen, (200, 255, 200), 
                             (center_x, center_y), int(inner_radius))
            
            # 绘制边框
            pygame.draw.circle(self.screen, (0, 150, 0), 
                             (center_x, center_y), int(inner_radius), 2)

    def step(self, action):
        """环境步进，支持并行执行"""
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            return self._was_dead_step(action)
        
        agent = self.agent_selection
        
        # 缓存当前智能体的动作
        self.cached_actions[agent] = action
        
        # 如果所有智能体都已经选择了动作，执行并行更新
        if len(self.cached_actions) == len(self.agents):
            # 同时更新所有智能体的位置
            for agent_name, act in self.cached_actions.items():
                agent_idx = self.agent_name_mapping[agent_name]
                new_pos = self.current_positions[agent_idx].copy()
                
                # 记录原始位置
                old_pos = new_pos.copy()
                
                if act == 0:  # 向左移动
                    new_pos[0] -= self.move_size
                elif act == 1:  # 向右移动
                    new_pos[0] += self.move_size
                elif act == 2:  # 向上移动
                    new_pos[1] += self.move_size
                elif act == 3:  # 向下移动
                    new_pos[1] -= self.move_size
                
                # 处理边界问题
                new_pos = self._handle_boundary(new_pos, old_pos, act)
                
                # 更新位置
                self.current_positions[agent_idx] = new_pos
            
            # 计算奖励
            reward = self._calculate_reward()
            
            # 更新所有智能体的奖励
            for agent_name in self.agents:
                self.rewards[agent_name] = reward
            
            # 清空动作缓存
            self.cached_actions.clear()
            
            # 更新步数
            self.step_count += 1
            if self.step_count >= self.max_steps:
                self.truncations = {agent: True for agent in self.agents}
        
        # 更新观察空间
        self._last_obs = self._get_obs()
        
        # 选择下一个智能体
        self.agent_selection = self._agent_selector.next()
        
        # 渲染环境
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "text":
            self.report()
        
        return self._last_obs

    def render(self):
        """使用 Pygame 渲染环境"""
        if self.render_mode is None:
            return
        
        import pygame
        import sys
        
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption(f"{self.num_rebars}点移动环境 - {self.config.boundary_shape.value}")
            self.clock = pygame.time.Clock()
            
            # 设置中文字体
            if sys.platform.startswith('win'):
                self.chinese_font = pygame.font.Font("C:\\Windows\\Fonts\\msyh.ttc", 20)
                self.small_chinese_font = pygame.font.Font("C:\\Windows\\Fonts\\msyh.ttc", 12)
            else:
                try:
                    self.chinese_font = pygame.font.SysFont("notosanscjksc", 36)
                    self.small_chinese_font = pygame.font.SysFont("notosanscjksc", 24)
                except:
                    self.chinese_font = pygame.font.Font(None, 36)
                    self.small_chinese_font = pygame.font.Font(None, 24)
        
        # 清空屏幕
        self.screen.fill((255, 255, 255))
        
        # 绘制外框
        pygame.draw.rect(self.screen, (100, 100, 100), (0, 0, self.window_width, self.window_height), 2)
        
        # 绘制坐标轴
        center_x, center_y = self._world_to_screen(0, 0)
        pygame.draw.line(self.screen, (0, 0, 0), (0, int(center_y)), 
                        (self.window_width, int(center_y)), 2)  # x轴
        pygame.draw.line(self.screen, (0, 0, 0), (int(center_x), 0),
                        (int(center_x), self.window_height), 2)  # y轴
        
        # 绘制边界
        self._draw_boundary()
        
        # 生成不同颜色的智能体
        colors = [
            (int(255 * (i / (self.num_rebars - 1))), 0, 255 - int(255 * (i / (self.num_rebars - 1))))
            for i in range(self.num_rebars)
        ]
        
        # 转换坐标并绘制智能体及其编号
        for i, pos in enumerate(self.current_positions):
            # 转换坐标到屏幕空间
            screen_x, screen_y = self._world_to_screen(pos[0], pos[1])
            
            # 绘制智能体圆点
            pygame.draw.circle(self.screen, colors[i], (int(screen_x), int(screen_y)), 10)
            pygame.draw.circle(self.screen, (0, 0, 0), (int(screen_x), int(screen_y)), 10, 1)
            
            # 绘制智能体编号和位置
            number_text = f"智能体{i}"
            text_surface = self.small_chinese_font.render(number_text, True, (0, 0, 0))
            text_rect = text_surface.get_rect()
            text_rect.center = (int(screen_x), int(screen_y) - 20)
            self.screen.blit(text_surface, text_rect)
            
            # 获取当前动作
            action_name = {
                0: "L",  # 左
                1: "R",  # 右
                2: "U",  # 上
                3: "D",  # 下
                4: "S"   # 不动
            }.get(self.cached_actions.get(f"agent_{i}", 4), "S")
            
            # 绘制位置和动作
            pos_text = f"({pos[0]:.1f}, {pos[1]:.1f}) - {action_name}"
            pos_surface = self.small_chinese_font.render(pos_text, True, (0, 0, 0))
            pos_rect = pos_surface.get_rect()
            pos_rect.center = (int(screen_x), int(screen_y) + 20)
            self.screen.blit(pos_surface, pos_rect)
        
        # 绘制当前奖励值和步数
        if len(self.rewards) > 0:
            reward = list(self.rewards.values())[0]
            reward_text = f"奖励值: {reward:.3f}"
            text = self.chinese_font.render(reward_text, True, (0, 0, 0))
            self.screen.blit(text, (10, 10))
        
        step_text = f"步数: {self.step_count}"
        text = self.chinese_font.render(step_text, True, (0, 0, 0))
        self.screen.blit(text, (10, 50))
        
        # 绘制边界形状信息
        shape_text = f"边界形状: {self.config.boundary_shape.value}"
        text = self.chinese_font.render(shape_text, True, (0, 0, 0))
        self.screen.blit(text, (10, 90))
        
        pygame.display.flip()
        self.clock.tick(self.metadata["render_fps"])
        
        # 处理事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

    def close(self):
        """关闭环境"""
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None

    def report(self):
        """以文字形式输出环境状态"""
        # 获取当前奖励值
        reward = list(self.rewards.values())[0] if self.rewards else 0
        
        # 获取当前动作
        current_actions = []
        for agent in self.agents:
            if agent in self.cached_actions:
                action = self.cached_actions[agent]
                action_name = {
                    0: "L",  # 左
                    1: "R",  # 右
                    2: "U",  # 上
                    3: "D",  # 下
                    4: "S"   # 不动
                }.get(action, "S")
                current_actions.append(f"{agent}: {action_name}")
        
        # 获取当前智能体
        current_agent = self.agent_selection
        
        # 计算最小距离
        min_distance = float('inf')
        for i in range(self.num_rebars):
            for j in range(i + 1, self.num_rebars):
                dist = np.linalg.norm(self.current_positions[i] - self.current_positions[j])
                min_distance = min(min_distance, dist)
        
        # 输出状态信息
        print("\n" + "="*50)
        print(f"步数: {self.step_count}")
        print(f"当前智能体: {current_agent}")
        print(f"当前动作: {', '.join(current_actions)}")
        print(f"奖励值: {reward:.3f}")
        print(f"最小距离: {min_distance:.3f}")
        print("\n智能体位置:")
        for i, pos in enumerate(self.current_positions):
            print(f"智能体{i}: ({pos[0]:.3f}, {pos[1]:.3f})")
        print("="*50 + "\n")