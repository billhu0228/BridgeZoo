import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .rebar import raw_env, EnvConfig, BoundaryShape

class RebarGymEnv(gym.Env):
    """钢筋布置环境的Gym包装类"""
    
    def __init__(self, config: EnvConfig = None, **kwargs):
        """初始化环境
        
        参数:
            config: EnvConfig实例，包含环境配置
            **kwargs: 其他配置参数，用于覆盖config中的默认值
        """
        if config is None:
            config = EnvConfig(**kwargs)
        elif kwargs:
            # 使用kwargs更新config
            for key, value in kwargs.items():
                if hasattr(config, key):
                    setattr(config, key, value)
        
        # 创建原始环境
        self.env = raw_env(config)
        
        # 设置动作空间
        self.action_space = spaces.Discrete(5)  # 0-左, 1-右, 2-上, 3-下, 4-不动
        
        # 设置观察空间
        # 观察空间包含所有智能体的位置信息
        obs_shape = (2 * config.num_agents,)  # [x1, y1, x2, y2, ...]
        self.observation_space = spaces.Box(
            low=np.array([config.min_x, config.min_y] * config.num_agents),
            high=np.array([config.max_x, config.max_y] * config.num_agents),
            shape=obs_shape,
            dtype=np.float32
        )
        
        # 保存配置
        self.config = config
        
        # 重置环境
        self.reset()
    
    def reset(self, seed=None, options=None):
        """重置环境
        
        参数:
            seed: 随机种子
            options: 其他选项
            
        返回:
            observation: 初始观察
            info: 额外信息
        """
        # 重置原始环境
        obs = self.env.reset(seed=seed)
        
        # 返回第一个智能体的观察
        return obs[self.env.agent_selection], {}
    
    def step(self, action):
        """执行一步动作
        
        参数:
            action: 要执行的动作
            
        返回:
            observation: 新的观察
            reward: 奖励值
            terminated: 是否结束
            truncated: 是否被截断
            info: 额外信息
        """
        # 执行动作
        obs, reward, done, info = self.env.step(action)
        
        # 获取当前智能体的观察和奖励
        current_agent = self.env.agent_selection
        current_obs = obs[current_agent]
        current_reward = reward[current_agent]
        
        # 检查是否所有智能体都完成了动作
        all_done = len(self.env.cached_actions) == self.config.num_agents
        
        # 返回结果
        return current_obs, current_reward, done[current_agent], False, info
    
    def render(self):
        """渲染环境"""
        self.env.render()
    
    def close(self):
        """关闭环境"""
        self.env.close()
    
    def seed(self, seed=None):
        """设置随机种子"""
        return self.env.reset(seed=seed) 