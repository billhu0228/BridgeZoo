import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
from gymnasium import spaces
from rebar import env, EnvConfig, BoundaryShape
import os
import subprocess
import webbrowser
import time
from typing import Dict, List, Tuple
from torch.utils.tensorboard import SummaryWriter

class MultiAgentEnvWrapper(gym.Env):
    """将多智能体环境包装为单智能体环境，便于SAC训练"""
    
    def __init__(self, num_agents=5, boundary_shape=BoundaryShape.RECTANGLE):
        super().__init__()
        
        # 创建原始环境
        self.env = env(
            config=EnvConfig(
                render_mode=None,
                num_agents=num_agents,
                boundary_shape=boundary_shape
            )
        )
        self.num_agents = num_agents
        
        # 设置动作空间和观察空间
        # 每个智能体有5个动作：左、右、上、下、不动
        self.action_space = spaces.MultiDiscrete([5] * num_agents)
        
        # 观察空间：所有智能体的x,y坐标
        self.observation_space = spaces.Box(
            low=np.array([self.env.min_x, self.env.min_y] * num_agents),
            high=np.array([self.env.max_x, self.env.max_y] * num_agents),
            shape=(2 * num_agents,),
            dtype=np.float32
        )
        
        # 用于记录统计信息
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.total_steps = 0
        self.render_mode = None
        
        # 记录上一步的状态，用于计算奖励
        self.last_positions = None
        self.last_min_distance = float('inf')
    
    def reset(self, seed=None):
        """重置环境"""
        obs = self.env.reset(seed=seed)
        # 重置episode统计信息
        if self.current_episode_length > 0:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        self.current_episode_reward = 0
        self.current_episode_length = 0
        self.last_positions = self.env.current_positions.copy()
        self.last_min_distance = self._get_min_distance()
        return list(obs.values())[0], {}
    
    def _get_min_distance(self) -> float:
        """计算智能体之间的最小距离"""
        min_distance = float('inf')
        positions = self.env.current_positions
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                dist = np.linalg.norm(positions[i] - positions[j])
                min_distance = min(min_distance, dist)
        return min_distance
    
    def _calculate_reward(self) -> float:
        """计算奖励值，考虑多个因素"""
        current_positions = self.env.current_positions
        current_min_distance = self._get_min_distance()
        
        # 1. 距离约束奖励
        distance_reward = 0
        if current_min_distance >= self.env.min_distance:
            distance_reward = 1.0
        else:
            distance_reward = -2.0 * (self.env.min_distance - current_min_distance)
        
        # 2. 边界利用率奖励
        max_x = self.env.config.max_absolute_x
        max_y = self.env.config.max_absolute_y
        boundary_usage = np.mean([
            (abs(pos[0])/max_x + abs(pos[1])/max_y)/2 
            for pos in current_positions
        ])
        boundary_reward = boundary_usage
        
        # 3. 移动效率奖励
        if self.last_positions is not None:
            movement = np.mean([
                np.linalg.norm(curr - last)
                for curr, last in zip(current_positions, self.last_positions)
            ])
            efficiency_reward = 0.1 * movement if current_min_distance >= self.env.min_distance else -0.1 * movement
        else:
            efficiency_reward = 0
        
        # 4. 最小距离改善奖励
        distance_improvement = current_min_distance - self.last_min_distance
        improvement_reward = 0.5 * distance_improvement if current_min_distance < self.env.min_distance else 0
        
        # 更新上一步状态
        self.last_positions = current_positions.copy()
        self.last_min_distance = current_min_distance
        
        # 综合奖励
        total_reward = (
            2.0 * distance_reward +  # 距离约束最重要
            1.0 * boundary_reward +  # 边界利用率次之
            0.5 * efficiency_reward +  # 移动效率影响较小
            0.5 * improvement_reward   # 改善奖励作为补充
        )
        
        return total_reward
    
    def step(self, actions):
        """执行动作"""
        # 依次为每个智能体执行动作
        for i in range(self.num_agents):
            obs = self.env.step(int(actions[i]))
        
        # 计算奖励
        reward = self._calculate_reward()
        
        # 获取最后一个智能体执行后的状态
        terminated = any(self.env.terminations.values())
        truncated = any(self.env.truncations.values())
        
        # 更新统计信息
        self.current_episode_reward += reward
        self.current_episode_length += 1
        self.total_steps += 1
        
        # 如果episode结束，记录统计信息
        if terminated or truncated:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
        
        return list(obs.values())[0], reward, terminated, truncated, {}

class CustomCNN(BaseFeaturesExtractor):
    """自定义特征提取器，使用残差连接和LayerNorm"""
    
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_input = observation_space.shape[0]
        
        self.layer1 = torch.nn.Sequential(
            torch.nn.Linear(n_input, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU()
        )
        
        self.layer2 = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU()
        )
        
        self.layer3 = torch.nn.Sequential(
            torch.nn.Linear(256, features_dim),
            torch.nn.LayerNorm(features_dim),
            torch.nn.ReLU()
        )
        
        # 残差连接的映射层
        self.residual = torch.nn.Linear(n_input, features_dim) if n_input != features_dim else None
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.layer1(observations)
        x = x + self.layer2(x)  # 第一个残差连接
        x = self.layer3(x)
        
        # 添加来自输入的残差连接
        if self.residual is not None:
            x = x + self.residual(observations)
        
        return x

class TensorboardCallback(BaseCallback):
    """用于记录额外训练信息的自定义回调"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []
        self.episode_boundaries = []
    
    def _on_training_start(self):
        """确保所有监控指标都有输出目录"""
        self.logger.record("train/episode_reward", 0)
        self.logger.record("train/episode_length", 0)
        self.logger.record("train/min_distance", 0)
        self.logger.record("train/boundary_usage", 0)
        self.logger.record("train/distance_violation_rate", 0)
    
    def _on_step(self) -> bool:
        """每步更新时记录信息"""
        # 获取当前环境
        env = self.training_env.envs[0].env
        
        # 记录当前episode的信息
        if len(env.episode_rewards) > 0:
            self.episode_rewards.append(env.episode_rewards[-1])
            self.episode_lengths.append(env.episode_lengths[-1])
        
        # 计算并记录当前状态的指标
        min_distance = env._get_min_distance()
        current_positions = env.env.current_positions
        max_x = env.env.config.max_absolute_x
        max_y = env.env.config.max_absolute_y
        boundary_usage = np.mean([
            (abs(pos[0])/max_x + abs(pos[1])/max_y)/2 
            for pos in current_positions
        ])
        
        # 计算距离违反率（有多少比例的智能体对之间的距离小于最小距离）
        num_violations = 0
        num_pairs = 0
        for i in range(env.num_agents):
            for j in range(i + 1, env.num_agents):
                dist = np.linalg.norm(current_positions[i] - current_positions[j])
                if dist < env.env.min_distance:
                    num_violations += 1
                num_pairs += 1
        violation_rate = num_violations / num_pairs if num_pairs > 0 else 0
        
        # 记录到TensorBoard
        self.logger.record("train/current_min_distance", min_distance)
        self.logger.record("train/current_boundary_usage", boundary_usage)
        self.logger.record("train/distance_violation_rate", violation_rate)
        
        # 如果有完整的episode，记录episode级别的统计信息
        if len(self.episode_rewards) > 0:
            self.logger.record("train/episode_reward", np.mean(self.episode_rewards[-100:]))
            self.logger.record("train/episode_length", np.mean(self.episode_lengths[-100:]))
            self.logger.record("train/episodes_completed", len(self.episode_rewards))
        
        return True

def make_env(num_agents=5, boundary_shape=BoundaryShape.RECTANGLE, rank=0):
    """创建环境的辅助函数"""
    def _init():
        env = MultiAgentEnvWrapper(num_agents, boundary_shape)
        env = Monitor(env)
        return env
    return _init

def train(num_agents=5, total_timesteps=1000000, save_freq=10000):
    """训练SAC模型"""
    # 创建日志和模型保存目录
    log_dir = "logs"
    tensorboard_log = os.path.join(log_dir, "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    
    # 创建多进程环境
    num_cpu = 4  # 根据您的CPU核心数调整
    env = SubprocVecEnv([make_env(num_agents, BoundaryShape.RECTANGLE, i) for i in range(num_cpu)])
    
    # 设置策略网络参数
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256, 256], qf=[256, 256]),
        activation_fn=torch.nn.ReLU
    )
    
    # 创建SAC模型
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        action_noise=None,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=1
    )
    
    # 设置检查点回调
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=log_dir,
        name_prefix="sac_rebar"
    )
    
    # 设置TensorBoard回调
    tensorboard_callback = TensorboardCallback()
    
    # 开始训练
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, tensorboard_callback],
        progress_bar=True
    )
    
    # 保存最终模型
    model.save(f"{log_dir}/final_model")
    
    return model

def start_tensorboard(logdir: str, port: int = 6006):
    """启动TensorBoard服务器
    
    Args:
        logdir: TensorBoard日志目录
        port: 端口号，默认6006
    
    Returns:
        subprocess.Popen: TensorBoard进程
    """
    tensorboard_process = subprocess.Popen(
        ["tensorboard", "--logdir", logdir, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    
    # 等待服务器启动
    time.sleep(3)
    
    # 在默认浏览器中打开TensorBoard
    webbrowser.open(f"http://localhost:{port}")
    
    return tensorboard_process

if __name__ == "__main__":
    # 训练参数
    NUM_AGENTS = 5
    TOTAL_TIMESTEPS = 2000000  # 增加训练步数
    SAVE_FREQ = 50000
    
    # 获取当前工作目录的绝对路径
    current_dir = os.path.abspath(os.path.dirname(__file__))
    tensorboard_dir = os.path.join(current_dir, "logs", "tensorboard")
    
    print("开始训练...")
    print("\n正在启动TensorBoard...")
    
    try:
        # 启动TensorBoard
        tensorboard_process = start_tensorboard(tensorboard_dir)
        print("TensorBoard已启动！")
        print("如果浏览器没有自动打开，请访问：http://localhost:6006")
        
        # 开始训练
        model = train(NUM_AGENTS, TOTAL_TIMESTEPS, SAVE_FREQ)
        print("\n训练完成！")
        print(f"模型已保存到：{os.path.join(current_dir, 'logs', 'final_model.zip')}")
        
    except Exception as e:
        print(f"\n启动TensorBoard时出错：{str(e)}")
        print("请手动运行以下命令：")
        print(f"tensorboard --logdir {tensorboard_dir}")
        
        # 继续训练
        model = train(NUM_AGENTS, TOTAL_TIMESTEPS, SAVE_FREQ)
        print("\n训练完成！")
        print(f"模型已保存到：{os.path.join(current_dir, 'logs', 'final_model.zip')}")
    
    finally:
        # 如果TensorBoard进程存在，则关闭它
        if 'tensorboard_process' in locals():
            tensorboard_process.terminate()
            print("\nTensorBoard已关闭。") 