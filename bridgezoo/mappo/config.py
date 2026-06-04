"""MAPPO 超参数配置（已实现）。

集中管理算法与训练超参，便于复现实验与做消融。所有脚本从这里取默认值。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MappoConfig:
    """MAPPO 训练超参数。"""

    # --- 采样 ---
    num_envs: int = 8              # 并行环境数
    rollout_len: int = 256         # 每次更新采样的步数（每环境）
    total_steps: int = 2_000_000   # 总环境步

    # --- 优化 ---
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    num_minibatches: int = 4
    clip_coef: float = 0.2         # PPO 策略裁剪
    clip_vloss: bool = True        # value 裁剪
    vf_coef: float = 0.5
    ent_coef: float = 0.01         # 熵正则
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03  # 提前停止阈值（None 关闭）

    # --- 网络 ---
    actor_hidden: tuple[int, ...] = (128, 128)
    critic_hidden: tuple[int, ...] = (128, 128)
    use_rnn: bool = False          # 是否用 GRU 处理部分可观测（先用前馈）
    share_actor: bool = True       # 参数共享
    use_agent_id: bool = True      # obs 拼接 agent one-hot/位置编码

    # --- 归一化技巧 ---
    norm_adv: bool = True          # 优势归一化
    value_norm: bool = True        # value/return 归一化（PopArt 风格）
    norm_obs: bool = True

    # --- 运行 ---
    seed: int = 0
    device: str = "cpu"            # 小网络 + 频繁 CPU env 交互，cpu 通常更快
    log_dir: str = "runs"
    save_interval: int = 50        # 每多少次更新存一次 checkpoint
    eval_interval: int = 25
    eval_episodes: int = 5

    extra: dict = field(default_factory=dict)  # 预留自定义项

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_len

    @property
    def minibatch_size(self) -> int:
        return self.batch_size // self.num_minibatches

    @property
    def num_updates(self) -> int:
        return self.total_steps // self.batch_size
