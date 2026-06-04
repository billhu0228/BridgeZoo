"""MAPPO 网络：共享 actor + 中心化 critic（里程碑 M3）。

- **Actor**（去中心化执行，参数共享）：输入局部观测 ``obs_i``（可选拼接 agent-id），
  输出离散动作分布；对非法动作用 action mask 将 logits 置 ``-inf``。
- **Critic**（集中训练）：输入全局状态 ``state``，输出团队价值 ``V(state)``。

实现要点（MAPPO tricks）：正交初始化、tanh 激活、可选 GRU、value normalization。
参见 ``docs/DESIGN_MAPPO.md`` 第 6 节。
"""

from __future__ import annotations

from bridgezoo.mappo.config import MappoConfig

# 说明：torch 为重依赖，仅在实现/调用时导入，保持骨架可被无 torch 环境 import。


class Actor:
    """共享策略网络（骨架）。

    TODO(M3):
      - __init__: 构建 MLP/GRU，输出每个离散动作头的 logits。
      - forward(obs, mask): 返回带掩码的动作分布。
      - act(obs, mask): 采样动作 + log_prob + entropy。
      - evaluate(obs, mask, action): 返回 log_prob / entropy（用于 PPO 更新）。
    """

    def __init__(self, obs_dim: int, action_nvec, cfg: MappoConfig):
        self.obs_dim = obs_dim
        self.action_nvec = action_nvec
        self.cfg = cfg
        raise NotImplementedError("TODO(M3): Actor.__init__")


class CentralCritic:
    """中心化价值网络（骨架）。

    TODO(M3):
      - __init__: 构建 MLP，输入全局状态维度。
      - forward(state): 返回 V(state)。
      - 支持 value normalization（PopArt 风格）。
    """

    def __init__(self, state_dim: int, cfg: MappoConfig):
        self.state_dim = state_dim
        self.cfg = cfg
        raise NotImplementedError("TODO(M3): CentralCritic.__init__")
