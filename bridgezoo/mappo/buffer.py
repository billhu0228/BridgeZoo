"""Rollout 缓冲与 GAE(λ) 优势估计（里程碑 M3）。

存储并行环境采样的 (obs, state, action, mask, logprob, reward, done, value)，在每次
更新前计算 GAE 优势与回报。合作设定下所有智能体共享团队奖励与同一 critic 价值。

参见 ``docs/DESIGN_MAPPO.md`` 第 6.2、6.3 节。
"""

from __future__ import annotations

from bridgezoo.mappo.config import MappoConfig


class RolloutBuffer:
    """按 (rollout_len, num_envs, num_agents, ...) 组织的采样缓冲（骨架）。

    TODO(M3):
      - __init__: 预分配张量。
      - add(...): 写入一步采样。
      - compute_gae(last_value, gamma, lam): 回填 advantages / returns。
      - minibatches(num_minibatches): 产出打平后的 minibatch 迭代器。
    """

    def __init__(self, cfg: MappoConfig, obs_dim: int, state_dim: int, action_dim: int, num_agents: int):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        raise NotImplementedError("TODO(M3): RolloutBuffer.__init__")
