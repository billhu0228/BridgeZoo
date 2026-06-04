"""自写精简 MAPPO 算法子包（CTDE：集中训练、分散执行）。

- :mod:`bridgezoo.mappo.config` —— 超参数 dataclass（已实现）。
- :mod:`bridgezoo.mappo.actor_critic` —— 共享 actor（去中心化，含 action mask）
  + 中心化 critic（看全局状态）。
- :mod:`bridgezoo.mappo.buffer` —— rollout 缓冲 + GAE(λ)。
- :mod:`bridgezoo.mappo.trainer` —— 训练主循环（PPO clip、value clip、熵正则、
  优势归一化、并行环境采样、日志与 checkpoint）。

实现要点参考 MAPPO（Yu et al., 2022）。参见 ``docs/DESIGN_MAPPO.md`` 第 6 节。
"""

from bridgezoo.mappo.config import MappoConfig

__all__ = ["MappoConfig", "actor_critic", "buffer", "trainer"]
