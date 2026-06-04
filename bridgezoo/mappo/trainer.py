"""MAPPO 训练主循环（里程碑 M3/M4）。

把环境、actor、critic、buffer 串起来：并行采样 → GAE → PPO 多轮 minibatch 更新
→ 日志（TensorBoard）→ 定期评估与 checkpoint。

训练循环伪码（详见 ``docs/DESIGN_MAPPO.md`` 第 6.3 节）::

    for update in range(num_updates):
        rollout = collect(envs, actor, critic)
        adv, ret = buffer.compute_gae(...)
        for epoch in range(ppo_epochs):
            for mb in buffer.minibatches(...):
                L = ppo_clip(actor, mb, adv) + vf_coef*value_loss(critic, mb, ret)
                     - ent_coef*entropy
                optimize(L)
        log(); maybe eval(); maybe save()
"""

from __future__ import annotations

from bridgezoo.mappo.config import MappoConfig


class MappoTrainer:
    """MAPPO 训练器（骨架）。

    TODO(M3/M4):
      - __init__: 构建向量化环境、Actor、CentralCritic、RolloutBuffer、优化器、logger。
      - collect(): 采样一个 rollout（含 action mask 与全局 state）。
      - update(): PPO 多轮 minibatch 更新（clip、value clip、熵、grad clip、KL 早停）。
      - learn(): 主循环（lr 退火、日志、评估、checkpoint）。
      - evaluate() / save() / load()。
    """

    def __init__(self, env_fn, cfg: MappoConfig):
        self.env_fn = env_fn
        self.cfg = cfg
        raise NotImplementedError("TODO(M3/M4): MappoTrainer.__init__")

    def learn(self):
        raise NotImplementedError("TODO(M3/M4): MappoTrainer.learn")
