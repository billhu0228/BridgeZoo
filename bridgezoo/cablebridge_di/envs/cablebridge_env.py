import copy
from typing import Any

import gymnasium as gym
import numpy as np
from ding.envs import BaseEnv, BaseEnvTimestep
from ding.torch_utils import to_ndarray
from ding.utils import ENV_REGISTRY
from easydict import EasyDict


@ENV_REGISTRY.register('cable_bridge_dix')
class CableBridgeDi(BaseEnv):
    config = dict(
        env_id='cable_bridge_di-v0',
        act_scale=True,
    )
    default_env_id = ['cable_bridge_di-v0', ]

    @classmethod
    def default_config(cls: type) -> EasyDict:
        cfg = EasyDict(copy.deepcopy(cls.config))
        cfg.cfg_type = cls.__name__ + 'Dict'
        return cfg

    def __init__(self, cfg: EasyDict) -> None:
        self._cfg = cfg
        self._env_id = cfg.env_id
        assert self._env_id in self.default_env_id
        self._act_scale = cfg.act_scale
        self._replay_path = None
        self._save_replay = False
        self._save_replay_count = 0
        self._init_flag = False

    def reset(self) -> np.ndarray:
        if not self._init_flag:
            self._env = gym.make(self._env_id)
            self._observation_space = self._env.observation_space
            self._action_space = self._env.action_space
            self._reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
            self._init_flag = True
        self._env.seed(0)
        self._eval_episode_return = 0
        obs, _ = self._env.reset()
        obs = to_ndarray(obs).astype(np.float32)
        return obs

    def close(self) -> None:
        if self._init_flag:
            self._env.close()
        self._init_flag = False

    def step(self, action: Any) -> 'BaseEnv.timestep':
        obs, rew, done, trunc, info = self._env.step(action)
        obs = to_ndarray(obs)
        obs = obs.astype(np.float32)
        rew = to_ndarray([rew])  # wrapped to be transferred to a numpy array with shape (1,)
        if isinstance(rew, list):
            rew = rew[0]
        assert isinstance(rew, np.ndarray) and rew.shape == (1,)
        self._eval_episode_return += rew.item()
        return BaseEnvTimestep(obs, rew, done, info)

    def seed(self, seed: int) -> None:
        self._seed = seed
        np.random.seed(self._seed)

    @property
    def observation_space(self) -> gym.spaces.Space:
        return self._observation_space

    @property
    def action_space(self) -> gym.spaces.Space:
        return self._action_space

    @property
    def reward_space(self) -> gym.spaces.Space:
        return self._reward_space

    def __repr__(self) -> str:
        pass
