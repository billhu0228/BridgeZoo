import numpy as np
import pygame
from gymnasium import spaces


class CableObj:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.stress_init = 0
        self.num_strands = 0
        self.stress_after = np.nan
        self._reset()
        self.deform = None
        self.action = None
        self._reward = None

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.float32(-10),
            high=np.float32(10),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.MultiDiscrete([3, ] * 2)

    def _reset(self):
        self.stress_init = 500
        self.num_strands = 20
        self.stress_after = np.nan
        self.deform = None
        self.action = None

    def step(self, stress_adj, num_adj=1):
        self.action = (stress_adj, num_adj)
        self.stress_init = self.stress_init + (stress_adj - 1) * 100
        self.num_strands = self.num_strands + (num_adj - 1)

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        self.deform = deform

    def done(self):
        return self.num_strands <= 0 or self.stress_init <= 0 or self.stress_init >= 2000

    def reward2(self):
        EPS = 0.01
        if abs(self.deform) <= EPS:
            s_score = {0: -1, 1: 2, 2: -1}
        else:
            if self.deform > 0:
                s_score = {0: 2, 1: -1, 2: -1}
            else:
                s_score = {0: -1, 1: -1, 2: 2}
        if self.stress_after <= 300:
            n_score = {0: 1, 1: 0, 2: -1}
        elif self.stress_after <= 800:
            n_score = {0: 0, 1: 1, 2: 0}
        else:
            n_score = {0: -2, 1: -1, 2: +2}
        self._reward = s_score[self.action[0]] + n_score[self.action[1]]
        return self._reward

    def reward3(self):
        self._reward = -abs(self.deform[-1]) * 100
        return self._reward


class CableAgent:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.stress_init = 0
        self.num_strands = 0
        self.stress_step = 20
        self.num_step = 2
        self.stress_after = np.nan
        self.stress_min = 100
        self.stress_max = 800
        self._reset()
        self.deform = None
        self.action = None
        self._reward = None

    @property
    def observation_space(self):
        return spaces.Box(
            low=np.float32(-10),
            high=np.float32(10),
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @property
    def action_space(self):
        return spaces.MultiDiscrete([3, ] * 2)

    def _reset(self):
        self.stress_init = 100  # 500
        self.num_strands = 60  # 20
        self.stress_after = np.nan
        self.deform = None
        self.action = None

    def step(self, stress_adj, num_adj=1):
        self.action = (stress_adj, num_adj)
        self.stress_init = self.stress_init + (stress_adj - 1) * 100
        self.num_strands = self.num_strands + (num_adj - 1)

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        self.deform = deform

    def done(self):
        return self.num_strands <= 0 or self.stress_init <= 0 or self.stress_init >= 2000

    def reward(self):
        EPS = 0.01
        if abs(self.deform) <= EPS:
            s_score = {0: -1, 1: 2, 2: -1}
        else:
            if self.deform > 0:
                s_score = {0: 2, 1: -1, 2: -1}
            else:
                s_score = {0: -1, 1: -1, 2: 2}
        if self.stress_after <= 450:
            n_score = {0: 1, 1: 0, 2: -1}
        elif self.stress_after <= 550:
            n_score = {0: 0, 1: 1, 2: 0}
        else:
            n_score = {0: -2, 1: -1, 2: +2}
        if self.action is None:
            debug = 1
        self._reward = s_score[self.action[0]] + n_score[self.action[1]]
        return self._reward

    def reward3(self):
        self._reward = -abs(self.deform[-1]) * 100
        return self._reward

    def reward2(self):
        self._reward = -abs(self.deform) * 1000 - abs(self.stress_after - 500) * 1e-1
        return self._reward

    def __repr__(self):
        return "S:%4.0f -> %4.0f | N:%2.0f" % (self.stress_init, self.stress_after, self.num_strands)
