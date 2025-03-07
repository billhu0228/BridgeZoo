import numpy as np
from gymnasium import spaces


class CableMoveY:
    def __init__(self, num_strands, stress_init, dy):
        # self.obs_dim = obs_dim
        self.stress_init = stress_init
        self.stress_exert = stress_init

        self.stress_after = np.nan
        self.action_history = []
        self.num_strands = num_strands

        self.position = 3
        self.delta_y = dy
        self.deform = None
        self.action = None
        self._reward = None

    def __str__(self):
        return "Cable(e:%i,a:%i)" % (self.stress_exert, self.stress_after)

    @property
    def action_space(self):
        stress_act_space = spaces.Discrete(3)
        return stress_act_space

    def reset(self):
        action = self.action_space.sample()
        self.stress_exert = self.stress_init
        self.stress_after = np.nan
        self.action = action
        self.position = 0
        self.deform = None

    def step(self, action):
        a = action - 1
        self.action_history.append(action)
        if len(self.action_history) == 4:
            debug = 1
        # self.stress_exert = self.stress_after + a * self.delta_y
        self.position = self.position + a * self.delta_y
        return

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        self.deform = deform

    def done(self):
        return False

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
        self._reward = -abs(self.deform) * 100
        return self._reward

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
