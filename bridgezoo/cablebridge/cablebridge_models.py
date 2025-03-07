import numpy as np
from gymnasium import spaces
from functools import cached_property


class CableAgent:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.stress_init = 1000
        self.stress_after = 1000
        self.num_strands = 20
        self.stress_step = 20
        self.num_step = 2

        # self.stress_min = 100
        # self.stress_max = 800
        self.observation = []
        self.deform = None
        self.action = None
        self.the_reward = None
        self.real_action = None
        self.reset()

    def reset(self):
        self.stress_init = 1000
        self.stress_after = 1000
        self.num_strands = 20
        self.stress_step = 20
        self.num_step = 2
        self.deform = None
        self.action = None
        self.the_reward = None
        self.real_action = None

    @cached_property
    def action_map(self):
        ret = {
            0: {'s': 0, 'n': 0},
            1: {'s': 1, 'n': 0},
            2: {'s': 2, 'n': 0},
            3: {'s': 0, 'n': 1},
            4: {'s': 1, 'n': 1},
            5: {'s': 2, 'n': 1},
            6: {'s': 0, 'n': 2},
            7: {'s': 1, 'n': 2},
            8: {'s': 2, 'n': 2},
        }
        return ret

    @cached_property
    def observation_space(self):
        return spaces.Box(
            low=-np.inf,
            high=+np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

    @cached_property
    def action_space(self):
        return spaces.Discrete(9)

    def step(self, act):
        self.action = act
        self.real_action = self.action_map[act]
        self.stress_init = self.stress_after + (self.real_action['s'] - 1) * self.stress_step
        self.num_strands = self.num_strands + (self.real_action['n'] - 1) * self.num_step

    def update(self, balance_stress, deform):
        self.stress_after = balance_stress
        self.stress_init = balance_stress
        self.deform = deform
        self.observation = [deform, balance_stress]

    def done(self):
        return bool(self.num_strands <= self.num_step or self.stress_after <= self.stress_step)

    def reward(self):
        EPS = 0.01
        if abs(self.observation[0]) <= EPS:
            s_score = {0: -1, 1: 2, 2: -1}
        else:
            if self.observation[0] > 0:
                s_score = {0: 2, 1: -1, 2: -1}
            else:
                s_score = {0: -1, 1: -1, 2: 2}
        if self.observation[1] <= 450:
            n_score = {0: 1, 1: 0, 2: -1}
        elif self.observation[1] <= 550:
            n_score = {0: 0, 1: 1, 2: 0}
        else:
            n_score = {0: -2, 1: -1, 2: +2}
        # if self.action is None:
        #     debug = 1
        self.the_reward = s_score[self.real_action['s']] + n_score[self.real_action['n']] + 1
        return self.the_reward

    def __repr__(self):
        return "S:%4.0f -> %4.0f | N:%2.0f" % (self.stress_init, self.stress_after, self.num_strands)
