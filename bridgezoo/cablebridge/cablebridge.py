from gymnasium.utils import EzPickle

from pettingzoo import AECEnv
from bridgezoo.cablebridge.cablebridge_base import CableBridgeBase as _env
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn


def env(**kwargs):
    the_env = raw_env(**kwargs)
    # the_env = wrappers.ClipOutOfBoundsWrapper(the_env)
    the_env = wrappers.OrderEnforcingWrapper(the_env)
    return the_env


parallel_env = parallel_wrapper_fn(env)


class raw_env(AECEnv, EzPickle):
    metadata = {
        "render_modes": ["human", "text"],
        "name": "cablebridge_v1",
        "is_parallelizable": True,
    }

    def __init__(self, *args, **kwargs):
        EzPickle.__init__(self, *args, **kwargs)
        AECEnv.__init__(self)
        self.env = _env(*args, **kwargs)
        self.agents = ["cable_" + str(r) for r in range(self.env.num_agents)]
        self.possible_agents = self.agents[:]
        self.agent_name_mapping = dict(zip(self.agents, list(range(self.num_agents))))
        self._agent_selector = agent_selector(self.agents)
        self.has_reset = False
        self.render_mode = self.env.render_mode

    def observation_space(self, agent):
        return self.env.observation_space[self.agent_name_mapping[agent]]

    def action_space(self, agent):
        return self.env.action_space[self.agent_name_mapping[agent]]

    # def convert_to_dict(self, list_of_list):
    #     return dict(zip(self.agents, list_of_list))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.seed(seed=seed)
        self.has_reset = True
        self.env.reset()
        self.agents = self.possible_agents[:]
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.next()
        self.rewards = dict(zip(self.agents, [0 for _ in self.agents]))
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

    def close(self):
        if self.has_reset:
            self.env.close()

    def render(self):
        return self.env.render()

    def step(self, action):
        if (
                self.terminations[self.agent_selection]
                or self.truncations[self.agent_selection]
        ):
            # self._was_dead_step(action)
            self.agents = []
            self.terminations = {}
            self.truncations = {}
            self.rewards = {}
            self._cumulative_rewards = {}
            self.infos = {}  #
            return

        agent = self.agent_selection
        is_last = self._agent_selector.is_last()
        self.env.step(action, self.agent_name_mapping[agent], is_last)
        self.infos = {agent: {} for agent in self.agents if not self.truncations[agent]}

        # if self.terminations[agent] or self.truncations[agent]:
        #     del self.infos[agent]
        if is_last:
            for r in self.rewards:
                self.rewards[r] += self.env.last_rewards[self.agent_name_mapping[r]]

        if self.env.frames >= self.env.max_cycles:
            self.truncations = dict(zip(self.agents, [True for _ in self.agents]))
        else:
            self.terminations = dict(zip(self.agents, self.env.last_dones))

        self._cumulative_rewards[self.agent_selection] = 0
        self.agent_selection = self._agent_selector.next()
        self._accumulate_rewards()
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == 'text':
            if True:
                cable_stress = [c.stress_init for c in self.env.cables.values()]
                beam_pos = [c.deform for c in self.env.cables.values()]
                cable_stress_after = [c.stress_after for c in self.env.cables.values()]
                cable_nos = [c.num_strands for c in self.env.cables.values()]
                actions = [c.action for c in self.env.cables.values()]
                # print(env.unwrapped.env.cables, obs_str)
                # print("BeamE:%.2e | Wg= %.2e | %s | %s" % (self.env.beam_E, self.env.wg, self.env.cables, obs_str))
                text_render = ""
                for s in cable_stress:
                    text_render += "%6i" % s
                text_render += "  |  "
                for s in beam_pos:
                    text_render += "%6.3f  " % s
                text_render += "  |  "
                for s in cable_stress_after:
                    text_render += "%6i" % s
                text_render += "  |  "
                for s in cable_nos:
                    text_render += "%3i" % s
                text_render += "  |  "
                for s in actions:
                    if s is None:
                        text_render += "  \033[91m×\033[0m"
                    else:
                        text_render += "%3i" % s

                print(text_render)
                # print(np.round(cable_stress_after, 0))

    def observe(self, agent):
        return self.env.observe(self.agent_name_mapping[agent])
