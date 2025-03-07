import numpy as np
import stable_baselines3
from stable_baselines3.common.env_checker import check_env
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.env_checker import check_env
from pettingzoo.test import parallel_api_test
from pettingzoo.test import api_test

from bridgezoo import cablebridge_v1

if __name__ == "__main__":
    env_fn = cablebridge_v1
    env_kwargs = dict(
        beam_w=10.0,
        beam_h=1.0,
        num_cables_per_side=6,
        anchor_height=20,
        max_cycles=100,
        render_mode="",
        DEF_SCALE=10,
        FPS=20,
    )
    # env_parallel = env_fn.parallel_env(**env_kwargs)
    # # Testing the parallel algorithm alone
    # parallel_api_test(env_parallel)

    env = cablebridge_v1.env(**env_kwargs)
    api_test(env, num_cycles=1000, verbose_progress=False)
