import numpy as np

import bridgezoo.cablebridge_di.envs
import gymnasium as gym

if __name__ == '__main__':
    env_kwargs = dict(
        beam_e=3.0e10,
        beam_w=20.0,
        beam_h=0.6,
        strands_init=50,
        stress_init=1000,
        delta_y=0.1,
        num_cables_per_side=12,
        middle_spacing=8,
        outside_spacing=8,
        end_to_first_spacing=4,
        center_to_adjacent_spacing=1,
        vertical_spacing=2,
        anchor_height=40,
        max_cycles=200,
        render_mode="text",
        DEF_SCALE=10,
        FPS=1,
    )

    env = gym.make('cable_bridge_di-v1', **env_kwargs)

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    for i in range(10):
        random_action = env.action_space.sample()
        s, r, d, t, info = env.step(random_action)
        if t or d:
            break
    env.close()
