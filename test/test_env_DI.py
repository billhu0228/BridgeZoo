import bridgezoo.cablebridge_di.envs
import gymnasium as gym

if __name__ == '__main__':
    env_kwargs = dict(
        beam_e=20e8,
        beam_w=20.0,
        beam_h=1.0,
        num_strands=100,
        stress_init=1000,
        num_cables_per_side=12,
        middle_spacing=10,
        outside_spacing=8,
        end_to_first_spacing=4,
        center_to_adjacent_spacing=2,
        vertical_spacing=2,
        anchor_height=40,
        max_cycles=3,
        render_mode="human",
        DEF_SCALE=10,
        FPS=1,
    )

    env = gym.make('cable_bridge_di-v2', **env_kwargs)

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    while True:
        random_action = env.action_space.sample()
        random_action.fill(2)
        s, r, d, t, _ = env.step(random_action)
        # print(s)
        # print(random_action.values()[0].shape)
        if t or d:
            break
    env.close()
