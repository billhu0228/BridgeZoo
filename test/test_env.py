from bridgezoo import cablebridge_v1


def run(env_func, num_games, **kwargs):
    env = env_func.env(**kwargs)
    env.reset()
    rewards = {agent: 0 for agent in env.possible_agents}
    for k in range(num_games):
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = env.unwrapped.action_space(agent).sample()
            env.step(act)
        print(rewards)
    env.close()


if __name__ == "__main__":
    env_fn = cablebridge_v1
    env_kwargs = dict(
        beam_w=10.0,
        beam_h=1.0,
        num_cables_per_side=6,
        anchor_height=20,
        max_cycles=60,
        render_mode="",
        DEF_SCALE=10,
        FPS=60,
    )

    env_kwargs['render_mode'] = 'text'
    run(env_fn, num_games=2, **env_kwargs)
