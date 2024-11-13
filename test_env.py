import bridgezoo.cablebridge_di.envs
import gymnasium as gym

env = gym.make('cable_bridge_di-v0',
               render_mode='human',
               DEF_SCALE=10,
               max_cycles=100,
               FPS=3, )

obs, _ = env.reset()
assert env.observation_space.contains(obs)
while True:
    random_action = env.action_space.sample()
    s, r, d, t, _ = env.step(random_action)
    # print(random_action.values()[0].shape)
    if t or d:
        break
env.close()
