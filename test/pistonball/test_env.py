from bridgezoo import pistonball_v6
from pettingzoo.test import api_test

if __name__ == "__main__":
    env = pistonball_v6.env(render_mode="human")
    env.reset()
    api_test(env, num_cycles=1000, verbose_progress=False)
