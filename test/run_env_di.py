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
        max_cycles=2,
        render_mode="human",
        DEF_SCALE=10,
        FPS=1,
    )

    env = gym.make('cable_bridge_di-v1', **env_kwargs)

    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    while True:
        random_action = env.action_space.sample()
        # 　random_action.fill(2)
        s, r, d, t, _ = env.step(random_action)
        # cable_y = [0., ] * 11 + [0.1, ]
        # 　print([np.round(s, 1) for s in cable_y])
        # 　print([np.round(s, 0) for s in cable_stress_after])
        # 　print([f"{np.round(s * 1000, 0):.0f}" for s in beam_pos[1:self.num_cables_per_side + 1]])
        rep = "-" * 100 + '\n'
        rep += "输入位置："
        rep += str([f"{np.round(a * 1000, 0):.0f}" for a in s[0:12]])
        rep += "\n"
        rep += "平衡位置："
        rep += str([f"{np.round(a * 1000, 0):.0f}" for a in s[12:24]])
        rep += "\n"
        rep += "平衡索力："
        rep += str([f"{np.round(a, 0):.0f}" for a in s[24:]])
        rep += "\n"
        rep += f"得分：{r:.0f}\n"
        rep += "-" * 100 + '\n'
        print(rep)
        if t or d:
            break
    env.close()
