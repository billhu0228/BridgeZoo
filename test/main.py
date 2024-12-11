import subprocess
from typing import Optional

import numpy as np

import bridgezoo.cablebridge_di.envs
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import time

from stable_baselines3.common.monitor import Monitor


def train(name, log_dir, **kwargs):
    # 创建Gym环境，并通过Monitor封装以记录训练过程
    env = gym.make(name, **kwargs)
    env = Monitor(env, log_dir)

    # 创建PPO模型，同时指定tensorboard_log参数以启用TensorBoard日志记录
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
    )

    # 开始训练，并在训练过程中自动记录到TensorBoard日志
    model.learn(total_timesteps=10000)

    env.close()


def run(name, num_games, **kwargs):
    kwargs['max_cycles'] = num_games
    env = gym.make(name, **kwargs)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    while True:
        random_action = env.action_space.sample()
        s, r, d, t, info = env.step(random_action)
        if t or d:
            break
    env.close()


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
        max_cycles=100,
        render_mode="text",
        DEF_SCALE=10,
        FPS=1,
    )

    run('cable_bridge_di-v1', 1, **env_kwargs)
    # train('cable_bridge_di-v1', "./", **env_kwargs)
