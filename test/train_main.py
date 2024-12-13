import subprocess

import numpy as np

import bridgezoo.cablebridge_di.envs
import gymnasium as gym
import os
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import time

from stable_baselines3.common.monitor import Monitor


def train(name, log_dir, with_board, **kwargs):
    env = gym.make(name, **kwargs)
    env = Monitor(env, log_dir)
    if with_board:
        tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--host", "0.0.0.0", "--port", "1235"])
    else:
        tensorboard_process = None
    model = PPO(
        MlpPolicy,
        env,
        learning_rate=5e-3,
        batch_size=512,
        # n_steps=1024,  # 增加 n_steps
        clip_range=0.1,  # 减小裁剪范围
        gae_lambda=0.95,  # 调整 GAE 参数
        device='cpu',
        verbose=2,
        tensorboard_log=log_dir,
    )
    try:
        model.learn(total_timesteps=100000)
        filename = "CableBridge_%s" % (time.strftime('%Y%m%d-%H%M%S'))
        model.save(os.path.join(log_dir, filename))
        print("训练完成, 模型已保存.")
    except KeyboardInterrupt:
        filename = "CableBridge_%s" % (time.strftime('%Y%m%d-%H%M%S'))
        model.save(os.path.join(log_dir, filename))
        print("提前终止, 模型已保存.")
    finally:
        env.close()
        _ = input("Press any key to exit..")
        if with_board:
            tensorboard_process.terminate()


def run(name, num_games, **kwargs):
    env = gym.make(name, **kwargs)
    obs, _ = env.reset()
    assert env.observation_space.contains(obs)
    for i in range(num_games):
        random_action = env.action_space.sample()
        s, r, d, t, info = env.step(random_action)
        if t or d:
            break
    env.close()


def evaluate(name, policy, num_games: int = 1, use_policy=False, **kwargs):
    env = gym.make(name, **kwargs)
    if use_policy:
        # policy = os.path.join(policy_folder, os.listdir(policy_folder)[0])
        model = PPO.load(policy)
    else:
        model = None
    # withPolicy = "载入策略%s" % latest_policy if use_policy else "无策略"
    # print(f"\n 评估环境： {str(env.metadata['name'])} (num_games={num_games}) | " + withPolicy)

    for i in range(num_games):
        rewards = []
        obs, _ = env.reset()
        while True:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs)
            obs, r, d, t, info = env.step(action)
            rewards.append(r)
            if t or d:
                break
        avg_reward = sum(rewards) / len(rewards)
        print(f"第{i + 1}次调索执行{len(rewards)}次，总得分：{sum(rewards)}，得分率{avg_reward:.3f}")
    env.close()
    return


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
        max_cycles=256,
        render_mode="empty",
        DEF_SCALE=20,
        FPS=1,
    )

    # run('cable_bridge_di-v1', 10, **env_kwargs)
    # train('cable_bridge_di-v1', "../bin/", with_board=True, **env_kwargs)
    evaluate('cable_bridge_di-v1', '../bin/CableBridge_20241212-154236.zip', 10, False, **env_kwargs)
    evaluate('cable_bridge_di-v1', '../bin/CableBridge_20241212-164638.zip', 10, True, **env_kwargs)
