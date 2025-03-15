from __future__ import annotations

import glob
import os
import subprocess
import time

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.ppo import MlpPolicy
from bridgezoo import cablebridge_v2,pistonball_v6


def train(env_func, log_dir, steps: int = 10000, seed: int | None = None, with_board=False, **kwargs):
    env = env_func.parallel_env(**kwargs)
    env.reset()
    os.makedirs(log_dir, exist_ok=True)
    if with_board:
        tensorboard_process = subprocess.Popen(["tensorboard", "--logdir", log_dir, "--host", "0.0.0.0", "--port", "1238"])
    print(f"{str(env.metadata['name'])} 环境训练开始 ... ")
    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=8, base_class="stable_baselines3")
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    model = PPO(
        MlpPolicy,
        env,
        policy_kwargs={"net_arch": [128, 128]},  # 增加神经网络层数
        verbose=2,
        learning_rate=5e-3,
        batch_size=512,
        device='cpu',
        tensorboard_log=log_dir,
    )
    try:
        model.learn(total_timesteps=steps)
    except KeyboardInterrupt:
        # filename = f"{env.unwrapped.metadata.get('name')}_{env.}_{time.strftime('%Y%m%d-%H%M%S')}"
        filename = "CableBridge_C%s_%s" % (len(env.unwrapped.agents), time.strftime('%Y%m%d-%H%M%S'))
        model.save(os.path.join(log_dir, filename))
        print("模型已保存, 训练结束.")
        env.close()
        if with_board:
            tensorboard_process.terminate()

    filename = f"{env.unwrapped.metadata.get('name')}_{time.strftime('%Y%m%d-%H%M%S')}"
    model.save(os.path.join(log_dir, filename))
    print("模型已保存, 训练结束.")
    env.close()
    if with_board:
        tensorboard_process.terminate()


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
                act = env.unwrapped.action_spaces[agent].sample()
            env.step(act)
        print(rewards)
    env.close()


def evaluate(env_func, policy_folder, policy_name="", num_games: int = 1, use_policy=False, **kwargs):
    env = env_func.env(**kwargs)
    latest_policy = ""
    if use_policy:
        if policy_name == "":
            try:
                file = os.path.join(policy_folder, f"{env.metadata['name']}*.zip")
                latest_policy = max(glob.glob(file), key=os.path.getctime)
            except ValueError:
                print("Policy not found.")
                exit(0)
        else:
            latest_policy = policy_name
        model = PPO.load(latest_policy)
    else:
        model = None
    withPolicy = "载入策略%s" % latest_policy if use_policy else "无策略"
    print(f"\n 评估环境： {str(env.metadata['name'])} (num_games={num_games}) | " + withPolicy)
    for i in range(num_games):
        info_s = "%i" % i
        rewards = {agent: 0 for agent in env.possible_agents}
        env.reset()
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            if termination or truncation:
                break
            else:
                if not use_policy:
                    act = env.unwrapped.action_space(agent).sample()
                else:
                    act = int(model.predict(obs, deterministic=False)[0])
            env.step(act)
            for a in env.agents:
                rewards[a] += env.rewards[a]
    env.close()
    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return


if __name__ == "__main__":
    env_fn = pistonball_v6
    # Train a model
    train(env_fn, '../train_folder/', steps=int(4e6), with_board=True)

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    # env_kwargs['render_mode'] = 'text'
    # env_kwargs['render_mode'] = 'human'
    # evaluate(env_fn, policy_folder="../train_folder", num_games=2, use_policy=False, **env_kwargs)
    # evaluate(env_fn, policy_folder="../train_folder", num_games=2, use_policy=False, **env_kwargs)
    # evaluate(env_fn, policy_folder="./train_folder", num_games=2, use_policy=True, **env_kwargs)

    # env_kwargs['render_mode'] = 'human'
    # run(env_fn, num_games=1, **env_kwargs)
