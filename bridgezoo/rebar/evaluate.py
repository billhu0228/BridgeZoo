import numpy as np
from stable_baselines3 import SAC
from rebar import env, EnvConfig, BoundaryShape
from train import MultiAgentEnvWrapper  # 导入包装器
import os
import time

def evaluate_model(
    model_path: str,
    num_episodes: int = 10,
    num_agents: int = 5,
    render: bool = True,
    boundary_shape: BoundaryShape = BoundaryShape.RECTANGLE
):
    """评估训练好的模型"""
    # 创建评估环境并包装
    base_env = env(
        config=EnvConfig(
            render_mode="human" if render else None,
            num_agents=num_agents,
            boundary_shape=boundary_shape,
            normal_x=-3.0,
        )
    )
    eval_env = MultiAgentEnvWrapper(
        num_agents=num_agents,
        boundary_shape=boundary_shape
    )
    eval_env.env = base_env  # 使用带渲染的环境替换原环境
    
    # 加载模型
    model = SAC.load(model_path)
    
    # 评估指标
    episode_rewards = []
    min_distances = []
    boundary_usages = []
    violation_rates = []
    
    print("\n开始评估...")
    for episode in range(num_episodes):
        obs, _ = eval_env.reset()
        episode_reward = 0
        episode_min_distances = []
        episode_boundary_usages = []
        episode_violations = []
        done = False
        
        while not done:
            # 模型预测动作
            action = model.predict(obs, deterministic=True)[0]
            
            # 执行动作
            obs, reward, term, trunc, _ = eval_env.step(action)
            episode_reward += reward
            done = term or trunc
            
            # 计算评估指标
            positions = eval_env.env.current_positions
            
            # 最小距离
            min_dist = float('inf')
            num_violations = 0
            num_pairs = 0
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    min_dist = min(min_dist, dist)
                    if dist < eval_env.env.min_distance:
                        num_violations += 1
                    num_pairs += 1
            
            episode_min_distances.append(min_dist)
            episode_violations.append(num_violations / num_pairs if num_pairs > 0 else 0)
            
            # 边界利用率
            max_x = eval_env.env.config.max_absolute_x
            max_y = eval_env.env.config.max_absolute_y
            boundary_usage = np.mean([
                (abs(pos[0])/max_x + abs(pos[1])/max_y)/2 
                for pos in positions
            ])
            episode_boundary_usages.append(boundary_usage)
            
            if render:
                eval_env.env.render()  # 显式调用渲染
                # time.sleep(0.01)
        
        # 记录每个回合的指标
        episode_rewards.append(episode_reward)
        min_distances.append(np.mean(episode_min_distances))
        boundary_usages.append(np.mean(episode_boundary_usages))
        violation_rates.append(np.mean(episode_violations))
        
        print(f"\n回合 {episode + 1}/{num_episodes} 评估结果：")
        print(f"总奖励: {episode_reward:.2f}")
        print(f"平均最小距离: {min_distances[-1]:.2f}")
        print(f"平均边界利用率: {boundary_usages[-1]:.2f}")
        print(f"平均违反率: {violation_rates[-1]:.2%}")
    
    # 打印总体评估结果
    print("\n=== 总体评估结果 ===")
    print(f"平均回合奖励: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"平均最小距离: {np.mean(min_distances):.2f} ± {np.std(min_distances):.2f}")
    print(f"平均边界利用率: {np.mean(boundary_usages):.2f} ± {np.std(boundary_usages):.2f}")
    print(f"平均违反率: {np.mean(violation_rates):.2%} ± {np.std(violation_rates):.2%}")
    
    eval_env.close()

if __name__ == "__main__":
    # 模型路径
    model_path = "logs/sac_model_20250316_114943_interrupted.zip"
    
    # 评估参数
    NUM_EPISODES = 10  # 评估回合数
    NUM_AGENTS = 4     # 智能体数量
    RENDER = True      # 是否渲染
    
    # 运行评估
    evaluate_model(
        model_path=model_path,
        num_episodes=NUM_EPISODES,
        num_agents=NUM_AGENTS,
        render=RENDER,
        boundary_shape=BoundaryShape.RECTANGLE
    )
