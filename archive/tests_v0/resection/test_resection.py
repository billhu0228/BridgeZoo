import numpy as np
import pygame
from bridgezoo.resection_v1 import env


def test_random_policy():
    """测试随机策略"""
    # 创建环境
    env_instance = env(render_mode="human")

    # 重置环境
    observations = env_instance.reset()

    # 跟踪每个智能体的位置和距离
    positions_agent0 = []
    positions_agent1 = []
    distances = []
    rewards_agent0 = []
    rewards_agent1 = []

    # 运行环境
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

        if not running:
            break

        # 选择当前智能体
        agent = env_instance.agent_selection

        # 随机选择动作
        action = env_instance.action_spaces[agent].sample()

        # 执行动作
        observations = env_instance.step(action)

        # 记录位置和距离
        if agent == "agent_0":
            positions_agent0.append(env_instance.current_positions[0])
            rewards_agent0.append(env_instance.rewards[agent])
        else:
            positions_agent1.append(env_instance.current_positions[1])
            rewards_agent1.append(env_instance.rewards[agent])
            distances.append(
                abs(env_instance.current_positions[0] - env_instance.current_positions[1]))

        # 控制帧率
        clock.tick(env_instance.metadata["render_fps"])

        # 如果所有智能体都结束了，则结束模拟
        if all(env_instance.terminations.values()) or all(env_instance.truncations.values()):
            break

    env_instance.close()


def test_intelligent_policy():
    """测试一个智能策略，尝试使坐标平方最大化，同时满足约束条件"""
    # 创建环境
    env_instance = env(render_mode="human")

    # 重置环境
    observations = env_instance.reset()

    # 跟踪每个智能体的位置和距离
    positions_agent0 = []
    positions_agent1 = []
    distances = []
    rewards_agent0 = []
    rewards_agent1 = []

    # 运行环境
    running = True
    clock = pygame.time.Clock()
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                    break

        if not running:
            break

        # 选择当前智能体
        agent = env_instance.agent_selection
        agent_idx = 0 if agent == "agent_0" else 1

        # 获取观察
        obs = observations[agent]

        # 计算当前距离和位置
        my_pos = obs[0]
        other_pos = obs[1]
        current_distance = abs(my_pos - other_pos)

        # 策略：
        # 1. 优先确保最小距离约束
        # 2. 然后尝试使自己的绝对值尽可能接近但不超过最大值

        if current_distance < env_instance.min_distance:
            # 如果距离太近，优先满足距离约束
            if my_pos < other_pos:  # 如果当前智能体在左边
                action = 0  # 向左移动
            else:
                action = 2  # 向右移动
        else:
            # 距离约束已满足，尝试最大化坐标的平方
            if abs(my_pos) < env_instance.max_absolute_value - 0.5:
                # 如果还没达到最大值限制，向绝对值更大的方向移动
                if my_pos > 0:
                    action = 2  # 向右移动
                elif my_pos < 0:
                    action = 0  # 向左移动
                else:
                    # 如果在原点，根据另一个智能体位置决定移动方向
                    if other_pos >= 0:
                        action = 0  # 向左移动
                    else:
                        action = 2  # 向右移动
            elif abs(my_pos) > env_instance.max_absolute_value:
                # 如果超过最大值限制，向原点方向移动
                if my_pos > 0:
                    action = 0  # 向左移动
                else:
                    action = 2  # 向右移动
            else:
                # 已经接近最大值，保持不动
                action = 1

        # 执行动作
        observations = env_instance.step(action)

        # 记录位置和距离
        if agent == "agent_0":
            positions_agent0.append(env_instance.current_positions[0])
            rewards_agent0.append(env_instance.rewards[agent])
        else:
            positions_agent1.append(env_instance.current_positions[1])
            rewards_agent1.append(env_instance.rewards[agent])
            distances.append(
                abs(env_instance.current_positions[0] - env_instance.current_positions[1]))

        # 控制帧率
        clock.tick(env_instance.metadata["render_fps"])

        # 如果所有智能体都结束了，则结束模拟
        if all(env_instance.terminations.values()) or all(env_instance.truncations.values()):
            break

    env_instance.close()


if __name__ == "__main__":
    print("Testing random policy...")
    test_random_policy()
    print("\nTesting intelligent policy...")
    test_intelligent_policy()
