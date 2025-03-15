import numpy as np
from rebar import env, EnvConfig, BoundaryShape

def test_random_policy(config: EnvConfig = None):
    """测试随机策略"""
    if config is None:
        config = EnvConfig(render_mode="human")
    
    # 创建环境
    env_instance = env(config=config)
    
    # 重置环境
    observations = env_instance.reset()
    
    # 运行环境
    for step in range(500):
        # 选择当前智能体
        agent = env_instance.agent_selection
        
        # 随机选择动作
        action = env_instance.action_spaces[agent].sample()
        
        # 执行动作
        observations = env_instance.step(action)
        
        # 如果所有智能体都结束了，则结束模拟
        if all(env_instance.terminations.values()) or all(env_instance.truncations.values()):
            break
    
    env_instance.close()

def test_intelligent_policy(config: EnvConfig = None):
    """测试一个智能策略，尝试使坐标平方最大化，同时满足约束条件"""
    if config is None:
        config = EnvConfig(render_mode="human")
    
    # 创建环境
    env_instance = env(config=config)
    
    # 重置环境
    observations = env_instance.reset()
    
    # 运行环境
    for step in range(500):
        # 选择当前智能体
        agent = env_instance.agent_selection
        agent_idx = int(agent.split('_')[1])
        
        # 获取观察
        obs = observations[agent]
        # 重塑观察为 (num_agents, 2) 的数组
        positions = obs.reshape(-1, 2)
        my_pos = positions[agent_idx]
        other_positions = np.delete(positions, agent_idx, axis=0)
        
        # 计算与其他智能体的最小距离
        min_distance = float('inf')
        nearest_pos = None
        for other_pos in other_positions:
            dist = np.linalg.norm(my_pos - other_pos)
            if dist < min_distance:
                min_distance = dist
                nearest_pos = other_pos
        
        # 策略：
        # 1. 优先确保最小距离约束
        # 2. 然后尝试使自己的坐标平方和最大化，但不超过边界
        
        if min_distance < env_instance.min_distance:
            # 如果距离太近，向远离最近智能体的方向移动
            diff = my_pos - nearest_pos
            if abs(diff[0]) > abs(diff[1]):  # x方向差异更大
                if diff[0] > 0:
                    action = 1  # 向右移动
                else:
                    action = 0  # 向左移动
            else:  # y方向差异更大
                if diff[1] > 0:
                    action = 2  # 向上移动
                else:
                    action = 3  # 向下移动
        else:
            # 距离约束已满足，尝试最大化坐标的平方
            if abs(my_pos[0]) < env_instance.config.max_absolute_x - 0.5 or abs(my_pos[1]) < env_instance.config.max_absolute_y - 0.5:
                # 选择可以移动的方向
                possible_moves = []
                if abs(my_pos[0]) < env_instance.config.max_absolute_x - 0.5:
                    if my_pos[0] > 0:
                        possible_moves.append(1)  # 向右
                    else:
                        possible_moves.append(0)  # 向左
                if abs(my_pos[1]) < env_instance.config.max_absolute_y - 0.5:
                    if my_pos[1] > 0:
                        possible_moves.append(2)  # 向上
                    else:
                        possible_moves.append(3)  # 向下
                
                if possible_moves:
                    # 随机选择一个可行的移动方向
                    action = np.random.choice(possible_moves)
                else:
                    action = 4  # 不动
            elif abs(my_pos[0]) > env_instance.config.max_absolute_x or abs(my_pos[1]) > env_instance.config.max_absolute_y:
                # 如果超过边界，向中心移动
                if abs(my_pos[0]) > env_instance.config.max_absolute_x:
                    action = 0 if my_pos[0] > 0 else 1
                else:
                    action = 3 if my_pos[1] > 0 else 2
            else:
                # 已经接近边界，保持不动
                action = 4
        
        # 执行动作
        observations = env_instance.step(action)
        
        # 如果所有智能体都结束了，则结束模拟
        if all(env_instance.terminations.values()) or all(env_instance.truncations.values()):
            break
    
    env_instance.close()

if __name__ == "__main__":
    # 测试不同的边界形状
    
    # 基础配置
    base_config = dict(
        num_agents=5,
        render_mode="human",
        render_fps=60,
        max_steps=1000,
        move_size=0.2,
        window_width=1024,
        window_height=768,
        max_absolute_x=9.0,        # 所有形状通用的x范围
        max_absolute_y=4.0,        # 所有形状正常区域的y范围
        normal_x=-8.0,              # T型和H型共用的正常区域x范围（T型为负，H型为正）
        narrow_y=2.0,               # T型和H型共用的窄区域y范围
        cover=0.3
    )
    
    # 4. 圆形边界
    print("\n测试圆形边界...")
    circle_config = EnvConfig(
        **base_config,
        boundary_shape=BoundaryShape.RECTANGLE,
    )
    # test_intelligent_policy(circle_config)
    test_random_policy(circle_config)

