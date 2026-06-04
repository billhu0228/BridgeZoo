from rebar import EnvConfig, BoundaryShape
from rebar.gym_env import RebarGymEnv
import time
import numpy as np

def test_boundary(boundary_shape: BoundaryShape):
    """测试指定边界形状的边界处理函数
    
    参数:
        boundary_shape: 要测试的边界形状
    """
    print(f"\n开始测试 {boundary_shape.value} 边界...")
    
    # 创建环境配置
    config = EnvConfig(
        num_agents=1,  # 只使用一个智能体
        render_mode="human",  # 使用可视化模式
        boundary_shape=boundary_shape,
        max_absolute_x=9.0,
        max_absolute_y=4.0,
        normal_x=3.0,
        narrow_y=2.0,
        move_size=0.1
    )
    
    # 创建Gym环境
    env = RebarGymEnv(config)
    
    # 重置环境
    obs, info = env.reset()
    
    # 根据不同的边界形状定义不同的测试动作序列
    if boundary_shape == BoundaryShape.RECTANGLE:
        # 矩形边界测试动作序列：顺时针绕边界移动
        action_sequence = [
            2,  # 上
            1,  # 右
            3,  # 下
            1,  # 右
            3,  # 下
            0,  # 左
            2,  # 上
            0,  # 左
        ]
    elif boundary_shape == BoundaryShape.T_SHAPE:
        # T型边界测试动作序列：先测试左侧区域，再测试右侧窄区域
        action_sequence = [
            2,  # 上
            1,  # 右
            3,  # 下
            1,  # 右
            3,  # 下
            0,  # 左
            2,  # 上
            0,  # 左
            1,  # 右
            2,  # 上
            3,  # 下
            0,  # 左
        ]
    elif boundary_shape == BoundaryShape.H_SHAPE:
        # H型边界测试动作序列：测试所有区域
        action_sequence = [
            2,  # 上
            1,  # 右
            3,  # 下
            1,  # 右
            3,  # 下
            0,  # 左
            2,  # 上
            0,  # 左
            1,  # 右
            2,  # 上
            3,  # 下
            0,  # 左
        ]
    else:  # CIRCLE
        # 圆形边界测试动作序列：顺时针绕边界移动
        action_sequence = [
            2,  # 上
            1,  # 右
            3,  # 下
            1,  # 右
            3,  # 下
            0,  # 左
            2,  # 上
            0,  # 左
        ]
    
    # 执行动作序列
    for i, action in enumerate(action_sequence):
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 打印当前状态
        print(f"\n执行第 {i+1} 个动作: {action}")
        print(f"当前位置: {env.env.current_positions[0]}")
        print(f"奖励值: {reward}")
        
        # 等待一段时间以便观察
        time.sleep(0.5)
        
        # 如果环境结束，退出循环
        if terminated:
            print("\n环境已结束")
            break
    
    # 关闭环境
    env.close()
    print(f"\n{boundary_shape.value} 边界测试完成")

def test_t_boundary():
    """测试T型边界的边界处理函数，使用2个智能体"""
    print("\n开始测试 T型边界（2个智能体）...")
    
    # 创建环境配置
    config = EnvConfig(
        num_agents=2,  # 使用2个智能体
        render_mode="human",  # 使用可视化模式
        boundary_shape=BoundaryShape.T_SHAPE,
        max_absolute_x=9.0,
        max_absolute_y=4.0,
        normal_x=3.0,
        narrow_y=2.0,
        move_size=0.1,
        random_seed=42  # 设置随机种子以确保初始位置一致
    )
    
    # 创建Gym环境
    env = RebarGymEnv(config)
    
    # 重置环境
    obs, info = env.reset()
    
    # 设置两个智能体的初始位置
    env.env.current_positions[0] = np.array([-8.0, 3.0])  # 第一个智能体在左上角
    env.env.current_positions[1] = np.array([-8.0, -3.0])  # 第二个智能体在左下角
    
    # 定义两个智能体的动作序列
    # 动作序列：0-左, 1-右, 2-上, 3-下, 4-不动
    action_sequence = []
    # 生成足够多的交替动作
    for _ in range(50):  # 设置较大的动作数量以确保充分测试
        # 第一个智能体：交替向右和向下移动
        # 第二个智能体：交替向右和向上移动
        action_sequence.extend([1, 3, 1, 2])  # [右, 下, 右, 上]
    
    # 执行动作序列
    for i, action in enumerate(action_sequence):
        # 执行动作
        obs, reward, terminated, truncated, info = env.step(action)
        
        # 打印当前状态
        print(f"\n执行第 {i+1} 个动作: {action}")
        print(f"智能体0位置: {env.env.current_positions[0]}")
        print(f"智能体1位置: {env.env.current_positions[1]}")
        print(f"奖励值: {reward}")
        
        # 等待一段时间以便观察
        time.sleep(0.1)  # 减小延时使移动更流畅
        
        # 如果两个智能体都到达目标位置，结束测试
        if (env.env.current_positions[0][0] >= 8.0 and env.env.current_positions[0][1] <= -3.0 and
            env.env.current_positions[1][0] >= 8.0 and env.env.current_positions[1][1] >= 3.0):
            print("\n两个智能体都已到达目标位置，结束测试")
            break
        
        # 如果环境结束，退出循环
        if terminated:
            print("\n环境已结束")
            break
    
    # 关闭环境
    env.close()
    print("\nT型边界测试完成")

def main():
    """主函数：测试所有边界形状"""
    # 测试所有边界形状
    for shape in BoundaryShape:
        test_boundary(shape)
        time.sleep(1)  # 在测试不同边界形状之间稍作暂停

if __name__ == "__main__":
    test_t_boundary() 