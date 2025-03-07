def score_by_displacement(value, eps=0.001, scale=5):
    """
    计算基于位移的得分，鼓励接近 0，偏离越多扣分越多。

    参数:
    - value: float, 观察值
    - eps: float, 允许接近 0 的误差范围
    - scale: float, 最大得分（接近 0 时）

    返回:
    - dict: {0: 放松得分, 1: 不变得分, 2: 加紧得分}
    """
    # 方式 1：二次损失（偏移平方衰减）
    score = scale - (value / eps) ** 2 * scale

    # 方式 2：线性衰减（可以替换上面这一行）
    # score = scale - abs(value / eps) * scale

    score = max(-scale, score)  # 限制最低得分

    if value > 0:
        return {0: scale, 1: score, 2: -scale}
    else:
        return {0: -scale, 1: score, 2: scale}

# 测试
print(score_by_displacement(0.0005))  # 接近 0，得分高
print(score_by_displacement(0.01))    # 偏正，扣分
print(score_by_displacement(-0.02))   # 偏负，扣分
