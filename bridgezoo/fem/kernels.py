"""二维杆系单元的底层数值核（与分析模式、求解器后端无关）。

这里集中放"单元层面的纯数学"，供**一次成桥**(:mod:`bridgezoo.fem.completed`)与
**分阶段施工**(:mod:`bridgezoo.fem.staged`)两套自研直接刚度法求解器共享，避免任一
求解器模块同时充当"核函数仓库"。

- :func:`_frame_local_stiffness` —— 2D Euler-Bernoulli 框架单元局部刚度 (6x6)。
- :func:`_frame_transform` —— 单元坐标变换矩阵 T (6x6)，``d_local = T @ d_global``。
- :func:`_udl_fixed_end_local` —— 局部横向均布荷载的一致等效节点荷载 (6,)。
- :func:`_gravity_feq_global` —— 全局竖向重力线荷载的一致等效节点荷载 (全局 6 向量)。
"""

from __future__ import annotations

import numpy as np


def _frame_local_stiffness(E, A, I, L):
    """2D 框架单元局部刚度矩阵 (6x6)，DOF 顺序 [u_i,v_i,θ_i,u_j,v_j,θ_j]。"""
    EA_L = E * A / L
    EI = E * I
    L2, L3 = L * L, L * L * L
    k = np.zeros((6, 6))
    # 轴向
    k[0, 0] = k[3, 3] = EA_L
    k[0, 3] = k[3, 0] = -EA_L
    # 弯曲 + 剪切（Euler-Bernoulli）
    k[1, 1] = k[4, 4] = 12 * EI / L3
    k[1, 4] = k[4, 1] = -12 * EI / L3
    k[1, 2] = k[2, 1] = 6 * EI / L2
    k[1, 5] = k[5, 1] = 6 * EI / L2
    k[4, 2] = k[2, 4] = -6 * EI / L2
    k[4, 5] = k[5, 4] = -6 * EI / L2
    k[2, 2] = k[5, 5] = 4 * EI / L
    k[2, 5] = k[5, 2] = 2 * EI / L
    return k


def _frame_transform(c, s):
    """单元坐标变换矩阵 T (6x6)，使 d_local = T @ d_global。"""
    T = np.zeros((6, 6))
    R = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
    T[0:3, 0:3] = R
    T[3:6, 3:6] = R
    return T


def _udl_fixed_end_local(wy, L):
    """局部横向均布荷载 wy 的一致等效节点荷载向量 (6,)。

    ``[0, wy*L/2, wy*L^2/12, 0, wy*L/2, -wy*L^2/12]``（与 OpenSees beamUniform Wy 一致）。
    """
    return np.array([0.0, wy * L / 2.0, wy * L * L / 12.0, 0.0, wy * L / 2.0, -wy * L * L / 12.0])


def _gravity_feq_global(gy: float, c: float, s: float, L: float) -> np.ndarray:
    """**全局竖向**重力线荷载 gy(向下为负)的一致等效节点荷载(全局 6 向量)。

    重力 (0, gy) 投影到单元局部:横向 Wy=gy*c、轴向 Wx=gy*s(局部 x 沿单元方向 (c,s),
    局部 y 为 (-s, c));再按一致固端力转回全局。水平梁 (s=0) 时退化为 Wy=gy*c。
    解决了"局部横向荷载"在反向(-x)梁上方向翻转、导致左右自重不对称的问题。
    """
    q_t = gy * c   # 局部横向
    q_a = gy * s   # 局部轴向
    feq_local = np.array([
        q_a * L / 2.0, q_t * L / 2.0, q_t * L * L / 12.0,
        q_a * L / 2.0, q_t * L / 2.0, -q_t * L * L / 12.0,
    ])
    return _frame_transform(c, s).T @ feq_local
