"""Euler-Bernoulli 一致的梁变形轴线插值（仅服务于绘图后处理）。

把求解结果的节点位移 ``(ux, uy, rz)`` 沿梁单元用三次 Hermite 形函数插值成
平顺的变形曲线：横向挠度由端部挠度与转角共同控制，与
:mod:`bridgezoo.fem.kernels` 中 Euler-Bernoulli 单元刚度的位移假设一致；
轴向位移按线性插值。

约定（与求解器一致，不得改动）：

- 单位 SI（m、rad）；二维，每节点 3 自由度 ``(ux, uy, rz)``，``rz`` 在
  平面内转轴变换下保持不变（同 ``kernels._frame_transform``）。
- 线性小位移假设，与自写 direct 求解器相同。``scale`` 仅用于绘图放大，
  直接乘在位移场上——位移场对节点自由度线性，先插值后放大与先放大后
  插值等价。
- 纯 numpy，不依赖 matplotlib，可被任意渲染端复用。
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np

__all__ = ["hermite_frame_shape", "deformed_chain_shape"]


def hermite_frame_shape(
    pi: tuple[float, float],
    pj: tuple[float, float],
    di: tuple[float, float, float],
    dj: tuple[float, float, float],
    scale: float = 1.0,
    samples: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """单根梁单元变形轴线的全局坐标采样。

    Parameters
    ----------
    pi, pj:
        单元两端未变形坐标 ``(x, y)`` [m]。
    di, dj:
        两端全局位移 ``(ux, uy, rz)`` [m, m, rad]。
    scale:
        位移绘图放大倍数（乘整个位移场；几何坐标不缩放）。
    samples:
        采样点数（含两端，>= 2）。

    Returns
    -------
    (xs, ys):
        变形轴线采样点的全局坐标。Hermite 形函数为插值基，端点严格等于
        ``p + scale * d[:2]``。曲线形状与单元 ``(i, j)`` 方向无关——同一组
        端部物理条件唯一确定该三次曲线。
    """
    if samples < 2:
        raise ValueError(f"samples must be >= 2, got {samples}")
    x_i, y_i = float(pi[0]), float(pi[1])
    x_j, y_j = float(pj[0]), float(pj[1])
    dx, dy = x_j - x_i, y_j - y_i
    length = math.hypot(dx, dy)
    if length <= 0.0:
        raise ValueError(f"zero-length element: pi={pi}, pj={pj}")
    c, s = dx / length, dy / length

    # 全局位移 -> 局部（轴向 u、横向 v），与 kernels._frame_transform 一致。
    u_i = c * float(di[0]) + s * float(di[1])
    v_i = -s * float(di[0]) + c * float(di[1])
    th_i = float(di[2])
    u_j = c * float(dj[0]) + s * float(dj[1])
    v_j = -s * float(dj[0]) + c * float(dj[1])
    th_j = float(dj[2])

    t = np.linspace(0.0, 1.0, samples)
    u = (1.0 - t) * u_i + t * u_j
    # 三次 Hermite 形函数；转角形函数带长度量纲（θ 控制局部斜率 dv/dx）。
    n1 = 1.0 - 3.0 * t**2 + 2.0 * t**3
    n2 = length * (t - 2.0 * t**2 + t**3)
    n3 = 3.0 * t**2 - 2.0 * t**3
    n4 = length * (t**3 - t**2)
    v = n1 * v_i + n2 * th_i + n3 * v_j + n4 * th_j

    # 局部位移场 -> 全局，放大后叠加到未变形轴线上。
    dux = c * u - s * v
    duy = s * u + c * v
    xs = x_i + t * dx + scale * dux
    ys = y_i + t * dy + scale * duy
    return xs, ys


def deformed_chain_shape(
    coords: Mapping[int, tuple[float, float]],
    disp: Mapping[int, tuple[float, float, float]],
    chain: Sequence[int],
    scale: float = 1.0,
    samples_per_element: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """沿一串首尾相连的梁单元节点链采样整条变形轴线。

    Parameters
    ----------
    coords:
        ``{node_id: (x, y)}`` 未变形坐标 [m]。
    disp:
        ``{node_id: (ux, uy, rz)}`` 全局位移。
    chain:
        有序节点 id。**约定相邻两个 id 之间恰好是一根梁单元**——例如
        builder 生成的主梁按 x 排序后的已安装节点链（任意施工记录下均
        连续无空洞）。
    scale, samples_per_element:
        同 :func:`hermite_frame_shape`。

    Returns
    -------
    (xs, ys):
        各单元曲线首尾拼接（接缝节点去重），可直接用于
        ``ax.plot`` / ``ax.fill_between``。空链返回空数组；单节点链返回
        该节点位移放大后的单点。
    """
    if len(chain) == 0:
        return np.empty(0), np.empty(0)
    if len(chain) == 1:
        nid = chain[0]
        x0, y0 = coords[nid]
        d = disp[nid]
        return (np.array([x0 + scale * float(d[0])]),
                np.array([y0 + scale * float(d[1])]))

    xs_parts: list[np.ndarray] = []
    ys_parts: list[np.ndarray] = []
    for k, (a, b) in enumerate(zip(chain[:-1], chain[1:])):
        exs, eys = hermite_frame_shape(
            coords[a], coords[b], disp[a], disp[b],
            scale=scale, samples=samples_per_element,
        )
        if k > 0:
            # 接缝节点在前一单元末尾已采样，去掉本单元首点避免重复 x。
            exs, eys = exs[1:], eys[1:]
        xs_parts.append(exs)
        ys_parts.append(eys)
    return np.concatenate(xs_parts), np.concatenate(ys_parts)
