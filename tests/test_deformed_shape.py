"""绘图用 Hermite 变形轴线插值的单元/集成测试。

被测对象：:mod:`bridgezoo.render.deformed_shape`。纯 numpy，不依赖
matplotlib / openseespy。
"""

from __future__ import annotations

import numpy as np
import pytest

from bridgezoo.fem.staged import StagedDirectSolver, build_staged_cantilever
from bridgezoo.render.deformed_shape import deformed_chain_shape, hermite_frame_shape

ZERO = (0.0, 0.0, 0.0)


def test_zero_displacement_is_straight_chord():
    # 零位移时插值应退化为未变形弦线；纯算术恒等，取机器精度容差。
    xs, ys = hermite_frame_shape((0.0, 0.0), (3.0, 4.0), ZERO, ZERO, samples=11)
    t = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(xs, 3.0 * t, atol=1e-12)
    np.testing.assert_allclose(ys, 4.0 * t, atol=1e-12)


def test_endpoints_match_scaled_nodal_displacements():
    # Hermite 形函数是插值基：端点必须严格还原 p + scale*d[:2]（机器精度）。
    pi, pj = (1.0, -2.0), (4.0, 2.0)
    di, dj = (0.01, -0.02, 0.003), (-0.004, 0.05, -0.001)
    scale = 7.0
    xs, ys = hermite_frame_shape(pi, pj, di, dj, scale=scale, samples=5)
    assert xs[0] == pytest.approx(pi[0] + scale * di[0], abs=1e-12)
    assert ys[0] == pytest.approx(pi[1] + scale * di[1], abs=1e-12)
    assert xs[-1] == pytest.approx(pj[0] + scale * dj[0], abs=1e-12)
    assert ys[-1] == pytest.approx(pj[1] + scale * dj[1], abs=1e-12)


def test_cantilever_tip_load_matches_analytic_cubic():
    # 悬臂端集中力的解析挠曲线 v(x) = -P x^2 (3L-x) / (6EI) 本身是三次式，
    # 落在 Hermite 插值空间内 —— 给定端部 (v, θ) 后应逐点一致到浮点舍入。
    L, E, I, P = 5.0, 2.0e11, 1.0e-4, 1.0e3
    tip_v = -P * L**3 / (3.0 * E * I)
    tip_th = -P * L**2 / (2.0 * E * I)
    xs, ys = hermite_frame_shape((0.0, 0.0), (L, 0.0), ZERO,
                                 (0.0, tip_v, tip_th), samples=41)
    analytic = -P * xs**2 * (3.0 * L - xs) / (6.0 * E * I)
    np.testing.assert_allclose(ys, analytic, rtol=1e-9, atol=1e-15)


def test_rotation_only_dofs_produce_curvature():
    # 端部挠度全零、仅有相同转角 θ 时 v(ξ) = θL(ξ - 3ξ² + 2ξ³)：曲线必须
    # 明显偏离直弦线，且最大偏离等于解析极值（ξ*=(3-√3)/6）。401 个采样点
    # 的离散低估远小于 0.1%，取 rel=1e-3。
    L, th = 8.0, 0.01
    xs, ys = hermite_frame_shape((0.0, 0.0), (L, 0.0), (0.0, 0.0, th),
                                 (0.0, 0.0, th), samples=401)
    max_dev = float(np.max(np.abs(ys)))
    assert max_dev > 0.05 * th * L
    xi_star = (3.0 - np.sqrt(3.0)) / 6.0
    peak = th * L * abs(xi_star - 3.0 * xi_star**2 + 2.0 * xi_star**3)
    assert max_dev == pytest.approx(peak, rel=1e-3)


def test_scale_is_linear_in_displacements():
    # 位移场对节点 DOF 线性 ⇒ shape(s) = chord + s*(shape(1) - chord)（机器精度）。
    pi, pj = (-2.0, 1.0), (6.0, 3.0)
    di, dj = (0.02, -0.03, 0.004), (-0.01, 0.06, -0.002)
    x0, y0 = hermite_frame_shape(pi, pj, ZERO, ZERO, samples=17)
    x1, y1 = hermite_frame_shape(pi, pj, di, dj, scale=1.0, samples=17)
    xs, ys = hermite_frame_shape(pi, pj, di, dj, scale=12.5, samples=17)
    np.testing.assert_allclose(xs, x0 + 12.5 * (x1 - x0), atol=1e-12)
    np.testing.assert_allclose(ys, y0 + 12.5 * (y1 - y0), atol=1e-12)


def test_reversed_element_gives_same_curve():
    # 同一组端部物理条件唯一确定三次曲线，(i,j) 翻转不得改变形状 ——
    # 覆盖左半主梁链逆向遍历单元的情形。坐标量级 ~5 m，1e-12 即机器精度。
    pi, pj = (0.0, 0.0), (5.0, 1.0)
    di, dj = (0.01, -0.04, 0.006), (-0.02, 0.03, -0.005)
    xf, yf = hermite_frame_shape(pi, pj, di, dj, scale=3.0, samples=25)
    xr, yr = hermite_frame_shape(pj, pi, dj, di, scale=3.0, samples=25)
    np.testing.assert_allclose(xr[::-1], xf, atol=1e-12)
    np.testing.assert_allclose(yr[::-1], yf, atol=1e-12)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        hermite_frame_shape((1.0, 1.0), (1.0, 1.0), ZERO, ZERO)
    with pytest.raises(ValueError):
        hermite_frame_shape((0.0, 0.0), (1.0, 0.0), ZERO, ZERO, samples=1)


def test_chain_empty_and_single_node():
    coords = {7: (2.0, 0.0)}
    disp = {7: (0.5, -0.25, 0.1)}
    xs, ys = deformed_chain_shape(coords, disp, [], scale=2.0)
    assert xs.size == 0 and ys.size == 0
    xs, ys = deformed_chain_shape(coords, disp, [7], scale=2.0)
    np.testing.assert_allclose(xs, [3.0], atol=1e-12)
    np.testing.assert_allclose(ys, [-0.5], atol=1e-12)


def test_chain_matches_nodes_on_real_staged_record():
    # 集成：真实分阶段求解记录上，链曲线必须严格经过每个节点的
    # (x + scale*ux, y + scale*uy)（插值基精确还原节点值，机器精度），
    # 且接缝去重后 x 单调（fill_between 的前提）。
    plan = build_staged_cantilever(n_seg=2)
    result = StagedDirectSolver().run(plan)
    rec = result.records[-1]
    chain = sorted((nid for nid in result.deck_ids if nid in rec.disp),
                   key=lambda nid: result.coords[nid][0])
    assert len(chain) >= 4
    scale, m = 10.0, 9
    xs, ys = deformed_chain_shape(result.coords, rec.disp, chain,
                                  scale=scale, samples_per_element=m)
    assert xs.size == (m - 1) * (len(chain) - 1) + 1
    for k, nid in enumerate(chain):
        x0, y0 = result.coords[nid]
        ux, uy, _ = rec.disp[nid]
        idx = k * (m - 1)
        assert xs[idx] == pytest.approx(x0 + scale * ux, abs=1e-12)
        assert ys[idx] == pytest.approx(y0 + scale * uy, abs=1e-12)
    assert np.all(np.diff(xs) > 0.0)
