"""几何模块测试（已实现，可通过）。

校核 :class:`bridgezoo.envs.geometry.BridgeGeometry` 的节点数、对称性、跨径推导等
不变量。这是当前唯一"实打实"运行的测试，作为重构后架构可用的冒烟验证。
"""

import numpy as np
import pytest

from bridgezoo.envs.geometry import BridgeGeometry


@pytest.mark.parametrize("n", [4, 6, 8, 12])
def test_node_count(n):
    """梁节点总数应为 2N+5。"""
    g = BridgeGeometry(num_cables_per_side=n)
    assert g.num_beam_nodes == 2 * n + 5
    assert len(g.x_positions) == 2 * n + 5


@pytest.mark.parametrize("n", [4, 6, 8])
def test_x_symmetry_and_monotonic(n):
    """节点 x 关于跨中对称且单调递增。"""
    g = BridgeGeometry(num_cables_per_side=n)
    x = g.x_positions
    assert np.all(np.diff(x) > 0), "x 必须严格递增"
    assert np.allclose(x, -x[::-1]), "x 必须关于 0 对称"
    assert np.isclose(x[0], -g.beam_length / 2)
    assert np.isclose(x[-1], g.beam_length / 2)


def test_span_and_side_span():
    """跨径/边跨推导与历史公式一致（N=6）。"""
    g = BridgeGeometry(num_cables_per_side=6)
    assert np.isclose(g.span, 2 * (6 * 0.5 * 10 + 2))        # 64
    assert np.isclose(g.side_span, 4 + 6 * 0.5 * 8)          # 28
    assert np.isclose(g.beam_length, g.side_span * 2 + g.span)  # 120


def test_tower_points():
    """塔上锚点数量 = N/2，自塔顶按 vertical_spacing 向下排列。"""
    g = BridgeGeometry(num_cables_per_side=6, anchor_height=20.0)
    assert g.left_tower_pts.shape == (3, 2)
    assert np.allclose(g.left_tower_pts[0], [-g.span / 2, 20.0])
    assert np.allclose(g.left_tower_pts[:, 1], [20.0, 18.0, 16.0])
    # 左右对称
    assert np.allclose(g.left_tower_pts[:, 0], -g.right_tower_pts[:, 0])


def test_section_properties():
    g = BridgeGeometry(num_cables_per_side=6, beam_w=10.0, beam_h=1.0)
    assert np.isclose(g.beam_area, 10.0)
    assert np.isclose(g.beam_Iz, 10.0 * 1.0 ** 3 / 12.0)
    assert g.wg > 0


def test_section_properties_direct_override():
    """beam_A / beam_I 直接指定时覆盖由 beam_w/beam_h 推导的值。"""
    g = BridgeGeometry(num_cables_per_side=6, beam_w=10.0, beam_h=1.0, beam_A=5.0, beam_I=2.0)
    assert np.isclose(g.beam_area, 5.0)
    assert np.isclose(g.beam_Iz, 2.0)
    # wg 仍由 beam_area（已 override）× density × gravity × load_factor 计算
    expected_wg = 5.0 * g.density * g.gravity * g.load_factor
    assert np.isclose(g.wg, expected_wg)


def test_invalid_cable_count():
    """非偶数 / 过小应报错。"""
    with pytest.raises(ValueError):
        BridgeGeometry(num_cables_per_side=5)
    with pytest.raises(ValueError):
        BridgeGeometry(num_cables_per_side=1)


def test_summary_runs():
    g = BridgeGeometry()
    assert isinstance(g.summary(), str) and g.summary()
