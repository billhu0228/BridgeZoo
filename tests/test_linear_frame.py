"""线性框架求解器测试（里程碑 M1，当前 skip）。

实现后应包含：

- 简支梁均布荷载挠度与解析解对比。
- 单索张拉的轴力/位移与手算对比。
- 逐阶段装配（变刚度 + 位移锁定）与 OpenSees 参考解对比。
"""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(M1): linear_frame 未实现")


def test_simply_supported_beam_udl():
    """简支梁均布荷载跨中挠度 5wL^4/384EI。"""
    raise NotImplementedError


def test_single_cable_pretension():
    """单索初张力下的轴力与端点位移。"""
    raise NotImplementedError


def test_staged_vs_opensees():
    """逐阶段累加结果 vs OpenSees 一次成桥（误差阈值内）。"""
    raise NotImplementedError
