"""pytest 配置与公共夹具。

确保从仓库根目录可直接 ``pytest`` 运行（无需先安装包）。
"""

import os
import sys

import pytest

# 把仓库根加入 sys.path，便于 `import bridgezoo` 在未安装时可用。
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture
def small_geometry():
    """N=6 的默认小算例几何，供多个测试复用。"""
    from bridgezoo.envs.geometry import BridgeGeometry

    return BridgeGeometry(num_cables_per_side=6, anchor_height=20.0)
