"""调索环境测试（里程碑 M2，当前 skip）。

实现后应包含：PettingZoo API 一致性（可借助 ``pettingzoo.test.parallel_api_test``）、
随机策略跑通完整施工序列、action mask 正确性、左右对称性、奖励/终止逻辑、
state() 维度与 state_space 一致。
"""

import pytest

pytestmark = pytest.mark.skip(reason="TODO(M2): cable_construction 未实现")


def test_parallel_api():
    raise NotImplementedError


def test_random_rollout_completes():
    raise NotImplementedError


def test_action_masks():
    raise NotImplementedError
