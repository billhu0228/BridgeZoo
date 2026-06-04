"""MAPPO 算法测试（里程碑 M3，当前 skip）。

实现后应包含：GAE 数值正确性（对照手算小例）、actor 掩码后非法动作概率为 0、
critic 前向维度、在玩具环境（如历史 resection 移植版）上能收敛的冒烟训练。
依赖 torch，未安装时自动跳过。
"""

import pytest

torch = pytest.importorskip("torch", reason="MAPPO 测试需要 torch")
pytestmark = pytest.mark.skip(reason="TODO(M3): mappo 未实现")


def test_gae_matches_manual():
    raise NotImplementedError


def test_actor_mask_zero_prob():
    raise NotImplementedError


def test_smoke_train_toy():
    raise NotImplementedError
