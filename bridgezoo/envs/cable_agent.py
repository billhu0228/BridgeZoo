"""单根索（索对）智能体的状态、动作与局部观测定义（里程碑 M2）。

每个智能体对应一对对称拉索，参数共享。智能体在施工过程中**恰好动作两次**：

1. 安装期（第一次张拉）：决策 ``(股数档位 Δn, 初应力档位 Δσ1)``。
2. 调索期（第二次张拉）：决策 ``(应力档位 Δσ2)``，股数此时冻结。

动作为离散（见 ``docs/DESIGN_MAPPO.md`` 第 5.4 节）。非法动作（越界/已松弛/非活动
阶段）由环境层的 action mask 屏蔽。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


@dataclass
class CableLimits:
    """安全与取值范围限制。"""

    sigma_min: float = 10.0     # 最小应力 (MPa)，低于则视为接近松弛
    sigma_allow: float = 1000.0  # 容许应力 (MPa)
    n_min: int = 4              # 最小股数
    n_max: int = 60             # 最大股数
    n_step: int = 2             # 股数调整步长
    sigma_step: float = 50.0    # 应力调整步长 (MPa)


class CableAgent:
    """单个索智能体的可变状态容器。

    Notes
    -----
    本类只持有状态与动作语义；真正的力学更新由环境层调用 FEM 完成后回填
    （:meth:`update`）。观测/奖励的具体张量在环境层组装，这里给出维度约定。
    """

    OBS_DIM = 9  # 局部观测维度（详见 build_observation 文档）

    def __init__(self, index: int, limits: CableLimits | None = None):
        self.index = index
        self.limits = limits or CableLimits()
        self.reset()

    def reset(self) -> None:
        self.num_strands: int = self.limits.n_min
        self.sigma: float = 0.0       # 当前索应力 (MPa)
        self.installed: bool = False  # 是否已安装（进入第一次张拉后为 True）
        self.deflection: float = 0.0  # 锚固点梁挠度 (m)
        self.last_action = None

    # ------------------------------------------------------------- 动作空间
    @staticmethod
    def erection_action_space() -> spaces.Space:
        """安装期动作空间：股数档 × 初应力档（各 3 档：减/不变/增）。"""
        return spaces.MultiDiscrete([3, 3])

    @staticmethod
    def adjustment_action_space() -> spaces.Space:
        """调索期动作空间：应力档（减/不变/增）。"""
        return spaces.Discrete(3)

    @staticmethod
    def observation_space() -> spaces.Space:
        return spaces.Box(low=-np.inf, high=np.inf, shape=(CableAgent.OBS_DIM,), dtype=np.float32)

    # --------------------------------------------------------------- 行为
    def apply_erection(self, dn_level: int, dsig_level: int) -> None:
        """应用安装期动作（档位 ∈ {0,1,2} 映射到 {-1,0,+1}）。

        TODO(M2): 更新 num_strands / sigma 并做范围裁剪，置 installed=True。
        """
        raise NotImplementedError("TODO(M2): CableAgent.apply_erection")

    def apply_adjustment(self, dsig_level: int) -> None:
        """应用调索期动作（股数冻结，仅调应力）。

        TODO(M2): 更新 sigma 并裁剪。
        """
        raise NotImplementedError("TODO(M2): CableAgent.apply_adjustment")

    def update(self, sigma_after: float, deflection: float) -> None:
        """FEM 求解后回填该索的平衡应力与锚固点挠度。"""
        self.sigma = sigma_after
        self.deflection = deflection

    def build_observation(self, phase_onehot, stage_frac: float, is_active: bool,
                          target_deflection: float, neighbor_defl) -> np.ndarray:
        """组装局部观测（OBS_DIM 维，已归一化）。

        建议布局::

            [ n/ n_max, sigma/sigma_allow, installed,
              defl, defl - target, neighbor_defl(2),
              stage_frac, is_active ]  + phase one-hot 由环境拼接

        TODO(M2): 按上面布局填充并归一化。
        """
        raise NotImplementedError("TODO(M2): CableAgent.build_observation")
