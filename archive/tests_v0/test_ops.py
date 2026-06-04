from dataclasses import dataclass
from typing import List

import numpy as np
import ezdxf
from ezdxf.math import Vec2

from bridgezoo.cablebridge.fem import FEM


@dataclass
class test_obj:
    num_cables_per_side: int
    wg: float
    beam_h: float
    beam_w: float
    beam_E: float
    beam_area: float
    beam_Iz: float
    num_beam_points: int
    left_tower_pts: List[int]
    right_tower_pts: List[int]
    x_positions: np.ndarray

    def __init__(self, num_cables, anchor_height, w, h, E):
        self.beam_h = h
        self.beam_w = w
        self.beam_E = E
        self.beam_area = h * w
        self.beam_Iz = w * h ** 3 / 12.0
        self.wg = w * h * 25000
        self.num_cables_per_side = num_cables
        self.middle_spacing = 10
        self.outside_spacing = 8
        self.end_to_first_spacing = 4
        self.center_to_adjacent_spacing = 2
        self.vertical_spacing = 2
        self.anchor_height = anchor_height  # 上塔柱高度
        self.num_beam_points = self.num_cables_per_side + 3  # 只考虑一侧的梁节点

        self.x_positions = np.zeros(self.num_cables_per_side * 2 + 5, dtype=np.float32)
        self.span = 2 * (self.num_cables_per_side * 0.5 * self.middle_spacing + self.center_to_adjacent_spacing)
        self.side_span = self.end_to_first_spacing + self.num_cables_per_side * 0.5 * self.outside_spacing
        self.beam_length = self.side_span * 2 + self.span
        x1 = np.linspace(-0.5 * self.beam_length + self.end_to_first_spacing, -self.outside_spacing - self.span * 0.5,
                         self.num_cables_per_side // 2)
        x2 = np.linspace(-self.span * 0.5, -self.center_to_adjacent_spacing, self.num_cables_per_side // 2 + 1)
        self.x_positions = np.hstack((-self.beam_length * 0.5, x1, x2, 0, x2 * -1, x1 * -1, 0.5 * self.beam_length))
        self.x_positions.sort()
        self.left_tower_top = Vec2([-0.5 * self.span, self.anchor_height])
        self.right_tower_top = Vec2([0.5 * self.span, self.anchor_height])
        self.left_tower_base = Vec2(-0.5 * self.span, 0)
        self.right_tower_base = Vec2(0.5 * self.span, 0)
        self.left_tower_pts = []
        self.right_tower_pts = []
        for i in range(self.num_cables_per_side // 2):
            left_anchor = self.left_tower_top + Vec2(0, -self.vertical_spacing) * i
            right_anchor = self.right_tower_top + Vec2(0, -self.vertical_spacing) * i
            self.left_tower_pts.append(left_anchor)
            self.right_tower_pts.append(right_anchor)

    def extract_fem_model(self, cable_sigma, cable_sizes):
        """
        从CableSystemEnv提取信息，生成FEM实例
        """
        # cable_tensions = self.state[self.num_beam_points:self.num_beam_points + self.num_cables_per_side]
        fem_model = FEM(self.num_cables_per_side, self.wg, cable_sigma)
        # 提取梁的节点信息
        for i, x in enumerate(self.x_positions):
            fem_model.add_node(i + 1, x, 0)
        # 提取梁的单元信息
        for i in range(len(self.x_positions) - 1):
            fem_model.add_element(i + 1, i + 1, i + 2, 1, 1)  # 梁单元类型和材料假设为1
        # 提取索塔坐标
        for i in range(self.num_cables_per_side // 2):
            fem_model.add_node(1001 + i, self.left_tower_pts[i].x, self.left_tower_pts[i].y)
            fem_model.add_node(3001 + i, self.right_tower_pts[i].x, self.right_tower_pts[i].y)
            beam_index_left = i + 2
            beam_index_right = self.num_beam_points + 1
            fem_model.add_element(1001 + i, 1001 + i, beam_index_left, 2, 1001 + i)  #
            fem_model.add_element(2001 + i, 1001 + i, self.num_beam_points - 1 - i, 2, 1001 + i)
            fem_model.add_element(3001 + i, 3001 + i, beam_index_right + i, 2, 1001 + i)
            fem_model.add_element(4001 + i, 3001 + i, self.num_beam_points * 2 - 2 - i, 2, 1001 + i)
            Ns1001 = cable_sizes[i]
            Nf1001 = cable_sigma[i]
            Ns2001 = cable_sizes[self.num_cables_per_side - 1 - i]
            Nf2001 = cable_sigma[self.num_cables_per_side - 1 - i]
            fem_model.add_material(1001 + i, {'Ns': Ns1001, 'sigma': Nf1001})
            fem_model.add_material(2001 + i, {'Ns': Ns2001, 'sigma': Nf2001})
            fem_model.add_material(3001 + i, {'Ns': Ns1001, 'sigma': Nf1001})
            fem_model.add_material(4001 + i, {'Ns': Ns2001, 'sigma': Nf2001})

        # 添加单元类型和材料信息
        fem_model.add_element_type(1, 'Beam')
        fem_model.add_element_type(2, 'Cable')
        fem_model.add_material(1, {'E': self.beam_E, 'A': self.beam_area, 'I': self.beam_Iz, 'W': self.beam_w, 'H': self.beam_h})
        return fem_model


if __name__ == '__main__':
    tst = test_obj(6, 20, 10, 1, 20e9)
    fem = tst.extract_fem_model([1000, ] * 6, [20, ] * 6)
    # fem.generate_dxf("OPS测试.dxf")
    pos, t = fem.opensees()

    print(pos, t)
