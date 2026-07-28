"""Physical materials and parameterised sections for the 3D single-tower model.

All values use SI units.  Unlike the legacy 2D model, the builder never accepts
pre-combined ``EA``/``EI`` values: section dimensions and material properties
remain explicit so that stiffness and self-weight have one physical source.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ElasticMaterial3D:
    """Isotropic linear-elastic material."""

    name: str
    E: float
    poisson: float
    density: float

    def __post_init__(self) -> None:
        if self.E <= 0.0:
            raise ValueError("material E must be positive")
        if not (-1.0 < self.poisson < 0.5):
            raise ValueError("material poisson ratio must lie between -1 and 0.5")
        if self.density <= 0.0:
            raise ValueError("material density must be positive")

    @property
    def G(self) -> float:
        """Shear modulus in Pa."""

        return self.E / (2.0 * (1.0 + self.poisson))


@dataclass(frozen=True)
class HSection3D:
    """Doubly symmetric welded H section.

    ``depth_z`` is the strong-axis depth, ``flange_width_y`` is the flange
    width, and the local member x-axis is normal to the section plane.
    """

    name: str
    depth_z: float
    flange_width_y: float
    web_thickness: float
    flange_thickness: float

    def __post_init__(self) -> None:
        values = (
            self.depth_z,
            self.flange_width_y,
            self.web_thickness,
            self.flange_thickness,
        )
        if any(value <= 0.0 for value in values):
            raise ValueError("H-section dimensions must be positive")
        if 2.0 * self.flange_thickness >= self.depth_z:
            raise ValueError("H-section flange thicknesses must leave a positive web depth")
        if self.web_thickness >= self.flange_width_y:
            raise ValueError("H-section web thickness must be smaller than flange width")

    @property
    def shape(self) -> str:
        return "H"

    @property
    def web_depth(self) -> float:
        return self.depth_z - 2.0 * self.flange_thickness

    @property
    def A(self) -> float:
        return (
            2.0 * self.flange_width_y * self.flange_thickness
            + self.web_depth * self.web_thickness
        )

    @property
    def Iy(self) -> float:
        flange_offset = 0.5 * (self.depth_z - self.flange_thickness)
        flange = self.flange_width_y * self.flange_thickness**3 / 12.0
        flange += self.flange_width_y * self.flange_thickness * flange_offset**2
        web = self.web_thickness * self.web_depth**3 / 12.0
        return 2.0 * flange + web

    @property
    def Iz(self) -> float:
        flange = self.flange_thickness * self.flange_width_y**3 / 12.0
        web = self.web_depth * self.web_thickness**3 / 12.0
        return 2.0 * flange + web

    @property
    def J(self) -> float:
        # Saint-Venant thin-plate approximation for an open H section.
        return (
            2.0 * self.flange_width_y * self.flange_thickness**3
            + self.web_depth * self.web_thickness**3
        ) / 3.0


@dataclass(frozen=True)
class HollowBoxSection3D:
    """Uniform-wall rectangular hollow box section."""

    name: str
    outer_width_y: float
    outer_depth_z: float
    wall_thickness: float

    def __post_init__(self) -> None:
        if min(self.outer_width_y, self.outer_depth_z, self.wall_thickness) <= 0.0:
            raise ValueError("box-section dimensions must be positive")
        if 2.0 * self.wall_thickness >= min(self.outer_width_y, self.outer_depth_z):
            raise ValueError("box-section wall thickness leaves no hollow core")

    @property
    def shape(self) -> str:
        return "hollow_box"

    @property
    def inner_width_y(self) -> float:
        return self.outer_width_y - 2.0 * self.wall_thickness

    @property
    def inner_depth_z(self) -> float:
        return self.outer_depth_z - 2.0 * self.wall_thickness

    @property
    def A(self) -> float:
        return (
            self.outer_width_y * self.outer_depth_z
            - self.inner_width_y * self.inner_depth_z
        )

    @property
    def Iy(self) -> float:
        return (
            self.outer_width_y * self.outer_depth_z**3
            - self.inner_width_y * self.inner_depth_z**3
        ) / 12.0

    @property
    def Iz(self) -> float:
        return (
            self.outer_depth_z * self.outer_width_y**3
            - self.inner_depth_z * self.inner_width_y**3
        ) / 12.0

    @property
    def J(self) -> float:
        # Bredt thin-wall closed-section torsion constant, using the wall
        # centreline dimensions and a uniform wall thickness.
        width_mid = self.outer_width_y - self.wall_thickness
        depth_mid = self.outer_depth_z - self.wall_thickness
        enclosed_mid_area = width_mid * depth_mid
        perimeter_over_t = 2.0 * (width_mid + depth_mid) / self.wall_thickness
        return 4.0 * enclosed_mid_area**2 / perimeter_over_t


@dataclass(frozen=True)
class RectangularSection3D:
    """Solid rectangular section used by equivalent deck-slab grillage strips."""

    name: str
    width_y: float
    depth_z: float

    def __post_init__(self) -> None:
        if self.width_y <= 0.0 or self.depth_z <= 0.0:
            raise ValueError("rectangular-section dimensions must be positive")

    @property
    def shape(self) -> str:
        return "rectangle"

    @property
    def A(self) -> float:
        return self.width_y * self.depth_z

    @property
    def Iy(self) -> float:
        return self.width_y * self.depth_z**3 / 12.0

    @property
    def Iz(self) -> float:
        return self.depth_z * self.width_y**3 / 12.0

    @property
    def J(self) -> float:
        long_side = max(self.width_y, self.depth_z)
        short_side = min(self.width_y, self.depth_z)
        ratio = short_side / long_side
        return long_side * short_side**3 * (
            1.0 / 3.0 - 0.21 * ratio * (1.0 - ratio**4 / 12.0)
        )


FrameSection3D = HSection3D | HollowBoxSection3D | RectangularSection3D


STEEL_Q345 = ElasticMaterial3D("Q345 structural steel", 206.0e9, 0.30, 7850.0)
CONCRETE_C50 = ElasticMaterial3D("C50 concrete", 34.5e9, 0.20, 2500.0)
CABLE_STEEL = ElasticMaterial3D("parallel strand cable steel", 195.0e9, 0.30, 7850.0)


__all__ = [
    "ElasticMaterial3D",
    "HSection3D",
    "HollowBoxSection3D",
    "RectangularSection3D",
    "FrameSection3D",
    "STEEL_Q345",
    "CONCRETE_C50",
    "CABLE_STEEL",
]
