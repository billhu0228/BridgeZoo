"""Build the detailed 3D single-tower staged bridge architecture.

The model retains the legacy single-staged longitudinal dimension semantics,
but uses physical materials and section dimensions.  Main girders and cross
girders share a two-line beam grid.  An equivalent deck-slab grillage spans the
full deck width, including the transverse cantilevers outside the main girders,
and sits on an eccentric reference plane connected by rigid links.
Each erection stage is split into steel/cable+A, wet-deck-load+B and composite
slab activation substeps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

from bridgezoo.envs.geometry import SingleTowerGeometry3D

from bridgezoo.fem.single_staged.model3d import (
    BridgeModel3D,
    CableElement3D,
    ConstructionStage3D,
    FrameElement3D,
    FrameLoad3D,
    Node3D,
    RigidLink3D,
    SingleStagedPlan3D,
    Support3D,
)
from bridgezoo.fem.single_staged.sections3d import (
    CABLE_STEEL,
    CONCRETE_C50,
    STEEL_Q345,
    ElasticMaterial3D,
    HSection3D,
    HollowBoxSection3D,
    RectangularSection3D,
)


_BEAM_NODE_BASE = 100_000
_SLAB_NODE_BASE = 200_000
_TOWER_NODE_BASE = 300_000
_GROUND_NODE_BASE = 400_000
_MAIN_FRAME_BASE = 1_000_000
_CROSS_FRAME_BASE = 1_100_000
_SLAB_LONG_FRAME_BASE = 1_200_000
_SLAB_CROSS_FRAME_BASE = 1_300_000
_TOWER_FRAME_BASE = 1_400_000
_BACKSTAY_BASE = 2_000_000
_MAIN_STAY_BASE = 2_100_000
_RIGID_LINK_BASE = 3_000_000


@dataclass(frozen=True)
class SingleStaged3DConfig(SingleTowerGeometry3D):
    """Physical inputs for the 3D single-tower bridge.

    The first group mirrors the established 2D dimension names.  New transverse
    dimensions and physical component definitions follow it.  SI units are
    mandatory: m, N, Pa and kg/m³.
    """

    gravity: float = 9.806
    secondary_main_girder_line_load: float = 0.0  # per main girder, N/m
    secondary_deck_pressure: float = 0.0  # full deck surface, N/m²
    superimposed_dead_load: float | None = None  # legacy alias for deck pressure
    flexible_birth_correction_factor: float = 1.0

    strand_area: float = 1.4e-4
    strands_per_cable: int | tuple[int, ...] | tuple[tuple[int, int], ...] = 55
    pretension_per_cable: (
        float | tuple[float, ...] | tuple[tuple[float, float], ...]
    ) = 3.5e6
    pretension_a_ratio: (
        float | tuple[float, ...] | tuple[tuple[float, float], ...]
    ) = 1.0

    steel: ElasticMaterial3D = STEEL_Q345
    concrete: ElasticMaterial3D = CONCRETE_C50
    cable_material: ElasticMaterial3D = CABLE_STEEL

    @property
    def main_girder_section(self) -> HSection3D:
        return HSection3D(
            f"main girder H-{self.main_girder_depth * 1000:.0f}",
            self.main_girder_depth,
            self.main_girder_flange_width,
            self.main_girder_web_thickness,
            self.main_girder_flange_thickness,
        )

    @property
    def cross_girder_section(self) -> HSection3D:
        return HSection3D(
            f"cross girder H-{self.cross_girder_depth * 1000:.0f}",
            self.cross_girder_depth,
            self.cross_girder_flange_width,
            self.cross_girder_web_thickness,
            self.cross_girder_flange_thickness,
        )

    @property
    def tower_section(self) -> HollowBoxSection3D:
        return HollowBoxSection3D(
            f"tower {self.tower_outer_width:.1f}x{self.tower_outer_depth:.1f} box",
            self.tower_outer_width,
            self.tower_outer_depth,
            self.tower_wall_thickness,
        )

    @property
    def resolved_secondary_deck_pressure(self) -> float:
        if self.superimposed_dead_load is not None:
            return float(self.superimposed_dead_load)
        return float(self.secondary_deck_pressure)

    def __post_init__(self) -> None:
        super().__post_init__()
        positive = {"gravity": self.gravity, "strand_area": self.strand_area}
        invalid = [
            name for name, value in positive.items() if not math.isfinite(value) or value <= 0.0
        ]
        if invalid:
            raise ValueError(f"positive finite values required for: {', '.join(invalid)}")
        secondary = {
            "secondary_main_girder_line_load": self.secondary_main_girder_line_load,
            "secondary_deck_pressure": self.secondary_deck_pressure,
        }
        if self.superimposed_dead_load is not None:
            secondary["superimposed_dead_load"] = self.superimposed_dead_load
            if self.secondary_deck_pressure != 0.0:
                raise ValueError(
                    "use secondary_deck_pressure or legacy superimposed_dead_load, not both"
                )
        invalid_secondary = [
            name
            for name, value in secondary.items()
            if not math.isfinite(value) or value < 0.0
        ]
        if invalid_secondary:
            raise ValueError(
                f"finite nonnegative values required for: {', '.join(invalid_secondary)}"
            )
        if not math.isfinite(self.flexible_birth_correction_factor) or not math.isclose(
            self.flexible_birth_correction_factor,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError(
                "flexible_birth_correction_factor is retired and must equal 1.0"
            )
        _stage_pair_values(
            self.strands_per_cable,
            self.n_seg,
            "strands_per_cable",
            integer=True,
        )
        _stage_pair_values(
            self.pretension_per_cable,
            self.n_seg,
            "pretension_per_cable",
        )
        ratios = _stage_pair_values(
            self.pretension_a_ratio,
            self.n_seg,
            "pretension_a_ratio",
        )
        if any(ratio > 1.0 for pair in ratios for ratio in pair):
            raise ValueError("pretension_a_ratio values must be between zero and one")


@dataclass(frozen=True)
class _Station:
    key: str
    x: float
    activation_stage: int
    role: str
    is_cross_grid: bool = False


def _steel_step(construction_stage: int) -> int:
    return 3 * construction_stage - 2


def _deck_weight_step(construction_stage: int) -> int:
    return 3 * construction_stage - 1


def _composite_step(construction_stage: int) -> int:
    return 3 * construction_stage


def _stage_pair_values(
    value,
    n_seg: int,
    name: str,
    integer: bool = False,
) -> tuple[tuple[float, float], ...]:
    """Normalize per-stage ``(backstay, main_stay)`` design values.

    Scalars and ``n_seg`` scalar values retain the original behavior by using
    the same value for both cable groups.  Optimization can additionally pass
    ``n_seg`` explicit pairs or one flat stage-major vector of length
    ``2 * n_seg``.
    """

    if isinstance(value, (int, float)):
        pairs = [(value, value)] * n_seg
    else:
        values = tuple(value)
        if len(values) == n_seg:
            pairs = []
            for item in values:
                if isinstance(item, Sequence) and not isinstance(item, (str, bytes)):
                    pair = tuple(item)
                    if len(pair) != 2:
                        raise ValueError(f"{name} stage pairs must contain two values")
                    pairs.append((pair[0], pair[1]))
                else:
                    pairs.append((item, item))
        elif len(values) == 2 * n_seg:
            pairs = [
                (values[2 * index], values[2 * index + 1])
                for index in range(n_seg)
            ]
        else:
            raise ValueError(
                f"{name} must be a scalar, contain n_seg values/pairs, "
                f"or contain 2*n_seg stage-major values"
            )

    numbers = tuple((float(back), float(main)) for back, main in pairs)
    flat = tuple(item for pair in numbers for item in pair)
    if any(not math.isfinite(item) or item < 0.0 for item in flat):
        raise ValueError(f"{name} values must be finite and nonnegative")
    if integer and any(not item.is_integer() or item <= 0.0 for item in flat):
        raise ValueError(f"{name} values must be positive integers")
    return numbers


def _tower_elevations(config: SingleStaged3DConfig) -> tuple[list[float], list[float]]:
    anchors = [
        config.anchor_base_height + index * config.anchor_spacing
        for index in range(config.n_seg)
    ]
    top = anchors[-1] + config.anchor_top_free
    critical = sorted(set([0.0, *anchors, top]))
    elevations = [critical[0]]
    for lower, upper in zip(critical, critical[1:]):
        subdivisions = max(1, math.ceil((upper - lower) / config.tower_element_size))
        for part in range(1, subdivisions + 1):
            elevations.append(lower + (upper - lower) * part / subdivisions)
    return elevations, anchors


def _key_stations(config: SingleStaged3DConfig) -> list[_Station]:
    stations = [_Station("tower_axis", 0.0, 1, "root")]
    if not math.isclose(config.resolved_right_fix, 0.0, abs_tol=1.0e-12):
        stations.append(
            _Station("right_bearing", config.resolved_right_fix, 1, "right_bearing")
        )
    for index in range(1, config.n_seg + 1):
        stations.append(
            _Station(
                f"cable_{index}",
                -(config.left_start + (index - 1) * config.left_spacing),
                index,
                "cable_station",
            )
        )
    tip_x = -(config.left_start + (config.n_seg - 1) * config.left_spacing + config.left_end)
    stations.append(_Station("free_tip", tip_x, config.n_seg + 1, "tip"))
    if config.left_span is not None:
        stations.append(
            _Station(
                "left_bearing",
                tip_x - config.left_span,
                config.n_seg + 2,
                "left_bearing",
            )
        )
    stations.sort(key=lambda station: station.x)
    if any(math.isclose(a.x, b.x, abs_tol=1.0e-12) for a, b in zip(stations, stations[1:])):
        raise ValueError("longitudinal station coordinates must be distinct")
    return stations


def _activation_stage_at_x(x: float, key_stations: list[_Station]) -> int:
    final_stage = max(station.activation_stage for station in key_stations)
    rightmost = max(station.x for station in key_stations)
    for stage in range(1, final_stage + 1):
        active = [station.x for station in key_stations if station.activation_stage <= stage]
        if active and min(active) - 1.0e-10 <= x <= rightmost + 1.0e-10:
            return stage
    return final_stage


def _stations(config: SingleStaged3DConfig) -> tuple[list[_Station], float]:
    """Merge construction/key nodes with an independent equal cross-beam grid."""

    key_stations = _key_stations(config)
    leftmost = min(station.x for station in key_stations)
    rightmost = max(station.x for station in key_stations)
    interval_count = max(
        1,
        math.ceil((rightmost - leftmost) / config.cross_girder_spacing),
    )
    actual_spacing = (rightmost - leftmost) / interval_count
    stations = list(key_stations)
    for cross_index in range(interval_count + 1):
        x = leftmost + cross_index * actual_spacing
        coincident_index = next(
            (
                index
                for index, station in enumerate(stations)
                if math.isclose(station.x, x, abs_tol=1.0e-10)
            ),
            None,
        )
        if coincident_index is not None:
            stations[coincident_index] = replace(stations[coincident_index], is_cross_grid=True)
        else:
            stations.append(
                _Station(
                    f"cross_{cross_index}",
                    x,
                    _activation_stage_at_x(x, key_stations),
                    "cross_grid",
                    True,
                )
            )
    stations.sort(key=lambda station: station.x)
    return stations, actual_spacing


def build_single_staged_3d(
    config: SingleStaged3DConfig | None = None,
    **overrides,
) -> SingleStagedPlan3D:
    """Build a solver-neutral 3D staged plan.

    Keyword overrides are applied with :func:`dataclasses.replace`, allowing a
    concise minimum input such as ``build_single_staged_3d(n_seg=3)``.
    """

    if config is None:
        config = SingleStaged3DConfig(**overrides)
    elif overrides:
        config = replace(config, **overrides)

    model = BridgeModel3D(name=f"single_tower_grillage_3d_N{config.n_seg}")
    stations, actual_cross_spacing = _stations(config)
    half_spacing = 0.5 * config.girder_spacing
    girder_y = (-half_spacing, half_spacing)
    half_deck_width = 0.5 * config.deck_width
    if config.deck_width > config.girder_spacing:
        deck_grid_y = (-half_deck_width, *girder_y, half_deck_width)
        girder_deck_lines = (1, 2)
    else:
        deck_grid_y = girder_y
        girder_deck_lines = (0, 1)
    deck_tributary_widths = tuple(
        0.5
        * (
            (deck_grid_y[line] - deck_grid_y[line - 1] if line > 0 else 0.0)
            + (
                deck_grid_y[line + 1] - deck_grid_y[line]
                if line + 1 < len(deck_grid_y)
                else 0.0
            )
        )
        for line in range(len(deck_grid_y))
    )

    previous_station_index: dict[int, int | None] = {}
    for station_index, station in enumerate(stations):
        earlier = [
            index
            for index, candidate in enumerate(stations)
            if candidate.activation_stage < station.activation_stage
        ]
        previous_station_index[station_index] = (
            min(earlier, key=lambda index: abs(stations[index].x - station.x))
            if earlier
            else None
        )

    # Beam-grid nodes are born with the steelwork.  Eccentric slab nodes are
    # introduced only in the third substep and inherit the current rigid-body
    # displacement of the nearest main-girder node.
    beam_nodes: dict[tuple[int, int], int] = {}
    slab_nodes: dict[tuple[int, int], int] = {}
    for station_index, station in enumerate(stations):
        steel_activation = _steel_step(station.activation_stage)
        slab_activation = _composite_step(station.activation_stage)
        for side in range(2):
            beam_id = _BEAM_NODE_BASE + 10 * station_index + side
            beam_nodes[station_index, side] = beam_id
            previous_index = previous_station_index[station_index]
            birth_master = (
                _BEAM_NODE_BASE + 10 * previous_index + side
                if previous_index is not None
                else None
            )
            model.add_node(
                Node3D(
                    beam_id,
                    station.x,
                    girder_y[side],
                    0.0,
                    f"main_girder_{side}",
                    steel_activation,
                    birth_master,
                )
            )
        for line, y in enumerate(deck_grid_y):
            slab_id = _SLAB_NODE_BASE + 10 * station_index + line
            slab_nodes[station_index, line] = slab_id
            nearest_side = min(range(2), key=lambda side: abs(girder_y[side] - y))
            model.add_node(
                Node3D(
                    slab_id,
                    station.x,
                    y,
                    config.deck_offset,
                    f"deck_slab_{line}",
                    slab_activation,
                    beam_nodes[station_index, nearest_side],
                )
            )
        for side, deck_line in enumerate(girder_deck_lines):
            model.add_rigid_link(
                RigidLink3D(
                    _RIGID_LINK_BASE + 10 * station_index + side,
                    beam_nodes[station_index, side],
                    slab_nodes[station_index, deck_line],
                    slab_activation,
                )
            )

    # Two longitudinal H main girders.  The slab has a longitudinal strip on
    # every transverse grid line; its tributary widths sum to the full deck
    # width, so adding explicit edge strips does not duplicate slab mass/load.
    slab_longitudinal_sections = tuple(
        RectangularSection3D(
            f"deck longitudinal strip {line}",
            tributary_width,
            config.deck_thickness,
        )
        for line, tributary_width in enumerate(deck_tributary_widths)
    )
    slab_longitudinal_ids: list[int] = []
    slab_longitudinal_width_by_id: dict[int, float] = {}
    main_girder_ids: list[int] = []
    construction_stage_by_main_id: dict[int, int] = {}
    for interval, (left, right) in enumerate(zip(stations, stations[1:])):
        construction_stage = max(left.activation_stage, right.activation_stage)
        for side in range(2):
            main_id = _MAIN_FRAME_BASE + 10 * interval + side
            model.add_frame(
                FrameElement3D(
                    main_id,
                    beam_nodes[interval, side],
                    beam_nodes[interval + 1, side],
                    config.steel,
                    config.main_girder_section,
                    group="main_girder",
                    activation_stage=_steel_step(construction_stage),
                )
            )
            main_girder_ids.append(main_id)
            construction_stage_by_main_id[main_id] = construction_stage
        for line, (section, tributary_width) in enumerate(
            zip(slab_longitudinal_sections, deck_tributary_widths)
        ):
            slab_id = _SLAB_LONG_FRAME_BASE + 10 * interval + line
            model.add_frame(
                FrameElement3D(
                    slab_id,
                    slab_nodes[interval, line],
                    slab_nodes[interval + 1, line],
                    config.concrete,
                    section,
                    group="deck_longitudinal",
                    activation_stage=_composite_step(construction_stage),
                )
            )
            slab_longitudinal_ids.append(slab_id)
            slab_longitudinal_width_by_id[slab_id] = tributary_width

    # Cross H girders share the beam-grid nodes.  Transverse slab strips use
    # station tributary lengths and the raised slab nodes.
    cross_station_indices = [
        index for index, station in enumerate(stations) if station.is_cross_grid
    ]
    for cross_order, station_index in enumerate(cross_station_indices):
        station = stations[station_index]
        cross_id = _CROSS_FRAME_BASE + cross_order
        model.add_frame(
            FrameElement3D(
                cross_id,
                beam_nodes[station_index, 0],
                beam_nodes[station_index, 1],
                config.steel,
                config.cross_girder_section,
                group="cross_girder",
                activation_stage=_steel_step(station.activation_stage),
            )
        )
        left_tributary = 0.0 if cross_order == 0 else 0.5 * actual_cross_spacing
        right_tributary = (
            0.0 if cross_order == len(cross_station_indices) - 1 else 0.5 * actual_cross_spacing
        )
        transverse_section = RectangularSection3D(
            f"deck transverse strip {station.key}",
            left_tributary + right_tributary,
            config.deck_thickness,
        )
        for transverse_span in range(len(deck_grid_y) - 1):
            model.add_frame(
                FrameElement3D(
                    _SLAB_CROSS_FRAME_BASE + 10 * cross_order + transverse_span,
                    slab_nodes[station_index, transverse_span],
                    slab_nodes[station_index, transverse_span + 1],
                    config.concrete,
                    transverse_section,
                    group="deck_transverse",
                    activation_stage=_composite_step(station.activation_stage),
                )
            )

    # Tower box members, with mesh boundaries at every cable anchor.
    elevations, anchor_elevations = _tower_elevations(config)
    tower_nodes: list[int] = []
    anchor_nodes: dict[float, int] = {}
    for index, elevation in enumerate(elevations):
        node_id = _TOWER_NODE_BASE + index
        role = "tower_base" if index == 0 else "tower"
        if any(math.isclose(elevation, anchor, abs_tol=1.0e-10) for anchor in anchor_elevations):
            role = "tower_anchor"
            anchor_nodes[next(anchor for anchor in anchor_elevations if math.isclose(elevation, anchor))] = node_id
        model.add_node(Node3D(node_id, 0.0, 0.0, elevation, role, 0))
        tower_nodes.append(node_id)
    for index, (node_i, node_j) in enumerate(zip(tower_nodes, tower_nodes[1:])):
        model.add_frame(
            FrameElement3D(
                _TOWER_FRAME_BASE + index,
                node_i,
                node_j,
                config.concrete,
                config.tower_section,
                orientation=(0.0, 1.0, 0.0),
                group="tower",
                activation_stage=0,
            )
        )

    # Two cable planes: paired main stays and paired backstays at each stage.
    strand_pairs = _stage_pair_values(
        config.strands_per_cable,
        config.n_seg,
        "strands_per_cable",
        integer=True,
    )
    pretension_pairs = _stage_pair_values(
        config.pretension_per_cable,
        config.n_seg,
        "pretension_per_cable",
    )
    pretension_a_ratios = _stage_pair_values(
        config.pretension_a_ratio,
        config.n_seg,
        "pretension_a_ratio",
    )
    station_by_key = {station.key: index for index, station in enumerate(stations)}
    ground_anchor_ids: list[int] = []
    for cable_index in range(1, config.n_seg + 1):
        activation = _steel_step(cable_index)
        second_pretension_stage = _deck_weight_step(cable_index)
        tower_anchor = anchor_nodes[anchor_elevations[cable_index - 1]]
        deck_station = station_by_key[f"cable_{cable_index}"]
        ground_x = config.right_start + (cable_index - 1) * config.right_spacing
        for side in range(2):
            ground_id = _GROUND_NODE_BASE + 10 * cable_index + side
            ground_anchor_ids.append(ground_id)
            model.add_node(
                Node3D(ground_id, ground_x, girder_y[side], 0.0, "ground_anchor", activation)
            )
            model.add_support(
                Support3D(ground_id, True, True, True, True, True, True, activation)
            )
            back_strands, main_strands = strand_pairs[cable_index - 1]
            back_tension, main_tension = pretension_pairs[cable_index - 1]
            back_a_ratio, main_a_ratio = pretension_a_ratios[cable_index - 1]
            model.add_cable(
                CableElement3D(
                    id=_BACKSTAY_BASE + 10 * cable_index + side,
                    i=tower_anchor,
                    j=ground_id,
                    material=config.cable_material,
                    area=config.strand_area * int(back_strands),
                    pretension=back_tension,
                    group="backstay",
                    activation_stage=activation,
                    pretension_a=back_tension * back_a_ratio,
                    second_pretension_stage=second_pretension_stage,
                    construction_stage=cable_index,
                )
            )
            model.add_cable(
                CableElement3D(
                    id=_MAIN_STAY_BASE + 10 * cable_index + side,
                    i=tower_anchor,
                    j=beam_nodes[deck_station, side],
                    material=config.cable_material,
                    area=config.strand_area * int(main_strands),
                    pretension=main_tension,
                    group="main_stay",
                    activation_stage=activation,
                    pretension_a=main_tension * main_a_ratio,
                    second_pretension_stage=second_pretension_stage,
                    construction_stage=cable_index,
                )
            )

    # Tower base and the fully fixed right girder end provide global stability.
    # At x=0 the right end reuses the tower-axis girder nodes, avoiding a
    # duplicate station and zero-length members.  An optional left auxiliary
    # span ends on vertical bearings.
    model.add_support(Support3D(tower_nodes[0], True, True, True, True, True, True, 0))
    right_index = station_by_key.get("right_bearing", station_by_key["tower_axis"])
    for side in range(2):
        model.add_support(
            Support3D(
                beam_nodes[right_index, side],
                True,
                True,
                True,
                True,
                True,
                True,
                _steel_step(1),
            )
        )
    if config.left_span is not None:
        left_index = station_by_key["left_bearing"]
        final_stage_index = _steel_step(config.n_seg + 2)
        for side in range(2):
            model.add_support(
                Support3D(
                    beam_nodes[left_index, side],
                    False,
                    side == 0,
                    True,
                    False,
                    False,
                    False,
                    final_stage_index,
                )
            )

    # Steel/tower self-weight is applied when each supporting member is born.
    # The concrete slab weight is deliberately not placed on slab members:
    # wet concrete is carried by the just-erected steel girders in substep 2,
    # then its already-committed effect remains in the steelwork when the slab
    # stiffness joins the composite system in substep 3.
    for frame in model.frames.values():
        if frame.group in {"deck_longitudinal", "deck_transverse"}:
            continue
        model.add_frame_load(
            FrameLoad3D(
                frame.id,
                qz=-frame.material.density * frame.section.A * config.gravity,
                load_case="self_weight",
                activation_stage=frame.activation_stage,
            )
        )

    temporary_slab_line_load = (
        config.concrete.density
        * config.deck_width
        * config.deck_thickness
        * config.gravity
        / 2.0
    )
    for member_id in main_girder_ids:
        construction_stage = construction_stage_by_main_id[member_id]
        model.add_frame_load(
            FrameLoad3D(
                member_id,
                qz=-temporary_slab_line_load,
                load_case="temporary_deck_self_weight",
                activation_stage=_deck_weight_step(construction_stage),
                deactivation_stage=_composite_step(construction_stage),
            )
        )

    geometry_final_stage = config.n_seg + 2 if config.left_span is not None else config.n_seg + 1
    deck_pressure = config.resolved_secondary_deck_pressure
    has_secondary_load = bool(config.secondary_main_girder_line_load or deck_pressure)
    secondary_stage_index = _composite_step(geometry_final_stage) + 1
    if config.secondary_main_girder_line_load:
        for member_id in main_girder_ids:
            model.add_frame_load(
                FrameLoad3D(
                    member_id,
                    qz=-config.secondary_main_girder_line_load,
                    load_case="secondary_main_girder_line",
                    activation_stage=secondary_stage_index,
                )
            )
    if deck_pressure:
        for member_id in slab_longitudinal_ids:
            model.add_frame_load(
                FrameLoad3D(
                    member_id,
                    qz=-deck_pressure * slab_longitudinal_width_by_id[member_id],
                    load_case="secondary_deck_pressure",
                    activation_stage=secondary_stage_index,
                )
            )

    stage_bases = [f"cable{index}" for index in range(1, config.n_seg + 1)] + ["tip"]
    if config.left_span is not None:
        stage_bases.append("left_span")
    stages: list[ConstructionStage3D] = []
    for construction_stage, base in enumerate(stage_bases, start=1):
        has_cable = construction_stage <= config.n_seg
        stages.extend(
            (
                ConstructionStage3D(
                    _steel_step(construction_stage),
                    f"{base}_steel_A",
                    (
                        "activate steel girders and stays; apply cable pretension A"
                        if has_cable
                        else "activate steel girders"
                    ),
                    construction_stage,
                    "steel_and_A",
                ),
                ConstructionStage3D(
                    _deck_weight_step(construction_stage),
                    f"{base}_deck_weight_B",
                    (
                        "apply temporary deck self-weight to the new steel girders "
                        "and cable pretension B"
                        if has_cable
                        else "apply temporary deck self-weight to the new steel girders"
                    ),
                    construction_stage,
                    "deck_weight_and_B",
                ),
                ConstructionStage3D(
                    _composite_step(construction_stage),
                    f"{base}_composite",
                    "retire the temporary load definition and activate the rigidly coupled deck slab",
                    construction_stage,
                    "composite",
                ),
            )
        )
    if has_secondary_load:
        stages.append(
            ConstructionStage3D(
                secondary_stage_index,
                "secondary_load",
                "apply main-girder line loads and deck-surface pressure",
                geometry_final_stage + 1,
                "secondary_load",
            )
        )

    metadata = {
        "coordinate_system": "x longitudinal, y transverse, z vertical",
        "analysis_scope": "path-dependent incremental linear staged analysis",
        "station_x": {station.key: station.x for station in stations},
        "cross_girder_x": tuple(stations[index].x for index in cross_station_indices),
        "actual_cross_girder_spacing": actual_cross_spacing,
        "beam_grid_node_ids": tuple(beam_nodes.values()),
        "slab_node_ids": tuple(slab_nodes.values()),
        "deck_grid_y": tuple(deck_grid_y),
        "deck_tributary_widths": deck_tributary_widths,
        "tower_node_ids": tuple(tower_nodes),
        "ground_anchor_ids": tuple(ground_anchor_ids),
        "girder_spacing": config.girder_spacing,
        "deck_width": config.deck_width,
        "deck_offset": config.deck_offset,
        "temporary_deck_line_load_per_main_girder": temporary_slab_line_load,
        "construction_substeps": ("steel_and_A", "deck_weight_and_B", "composite"),
        "secondary_load_stage": secondary_stage_index if has_secondary_load else None,
        "config": config,
    }
    return SingleStagedPlan3D(model=model, stages=stages, metadata=metadata)


__all__ = ["SingleStaged3DConfig", "build_single_staged_3d"]
