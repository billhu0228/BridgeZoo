"""Matplotlib stage renderer for the 3D single-tower grillage model.

The renderer consumes the solver-neutral plan/result pair, so OpenSees and the
self-written backend produce identical visual semantics.  It mirrors the 2D
workflow: optional per-stage PNGs plus a staged GIF (or one final PNG).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from bridgezoo.fem.single_staged.model3d import SingleStagedPlan3D, StagedResult3D
from bridgezoo.render.deformed_shape import hermite_frame_shape_3d


_GROUP_STYLE = {
    "main_girder": ("#d55e00", 2.8),
    "cross_girder": ("#e69f00", 1.8),
    "deck_longitudinal": ("#009e73", 1.2),
    "deck_transverse": ("#009e73", 1.2),
    "tower": ("#404040", 4.0),
}

_DXF_LAYERS = {
    "main_girder": ("BZ_MAIN_GIRDER", 1),
    "cross_girder": ("BZ_CROSS_GIRDER", 30),
    "deck_longitudinal": ("BZ_DECK_LONGITUDINAL", 3),
    "deck_transverse": ("BZ_DECK_TRANSVERSE", 3),
    "tower": ("BZ_TOWER", 8),
    "main_stay": ("BZ_MAIN_STAY", 5),
    "backstay": ("BZ_BACKSTAY", 3),
    "rigid_link": ("BZ_RIGID_LINK", 6),
    "deck_panel": ("BZ_DECK_PANEL", 121),
    "node": ("BZ_NODE", 7),
    "other": ("BZ_FRAME_OTHER", 9),
}


def _deformed_point(model, record, node_id: int, scale: float) -> np.ndarray:
    node = model.nodes[node_id]
    return np.asarray(node.xyz, dtype=float) + scale * np.asarray(
        record.displacement[node_id][:3], dtype=float
    )


def _deck_panel_vertices(model, record, scale: float) -> list[list[np.ndarray]]:
    by_x: dict[float, dict[int, int]] = {}
    for node_id in record.displacement:
        node = model.nodes[node_id]
        if not node.role.startswith("deck_slab_"):
            continue
        side = int(node.role.rsplit("_", 1)[1])
        by_x.setdefault(node.x, {})[side] = node_id
    complete = [(x, sides) for x, sides in sorted(by_x.items()) if set(sides) == {0, 1}]
    panels = []
    for (_, left), (_, right) in zip(complete, complete[1:]):
        panels.append(
            [
                _deformed_point(model, record, left[0], scale),
                _deformed_point(model, record, left[1], scale),
                _deformed_point(model, record, right[1], scale),
                _deformed_point(model, record, right[0], scale),
            ]
        )
    return panels


def _axis_limits(plan: SingleStagedPlan3D, result: StagedResult3D, scale: float):
    model = plan.model
    points = [np.asarray(node.xyz, dtype=float) for node in model.nodes.values()]
    for record in result.records:
        points.extend(
            _deformed_point(model, record, node_id, scale)
            for node_id in record.displacement
        )
    coordinates = np.vstack(points)
    low = coordinates.min(axis=0)
    high = coordinates.max(axis=0)
    ranges = np.maximum(high - low, 1.0)
    margin = np.array((0.04, 0.12, 0.08)) * ranges
    return low - margin, high + margin


def export_final_3d_dxf(
    plan: SingleStagedPlan3D,
    result: StagedResult3D,
    out: str | Path,
) -> Path:
    """Export the final active 3D topology as a meter-unit DXF.

    Coordinates are the true, undeformed solver-neutral model coordinates.
    The plot displacement scale is deliberately not applied, so CAD distance
    measurements remain physical.  Only objects active in the final result
    record are emitted.
    """

    if not result.records:
        raise ValueError("cannot export DXF from an empty 3D staged result")

    import ezdxf
    from ezdxf import units

    output_path = Path(out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = ezdxf.new("R2010")
    document.units = units.M
    document.header["$MEASUREMENT"] = 1
    modelspace = document.modelspace()
    for layer_name, color in _DXF_LAYERS.values():
        if layer_name not in document.layers:
            document.layers.add(layer_name, color=color)

    model = plan.model
    final = result.final
    active_node_ids = set(final.displacement)
    for node_id in sorted(active_node_ids):
        node = model.nodes[node_id]
        modelspace.add_point(
            node.xyz,
            dxfattribs={"layer": _DXF_LAYERS["node"][0]},
        )

    for frame in model.frames.values():
        if (
            frame.activation_stage > final.stage_index
            or frame.i not in active_node_ids
            or frame.j not in active_node_ids
        ):
            continue
        layer = _DXF_LAYERS.get(frame.group, _DXF_LAYERS["other"])[0]
        modelspace.add_line(
            model.nodes[frame.i].xyz,
            model.nodes[frame.j].xyz,
            dxfattribs={"layer": layer},
        )

    for cable_id in final.cable_force:
        cable = model.cables[cable_id]
        layer = _DXF_LAYERS.get(cable.group, _DXF_LAYERS["other"])[0]
        modelspace.add_line(
            model.nodes[cable.i].xyz,
            model.nodes[cable.j].xyz,
            dxfattribs={"layer": layer},
        )

    for link in model.rigid_links.values():
        if (
            link.activation_stage > final.stage_index
            or link.master not in active_node_ids
            or link.slave not in active_node_ids
        ):
            continue
        modelspace.add_line(
            model.nodes[link.master].xyz,
            model.nodes[link.slave].xyz,
            dxfattribs={"layer": _DXF_LAYERS["rigid_link"][0]},
        )

    for panel in _deck_panel_vertices(model, final, scale=0.0):
        modelspace.add_3dface(
            [tuple(float(value) for value in vertex) for vertex in panel],
            dxfattribs={"layer": _DXF_LAYERS["deck_panel"][0]},
        )

    document.saveas(output_path)
    return output_path


def render_staged_3d(
    plan: SingleStagedPlan3D,
    result: StagedResult3D,
    *,
    scale: float = 10.0,
    out: str | Path | None = None,
    frames_dir: str | Path | None = None,
    fps: int = 1,
    elevation: float = 22.0,
    azimuth: float = -62.0,
    dxf_out: str | Path | None = None,
) -> dict[str, object]:
    """Render staged graphics and export the final active topology to DXF."""

    if not result.records:
        raise ValueError("cannot render an empty 3D staged result")
    if scale < 0.0:
        raise ValueError("render displacement scale must be nonnegative")
    if fps < 1:
        raise ValueError("render fps must be at least 1")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    from bridgezoo.render.mpl_cjk import use_cjk_font

    use_cjk_font()
    model = plan.model
    output_path = Path(out) if out is not None else None
    frame_path = Path(frames_dir) if frames_dir is not None else None
    dxf_path = (
        Path(dxf_out)
        if dxf_out is not None
        else output_path.with_suffix(".dxf") if output_path is not None else None
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    if frame_path is not None:
        frame_path.mkdir(parents=True, exist_ok=True)
        for stale in frame_path.glob("stage_*.png"):
            stale.unlink()

    low, high = _axis_limits(plan, result, scale)
    axis_ranges = np.maximum(high - low, 1.0)
    figure = plt.figure(figsize=(14.0, 8.5))
    axis = figure.add_subplot(111, projection="3d")

    legend = [
        Line2D([0], [0], color="0.72", lw=1.0, ls="--", label="未变形构件"),
        Line2D([0], [0], color="#d55e00", lw=2.8, label="主梁"),
        Line2D([0], [0], color="#e69f00", lw=1.8, label="横梁"),
        Patch(facecolor="#56b4a9", alpha=0.18, label="桥面板"),
        Line2D([0], [0], color="#404040", lw=4.0, label="空心箱塔轴线"),
        Line2D([0], [0], color="#4f8fba", lw=1.2, label="主跨索"),
        Line2D([0], [0], color="#6a9f58", lw=1.2, label="背索"),
        Line2D([0], [0], color="#b2182b", lw=4.0, label="本阶段新增"),
    ]

    def draw(record_index: int):
        record = result.records[record_index]
        axis.clear()
        active_frames = [
            frame
            for frame in model.frames.values()
            if frame.activation_stage <= record.stage_index
        ]

        # Undeformed reference grid and tower.
        for frame in active_frames:
            node_i, node_j = model.nodes[frame.i], model.nodes[frame.j]
            axis.plot(
                [node_i.x, node_j.x],
                [node_i.y, node_j.y],
                [node_i.z, node_j.z],
                color="0.72",
                lw=0.8,
                ls="--",
                alpha=0.45,
            )

        panels = _deck_panel_vertices(model, record, scale)
        if panels:
            axis.add_collection3d(
                Poly3DCollection(
                    panels,
                    facecolors="#56b4a9",
                    edgecolors="#25887f",
                    linewidths=0.25,
                    alpha=0.18,
                )
            )

        for frame in active_frames:
            curve = hermite_frame_shape_3d(
                model.nodes[frame.i].xyz,
                model.nodes[frame.j].xyz,
                record.displacement[frame.i],
                record.displacement[frame.j],
                frame.orientation,
                scale=scale,
            )
            color, width = _GROUP_STYLE.get(frame.group, ("#555555", 1.2))
            if frame.activation_stage == record.stage_index:
                color, width = "#b2182b", max(width, 3.5)
            axis.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=color, lw=width)

        for cable_id in record.cable_force:
            cable = model.cables[cable_id]
            point_i = _deformed_point(model, record, cable.i, scale)
            point_j = _deformed_point(model, record, cable.j, scale)
            color = "#6a9f58" if cable.group == "backstay" else "#4f8fba"
            width = 2.6 if cable.activation_stage == record.stage_index else 1.2
            axis.plot(
                [point_i[0], point_j[0]],
                [point_i[1], point_j[1]],
                [point_i[2], point_j[2]],
                color="#b2182b" if cable.activation_stage == record.stage_index else color,
                lw=width,
                alpha=0.86,
            )

        for link in model.rigid_links.values():
            if link.activation_stage > record.stage_index:
                continue
            master = _deformed_point(model, record, link.master, scale)
            slave = _deformed_point(model, record, link.slave, scale)
            axis.plot(
                [master[0], slave[0]],
                [master[1], slave[1]],
                [master[2], slave[2]],
                color="#8c6bb1",
                lw=0.7,
                alpha=0.45,
            )

        support_points = [
            _deformed_point(model, record, support.node, scale)
            for support in model.supports.values()
            if support.activation_stage <= record.stage_index
            and support.node in record.displacement
        ]
        if support_points:
            supports = np.vstack(support_points)
            axis.scatter(
                supports[:, 0],
                supports[:, 1],
                supports[:, 2],
                marker="^",
                s=24,
                color="#7b3294",
                depthshade=False,
            )

        deck_uz = [
            values[2]
            for node_id, values in record.displacement.items()
            if model.nodes[node_id].role.startswith("main_girder")
        ]
        max_uz_mm = max((abs(value) for value in deck_uz), default=0.0) * 1000.0
        axis.set_title(
            f"OMO 单塔 3D 逐阶段施工：{record.stage_label}   "
            f"backend={result.backend}   max|uz|={max_uz_mm:.1f} mm   位移×{scale:g}"
        )
        axis.set_xlabel("纵桥向 x [m]")
        axis.set_ylabel("横桥向 y [m]")
        axis.set_zlabel("竖向 z [m]")
        axis.set_xlim(low[0], high[0])
        axis.set_ylim(low[1], high[1])
        axis.set_zlim(low[2], high[2])
        axis.set_box_aspect(axis_ranges)
        axis.view_init(elev=elevation, azim=azimuth)
        axis.grid(True, alpha=0.25)
        axis.legend(handles=legend, loc="upper left", fontsize=8, ncol=2)
        figure.tight_layout()
        return axis.lines

    written_frames: list[Path] = []
    if frame_path is not None:
        for index, record in enumerate(result.records):
            draw(index)
            safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.stage_label)
            path = frame_path / f"stage_{index + 1:02d}_{safe_label}.png"
            figure.savefig(path, dpi=140)
            written_frames.append(path)

    if output_path is not None:
        if output_path.suffix.lower() == ".gif":
            animation = FuncAnimation(
                figure,
                draw,
                frames=len(result.records),
                interval=1000 / fps,
                blit=False,
            )
            animation.save(output_path, writer=PillowWriter(fps=fps), dpi=125)
        else:
            draw(len(result.records) - 1)
            figure.savefig(output_path, dpi=150)

    plt.close(figure)
    written_dxf = (
        export_final_3d_dxf(plan, result, dxf_path)
        if dxf_path is not None
        else None
    )
    return {
        "output": output_path,
        "frames": tuple(written_frames),
        "dxf": written_dxf,
    }


__all__ = ["export_final_3d_dxf", "render_staged_3d"]
