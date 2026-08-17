from __future__ import annotations

from math import hypot
from typing import Any, Mapping, Sequence


_BADGE_OFFSETS = (
    (24, -24),
    (-28, -24),
    (26, 26),
    (-30, 26),
    (0, -38),
    (38, 0),
    (0, 38),
    (-40, 0),
    (36, -14),
    (-40, -14),
    (36, 18),
    (-40, 18),
)


def recommendation_badges(
    point_ids: Sequence[str],
    plot_points: Sequence[Mapping[str, Any]],
    *,
    plot_width: int,
    plot_height: int,
) -> dict[str, dict[str, Any]]:
    point_by_id = {point["point_id"]: point for point in plot_points}
    all_positions = [
        (float(point["screen_x"]), float(point["screen_y"]))
        for point in plot_points
    ]
    occupied: list[tuple[float, float]] = []
    badges: dict[str, dict[str, Any]] = {}
    for index, point_id in enumerate(point_ids):
        point = point_by_id.get(point_id)
        if point is None:
            continue
        point_x = float(point["screen_x"])
        point_y = float(point["screen_y"])
        label_x, label_y = _best_position(
            point_x,
            point_y,
            index=index,
            all_positions=all_positions,
            occupied=occupied,
            plot_width=plot_width,
            plot_height=plot_height,
        )
        occupied.append((label_x, label_y))
        badges[point_id] = {
            "number": index + 1,
            "label": str(index + 1),
            "label_x": round(label_x, 2),
            "label_y": round(label_y, 2),
        }
    return badges


def _best_position(
    point_x: float,
    point_y: float,
    *,
    index: int,
    all_positions: Sequence[tuple[float, float]],
    occupied: Sequence[tuple[float, float]],
    plot_width: int,
    plot_height: int,
) -> tuple[float, float]:
    best = (point_x, point_y)
    best_score = float("-inf")
    for offset_index in range(len(_BADGE_OFFSETS)):
        dx, dy = _BADGE_OFFSETS[(index + offset_index) % len(_BADGE_OFFSETS)]
        label_x = _clamp(point_x + dx, 18, plot_width - 18)
        label_y = _clamp(point_y + dy, 18, plot_height - 18)
        nearby_points = [
            position
            for position in all_positions
            if position != (point_x, point_y)
        ]
        score = (
            min(_nearest(label_x, label_y, occupied, 48.0), 48.0) * 2.0
            + min(_nearest(label_x, label_y, nearby_points, 28.0), 32.0)
            + min(hypot(label_x - point_x, label_y - point_y), 42.0) * 0.25
        )
        if score > best_score:
            best_score = score
            best = (label_x, label_y)
    return best


def _nearest(
    x: float,
    y: float,
    positions: Sequence[tuple[float, float]],
    default: float,
) -> float:
    if not positions:
        return default
    return min(hypot(x - other_x, y - other_y) for other_x, other_y in positions)


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
